"""
finetune.py — Alpamayo VLA Training (DGX / V100 DDP + live vision)

Changes vs the RTX-workstation version:
  - LIVE vision: the 6 images are run through the FROZEN encoder every step
    (vision_live.encode_batch, no_grad) instead of loading a token cache.
  - torch.amp.GradScaler for stable fp16 mixed precision (fp32 adapter masters).
  - Live encode in BOTH train and validate.
  - Smoke / gate args: --batch_size --grad_accum_steps --max_steps --augment
    --find_unused (+ --overfit_one_batch for Gate 3).
  - Per-step peak-VRAM and sec/step reporting.
  - Paths repointed drive1 -> DGX. No bf16, no flash-attn (see model.py).

Launch (8-GPU DDP, Gate 4 smoke):
    cd /home/dgx1user/Alpamayo-Kushal/code
    python -m torch.distributed.run --nproc_per_node=8 finetune.py \
        --batch_size 2 --grad_accum_steps 1 --max_steps 60 --augment

Launch (single-GPU Gate 3, overfit one batch, aug off):
    python -m torch.distributed.run --nproc_per_node=1 finetune.py \
        --batch_size 2 --grad_accum_steps 1 --max_steps 200 --overfit_one_batch
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_DIR))

from dataset import (get_class_weights, NuScenesVLADataset, build_scene_split)
from model import AlpamayoVLA, load_model, TRAJ_VOCAB, TRAJ_LEN
from vision_live import encode_batch
import pickle

# ── Config ────────────────────────────────────────────────────────────────────

CFG = {
    'trajectories_path': '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/trajectories_full.pkl',
    'nuscenes_root':     '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes',
    'checkpoint_dir':    '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints',
    'cosmos_path':       '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason',

    # Training
    'epochs':            15,
    'batch_size':        1,           # per GPU
    'grad_accum_steps':  8,
    'num_workers':       4,
    'grad_clip':         1.0,
    'seed':              42,
    'patience':          7,

    'lr':                5e-5,
    'warmup_steps':      100,
    'min_lr_ratio':      0.1,         # cosine floor = 5e-6

    'weight_decay':      0.05,

    'lora_rank':         16,
    'lora_alpha':        32,
    'lora_dropout':      0.1,

    'log_every':         50,
}

# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main():
    return int(os.environ.get('LOCAL_RANK', 0)) == 0

# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step, total_steps, cfg):
    peak   = cfg['lr']
    min_lr = peak * cfg['min_lr_ratio']
    warmup = cfg['warmup_steps']
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return min_lr + (peak - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, step, val_loss, cfg, tag='best'):
    if not is_main():
        return
    ckpt_dir = Path(cfg['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f'alpamayo_{tag}.pt'

    raw = model.module if isinstance(model, DDP) else model
    trainable_state = {
        k: v for k, v in raw.state_dict().items()
        if 'lora_' in k
        or any(k.startswith(m) for m in ['ego_encoder', 'traj_embed', 'output_head'])
    }
    torch.save({
        'epoch': epoch, 'step': step, 'val_loss': val_loss,
        'model_state': trainable_state,
        'optimizer_state': optimizer.state_dict(),
        'cfg': cfg,
    }, path)
    print(f"[ckpt] Saved {tag} → {path}  (val_loss={val_loss:.4f})")


def load_checkpoint(model, optimizer, cfg, tag='latest'):
    path = Path(cfg['checkpoint_dir']) / f'alpamayo_{tag}.pt'
    if not path.exists():
        if is_main():
            print(f"[ckpt] No checkpoint at {path}, starting fresh.")
        return 0, 0, float('inf')
    ckpt = torch.load(path, map_location='cpu')
    raw  = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(ckpt['model_state'], strict=False)
    optimizer.load_state_dict(ckpt['optimizer_state'])
    if is_main():
        print(f"[ckpt] Resumed from {path}  epoch={ckpt['epoch']} val_loss={ckpt['val_loss']:.4f}")
    return ckpt['epoch'], ckpt['step'], ckpt['val_loss']

# ── Live encode helper ────────────────────────────────────────────────────────

def encode(model, batch, device):
    """Run the FROZEN encoder live on this batch's images -> fp16 visual tokens."""
    raw    = model.module if isinstance(model, DDP) else model
    visual = raw.cosmos.model.visual
    images = batch['images'].to(device, non_blocking=True)
    with torch.no_grad():
        return encode_batch(visual, images)          # [B, 1536, 3584] fp16

# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in val_loader:
        visual_tokens = encode(model, batch, device)
        ego_state     = batch['ego_state'].to(device)          # fp32
        traj_tokens   = batch['traj_tokens'].to(device)
        logits = model(visual_tokens, ego_state, traj_tokens)
        loss   = criterion(logits.reshape(-1, TRAJ_VOCAB), traj_tokens.reshape(-1))
        total_loss += loss.item()
        n += 1

    total_tensor = torch.tensor([total_loss, float(n)], device=device)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    model.train()
    return (total_tensor[0] / total_tensor[1]).item()

# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg, resume=False, max_steps=None, augment=False,
          find_unused=False, overfit_one_batch=False):
    local_rank = setup_ddp()
    device     = f'cuda:{local_rank}'
    torch.manual_seed(cfg['seed'] + local_rank)
    torch.cuda.reset_peak_memory_stats(device)

    model = load_model(
        cosmos_path  = cfg['cosmos_path'],
        lora_rank    = cfg['lora_rank'],
        lora_alpha   = cfg['lora_alpha'],
        lora_dropout = cfg['lora_dropout'],
        device       = device,
    )

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if is_main():
        print(f"[train] Trainable params: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler    = torch.amp.GradScaler('cuda')

    # Data — build scene split (also resolves the 6 cam paths per sample)
    with open(cfg['trajectories_path'], 'rb') as f:
        all_trajs = pickle.load(f)
    train_trajs, val_trajs, _ = build_scene_split(all_trajs, cfg['nuscenes_root'])

    train_dataset = NuScenesVLADataset(train_trajs, split='train', augment=augment)
    val_dataset   = NuScenesVLADataset(val_trajs,   split='val',   augment=False)

    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=cfg['seed'])
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['batch_size'], sampler=train_sampler,
        num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg['batch_size'], sampler=val_sampler,
        num_workers=cfg['num_workers'], pin_memory=True)

    # Class weights (rank 0, broadcast)
    if local_rank == 0:
        class_weights = get_class_weights(train_dataset, device=device)
    else:
        class_weights = torch.zeros(TRAJ_VOCAB, device=device)
    dist.broadcast(class_weights, src=0)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    steps_per_epoch     = len(train_loader)
    effective_per_epoch = max(steps_per_epoch // cfg['grad_accum_steps'], 1)
    world_size          = dist.get_world_size()
    total_eff_steps     = effective_per_epoch * cfg['epochs']

    if is_main():
        print(f"[train] World size: {world_size} GPUs")
        print(f"[train] {steps_per_epoch} steps/epoch/GPU, {effective_per_epoch} effective/epoch, "
              f"effective batch = {cfg['batch_size'] * world_size * cfg['grad_accum_steps']}")
        if max_steps:
            print(f"[train] max_steps={max_steps} (smoke/gate mode)")
        if overfit_one_batch:
            print("[train] OVERFIT-ONE-BATCH mode (Gate 3)")

    start_epoch = 0
    eff_step    = 0
    best_val    = float('inf')
    no_improve  = 0
    fixed_batch = None

    if resume:
        start_epoch, eff_step, best_val = load_checkpoint(model, optimizer, cfg)

    stop = False
    for epoch in range(start_epoch, cfg['epochs']):
        train_sampler.set_epoch(epoch)
        if is_main():
            print(f"── Epoch {epoch+1}/{cfg['epochs']}  |  no_improve={no_improve}/{cfg['patience']}")

        epoch_loss = 0.0
        optimizer.zero_grad()
        t_step = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if overfit_one_batch:
                if fixed_batch is None:
                    fixed_batch = batch
                batch = fixed_batch

            visual_tokens = encode(model, batch, device)
            ego_state     = batch['ego_state'].to(device)          # fp32
            traj_tokens   = batch['traj_tokens'].to(device)

            logits = model(visual_tokens, ego_state, traj_tokens)
            loss   = criterion(logits.reshape(-1, TRAJ_VOCAB), traj_tokens.reshape(-1))

            scaler.scale(loss / cfg['grad_accum_steps']).backward()
            epoch_loss += loss.item()

            if (batch_idx + 1) % cfg['grad_accum_steps'] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, cfg['grad_clip'])
                lr = get_lr(eff_step, total_eff_steps, cfg)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                eff_step += 1

                if is_main():
                    dt        = time.time() - t_step
                    peak_gb   = torch.cuda.max_memory_allocated(device) / 1e9
                    n_nonfin  = int((~torch.isfinite(loss)).sum().item())
                    print(f"  eff_step {eff_step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                          f"{dt:.2f}s/step | peakVRAM {peak_gb:.1f}G | nonfinite {n_nonfin}")
                t_step = time.time()

                if max_steps and eff_step >= max_steps:
                    stop = True
                    break

        if stop or overfit_one_batch:
            # smoke/gate/overfit runs don't do the full validate/early-stop cycle
            if not stop:
                stop = (max_steps is not None and eff_step >= max_steps)
            if stop:
                break

        # Full training path: validate + checkpoint + early stopping
        avg_train = epoch_loss / max(steps_per_epoch, 1)
        if is_main():
            print(f"\n[epoch {epoch+1}] train_loss={avg_train:.4f} — validating...")
        val_loss = validate(model, val_loader, criterion, device)

        if is_main():
            gap = val_loss - avg_train
            print(f"[epoch {epoch+1}] val_loss={val_loss:.4f}  train_loss={avg_train:.4f}  "
                  f"gap={gap:.4f}  best={best_val:.4f}")
            if val_loss < best_val:
                best_val, no_improve = val_loss, 0
                save_checkpoint(model, optimizer, epoch+1, eff_step, val_loss, cfg, 'best')
            else:
                no_improve += 1
                print(f"[epoch {epoch+1}] No improvement ({no_improve}/{cfg['patience']})")
            save_checkpoint(model, optimizer, epoch+1, eff_step, val_loss, cfg, 'latest')

        no_improve_tensor = torch.tensor(no_improve, device=device)
        dist.broadcast(no_improve_tensor, src=0)
        no_improve = no_improve_tensor.item()
        if no_improve >= cfg['patience']:
            if is_main():
                print("\n[train] Early stopping.")
            break

    if is_main():
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[train] Done. best_val={best_val:.4f}  final_eff_step={eff_step}  peakVRAM={peak_gb:.1f}G")

    cleanup_ddp()

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--epochs', type=int, default=CFG['epochs'])
    parser.add_argument('--lr',     type=float, default=CFG['lr'])
    parser.add_argument('--batch_size',      type=int, default=CFG['batch_size'])
    parser.add_argument('--grad_accum_steps', type=int, default=CFG['grad_accum_steps'])
    parser.add_argument('--max_steps',       type=int, default=None)
    parser.add_argument('--augment',         action='store_true')
    parser.add_argument('--find_unused',     action='store_true')
    parser.add_argument('--overfit_one_batch', action='store_true')
    args = parser.parse_args()

    CFG['epochs']           = args.epochs
    CFG['lr']               = args.lr
    CFG['batch_size']       = args.batch_size
    CFG['grad_accum_steps'] = args.grad_accum_steps

    train(CFG, resume=args.resume, max_steps=args.max_steps, augment=args.augment,
          find_unused=args.find_unused, overfit_one_batch=args.overfit_one_batch)
