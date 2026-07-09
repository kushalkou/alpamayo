"""
finetune.py — Alpamayo VLA Training (DDP version)

Changes vs previous version:
  - DistributedDataParallel via torchrun (symmetric all-reduce, no primary GPU OOM)
  - lr=5e-5 (was 2e-4, too aggressive)
  - weight_decay=0.05 (was 0.01)
  - FlashAttention2 + gradient checkpointing (in model.py)
  - Ego noise augmentation (in model.py)
  - Early stopping patience=3
  - DistributedSampler for proper data distribution across GPUs

Launch (3 GPUs):
    cd /home/drive1/Alpamayo
    PYTHONPATH=data:tokenization:training:/home/drive1/python_packages \
    torchrun --nproc_per_node=3 training/finetune.py

Launch (2 GPUs):
    torchrun --nproc_per_node=2 training/finetune.py

Launch (1 GPU, debug):
    torchrun --nproc_per_node=1 training/finetune.py

Resume:
    torchrun --nproc_per_node=3 training/finetune.py --resume
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

sys.path.insert(0, '/home/drive1/Alpamayo/data')
sys.path.insert(0, '/home/drive1/Alpamayo/tokenization')
sys.path.insert(0, '/home/drive1/Alpamayo/training')

from dataset import get_dataloaders, get_class_weights, NuScenesVLADataset, build_scene_split
from model import AlpamayoVLA, load_model, TRAJ_VOCAB, TRAJ_LEN
import pickle

# ── Config ────────────────────────────────────────────────────────────────────

CFG = {
    'trajectories_path': '/home/drive1/Alpamayo/data/trajectories_full.pkl',
    'nuscenes_root':     '/home/drive1/Alpamayo/nuscenes_full',
    'checkpoint_dir':    '/home/drive1/Alpamayo/models/checkpoints',
    'cosmos_path':       '/home/drive1/Alpamayo/models/cosmos_reason',

    # Training
    'epochs':            15,
    'batch_size':        1,           # per GPU
    'grad_accum_steps':  8,           # effective batch = 1 * n_gpus * 8
    'num_workers':       4,
    'grad_clip':         1.0,
    'seed':              42,
    'patience':          7,

    # LR — reduced from 2e-4 to 5e-5
    'lr':                5e-5,
    'warmup_steps':      100,
    'min_lr_ratio':      0.1,

    # Regularization
    'weight_decay':      0.05,        # increased from 0.01

    # LoRA
    'lora_rank':         16,
    'lora_alpha':        32,
    'lora_dropout':      0.1,

    # Logging
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

# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in val_loader:
        visual_tokens = batch['visual_tokens'].to(device)
        ego_state     = batch['ego_state'].to(device, dtype=torch.bfloat16)
        traj_tokens   = batch['traj_tokens'].to(device)
        logits = model(visual_tokens, ego_state, traj_tokens)
        loss   = criterion(logits.reshape(-1, TRAJ_VOCAB).float(), traj_tokens.reshape(-1))
        total_loss += loss.item()
        n += 1

    # Average across all DDP ranks
    total_tensor = torch.tensor([total_loss, float(n)], device=device)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    model.train()
    return (total_tensor[0] / total_tensor[1]).item()

# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg, resume=False):
    local_rank = setup_ddp()
    device     = f'cuda:{local_rank}'
    torch.manual_seed(cfg['seed'] + local_rank)

    # Model on this GPU
    model = load_model(
        cosmos_path  = cfg['cosmos_path'],
        lora_rank    = cfg['lora_rank'],
        lora_alpha   = cfg['lora_alpha'],
        lora_dropout = cfg['lora_dropout'],
        device       = device,
    )

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    model.train()

    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    if is_main():
        print(f"[train] Trainable params: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    # Data — build scene split once on main rank, share via broadcast
    if is_main():
        print("[train] Building scene split...")
    with open(cfg['trajectories_path'], 'rb') as f:
        all_trajs = pickle.load(f)
    train_trajs, val_trajs, _ = build_scene_split(all_trajs, cfg['nuscenes_root'])

    train_dataset = NuScenesVLADataset(train_trajs, split='train')
    val_dataset   = NuScenesVLADataset(val_trajs,   split='val')

    # DistributedSampler ensures each GPU gets different data
    train_sampler = DistributedSampler(train_dataset, shuffle=True,
                                       seed=cfg['seed'])
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['batch_size'],
        sampler=train_sampler, num_workers=cfg['num_workers'],
        pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg['batch_size'],
        sampler=val_sampler, num_workers=cfg['num_workers'],
        pin_memory=True,
    )

    # Class weights (compute on train set, same across ranks)
    # Compute class weights on rank 0 only, broadcast to all ranks
    if local_rank == 0:
        class_weights = get_class_weights(train_dataset, device=device)
    else:
        class_weights = torch.zeros(TRAJ_VOCAB, device=device)
    dist.broadcast(class_weights, src=0)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    steps_per_epoch     = len(train_loader)
    effective_per_epoch = steps_per_epoch // cfg['grad_accum_steps']
    world_size          = dist.get_world_size()
    total_eff_steps     = effective_per_epoch * cfg['epochs']

    if is_main():
        print(f"[train] World size: {world_size} GPUs")
        print(f"[train] {steps_per_epoch} steps/epoch/GPU, "
              f"{effective_per_epoch} effective/epoch, "
              f"effective batch = {cfg['batch_size'] * world_size * cfg['grad_accum_steps']}")

    start_epoch = 0
    eff_step    = 0
    best_val    = float('inf')
    no_improve  = 0

    if resume:
        start_epoch, eff_step, best_val = load_checkpoint(model, optimizer, cfg)

    if is_main():
        print(f"\n[train] Starting from epoch {start_epoch}...\n")

    for epoch in range(start_epoch, cfg['epochs']):
        train_sampler.set_epoch(epoch)   # ensures different shuffle per epoch
        lr = get_lr(eff_step, total_eff_steps, cfg)

        if is_main():
            print(f"── Epoch {epoch+1}/{cfg['epochs']}  |  lr={lr:.2e}  |  "
                  f"no_improve={no_improve}/{cfg['patience']}")

        epoch_loss = 0.0
        t_start    = time.time()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            visual_tokens = batch['visual_tokens'].to(device)
            ego_state     = batch['ego_state'].to(device, dtype=torch.bfloat16)
            traj_tokens   = batch['traj_tokens'].to(device)

            logits = model(visual_tokens, ego_state, traj_tokens)
            loss   = criterion(logits.reshape(-1, TRAJ_VOCAB).float(), traj_tokens.reshape(-1))

            (loss / cfg['grad_accum_steps']).backward()
            epoch_loss += loss.item()

            if (batch_idx + 1) % cfg['grad_accum_steps'] == 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                lr = get_lr(eff_step, total_eff_steps, cfg)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()
                eff_step += 1

                if is_main() and eff_step % cfg['log_every'] == 0:
                    avg = epoch_loss / (batch_idx + 1)
                    print(f"  eff_step {eff_step:5d} | loss {loss.item():.4f} | "
                          f"avg {avg:.4f} | lr {lr:.2e} | "
                          f"{(time.time()-t_start)/60:.1f}min")

        # End of epoch
        avg_train = epoch_loss / steps_per_epoch
        if is_main():
            print(f"\n[epoch {epoch+1}] train_loss={avg_train:.4f} — validating...")

        val_loss = validate(model, val_loader, criterion, device)

        if is_main():
            gap = val_loss - avg_train
            print(f"[epoch {epoch+1}] val_loss={val_loss:.4f}  "
                  f"train_loss={avg_train:.4f}  gap={gap:.4f}  best={best_val:.4f}")

            if val_loss < best_val:
                best_val   = val_loss
                no_improve = 0
                save_checkpoint(model, optimizer, epoch+1, eff_step, val_loss, cfg, 'best')
            else:
                no_improve += 1
                print(f"[epoch {epoch+1}] No improvement ({no_improve}/{cfg['patience']})")

            save_checkpoint(model, optimizer, epoch+1, eff_step, val_loss, cfg, 'latest')

        # Broadcast no_improve to all ranks for early stopping
        no_improve_tensor = torch.tensor(no_improve, device=device)
        dist.broadcast(no_improve_tensor, src=0)
        no_improve = no_improve_tensor.item()

        if no_improve >= cfg['patience']:
            if is_main():
                print(f"\n[train] Early stopping.")
            break

        if is_main():
            print()

    if is_main():
        print(f"[train] Done. Best val loss: {best_val:.4f}")

    cleanup_ddp()

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--epochs', type=int, default=CFG['epochs'])
    parser.add_argument('--lr',     type=float, default=CFG['lr'])
    args = parser.parse_args()

    CFG['epochs'] = args.epochs
    CFG['lr']     = args.lr

    train(CFG, resume=args.resume)