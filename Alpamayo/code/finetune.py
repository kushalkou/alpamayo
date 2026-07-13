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

Launch (8 GPUs, full):
    cd /home/dgx1user/Alpamayo-Kushal/Alpamayo/code
    python -m torch.distributed.run --nproc_per_node=8 finetune.py

Gate-4 smoke (8 GPU, batch=2/GPU, aug ON, no grad-accum, 60 steps):
    cd /home/dgx1user/Alpamayo-Kushal/Alpamayo/code
    python -m torch.distributed.run --nproc_per_node=8 finetune.py \
        --batch_size 2 --grad_accum_steps 1 --max_steps 60 --augment

Resume:
    python -m torch.distributed.run --nproc_per_node=8 finetune.py --resume
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')

from dataset import get_dataloaders, get_class_weights, NuScenesVLADataset, build_scene_split
from model import AlpamayoVLA, load_model, TRAJ_VOCAB, TRAJ_LEN
from vision_live import encode_normalized_images
from ar_eval import compute_val_ade, fixed_val_indices
from tokenizer import TrajectoryTokenizer
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
    'grad_accum_steps':  8,           # effective batch = 1 * n_gpus * 8
    'num_workers':       4,
    'augment':           False,       # image-space photometric aug (dataloader)
    'max_steps':         0,           # >0 => smoke mode: stop after N optimizer steps
    'find_unused':       False,       # set True only if DDP hangs on unused params
    'zero_vision':       False,       # ablation: feed zeros for the 1536 visual tokens (ego-only)
    'zero_ego':          False,       # ablation: feed zeros for the 4 ego tokens (vision-only)
    # P3: position-weighted loss. Concentrate gradient on the perception-dependent
    # accel_1..11 slots (weight 1.0); down-weight the trivially-solvable curv slots
    # and accel_0 (weight 0.2). Multiplied with sqrt inv-freq class weights, not replaced.
    'pos_weighted':      False,
    'pos_w_high':        1.0,         # accel_1..11
    'pos_w_low':         0.2,         # accel_0 + curv_0..11
    # P4: scheduled sampling. Feed model's own predicted token w.p. p, ramped 0->ss_max
    # linearly over the run. Breaks the free copy/persistence shortcut.
    'scheduled_sampling': False,
    'ss_max':            0.25,
    'grad_clip':         1.0,
    'seed':              42,
    'patience':          7,

    # Model selection: autoregressive val-ADE (teacher-forced val loss is DISQUALIFIED).
    # After each epoch, AR-decode a seed-fixed val subset and select on median ADE@6s.
    'val_ade_k':         400,         # # val samples AR-decoded each epoch for selection
    'val_ade_seed':      1234,        # fixed => comparable subset across runs

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

def save_checkpoint(model, optimizer, epoch, step, val_loss, cfg, tag='best',
                    sel_ade=None):
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
        'epoch': epoch, 'step': step,
        'val_loss': val_loss,        # teacher-forced (record only)
        'val_ade6': sel_ade,         # AR median ADE@6s — the SELECTION metric
        'model_state': trainable_state,
        'optimizer_state': optimizer.state_dict(),
        'cfg': cfg,
    }, path)
    ade_str = f", val_ade6={sel_ade:.4f}" if sel_ade is not None else ""
    print(f"[ckpt] Saved {tag} → {path}  (val_loss={val_loss:.4f}{ade_str})")


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
def encode_live(visual, images, device):
    """images [B,6,3,448,448] -> visual_tokens [B,1536,3584] via the frozen encoder."""
    B = images.shape[0]
    flat = images.reshape(B * 6, *images.shape[2:]).to(device)   # [B*6,3,448,448]
    pooled = encode_normalized_images(visual, flat)              # [B*6*256, 3584]
    return pooled.reshape(B, 6 * 256, -1)                        # [B,1536,3584]


@torch.no_grad()
def validate(model, val_loader, criterion, device, visual, zero_vision=False):
    model.eval()
    total_loss, n = 0.0, 0
    tok_correct, tok_total = 0, 0
    for batch in val_loader:
        visual_tokens = encode_live(visual, batch['images'], device)
        if zero_vision:
            visual_tokens = torch.zeros_like(visual_tokens)
        ego_state     = batch['ego_state'].to(device, dtype=torch.float32)
        traj_tokens   = batch['traj_tokens'].to(device)
        logits = model(visual_tokens, ego_state, traj_tokens)
        loss   = criterion(logits.reshape(-1, TRAJ_VOCAB).float(), traj_tokens.reshape(-1))
        total_loss += loss.item()
        n += 1
        # Observability only (no grad, eval mode): per-token argmax accuracy.
        preds = logits.argmax(dim=-1)                     # [B, 24]
        tok_correct += (preds == traj_tokens).sum().item()
        tok_total   += traj_tokens.numel()

    # Average across all DDP ranks
    stats = torch.tensor([total_loss, float(n), float(tok_correct), float(tok_total)],
                         device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    model.train()
    val_loss = (stats[0] / stats[1]).item()
    tok_acc  = (stats[2] / stats[3]).item() if stats[3] > 0 else 0.0
    return val_loss, tok_acc

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

    # Ablation: zero the 4 ego tokens post-encoder (vision-only mirror of --zero_vision)
    model.zero_ego = cfg['zero_ego']

    # Frozen visual tower handle for live encoding (runs under no_grad, not in DDP graph)
    visual = model.cosmos.model.visual

    # GradScaler — mandatory for the fp16-backbone / fp32-adapter recipe on V100
    scaler = torch.amp.GradScaler('cuda')

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank],
                find_unused_parameters=cfg['find_unused'])
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

    train_dataset = NuScenesVLADataset(train_trajs, split='train', augment=cfg['augment'])
    val_dataset   = NuScenesVLADataset(val_trajs,   split='val',   augment=False)

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

    # P3 position weights [24]: accel_0=low, accel_1..11=high, curv_0..11=low.
    pos_weights = torch.full((TRAJ_LEN,), cfg['pos_w_low'], device=device)
    pos_weights[1:12] = cfg['pos_w_high']       # accel_1..accel_11
    if is_main() and cfg['pos_weighted']:
        print(f"[loss] POSITION-WEIGHTED: accel_1..11={cfg['pos_w_high']}, "
              f"accel_0+curv={cfg['pos_w_low']} (x sqrt inv-freq class weights)")

    def compute_loss(logits, traj_tokens):
        """Weighted-mean CE. Extends CrossEntropyLoss(weight=class_weights) with
        per-position weights when cfg['pos_weighted']; identical to plain criterion
        otherwise."""
        if not cfg['pos_weighted']:
            return criterion(logits.reshape(-1, TRAJ_VOCAB).float(),
                             traj_tokens.reshape(-1))
        B = traj_tokens.shape[0]
        tgt = traj_tokens.reshape(-1)                                   # [B*24]
        ce  = F.cross_entropy(logits.reshape(-1, TRAJ_VOCAB).float(), tgt,
                              weight=class_weights, reduction='none')   # cw[tgt]*CE
        pw  = pos_weights.unsqueeze(0).expand(B, -1).reshape(-1)        # [B*24]
        num = (ce * pw).sum()
        den = (class_weights[tgt] * pw).sum().clamp_min(1e-6)
        return num / den

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
    best_val    = float('inf')      # best teacher-forced val loss (record only)
    best_ade    = float('inf')      # best AR val median ADE@6s (SELECTION metric)
    no_improve  = 0

    # Selection infra: seed-fixed val subset + tokenizer (shared across all ranks)
    val_ade_idx  = fixed_val_indices(len(val_dataset), k=cfg['val_ade_k'],
                                     seed=cfg['val_ade_seed'])
    ar_tokenizer = TrajectoryTokenizer()
    if is_main():
        print(f"[select] Metric = AR median ADE@6s on {len(val_ade_idx)} fixed val samples "
              f"(seed {cfg['val_ade_seed']}). Teacher-forced val loss logged for record only.")

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
        torch.cuda.reset_peak_memory_stats(device)
        t_step     = time.time()

        for batch_idx, batch in enumerate(train_loader):
            visual_tokens = encode_live(visual, batch['images'], device)   # frozen, no_grad
            if cfg['zero_vision']:
                visual_tokens = torch.zeros_like(visual_tokens)            # ego-only ablation
            ego_state     = batch['ego_state'].to(device, dtype=torch.float32)
            traj_tokens   = batch['traj_tokens'].to(device)

            if cfg['scheduled_sampling']:
                ss_p = cfg['ss_max'] * min(1.0, eff_step / max(total_eff_steps, 1))
                (model.module if isinstance(model, DDP) else model).scheduled_sampling_p = ss_p

            logits = model(visual_tokens, ego_state, traj_tokens)
            loss   = compute_loss(logits, traj_tokens)

            # GradScaler: fp16 backbone forward, fp32 adapter grads, scaled loss
            scaler.scale(loss / cfg['grad_accum_steps']).backward()
            epoch_loss += loss.item()

            if (batch_idx + 1) % cfg['grad_accum_steps'] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                lr = get_lr(eff_step, total_eff_steps, cfg)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                eff_step += 1

                sec_step = time.time() - t_step
                t_step   = time.time()

                if is_main() and (eff_step % cfg['log_every'] == 0 or cfg['max_steps']):
                    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
                    avg = epoch_loss / (batch_idx + 1)
                    ss_disp = (model.module if isinstance(model, DDP) else model).scheduled_sampling_p
                    ss_str = f" | ss_p {ss_disp:.3f}" if cfg['scheduled_sampling'] else ""
                    print(f"  eff_step {eff_step:5d} | loss {loss.item():.4f} | "
                          f"avg {avg:.4f} | lr {lr:.2e} | scale {scaler.get_scale():.0f} | "
                          f"{sec_step:.2f}s/step | peak {peak_gb:.1f}GB{ss_str}")

                if cfg['max_steps'] and eff_step >= cfg['max_steps']:
                    if is_main():
                        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
                        print(f"\n[smoke] reached max_steps={cfg['max_steps']} | "
                              f"peak VRAM/GPU {peak_gb:.1f}GB | final scale {scaler.get_scale():.0f}")
                    cleanup_ddp()
                    return

        # End of epoch
        avg_train = epoch_loss / steps_per_epoch
        if is_main():
            print(f"\n[epoch {epoch+1}] train_loss={avg_train:.4f} — validating...")

        # Teacher-forced val loss — RECORD ONLY, must not drive selection.
        val_loss, tok_acc = validate(model, val_loader, criterion, device, visual,
                                     zero_vision=cfg['zero_vision'])

        # Autoregressive val-ADE — THE selection metric. All ranks participate
        # (internal shard + all_gather). Grad-checkpointing must be off for use_cache.
        raw = model.module if isinstance(model, DDP) else model
        raw.cosmos.model.language_model.gradient_checkpointing_disable()
        ade_stats = compute_val_ade(
            raw, val_dataset, val_trajs, val_ade_idx, device, visual,
            tokenizer=ar_tokenizer, world_size=world_size, rank=local_rank,
            zero_vision=cfg['zero_vision'], zero_ego=cfg['zero_ego'])
        raw.cosmos.model.language_model.gradient_checkpointing_enable()
        val_ade6 = ade_stats['ade'].get('6s', {}).get('median', float('inf'))
        val_ade6_mean = ade_stats['ade'].get('6s', {}).get('mean', float('nan'))

        if is_main():
            gap = val_loss - avg_train
            print(f"[epoch {epoch+1}] SELECT val_ADE@6s median={val_ade6:.4f} "
                  f"(mean={val_ade6_mean:.4f}, n={ade_stats['n']})  best_ADE={best_ade:.4f}")
            print(f"[epoch {epoch+1}] [record] TF val_loss={val_loss:.4f} "
                  f"tok_acc={100*tok_acc:.2f}% train_loss={avg_train:.4f} gap={gap:.4f}")

            if val_ade6 < best_ade:
                best_ade   = val_ade6
                best_val   = val_loss
                no_improve = 0
                save_checkpoint(model, optimizer, epoch+1, eff_step, val_loss, cfg,
                                'best', sel_ade=val_ade6)
            else:
                no_improve += 1
                print(f"[epoch {epoch+1}] No ADE improvement ({no_improve}/{cfg['patience']})")

            save_checkpoint(model, optimizer, epoch+1, eff_step, val_loss, cfg,
                            'latest', sel_ade=val_ade6)

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
    parser.add_argument('--epochs',          type=int,   default=CFG['epochs'])
    parser.add_argument('--lr',              type=float, default=CFG['lr'])
    parser.add_argument('--batch_size',      type=int,   default=CFG['batch_size'])
    parser.add_argument('--grad_accum_steps',type=int,   default=CFG['grad_accum_steps'])
    parser.add_argument('--num_workers',     type=int,   default=CFG['num_workers'])
    parser.add_argument('--max_steps',       type=int,   default=CFG['max_steps'],
                        help='>0 => smoke mode: stop after N optimizer steps')
    parser.add_argument('--augment',     action='store_true', help='image-space photometric aug ON')
    parser.add_argument('--find_unused', action='store_true', help='DDP find_unused_parameters (use if it hangs)')
    parser.add_argument('--zero_vision', action='store_true', help='ablation: zero the 1536 visual tokens (ego-only)')
    parser.add_argument('--zero_ego', action='store_true', help='ablation: zero the 4 ego tokens (vision-only)')
    parser.add_argument('--pos_weighted', action='store_true',
                        help='P3: weight accel_1..11 loss 1.0, curv+accel_0 0.2 (x class weights)')
    parser.add_argument('--scheduled_sampling', action='store_true',
                        help='P4: feed own predicted token w.p. p ramped 0->ss_max')
    parser.add_argument('--ss_max', type=float, default=CFG['ss_max'])
    parser.add_argument('--patience', type=int, default=CFG['patience'],
                        help='early-stopping patience on AR val-ADE')
    args = parser.parse_args()

    CFG['epochs']           = args.epochs
    CFG['lr']               = args.lr
    CFG['batch_size']       = args.batch_size
    CFG['grad_accum_steps'] = args.grad_accum_steps
    CFG['num_workers']      = args.num_workers
    CFG['max_steps']        = args.max_steps
    CFG['augment']          = args.augment
    CFG['find_unused']      = args.find_unused
    CFG['zero_vision']      = args.zero_vision
    CFG['zero_ego']         = args.zero_ego
    CFG['pos_weighted']     = args.pos_weighted
    CFG['scheduled_sampling'] = args.scheduled_sampling
    CFG['ss_max']           = args.ss_max
    CFG['patience']         = args.patience

    train(CFG, resume=args.resume)