"""gate3_overfit.py — overfit-one-batch sanity (single GPU).

Confirms the reconstructed stack is numerically clean: live vision -> fp16 backbone
forward -> fp32 adapters -> fp32 CE -> GradScaler. A correct model must drive a fixed
tiny batch to ~0 loss / 100% token acc with ZERO loss nan/inf. GradScaler scale-downs
in the first few steps are the expected warm-up mechanism, not failures.

Usage:
    cd /home/dgx1user/Alpamayo-Kushal/Alpamayo/code
    python gate3_overfit.py            # 4 samples, 200 steps, lr 5e-4
"""
import sys, time, pickle, argparse
import torch
import torch.nn as nn

sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model, TRAJ_VOCAB
from dataset import compute_ego_state, NUSCENES_ROOT, TRAJ_PATH
from tokenizer import TrajectoryTokenizer
from vision_live import LiveVisionEncoder, get_cam_paths, CAMERAS
import os

DEVICE = 'cuda:0'


def build_fixed_batch(nusc, trajs, tokenizer, n):
    """First n trajectories whose 6 camera images all exist on disk."""
    picks = []
    for traj in trajs:
        cam_paths = get_cam_paths(nusc, traj['sample_token'])
        if all(os.path.exists(cam_paths[c]) for c in CAMERAS):
            picks.append((traj, cam_paths))
        if len(picks) == n:
            break
    if len(picks) < n:
        raise RuntimeError(f"only found {len(picks)}/{n} samples with all cameras")
    ego = torch.stack([compute_ego_state(t) for t, _ in picks], 0).to(DEVICE)  # [n,4,4]
    traj_tokens = []
    for t, _ in picks:
        pairs = tokenizer.tokenize(t)
        traj_tokens.append([p[0] for p in pairs] + [p[1] for p in pairs])
    traj_tokens = torch.tensor(traj_tokens, dtype=torch.long, device=DEVICE)    # [n,24]
    cam_paths_list = [cp for _, cp in picks]
    return ego, traj_tokens, cam_paths_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr',    type=float, default=5e-4)
    args = ap.parse_args()

    model = load_model(device=DEVICE)
    model.train()
    model.augment = False
    # keep gradient checkpointing alive (needs train mode) but kill dropout noise
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()

    enc = LiveVisionEncoder(model.cosmos.model.visual, device=DEVICE)

    from nuscenes.nuscenes import NuScenes
    print("[gate3] loading nuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSCENES_ROOT, verbose=False)
    with open(TRAJ_PATH, 'rb') as f:
        trajs = pickle.load(f)
    tokenizer = TrajectoryTokenizer()

    ego, traj_tokens, cam_paths_list = build_fixed_batch(nusc, trajs, tokenizer, args.batch)
    print(f"[gate3] encoding {args.batch} samples live (aug OFF)...")
    visual_tokens = enc.encode_batch(cam_paths_list, augment=False)   # [n,1536,3584]

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler    = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    print(f"[gate3] overfitting {args.batch} samples for {args.steps} steps (lr={args.lr}) ...")
    loss_nan_inf = 0
    scaler_skips = 0
    prev_scale   = scaler.get_scale()
    losses       = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        logits = model(visual_tokens, ego, traj_tokens)
        loss   = criterion(logits.reshape(-1, TRAJ_VOCAB).float(), traj_tokens.reshape(-1))
        if not torch.isfinite(loss):
            loss_nan_inf += 1
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        if new_scale < prev_scale:
            scaler_skips += 1
        prev_scale = new_scale

        with torch.no_grad():
            acc = (logits.argmax(-1) == traj_tokens).float().mean().item()
        losses.append(loss.item())
        if step == 1 or step % 10 == 0:
            print(f"  step {step:4d} | loss {loss.item():.4f} | tok_acc {acc*100:5.1f}% | scale {new_scale:.0f}")

    drops = sum(1 for i in range(1, len(losses)) if losses[i] <= losses[i-1])
    print("\n=== GATE 3 SUMMARY ===")
    print(f"  final loss        : {losses[-1]:.4f}")
    print(f"  loss nan/inf      : {loss_nan_inf}")
    print(f"  GradScaler skips  : {scaler_skips}  (early warm-up expected)")
    print(f"  monotonic-down    : {100*drops/(len(losses)-1):.0f}%")
    print(f"  peak VRAM         : {torch.cuda.max_memory_allocated(DEVICE)/1e9:.1f} GB")


if __name__ == '__main__':
    main()
