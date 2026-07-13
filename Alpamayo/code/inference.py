"""
inference.py — Alpamayo VLA Full Evaluation  (DGX / V100 live-vision build)

Autoregressive decode => NO GT-token leak (unlike teacher-forced val loss), so
ADE/FDE is the only metric that can actually separate the input modalities.

Ported for the current stack:
  - fp16 backbone (NOT bf16 — V100/Volta has no bf16 hardware)
  - LIVE vision: the 6 camera images are encoded on the fly by the frozen visual
    tower (the 246GB precomputed token cache is gone)
  - gradient checkpointing OFF (inference)
  - current AlpamayoVLA._build_context(visual_tokens, ego_state) signature
  - predicted accel/curv tokens clamped to 0..63 before detokenize (STOP=128 -> center)

Usage:
    cd /home/dgx1user/Alpamayo-Kushal/Alpamayo/code
    python inference.py --n_samples 200 \
        --checkpoint /home/dgx1user/.../checkpoints/_livevision_run_jul9/alpamayo_best_e1_val2.0806.pt

IMPORTANT: Only run on TEST set. Never tune hyperparameters based on these numbers.
"""

import sys
import os
import pickle
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')

from dataset import build_scene_split, NuScenesVLADataset
from model import load_model, TRAJ_VOCAB, TRAJ_LEN, TEXT_DIM
from tokenizer import TrajectoryTokenizer
from vision_live import encode_normalized_images

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CKPT      = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints/_livevision_run_jul9/alpamayo_best_e1_val2.0806.pt'
TRAJECTORIES_PATH = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/trajectories_full.pkl'
NUSCENES_ROOT     = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
COSMOS_PATH       = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason'
OUTPUT_DIR        = Path('/home/dgx1user/Alpamayo-Kushal/Alpamayo/results')
DEVICE            = 'cuda:0'
DTYPE             = torch.float16   # V100: fp16, never bf16

DT       = 0.5   # nuScenes @ 2Hz
N_STEPS  = 12    # 6 seconds
HORIZONS = {2: '1s', 4: '2s', 6: '3s', 12: '6s'}

# ── Unicycle model ────────────────────────────────────────────────────────────

def unicycle_rollout(accels, curvatures, v0, yaw0, x0=0.0, y0=0.0, dt=DT):
    positions, yaws = [], []
    x, y, yaw, v = x0, y0, yaw0, v0
    for a, k in zip(accels, curvatures):
        v   = max(0.0, v + a * dt)
        yaw = yaw + v * k * dt
        x   = x + v * math.cos(yaw) * dt
        y   = y + v * math.sin(yaw) * dt
        positions.append([x, y])
        yaws.append(yaw)
    return np.array(positions), np.array(yaws)

# ── Live vision encode ────────────────────────────────────────────────────────

@torch.no_grad()
def encode_live(visual, images):
    """images [6,3,448,448] (one sample) -> visual_tokens [1,1536,3584] fp16."""
    flat   = images.to(DEVICE)                        # [6,3,448,448]
    pooled = encode_normalized_images(visual, flat)   # [6*256, 3584]
    return pooled.reshape(1, 6 * 256, -1)             # [1,1536,3584]

# ── Autoregressive decoding ───────────────────────────────────────────────────

@torch.no_grad()
def decode_trajectory(model, visual_tokens, ego_state, tokenizer):
    """Autoregressively decode 24 trajectory tokens (NO GT-token leak)."""
    raw = model.module if hasattr(model, 'module') else model
    raw.eval()

    vis = visual_tokens.to(DEVICE, dtype=DTYPE)               # [1,1536,3584]
    ego = ego_state.to(DEVICE, dtype=torch.float32).unsqueeze(0)  # [1,4,4]

    context = raw._build_context(vis, ego)                    # [1,1540,3584]
    lm = raw.cosmos.model.language_model

    generated_tokens = []

    # Prefill the context once, then decode with a KV cache: each step feeds only
    # the single new token embedding instead of recomputing the whole prefix.
    # Greedy argmax => output-identical to the full-recompute path, ~24x cheaper.
    lm_out = lm(inputs_embeds=context, use_cache=True)
    past   = lm_out.past_key_values
    hidden = lm_out.last_hidden_state
    step_logits = raw.output_head(hidden[:, -1, :].float())        # [1,129]
    next_token  = step_logits.argmax(dim=-1)                       # [1]
    generated_tokens.append(int(next_token.item()))

    for _ in range(TRAJ_LEN - 1):
        next_embed = raw.traj_embed(next_token).unsqueeze(1).to(context.dtype)  # [1,1,3584]
        lm_out = lm(inputs_embeds=next_embed, past_key_values=past, use_cache=True)
        past   = lm_out.past_key_values
        hidden = lm_out.last_hidden_state
        step_logits = raw.output_head(hidden[:, -1, :].float())    # [1,129]
        next_token  = step_logits.argmax(dim=-1)                   # [1]
        generated_tokens.append(int(next_token.item()))

    pred_tokens  = generated_tokens          # [24] = [accel_0..11, curv_0..11]
    accel_tokens = pred_tokens[:12]
    curv_tokens  = pred_tokens[12:]

    accels, curvatures = [], []
    for a_tok, k_tok in zip(accel_tokens, curv_tokens):
        # accel/curv tokens must be in 0..63; STOP(128) -> bin center (32)
        a_tok = 32 if a_tok == 128 else min(max(int(a_tok), 0), 63)
        k_tok = 32 if k_tok == 128 else min(max(int(k_tok), 0), 63)
        a, k = tokenizer.detokenize_step(a_tok, k_tok)
        accels.append(a)
        curvatures.append(k)

    return pred_tokens, np.array(accels), np.array(curvatures)

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_ade_fde(pred_positions, gt_positions, horizon_steps=None):
    if horizon_steps is not None:
        pred_positions = pred_positions[:horizon_steps]
        gt_positions   = gt_positions[:horizon_steps]
    errors = np.linalg.norm(pred_positions - gt_positions, axis=1)
    return float(errors.mean()), float(errors[-1])

# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(n_samples=None, checkpoint=DEFAULT_CKPT, zero_vision=False, zero_ego=False, out=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[inference] Loading trajectories...")
    with open(TRAJECTORIES_PATH, 'rb') as f:
        all_trajs = pickle.load(f)
    _, _, test_trajs = build_scene_split(all_trajs, NUSCENES_ROOT)
    print(f"[inference] Test set: {len(test_trajs)} trajectories")

    if n_samples is not None:
        test_trajs = test_trajs[:n_samples]
        print(f"[inference] Limiting to {n_samples} samples")

    print("[inference] Loading model...")
    model = load_model(cosmos_path=COSMOS_PATH, device=DEVICE)
    ckpt = torch.load(checkpoint, map_location='cpu')
    raw  = model.module if hasattr(model, 'module') else model
    raw.load_state_dict(ckpt['model_state'], strict=False)
    # Inference must zero the SAME modality the checkpoint was trained with
    # (mirror of finetune.py --zero_vision / --zero_ego). zero_ego handled inside
    # _build_context; zero_vision applied to visual_tokens right after encode_live.
    raw.zero_ego = zero_ego
    print(f"[inference] Checkpoint: {checkpoint}")
    print(f"[inference]   epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    print(f"[inference]   zero_vision={zero_vision}  zero_ego={zero_ego}")
    model.eval()

    raw.cosmos.model.language_model.gradient_checkpointing_disable()
    print("[inference] Gradient checkpointing disabled")

    visual    = raw.cosmos.model.visual
    tokenizer = TrajectoryTokenizer()
    dataset   = NuScenesVLADataset(test_trajs, split='test', augment=False)

    all_ade, all_fde = defaultdict(list), defaultdict(list)
    token_correct = token_total = seq_correct = 0

    print(f"\n[inference] Evaluating {len(test_trajs)} test trajectories (autoregressive)...\n")

    for idx, traj in enumerate(test_trajs):
        if idx % 50 == 0:
            print(f"  [{idx}/{len(test_trajs)}]", flush=True)

        item      = dataset[idx]
        images    = item['images']         # [6,3,448,448]
        ego_state = item['ego_state']      # [4,4]
        gt_tok    = item['traj_tokens'].tolist()

        gt_positions = np.array(traj['future_positions'])[:N_STEPS]        # [12,2] global
        cx, cy       = traj['current_pose']['translation'][0], traj['current_pose']['translation'][1]
        gt_local     = gt_positions - np.array([cx, cy])                    # global-axes, origin at ego

        # Initial speed MUST be the true current speed. ego_state[3,0] is a BACKWARD
        # difference over past poses — it is 0 when past_poses are missing and
        # systematically under-estimates speed at trajectory starts, which made the
        # rollout undershoot and inflated the floor from 0.885m to ~2.5m (V1 bug).
        # future_speeds[0] is the validated reference used by test_roundtrip.py.
        v0   = float(traj['future_speeds'][0])   # true current speed (V1 fix)
        yaw0 = float(ego_state[3, 1])            # current (global) yaw

        try:
            visual_tokens = encode_live(visual, images)
            if zero_vision:
                visual_tokens = torch.zeros_like(visual_tokens)   # ego-only ablation
            pred_tok, pred_accels, pred_curvs = decode_trajectory(
                model, visual_tokens, ego_state, tokenizer)
        except Exception as e:
            print(f"  [WARNING] failed on sample {idx}: {e}")
            continue

        pred_positions, _ = unicycle_rollout(pred_accels, pred_curvs, v0, yaw0)

        for step, label in HORIZONS.items():
            if step <= len(pred_positions) and step <= len(gt_local):
                ade, fde = compute_ade_fde(pred_positions, gt_local, horizon_steps=step)
                all_ade[label].append(ade)
                all_fde[label].append(fde)

        for gt_t, pred_t in zip(gt_tok, pred_tok):
            token_correct += int(gt_t == pred_t)
            token_total   += 1
        if gt_tok == list(pred_tok):
            seq_correct += 1

    # ── Results ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("INFERENCE RESULTS — TEST SET (autoregressive, live vision)")
    print("="*60)
    print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    print(f"Trajectories evaluated: {len(all_ade['6s'])}")
    print()
    print("── ADE (mean / median, meters) ──")
    for label in ['1s', '2s', '3s', '6s']:
        v = all_ade[label]
        if v:
            print(f"  {label:<4} ADE  mean={np.mean(v):.3f}  median={np.median(v):.3f}")
    print()
    print("── FDE (mean / median, meters) ──")
    for label in ['1s', '2s', '3s', '6s']:
        v = all_fde[label]
        if v:
            print(f"  {label:<4} FDE  mean={np.mean(v):.3f}  median={np.median(v):.3f}")
    print()
    print("── Token Metrics (autoregressive) ──")
    print(f"  Per-token accuracy: {100*token_correct/max(token_total,1):.2f}%")
    print(f"  Sequence accuracy:  {100*seq_correct/max(len(test_trajs),1):.2f}%")
    print()
    print("── Reference ── roundtrip ADE floor 0.885m ; old baseline tok-acc 41.49%")

    import json
    results = {
        'ade': {k: {'mean': float(np.mean(v)), 'median': float(np.median(v))}
                for k, v in all_ade.items() if v},
        'fde': {k: {'mean': float(np.mean(v)), 'median': float(np.median(v))}
                for k, v in all_fde.items() if v},
        'token_accuracy':    float(token_correct/max(token_total,1)),
        'sequence_accuracy': float(seq_correct/max(len(test_trajs),1)),
        'n_evaluated':       len(all_ade['6s']),
        'checkpoint':        str(checkpoint),
        'checkpoint_epoch':  ckpt['epoch'],
        'checkpoint_val_loss': ckpt['val_loss'],
    }
    rp = OUTPUT_DIR / (out if out else 'inference_results.json')
    with open(rp, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[inference] Results saved → {rp}")
    print("[inference] DONE")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Evaluate on first N test samples (default: all)')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CKPT)
    parser.add_argument('--zero_vision', action='store_true',
                        help='ablation: zero the 1536 visual tokens (ego-only ckpts)')
    parser.add_argument('--zero_ego', action='store_true',
                        help='ablation: zero the 4 ego tokens (vision-only ckpts)')
    parser.add_argument('--out', type=str, default=None,
                        help='results json filename (under results/); default inference_results.json')
    args = parser.parse_args()
    evaluate(n_samples=args.n_samples, checkpoint=args.checkpoint,
             zero_vision=args.zero_vision, zero_ego=args.zero_ego, out=args.out)
