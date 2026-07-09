"""
inference.py — Alpamayo VLA Full Evaluation

Metrics:
  - ADE/FDE at 1s, 2s, 3s, 6s horizons
  - Mean and median ADE/FDE
  - Per-token accuracy (overall + per position)
  - Token sequence accuracy
  - Trajectory plots (GT vs predicted)

Usage:
    cd /home/drive1/Alpamayo
    PYTHONPATH=data:tokenization:training:/home/drive1/python_packages \
    /opt/miniconda3/bin/python training/inference.py

IMPORTANT: Only run on TEST set. Never tune hyperparameters based on these numbers.
"""

import sys
import os
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/home/drive1/Alpamayo/data')
sys.path.insert(0, '/home/drive1/Alpamayo/tokenization')
sys.path.insert(0, '/home/drive1/Alpamayo/training')

from dataset import build_scene_split, NuScenesVLADataset, VISUAL_TOKENS_DIR
from model import load_model, TRAJ_VOCAB, TRAJ_LEN, TEXT_DIM
from tokenizer import TrajectoryTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH  = '/home/drive1/Alpamayo/models/checkpoints/alpamayo_best.pt'
TRAJECTORIES_PATH = '/home/drive1/Alpamayo/data/trajectories_full.pkl'
NUSCENES_ROOT    = '/home/drive1/Alpamayo/nuscenes_full'
COSMOS_PATH      = '/home/drive1/Alpamayo/models/cosmos_reason'
OUTPUT_DIR       = Path('/home/drive1/Alpamayo/results')
DEVICE           = 'cuda:0'

DT          = 0.5   # nuScenes @ 2Hz
N_STEPS     = 12    # 6 seconds
HORIZONS    = {     # step index -> label
    2:  '1s',
    4:  '2s',
    6:  '3s',
    12: '6s',
}

# ── Unicycle model ────────────────────────────────────────────────────────────

def unicycle_rollout(accels, curvatures, v0, yaw0, x0=0.0, y0=0.0, dt=DT):
    """
    Integrate unicycle dynamics from initial state.
    accels:     [12] m/s²
    curvatures: [12] rad/m
    v0:         initial speed (m/s)
    yaw0:       initial yaw (rad)
    Returns:    positions [12, 2], yaws [12]
    """
    positions = []
    yaws      = []
    x, y, yaw, v = x0, y0, yaw0, v0

    for a, k in zip(accels, curvatures):
        v   = max(0.0, v + a * dt)
        yaw = yaw + v * k * dt
        x   = x + v * math.cos(yaw) * dt
        y   = y + v * math.sin(yaw) * dt
        positions.append([x, y])
        yaws.append(yaw)

    return np.array(positions), np.array(yaws)

# ── Autoregressive decoding ───────────────────────────────────────────────────

@torch.no_grad()
def decode_trajectory(model, visual_tokens, ego_state, tokenizer):
    """
    Autoregressively decode 24 trajectory tokens from the model.
    At each step, feed previously generated tokens as context.

    Returns:
        pred_tokens: [24] int
        accels:      [12] float (m/s²)
        curvatures:  [12] float (rad/m)
    """
    model.eval()
    dtype = torch.bfloat16

    # Build visual + ego context
    vis = visual_tokens.to(DEVICE, dtype=dtype).unsqueeze(0)   # [1, 1536, 3584]
    ego = ego_state.to(DEVICE, dtype=dtype).unsqueeze(0)        # [1, 4, 4]

    raw = model.module if hasattr(model, 'module') else model
    context = raw._build_context(vis, ego, add_noise=False)     # [1, 1540, 3584]
    ctx_len = context.shape[1]

    generated_tokens  = []
    generated_embeds  = torch.zeros(1, 0, TEXT_DIM, device=DEVICE, dtype=dtype)

    for step in range(TRAJ_LEN):
        lm_input = torch.cat([context, generated_embeds], dim=1)
        lm_out   = raw.cosmos.model.language_model(
            inputs_embeds=lm_input,
            use_cache=False,
        )
        hidden      = lm_out.last_hidden_state
        step_logits = raw.output_head(hidden[:, -1, :])         # [1, 129]
        next_token  = step_logits.argmax(dim=-1)                # [1]
        generated_tokens.append(next_token.item())

        next_embed      = raw.traj_embed(next_token).unsqueeze(1)  # [1, 1, 3584]
        generated_embeds = torch.cat([generated_embeds, next_embed], dim=1)

    pred_tokens = generated_tokens   # [24]
    accel_tokens = pred_tokens[:12]
    curv_tokens  = pred_tokens[12:]

    # Detokenize — detokenize_step(accel_token, curv_token) -> (accel, curv)
    accels     = []
    curvatures = []
    for a_tok, k_tok in zip(accel_tokens, curv_tokens):
    # Both accel and curv tokens must be in 0-63 range
    # STOP=128 handled separately
        a_tok = 32 if a_tok == 128 else min(int(a_tok), 63)
        k_tok = 32 if k_tok == 128 else min(int(k_tok), 63)
        a, k = tokenizer.detokenize_step(a_tok, k_tok)
        accels.append(a)
        curvatures.append(k)

    return pred_tokens, np.array(accels), np.array(curvatures)

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_ade_fde(pred_positions, gt_positions, horizon_steps=None):
    """
    pred_positions: [N, 2]
    gt_positions:   [N, 2]
    horizon_steps:  if set, only evaluate up to this step
    Returns: ADE (float), FDE (float)
    """
    if horizon_steps is not None:
        pred_positions = pred_positions[:horizon_steps]
        gt_positions   = gt_positions[:horizon_steps]

    errors = np.linalg.norm(pred_positions - gt_positions, axis=1)  # [N]
    ade    = errors.mean()
    fde    = errors[-1]
    return float(ade), float(fde)

# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(n_samples=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load test trajectories
    print("[inference] Loading trajectories...")
    with open(TRAJECTORIES_PATH, 'rb') as f:
        all_trajs = pickle.load(f)
    _, _, test_trajs = build_scene_split(all_trajs, NUSCENES_ROOT)
    print(f"[inference] Test set: {len(test_trajs)} trajectories")

    if n_samples is not None:
        test_trajs = test_trajs[:n_samples]
        print(f"[inference] Limiting to {n_samples} samples")

    # Load model
    print("[inference] Loading model...")
    model = load_model(cosmos_path=COSMOS_PATH, device=DEVICE)

    # Load checkpoint
    print(f"[inference] Loading checkpoint from {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
    raw  = model.module if hasattr(model, 'module') else model
    raw.load_state_dict(ckpt['model_state'], strict=False)
    print(f"[inference] Checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    model.eval()

    raw.cosmos.model.language_model.gradient_checkpointing_disable()
    print("[inference] Gradient checkpointing disabled for inference")

    tokenizer = TrajectoryTokenizer()

    # Metrics storage
    all_ade   = defaultdict(list)   # horizon -> [ade per sample]
    all_fde   = defaultdict(list)
    token_correct = 0
    token_total   = 0
    seq_correct   = 0
    gt_tokens_all = []
    pred_tokens_all = []

    # Also store for plotting
    plot_samples = []

    print(f"\n[inference] Evaluating {len(test_trajs)} test trajectories...\n")

    dataset = NuScenesVLADataset(test_trajs, split='test')

    for idx, traj in enumerate(test_trajs):
        if idx % 100 == 0:
            print(f"  [{idx}/{len(test_trajs)}]")

        sample_token = traj['sample_token']

        # Load visual tokens
        token_path = VISUAL_TOKENS_DIR / f"{sample_token}.pt"
        if not token_path.exists():
            continue
        visual_tokens = torch.load(token_path, map_location='cpu', weights_only=True)

        # Get ego state and GT tokens from dataset
        item        = dataset[idx]
        ego_state   = item['ego_state']
        gt_tok      = item['traj_tokens'].tolist()   # [24]

        # Get GT trajectory for ADE/FDE
        gt_positions = np.array(traj['future_positions'])[:N_STEPS]  # [12, 2] global coords
        # Convert to local (ego-centric) frame
        current_x   = traj['current_pose']['translation'][0]
        current_y   = traj['current_pose']['translation'][1]
        gt_local    = gt_positions - np.array([current_x, current_y])

        # Initial ego state for unicycle
        v0   = float(ego_state[3, 0])   # speed at current timestep
        yaw0 = float(ego_state[3, 1])   # yaw at current timestep

        # Decode
        try:
            pred_tok, pred_accels, pred_curvatures = decode_trajectory(
                model, visual_tokens, ego_state, tokenizer
            )
        except Exception as e:
            print(f"  [WARNING] Failed on {sample_token}: {e}")
            continue

        # Rollout unicycle
        pred_positions, _ = unicycle_rollout(pred_accels, pred_curvatures, v0, yaw0)

        # ADE/FDE at multiple horizons
        for step, label in HORIZONS.items():
            if step <= len(pred_positions) and step <= len(gt_local):
                ade, fde = compute_ade_fde(pred_positions, gt_local, horizon_steps=step)
                all_ade[label].append(ade)
                all_fde[label].append(fde)

        # Token accuracy
        for gt_t, pred_t in zip(gt_tok, pred_tok):
            token_correct += int(gt_t == pred_t)
            token_total   += 1
        if gt_tok == list(pred_tok):
            seq_correct += 1

        gt_tokens_all.extend(gt_tok)
        pred_tokens_all.extend(pred_tok)

        # Store first 10 for plotting
        if len(plot_samples) < 10:
            plot_samples.append({
                'gt_positions':   gt_local,
                'pred_positions': pred_positions,
                'gt_tokens':      gt_tok,
                'pred_tokens':    pred_tok,
                'sample_token':   sample_token,
            })

    # ── Print results ─────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("INFERENCE RESULTS — TEST SET")
    print("="*60)
    print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    print(f"Trajectories evaluated: {len(all_ade['6s'])}")
    print()

    print("── ADE (Average Displacement Error) ──")
    print(f"{'Horizon':<10} {'Mean (m)':<12} {'Median (m)':<12}")
    print("-"*34)
    for label in ['1s', '2s', '3s', '6s']:
        vals = all_ade[label]
        if vals:
            print(f"{label:<10} {np.mean(vals):<12.3f} {np.median(vals):<12.3f}")

    print()
    print("── FDE (Final Displacement Error) ──")
    print(f"{'Horizon':<10} {'Mean (m)':<12} {'Median (m)':<12}")
    print("-"*34)
    for label in ['1s', '2s', '3s', '6s']:
        vals = all_fde[label]
        if vals:
            print(f"{label:<10} {np.mean(vals):<12.3f} {np.median(vals):<12.3f}")

    print()
    print("── Token Metrics ──")
    print(f"Per-token accuracy:     {100*token_correct/max(token_total,1):.2f}%")
    print(f"Sequence accuracy:      {100*seq_correct/max(len(test_trajs),1):.2f}%")

    # Roundtrip ADE baseline reminder
    print()
    print("── Baseline Reference ──")
    print(f"Roundtrip ADE (tokenizer):  0.885m (theoretical floor)")
    print(f"Modal token baseline loss:  ~2.33 nats")

    # ── Save results ──────────────────────────────────────────────────────────

    results = {
        'ade': {k: {'mean': float(np.mean(v)), 'median': float(np.median(v))}
                for k, v in all_ade.items()},
        'fde': {k: {'mean': float(np.mean(v)), 'median': float(np.median(v))}
                for k, v in all_fde.items()},
        'token_accuracy':    float(token_correct / max(token_total, 1)),
        'sequence_accuracy': float(seq_correct / max(len(test_trajs), 1)),
        'n_evaluated':       len(all_ade['6s']),
        'checkpoint_epoch':  ckpt['epoch'],
        'checkpoint_val_loss': ckpt['val_loss'],
    }

    import json
    results_path = OUTPUT_DIR / 'inference_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[inference] Results saved → {results_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Plot 1: trajectory comparison grid
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for i, sample in enumerate(plot_samples):
            ax = axes[i]
            gt  = sample['gt_positions']
            pred = sample['pred_positions']

            ax.plot(gt[:, 0],   gt[:, 1],   'b.-', label='GT',   linewidth=2, markersize=4)
            ax.plot(pred[:, 0], pred[:, 1], 'r.-', label='Pred', linewidth=2, markersize=4)
            ax.plot(0, 0, 'ko', markersize=6)  # ego start

            ade_val = np.linalg.norm(pred - gt, axis=1).mean()
            ax.set_title(f'ADE={ade_val:.2f}m', fontsize=9)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
            ax.set_xlabel('x (m)', fontsize=7)
            ax.set_ylabel('y (m)', fontsize=7)

        plt.suptitle('GT vs Predicted Trajectories (Test Set, First 10 Samples)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        traj_plot_path = OUTPUT_DIR / 'trajectory_comparison.png'
        plt.savefig(traj_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[inference] Trajectory plot saved → {traj_plot_path}")

        # Plot 2: ADE distribution
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, label in enumerate(['1s', '2s', '3s', '6s']):
            vals = all_ade[label]
            if vals:
                axes[i].hist(vals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
                axes[i].axvline(np.mean(vals), color='red', linestyle='--',
                                label=f'Mean={np.mean(vals):.2f}m')
                axes[i].axvline(np.median(vals), color='orange', linestyle='--',
                                label=f'Median={np.median(vals):.2f}m')
                axes[i].set_title(f'ADE @ {label}')
                axes[i].set_xlabel('ADE (m)')
                axes[i].set_ylabel('Count')
                axes[i].legend(fontsize=8)

        plt.suptitle('ADE Distribution at Multiple Horizons (Test Set)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        ade_plot_path = OUTPUT_DIR / 'ade_distribution.png'
        plt.savefig(ade_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[inference] ADE distribution plot saved → {ade_plot_path}")

        # Plot 3: per-position token accuracy
        pos_acc = []
        gt_arr   = np.array(gt_tokens_all).reshape(-1, TRAJ_LEN)
        pred_arr = np.array(pred_tokens_all).reshape(-1, TRAJ_LEN)
        for pos in range(TRAJ_LEN):
            acc = (gt_arr[:, pos] == pred_arr[:, pos]).mean()
            pos_acc.append(acc)

        fig, ax = plt.subplots(figsize=(10, 4))
        positions = list(range(TRAJ_LEN))
        colors = ['#e74c3c'] * 12 + ['#3498db'] * 12
        ax.bar(positions, [a * 100 for a in pos_acc], color=colors, edgecolor='black', alpha=0.8)
        ax.axvline(11.5, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Per-Position Token Accuracy\n(Red=Acceleration tokens, Blue=Curvature tokens)')
        ax.set_xticks(range(TRAJ_LEN))
        ax.set_xticklabels([f'a{i+1}' if i < 12 else f'k{i-11}' for i in range(TRAJ_LEN)],
                           rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        token_plot_path = OUTPUT_DIR / 'token_accuracy.png'
        plt.savefig(token_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[inference] Token accuracy plot saved → {token_plot_path}")

    except ImportError:
        print("[inference] matplotlib not available — skipping plots")

    print("\n[inference] DONE")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Evaluate on first N test samples (default: all)')
    args = parser.parse_args()
    evaluate(n_samples=args.n_samples)