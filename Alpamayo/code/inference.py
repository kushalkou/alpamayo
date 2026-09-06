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

# ── Generalised decode: mode / mean / sampling ────────────────────────────────
#
# Motivation (decode-rule fix): argmax on a near-marginal distribution emits an
# independent noisy commitment per step, and the unicycle double-integrates those
# errors. Decoding the conditional MEAN instead degrades gracefully toward CV
# where the model has no information.
#
# Token layout: slots 0-11 = accel, slots 12-23 = curvature. BOTH use bin ids
# 0..63 (the slot position disambiguates); id 128 = STOP. Ids 64..127 are never
# targets — dead classes. So the per-slot distribution is softmax over ids 0..63.
#
# Context is ALWAYS fed the ARGMAX token (keeps the token stream on-manifold),
# except in 'sample' mode where the sampled token is fed back (true ancestral
# sampling). Only the CONTINUOUS value handed to the unicycle changes.

DECODE_MODES = ('argmax', 'expect', 'expect_topk', 'expect_temp', 'sample')
N_ACT = 64          # active bin ids per slot
STOP_ID = 128


def slot_centers(tokenizer, step):
    """Bin centers for AR step `step` (0..23). Imported from the tokenizer — never re-derived."""
    return tokenizer.accel_centers if step < TRAJ_LEN // 2 else tokenizer.curv_centers


def _step_value(logits, tokenizer, step, decode_mode, topk, temperature, generator,
                stop_aware=False):
    """logits [129] fp32 -> (continuous value, token to feed back, stats dict).

    stats: entropy (nats) and top-1 mass of the 64-way slot distribution, plus the
    softmax mass the full 129-way head puts on the dead ids 64..127 (diagnostic).
    """
    centers = torch.as_tensor(slot_centers(tokenizer, step),
                              dtype=torch.float32, device=logits.device)
    argmax_tok = int(logits.argmax(-1).item())            # full 129-way argmax (unchanged path)

    act = logits[:N_ACT]                                   # ids 0..63
    p = torch.softmax(act, dim=-1)                         # renormalised over the 64 active bins

    full_p = torch.softmax(logits, dim=-1)
    stats = {
        'entropy':  float(-(p * (p + 1e-12).log()).sum().item()),
        'top1':     float(p.max().item()),
        'dead_mass': float(full_p[N_ACT:STOP_ID].sum().item()),
        'stop_mass': float(full_p[STOP_ID].item()),
        'p': p.detach().cpu().numpy(),          # 64-way slot distribution (diagnostics)
    }

    if stop_aware:
        # STOP-AWARE expectation. Dropping id 128 and renormalising over 0..63 (the
        # literal V1/V2/V3 rule) is WRONG for a stopped vehicle: the model puts its
        # mass on STOP, so the residual over 0..63 is arbitrary leftover noise and
        # the decoded mean drives a parked car away. Here the support is
        # {0..63} U {128}; STOP contributes the tokenizer's own detokenize_step(STOP)
        # value of 0.0 for both accel and curvature. Dead ids 64..127 stay excluded.
        vlog = torch.cat([act, logits[STOP_ID:STOP_ID + 1]])          # [65]
        vcent = torch.cat([centers, torch.zeros(1, device=logits.device)])
        if decode_mode == 'expect':
            q = torch.softmax(vlog, dim=-1)
            value = float((q * vcent).sum().item())
        elif decode_mode == 'expect_topk':
            q = torch.softmax(vlog, dim=-1)
            tv, ti = torch.topk(q, min(topk, vlog.numel()))
            tv = tv / tv.sum()
            value = float((tv * vcent[ti]).sum().item())
        elif decode_mode == 'expect_temp':
            q = torch.softmax(vlog / temperature, dim=-1)
            value = float((q * vcent).sum().item())
        else:
            raise ValueError(f'stop_aware not defined for decode_mode {decode_mode}')
        return value, argmax_tok, stats

    fed_tok = argmax_tok
    if decode_mode == 'argmax':
        t = 32 if argmax_tok == STOP_ID else min(max(argmax_tok, 0), N_ACT - 1)
        value = float(centers[t].item())
    elif decode_mode == 'expect':
        value = float((p * centers).sum().item())
    elif decode_mode == 'expect_topk':
        k = min(topk, N_ACT)
        tv, ti = torch.topk(p, k)
        tv = tv / tv.sum()
        value = float((tv * centers[ti]).sum().item())
    elif decode_mode == 'expect_temp':
        pt = torch.softmax(act / temperature, dim=-1)
        value = float((pt * centers).sum().item())
    elif decode_mode == 'sample':
        ps = torch.softmax(act / temperature, dim=-1)
        tok = int(torch.multinomial(ps, 1, generator=generator).item())
        value = float(centers[tok].item())
        fed_tok = tok                                      # ancestral: feed the sample back
    else:
        raise ValueError(f'unknown decode_mode {decode_mode}')

    return value, fed_tok, stats


@torch.no_grad()
def prefill_context(raw, visual_tokens, ego_state, device=DEVICE, dtype=DTYPE):
    """Run the 1,540-token context ONCE and return a reusable prefill.

    The 1,540-token prefill dominates per-rollout cost (the 23 follow-on steps are
    single-token). decode_trajectory_ex crops the cache back to ctx_len when it is
    done, so one prefill can serve many rollouts of the same sample — which is what
    makes K-sample decoding (minADE@K) affordable.
    """
    raw = raw.module if hasattr(raw, 'module') else raw
    vis = visual_tokens.to(device, dtype=dtype)
    ego = ego_state.to(device, dtype=torch.float32).unsqueeze(0)
    context = raw._build_context(vis, ego)
    out = raw.cosmos.model.language_model(inputs_embeds=context, use_cache=True)
    return {'past': out.past_key_values,
            'logits0': raw.output_head(out.last_hidden_state[:, -1, :].float())[0],
            'ctx_len': int(context.shape[1]),
            'ctx_dtype': context.dtype}


@torch.no_grad()
def decode_trajectory_ex(raw, visual_tokens, ego_state, tokenizer, device=DEVICE,
                         dtype=DTYPE, decode_mode='argmax', topk=5, temperature=1.0,
                         collect_stats=False, generator=None, pre=None,
                         stop_aware=False):
    """KV-cached AR decode of 24 slots under `decode_mode`.

    `pre`: an optional prefill_context() result to reuse (visual_tokens/ego_state are
    then ignored). Returns (tokens[24], accels[12], curvs[12], stats[24] or None).
    With decode_mode='argmax' this is output-identical to decode_trajectory().
    """
    if decode_mode not in DECODE_MODES:
        raise ValueError(f'decode_mode must be one of {DECODE_MODES}')
    raw = raw.module if hasattr(raw, 'module') else raw
    lm = raw.cosmos.model.language_model

    if pre is None:
        pre = prefill_context(raw, visual_tokens, ego_state, device=device, dtype=dtype)
    past, ctx_len, ctx_dtype = pre['past'], pre['ctx_len'], pre['ctx_dtype']
    past.crop(ctx_len)                       # drop any tokens left by a previous rollout

    toks, values, stats = [], [], []
    logits = pre['logits0']                  # [129]

    for step in range(TRAJ_LEN):
        v, fed, st = _step_value(logits, tokenizer, step, decode_mode,
                                 topk, temperature, generator, stop_aware=stop_aware)
        values.append(v); toks.append(fed)
        if collect_stats:
            stats.append(st)
        if step == TRAJ_LEN - 1:
            break
        t = torch.tensor([fed], device=device, dtype=torch.long)
        emb = raw.traj_embed(t).unsqueeze(1).to(ctx_dtype)
        out = lm(inputs_embeds=emb, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = raw.output_head(out.last_hidden_state[:, -1, :].float())[0]

    past.crop(ctx_len)                       # leave the prefill reusable
    accels = np.array(values[:TRAJ_LEN // 2])
    curvs  = np.array(values[TRAJ_LEN // 2:])
    return toks, accels, curvs, (stats if collect_stats else None)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_ade_fde(pred_positions, gt_positions, horizon_steps=None):
    if horizon_steps is not None:
        pred_positions = pred_positions[:horizon_steps]
        gt_positions   = gt_positions[:horizon_steps]
    errors = np.linalg.norm(pred_positions - gt_positions, axis=1)
    return float(errors.mean()), float(errors[-1])

# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(n_samples=None, checkpoint=DEFAULT_CKPT, zero_vision=False, zero_ego=False, out=None,
             decode_mode='argmax', topk=5, temperature=1.0):
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
            if decode_mode == 'argmax':
                pred_tok, pred_accels, pred_curvs = decode_trajectory(
                    model, visual_tokens, ego_state, tokenizer)
            else:
                pred_tok, pred_accels, pred_curvs, _ = decode_trajectory_ex(
                    model, visual_tokens, ego_state, tokenizer, device=DEVICE,
                    decode_mode=decode_mode, topk=topk, temperature=temperature)
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
    parser.add_argument('--decode_mode', type=str, default='argmax', choices=list(DECODE_MODES),
                        help='argmax (default, unchanged) | expect | expect_topk | expect_temp | sample')
    parser.add_argument('--topk', type=int, default=5, help='k for expect_topk')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='T for expect_temp / sample')
    args = parser.parse_args()
    evaluate(n_samples=args.n_samples, checkpoint=args.checkpoint,
             zero_vision=args.zero_vision, zero_ego=args.zero_ego, out=args.out,
             decode_mode=args.decode_mode, topk=args.topk, temperature=args.temperature)
