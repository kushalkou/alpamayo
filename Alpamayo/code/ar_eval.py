"""ar_eval.py — Autoregressive val-ADE for model selection (shared by finetune.py).

Teacher-forced val loss is DISQUALIFIED as a selection metric (a zero-input model
matches it — see RESULTS_OVERNIGHT.md). Selection/early-stopping must use the
leak-free autoregressive ADE instead.

Decode logic here is byte-for-byte the same as inference.py.decode_trajectory
(KV-cache greedy), but device-parametrized so each DDP rank can decode its own
shard on its own GPU. unicycle_rollout is copied verbatim from inference.py — the
decode/rollout must NOT change.
"""
import math
import numpy as np
import torch

from tokenizer import TrajectoryTokenizer
from model import TRAJ_LEN, TEXT_DIM
from vision_live import encode_normalized_images

DT      = 0.5
N_STEPS = 12
HORIZONS = {2: '1s', 4: '2s', 6: '3s', 12: '6s'}


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


@torch.no_grad()
def encode_live_one(visual, images, device, dtype=torch.float16):
    """images [6,3,448,448] -> [1,1536,3584]."""
    flat   = images.to(device)
    pooled = encode_normalized_images(visual, flat)
    return pooled.reshape(1, 6 * 256, -1).to(dtype)


@torch.no_grad()
def ar_decode(raw, visual_tokens, ego_state, tokenizer, device, dtype=torch.float16):
    """KV-cache greedy decode of 24 tokens. Identical to inference.decode_trajectory."""
    vis = visual_tokens.to(device, dtype=dtype)
    ego = ego_state.to(device, dtype=torch.float32).unsqueeze(0)
    context = raw._build_context(vis, ego)
    lm = raw.cosmos.model.language_model

    toks = []
    out  = lm(inputs_embeds=context, use_cache=True)
    past = out.past_key_values
    t = raw.output_head(out.last_hidden_state[:, -1, :].float()).argmax(-1)
    toks.append(int(t.item()))
    for _ in range(TRAJ_LEN - 1):
        emb = raw.traj_embed(t).unsqueeze(1).to(context.dtype)
        out = lm(inputs_embeds=emb, past_key_values=past, use_cache=True)
        past = out.past_key_values
        t = raw.output_head(out.last_hidden_state[:, -1, :].float()).argmax(-1)
        toks.append(int(t.item()))

    a_toks, k_toks = toks[:12], toks[12:]
    accels, curvs = [], []
    for a_tok, k_tok in zip(a_toks, k_toks):
        a_tok = 32 if a_tok == 128 else min(max(int(a_tok), 0), 63)
        k_tok = 32 if k_tok == 128 else min(max(int(k_tok), 0), 63)
        a, k = tokenizer.detokenize_step(a_tok, k_tok)
        accels.append(a); curvs.append(k)
    return toks, np.array(accels), np.array(curvs)


def fixed_val_indices(n_val, k=400, seed=1234):
    """Seed-fixed subset of val indices — comparable across runs."""
    k = min(k, n_val)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_val, size=k, replace=False))


@torch.no_grad()
def compute_val_ade(raw, val_dataset, val_trajs, indices, device, visual,
                    tokenizer=None, world_size=1, rank=0, zero_vision=False,
                    zero_ego=False):
    """AR-decode `indices` of the val set, sharded across ranks. Returns a dict of
    ADE/FDE mean+median per horizon (computed on rank 0 after all_gather).

    Uses global-axes ego-origin frame identical to inference.py:
      gt_local = future_positions - current_pose.translation[:2]
    """
    if tokenizer is None:
        tokenizer = TrajectoryTokenizer()

    was_training = raw.training
    raw.eval()
    prev_zero_ego = getattr(raw, 'zero_ego', False)
    raw.zero_ego = zero_ego

    my_idx = indices[rank::world_size]
    local_ade = {h: [] for h in HORIZONS.values()}
    local_fde = {h: [] for h in HORIZONS.values()}

    for i in my_idx:
        i = int(i)
        item = val_dataset[i]
        traj = val_trajs[i]
        vt = encode_live_one(visual, item['images'], device)
        if zero_vision:
            vt = torch.zeros_like(vt)
        _, accels, curvs = ar_decode(raw, vt, item['ego_state'], tokenizer, device)

        ego = item['ego_state']
        v0, yaw0 = float(ego[3, 0]), float(ego[3, 1])
        pred, _ = unicycle_rollout(accels, curvs, v0, yaw0)
        gt = np.array(traj['future_positions'])[:N_STEPS]
        cx, cy = traj['current_pose']['translation'][0], traj['current_pose']['translation'][1]
        gtl = gt - np.array([cx, cy])
        for step, lab in HORIZONS.items():
            if step <= len(pred) and step <= len(gtl):
                e = np.linalg.norm(pred[:step] - gtl[:step], axis=1)
                local_ade[lab].append(float(e.mean()))
                local_fde[lab].append(float(e[-1]))

    # Gather per-sample values across ranks
    if world_size > 1:
        import torch.distributed as dist
        gathered_ade = [None] * world_size
        gathered_fde = [None] * world_size
        dist.all_gather_object(gathered_ade, local_ade)
        dist.all_gather_object(gathered_fde, local_fde)
    else:
        gathered_ade = [local_ade]
        gathered_fde = [local_fde]

    raw.zero_ego = prev_zero_ego
    if was_training:
        raw.train()

    # Merge
    ade = {h: [] for h in HORIZONS.values()}
    fde = {h: [] for h in HORIZONS.values()}
    for g in gathered_ade:
        for h in ade: ade[h].extend(g[h])
    for g in gathered_fde:
        for h in fde: fde[h].extend(g[h])

    out = {'n': len(ade['6s']), 'ade': {}, 'fde': {}}
    for h in HORIZONS.values():
        if ade[h]:
            out['ade'][h] = {'mean': float(np.mean(ade[h])), 'median': float(np.median(ade[h]))}
            out['fde'][h] = {'mean': float(np.mean(fde[h])), 'median': float(np.median(fde[h]))}
    return out
