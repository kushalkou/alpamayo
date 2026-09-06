"""dump_decode.py — dump per-slot decode values so every downstream sweep is offline.

KEY: the argmax decode and the STOP-aware expectation decode feed back the SAME
token stream (both condition on the argmax token). So a single AR pass yields the
value sequence for BOTH, plus p(STOP) per slot. That halves GPU cost and lets the
tau / alpha / lambda / gamma sweeps run offline on CPU from the dump.

Per sample we store: argmax_val[24], expect_val[24], p_stop[24], gt_local[12,2],
v0, yaw0, max|GT curv|, max|pred curv|, and the number of real past poses.

Launch:
  python -m torch.distributed.run --nproc_per_node=8 dump_decode.py --split val
"""
import os, sys, json, pickle, time, argparse
import numpy as np, torch
import torch.distributed as dist

sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model
from dataset import build_scene_split, NuScenesVLADataset, compute_ego_state
from tokenizer import TrajectoryTokenizer
from ar_eval import encode_live_one
import inference as INF
from inference import prefill_context, slot_centers, N_ACT, STOP_ID
from model import TRAJ_LEN

CK = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'
OUT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
MODELS = {
    'y1_full': (f'{CK}/_y1_full_turnw/alpamayo_best.pt',    False, False),
    'y1_ego':  (f'{CK}/_y1_egoonly_turnw/alpamayo_best.pt', True,  False),
    'zeroboth_jul12': (f'{CK}/_zeroboth_run_jul12/alpamayo_best_e1_val2.0392.pt', True, True),
}


@torch.no_grad()
def dump_one(raw, vt, ego, tok, device):
    """One AR pass (argmax-conditioned). Returns argmax_val[24], expect_val[24], p_stop[24]."""
    lm = raw.cosmos.model.language_model
    pre = prefill_context(raw, vt, ego, device=device)
    past, ctx_len, ctx_dtype = pre['past'], pre['ctx_len'], pre['ctx_dtype']
    past.crop(ctx_len)
    logits = pre['logits0']
    av, ev, ps = [], [], []
    for step in range(TRAJ_LEN):
        centers = torch.as_tensor(slot_centers(tok, step), dtype=torch.float32, device=logits.device)
        tokid = int(logits.argmax(-1).item())
        act = logits[:N_ACT]
        # STOP-aware support {0..63} U {128}; STOP -> 0.0 (tokenizer detokenize_step)
        vlog  = torch.cat([act, logits[STOP_ID:STOP_ID + 1]])
        vcent = torch.cat([centers, torch.zeros(1, device=logits.device)])
        q = torch.softmax(vlog, dim=-1)
        ev.append(float((q * vcent).sum().item()))
        ps.append(float(q[-1].item()))
        av.append(0.0 if tokid == STOP_ID else float(centers[min(max(tokid, 0), N_ACT - 1)].item()))
        if step == TRAJ_LEN - 1:
            break
        t = torch.tensor([tokid], device=device, dtype=torch.long)
        emb = raw.traj_embed(t).unsqueeze(1).to(ctx_dtype)
        out = lm(inputs_embeds=emb, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = raw.output_head(out.last_hidden_state[:, -1, :].float())[0]
    past.crop(ctx_len)
    return np.array(av), np.array(ev), np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['val', 'test'])
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    shard = f'/tmp/claude-1000/dump_shards/{args.split}'

    dist.init_process_group(backend='nccl')
    lr = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    device = f'cuda:{lr}'; ws = dist.get_world_size(); m0 = (lr == 0)
    os.makedirs(shard, exist_ok=True)

    with open(INF.TRAJECTORIES_PATH, 'rb') as f: allt = pickle.load(f)
    train, val, test = build_scene_split(allt, INF.NUSCENES_ROOT)
    trajs = val if args.split == 'val' else test
    ds = NuScenesVLADataset(trajs, split=args.split, augment=False)
    idx_all = list(range(len(trajs)))
    if args.limit: idx_all = idx_all[:args.limit]
    if m0: print(f"[dump] split={args.split} n={len(idx_all)}", flush=True)

    tok = TrajectoryTokenizer()
    model = load_model(device=device)
    model.cosmos.model.language_model.gradient_checkpointing_disable(); model.eval()
    visual = model.cosmos.model.visual
    torch.set_grad_enabled(False)
    my = idx_all[lr::ws]

    meta = {}
    for i in my:
        t = trajs[i]; ego = compute_ego_state(t)
        c = np.abs(np.array(t.get('future_curvatures', [0])))
        sp = np.maximum(np.abs(ego.numpy()[:, 0]), 1e-3)
        meta[i] = {
            'gt': (np.array(t['future_positions'])[:12] -
                   np.array(t['current_pose']['translation'][:2])).tolist(),
            'v0': float(t['future_speeds'][0]),
            'yaw0': float(ego[3, 1]),
            'maxcurv': float(c.max()) if len(c) else 0.0,
            'past_curv': float(np.max(np.abs(ego.numpy()[:, 2] / sp))),
            'n_hist': int(len(t.get('past_poses', []))),
        }

    data = {}
    t0 = time.time()
    for mname, (path, zv, ze) in MODELS.items():
        ck = torch.load(path, map_location='cpu')
        model.load_state_dict(ck['model_state'], strict=False); model.zero_ego = ze
        d = {}
        for c, i in enumerate(my):
            item = ds[i]
            vt = encode_live_one(visual, item['images'], device)
            if zv: vt = torch.zeros_like(vt)
            av, ev, ps = dump_one(model, vt, item['ego_state'], tok, device)
            d[i] = {'argmax': av.tolist(), 'expect': ev.tolist(), 'p_stop': ps.tolist()}
            if m0 and c % 50 == 0:
                print(f"  [{mname}] {c}/{len(my)}  {time.time()-t0:.0f}s", flush=True)
        data[mname] = d

    tmp = os.path.join(shard, f's_{lr}.pkl.tmp')
    with open(tmp, 'wb') as f: pickle.dump({'meta': meta, 'data': data}, f)
    os.replace(tmp, os.path.join(shard, f's_{lr}.pkl'))
    if not m0:
        dist.destroy_process_group(); return
    while sum(os.path.exists(os.path.join(shard, f's_{r}.pkl')) for r in range(ws)) < ws:
        time.sleep(5)
    META, DATA = {}, {m: {} for m in MODELS}
    for r in range(ws):
        g = pickle.load(open(os.path.join(shard, f's_{r}.pkl'), 'rb'))
        META.update(g['meta'])
        for m in g['data']: DATA[m].update(g['data'][m])
    op = os.path.join(OUT, f'dump_{args.split}.pkl')
    with open(op, 'wb') as f: pickle.dump({'meta': META, 'data': DATA}, f)
    print(f"[dump] saved -> {op}  n={len(META)}", flush=True)
    print("DUMP_DONE", flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
