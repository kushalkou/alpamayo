"""decode_eval.py — sharded (8-GPU) AR evaluation harness for the decode-rule study.

One harness for every gate: it AR-decodes each sample once per (model, variant),
rolls out the unicycle, and stores PER-SAMPLE ADE/FDE so any stratification or
min-over-K reduction can be done afterwards from the saved json.

A *variant* is a decode rule (see inference.DECODE_MODES):
  argmax                       — the existing/unchanged path
  expect                       — E[value] over all 64 bins            (V1)
  expect_topk  (topk=5)        — E[value] over top-k bins, renormed   (V2)
  expect_temp  (temperature)   — E[value] with tempered logits        (V3)
  sample       (temperature,K) — ancestral sampling, K rollouts       (Gate 3)

The CV (constant-velocity) reference is computed for every sample, always.

Launch:
  cd .../code && python -m torch.distributed.run --nproc_per_node=8 \
      --master_port=29571 decode_eval.py --job gate0
"""
import os, sys, json, pickle, time, argparse
import numpy as np, torch
import torch.distributed as dist

sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model
from dataset import build_scene_split, NuScenesVLADataset, compute_ego_state
from tokenizer import TrajectoryTokenizer
from ar_eval import encode_live_one, fixed_val_indices
import inference as INF
from inference import (decode_trajectory_ex, prefill_context, unicycle_rollout,
                       HORIZONS, N_STEPS)

CK = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
SHARD_ROOT = '/tmp/claude-1000/decode_shards'
CURV_THRESH = 0.05

# name -> (checkpoint, zero_vision)
# name -> (checkpoint, zero_vision, zero_ego)
MODELS = {
    'y1_full': (f'{CK}/_y1_full_turnw/alpamayo_best.pt',    False, False),
    'y1_ego':  (f'{CK}/_y1_egoonly_turnw/alpamayo_best.pt', True,  False),
    # 2.1c zero-input transfer: BOTH modalities zeroed -> the model has no
    # information, so any decode gain must come from the decode rule alone.
    'y1_full_zeroboth': (f'{CK}/_y1_full_turnw/alpamayo_best.pt', True, True),
    'zeroboth_jul12':   (f'{CK}/_zeroboth_run_jul12/alpamayo_best_e1_val2.0392.pt', True, True),
}

def V(name, mode, **kw):
    d = {'name': name, 'decode_mode': mode, 'topk': 5, 'temperature': 1.0, 'K': 1,
         'stop_aware': False, 'a': None, 'k': None, 'stop_bin32': False, 'coarsen': 1}
    d.update(kw); return d

def H(mode='argmax', stop_aware=False, topk=5, temperature=1.0, stop_bin32=False, coarsen=1):
    """per-half decode rule (accel half / curvature half)"""
    return {'mode': mode, 'stop_aware': stop_aware, 'topk': topk, 'temperature': temperature,
            'stop_bin32': stop_bin32, 'coarsen': coarsen}

JOBS = {
    # GATE 0 — reproduce argmax on the full test set, best turn-weighted full-vision ckpt
    'gate0':  dict(split='test', models=['y1_full'], variants=[V('argmax', 'argmax')]),
    # GATE 1 — decode-rule variants on the fixed 400-sample val subset
    'gate1':  dict(split='val', n_val=400, models=['y1_full'], stats_variant='argmax',
                   variants=[
                     V('argmax', 'argmax'),
                     # CONTROL family — literal spec: drop STOP, renormalise over 0..63
                     V('V1_expect', 'expect'),
                     V('V2_topk5', 'expect_topk', topk=5),
                     V('V3_T0.5', 'expect_temp', temperature=0.5),
                     # PRIMARY family — STOP-aware: support {0..63} U {128}, STOP -> 0.0
                     V('V1s_expect', 'expect', stop_aware=True),
                     V('V2s_topk5', 'expect_topk', topk=5, stop_aware=True),
                     V('V3s_T0.5', 'expect_temp', temperature=0.5, stop_aware=True),
                     # MIXED — accel half vs curvature half in opposite regimes
                     V('V4_aExp_kArg_T1.0', 'argmax',
                       a=H('expect', stop_aware=True),        k=H('argmax')),
                     V('V4_aExp_kArg_T0.5', 'argmax',
                       a=H('expect_temp', stop_aware=True, temperature=0.5), k=H('argmax')),
                     V('V5_aArg_kExp', 'argmax',
                       a=H('argmax'), k=H('expect', stop_aware=True)),
                   ]),
    # GATE 2.0/2.1 — STOP-fix fairness delta + mechanism tests (val subset)
    'gate21': dict(split='val', n_val=400, models=['y1_full'], stats_variant='argmax_corrected',
                   variants=[
                     V('argmax_legacy_bin32', 'argmax', stop_bin32=True),
                     V('argmax_corrected',    'argmax'),
                     V('V1s_expect',          'expect', stop_aware=True),
                     V('V2s_topk5',           'expect_topk', topk=5, stop_aware=True),
                     # 2.1b coarsening dose-response (decode-time bin merging)
                     V('argmax_c2', 'argmax', a=H('argmax', coarsen=2), k=H('argmax', coarsen=2)),
                     V('expect_c2', 'expect', a=H('expect', stop_aware=True, coarsen=2),
                                              k=H('expect', stop_aware=True, coarsen=2)),
                     V('argmax_c4', 'argmax', a=H('argmax', coarsen=4), k=H('argmax', coarsen=4)),
                     V('expect_c4', 'expect', a=H('expect', stop_aware=True, coarsen=4),
                                              k=H('expect', stop_aware=True, coarsen=4)),
                   ]),
    # 2.1c — zero-input / ego-only transfer of the decode fix
    'gate21c': dict(split='val', n_val=400,
                    models=['y1_full_zeroboth', 'zeroboth_jul12', 'y1_ego'],
                    variants=[V('argmax_corrected', 'argmax'),
                              V('V1s_expect', 'expect', stop_aware=True)]),
    # GATE 2 — winning variant, full test set, both models (variant filled in at launch)
    'gate2':  dict(split='test', models=['y1_full', 'y1_ego'],
                   variants=[V('argmax', 'argmax')]),
    # GATE 3 — ancestral sampling, K=10 rollouts, two temperatures, full test set
    'gate3':  dict(split='test', models=['y1_full', 'y1_ego'],
                   variants=[V('sample_T1.0', 'sample', temperature=1.0, K=10),
                             V('sample_T0.7', 'sample', temperature=0.7, K=10)]),
}


def gt_local(t):
    return np.array(t['future_positions'])[:N_STEPS] - np.array(t['current_pose']['translation'][:2])

def errs(pred, gt):
    ade, fde = {}, {}
    for s, lab in HORIZONS.items():
        e = np.linalg.norm(pred[:s] - gt[:s], axis=1)
        ade[lab] = float(e.mean()); fde[lab] = float(e[-1])
    return {'ade': ade, 'fde': fde}

def maxcurv(t):
    c = np.abs(np.array(t.get('future_curvatures', [0])))
    return float(c.max()) if len(c) else 0.0

def cv_pred(v0, yaw0):
    return np.array([[v0 * 0.5 * (k + 1) * np.cos(yaw0),
                      v0 * 0.5 * (k + 1) * np.sin(yaw0)] for k in range(N_STEPS)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True, choices=list(JOBS))
    ap.add_argument('--variant_override', type=str, default=None,
                    help="json list of variant dicts, replaces the job's variants")
    ap.add_argument('--models_override', type=str, default=None, help='comma-separated model names')
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--limit', type=int, default=None, help='smoke test: first N samples only')
    args = ap.parse_args()

    job = dict(JOBS[args.job])
    if args.variant_override:
        job['variants'] = json.loads(args.variant_override)
    if args.models_override:
        job['models'] = args.models_override.split(',')
    out_path = os.path.join(RES, args.out or f'res_decode_{args.job}.json')
    shard_dir = os.path.join(SHARD_ROOT, args.job)

    dist.init_process_group(backend='nccl')
    lr = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    device = f'cuda:{lr}'; ws = dist.get_world_size(); m0 = (lr == 0)
    os.makedirs(shard_dir, exist_ok=True)

    with open(INF.TRAJECTORIES_PATH, 'rb') as f:
        allt = pickle.load(f)
    train, val, test = build_scene_split(allt, INF.NUSCENES_ROOT)
    if job['split'] == 'test':
        trajs, ds = test, NuScenesVLADataset(test, split='test', augment=False)
        idx_all = list(range(len(test)))
    else:
        trajs, ds = val, NuScenesVLADataset(val, split='val', augment=False)
        idx_all = [int(i) for i in fixed_val_indices(len(val), k=job.get('n_val', 400))]
    if args.limit:
        idx_all = idx_all[:args.limit]
    if m0:
        print(f"[decode_eval] job={args.job} split={job['split']} n={len(idx_all)} "
              f"models={job['models']} variants={[v['name'] for v in job['variants']]}", flush=True)

    tok = TrajectoryTokenizer()
    model = load_model(device=device)
    model.cosmos.model.language_model.gradient_checkpointing_disable(); model.eval()
    visual = model.cosmos.model.visual
    torch.set_grad_enabled(False)
    my = idx_all[lr::ws]

    # CV reference + curvature (cheap, model-free)
    cv, curv, spd = {}, {}, {}
    for i in my:
        t = trajs[i]; ego = compute_ego_state(t)
        v0, yaw0 = float(t['future_speeds'][0]), float(ego[3, 1])
        cv[i] = errs(cv_pred(v0, yaw0), gt_local(t)); curv[i] = maxcurv(t); spd[i] = v0

    stats_variant = job.get('stats_variant')
    per = {}            # per[model][variant][idx] = list of K {'ade','fde'}
    slot_stats = {}     # slot_stats[model] = [24 x {entropy,top1,dead_mass,stop_mass} sums], count
    t0 = time.time()
    for mname in job['models']:
        path, zv, ze = MODELS[mname]
        ck = torch.load(path, map_location='cpu')
        model.load_state_dict(ck['model_state'], strict=False); model.zero_ego = ze
        per[mname] = {v['name']: {} for v in job['variants']}
        SKEYS = ('entropy', 'top1', 'dead_mass', 'stop_mass', 'argmax_is_stop', 'bimodal',
                 'resid_bw')
        sacc = [{k: 0. for k in SKEYS} for _ in range(24)]
        resid = [[] for _ in range(24)]      # full residual distribution (2.1a)
        sn = 0
        for c, i in enumerate(my):
            t = trajs[i]; item = ds[i]
            vt = encode_live_one(visual, item['images'], device)
            if zv: vt = torch.zeros_like(vt)
            ego = item['ego_state']
            v0, yaw0 = float(t['future_speeds'][0]), float(ego[3, 1])
            gl = gt_local(t)
            # ONE prefill per sample, reused by every variant and every K-rollout
            pre = prefill_context(model, vt, ego, device=device)
            for v in job['variants']:
                runs = []
                K = int(v.get('K', 1))
                for r in range(K):
                    gen = None
                    if v['decode_mode'] == 'sample':
                        gen = torch.Generator(device=device)
                        gen.manual_seed(1000003 * i + 7919 * r + 13)
                    want_stats = (v['name'] == stats_variant and r == 0)
                    _, acc, cur, st = decode_trajectory_ex(
                        model, vt, ego, tok, device=device,
                        decode_mode=v['decode_mode'], topk=v['topk'],
                        temperature=v['temperature'], collect_stats=want_stats,
                        generator=gen, pre=pre, stop_aware=v.get('stop_aware', False),
                        a_spec=v.get('a'), k_spec=v.get('k'),
                        stop_bin32=v.get('stop_bin32', False), coarsen=v.get('coarsen', 1))
                    pred, _ = unicycle_rollout(acc, cur, v0, yaw0)
                    runs.append(errs(pred, gl))
                    if want_stats:
                        for s_i in range(24):
                            for k_ in sacc[s_i]: sacc[s_i][k_] += st[s_i][k_]
                            resid[s_i].append(st[s_i]['resid_bw'])
                        sn += 1
                per[mname][v['name']][i] = runs
            del pre
            if m0 and c % 50 == 0:
                el = time.time() - t0
                print(f"  [{mname}] {c}/{len(my)}  {el:.0f}s", flush=True)
        if sn:
            slot_stats[mname] = {'n': sn, 'sums': sacc, 'resid': resid}

    payload = {'per': per, 'cv': cv, 'curv': curv, 'v0': spd, 'slot_stats': slot_stats}
    tmp = os.path.join(shard_dir, f's_{lr}.pkl.tmp')
    with open(tmp, 'wb') as f: pickle.dump(payload, f)
    os.replace(tmp, os.path.join(shard_dir, f's_{lr}.pkl'))
    if not m0:
        dist.destroy_process_group(); return

    while sum(os.path.exists(os.path.join(shard_dir, f's_{r}.pkl')) for r in range(ws)) < ws:
        time.sleep(5)
    G = [pickle.load(open(os.path.join(shard_dir, f's_{r}.pkl'), 'rb')) for r in range(ws)]
    PER = {m: {v['name']: {} for v in job['variants']} for m in job['models']}
    CV, CURV, V0, SS = {}, {}, {}, {}
    for g in G:
        for m in g['per']:
            for vn in g['per'][m]: PER[m][vn].update(g['per'][m][vn])
        CV.update(g['cv']); CURV.update(g['curv']); V0.update(g['v0'])
        for m, s in g['slot_stats'].items():
            if m not in SS: SS[m] = {'n': 0, 'sums': [{k: 0. for k in s['sums'][0]} for _ in range(24)],
                                     'resid': [[] for _ in range(24)]}
            SS[m]['n'] += s['n']
            for i in range(24):
                for k in s['sums'][i]: SS[m]['sums'][i][k] += s['sums'][i][k]
                SS[m]['resid'][i].extend(s.get('resid', [[]]*24)[i])

    res = {'job': args.job, 'split': job['split'], 'n': len(CURV),
           'variants': job['variants'], 'models': job['models'],
           'curv': {str(i): CURV[i] for i in CURV},
           'v0': {str(i): V0[i] for i in V0},
           'cv': {str(i): CV[i] for i in CV},
           'per': {m: {vn: {str(i): PER[m][vn][i] for i in PER[m][vn]} for vn in PER[m]} for m in PER},
           'slot_stats': {m: {'n': SS[m]['n'],
                              'mean': [{k: SS[m]['sums'][i][k] / SS[m]['n'] for k in SS[m]['sums'][i]}
                                       for i in range(24)],
                              'resid': SS[m]['resid']} for m in SS}}
    json.dump(res, open(out_path, 'w'))
    print(f"[decode_eval] saved -> {out_path}", flush=True)
    report(res)
    print("DECODE_EVAL_DONE", flush=True)
    dist.destroy_process_group()


def report(res):
    idx  = sorted(int(i) for i in res['curv'])
    curv = {int(i): v for i, v in res['curv'].items()}
    cvd  = {int(i): v for i, v in res['cv'].items()}
    v0d  = {int(i): v for i, v in res.get('v0', {}).items()}
    subsets = [('ALL', idx),
               ('STRAIGHT', [i for i in idx if curv[i] <= CURV_THRESH]),
               ('TURNING',  [i for i in idx if curv[i] > CURV_THRESH])]
    if v0d:
        subsets.append(('STATIONARY (|v0|<0.5 m/s)',
                        [i for i in idx if abs(v0d.get(i, 9e9)) < 0.5]))
    labs = ['1s', '2s', '3s', '6s']
    for sname, sub in subsets:
        if not sub:
            print(f"\n-- {sname} (n=0) — empty subset --"); continue
        print(f"\n-- {sname} (n={len(sub)}) --")
        print(f"{'model/variant':24} " + " ".join(f"{'ADE'+l:>8}" for l in labs) +
              f" {'medADE6s':>9} {'FDE6s':>8}")
        row = [np.mean([cvd[i]['ade'][l] for i in sub]) for l in labs]
        print(f"{'CV (baseline)':24} " + " ".join(f"{v:8.3f}" for v in row) +
              f" {np.median([cvd[i]['ade']['6s'] for i in sub]):9.3f} "
              f"{np.mean([cvd[i]['fde']['6s'] for i in sub]):8.3f}")
        for m in res['per']:
            for vn in res['per'][m]:
                P = {int(i): v for i, v in res['per'][m][vn].items()}
                K = max(len(P[i]) for i in P)
                ks = sorted({1, 5, 10} & set(range(1, K + 1))) if K > 1 else [1]
                for k in ks:
                    a = {l: [min(r['ade'][l] for r in P[i][:k]) for i in sub if i in P] for l in labs}
                    f6 = np.mean([min(r['fde']['6s'] for r in P[i][:k]) for i in sub if i in P])
                    tag = f"{m}/{vn}" + (f"@{k}" if K > 1 else "")
                    print(f"{tag:24} " + " ".join(f"{np.mean(a[l]):8.3f}" for l in labs) +
                          f" {np.median(a['6s']):9.3f} {f6:8.3f}")

    for m, st in res.get('slot_stats', {}).items():
        n = st['n']; S = st['mean']
        print(f"\n=== DIAGNOSTICS [{m}] (n={n} samples, greedy-argmax token stream) ===")
        print(f"{'slot':5} {'kind':6} {'entropy':>8} {'top1':>7} {'dead64_127':>11} "
              f"{'p(STOP)':>8} {'argmax=STOP':>12} {'bimodal':>8}")
        for i, d in enumerate(S):
            flag = '  <-- DEAD>1%' if d['dead_mass'] > 0.01 else ''
            print(f"{i:<5} {'accel' if i < 12 else 'curv':6} {d['entropy']:8.3f} {d['top1']:7.3f} "
                  f"{d['dead_mass']:11.5f} {d['stop_mass']:8.4f} {d['argmax_is_stop']:12.4f} "
                  f"{d['bimodal']:8.4f}{flag}")
        dead_max = max(d['dead_mass'] for d in S)
        print(f"\n[DIAG 1] dead-class mass ids 64-127: max over slots = {dead_max:.6f} "
              f"({'FLAG: exceeds 1%' if dead_max > 0.01 else 'OK, <1% everywhere'})")
        bm = [d['bimodal'] for d in S[12:]]
        print(f"[DIAG 2] curvature bimodality (top-2 bins non-adjacent, gap>3): "
              f"mean {np.mean(bm):.4f}, max slot {np.max(bm):.4f}, min slot {np.min(bm):.4f}")
        st_a = [d['argmax_is_stop'] for d in S[:12]]; st_k = [d['argmax_is_stop'] for d in S[12:]]
        print(f"[DIAG 3] argmax selects STOP(128): accel slots mean {np.mean(st_a):.4f}, "
              f"curv slots mean {np.mean(st_k):.4f}, max any slot {max(max(st_a), max(st_k)):.4f}")
        R = st.get('resid')
        if R:
            print(f"\n[DIAG 2.1a] |E[value] - argmax_center| in BIN WIDTHS")
            print(f"{'half':8} {'n':>7} {'mean':>7} {'median':>7} {'p90':>7} {'p99':>7} "
                  f"{'frac>0.5':>9} {'frac>1.0':>9}")
            for lab, rng_ in [('accel', range(0, 12)), ('curv', range(12, 24))]:
                v = np.array([x for i in rng_ for x in R[i]])
                if not len(v): continue
                print(f"{lab:8} {len(v):7d} {v.mean():7.3f} {np.median(v):7.3f} "
                      f"{np.percentile(v,90):7.3f} {np.percentile(v,99):7.3f} "
                      f"{(v>0.5).mean():9.4f} {(v>1.0).mean():9.4f}")
        ea = [d['entropy'] for d in S[:12]]; ek = [d['entropy'] for d in S[12:]]
        t1a = [d['top1'] for d in S[:12]]; t1k = [d['top1'] for d in S[12:]]
        print(f"[DIAG 4] entropy: accel mean {np.mean(ea):.3f} (slots 1-11 "
              f"{np.mean(ea[1:]):.3f}), curv mean {np.mean(ek):.3f} | "
              f"top-1 mass: accel {np.mean(t1a):.3f} (slots 1-11 {np.mean(t1a[1:]):.3f}), "
              f"curv {np.mean(t1k):.3f}")


if __name__ == '__main__':
    main()
