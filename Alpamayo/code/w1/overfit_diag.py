"""w1/overfit_diag.py -- STEP 3.5a FAIL diagnosis on the 256 overfit samples.
Free-running (argmax fed back) dumps of the best (selected) and latest (epoch 150)
checkpoints. Reports: floor on the same 256; AR ADE@6s under argmax / expectation /
hybrid; per-slot AR token accuracy; exact-sequence rate; slot of the FIRST wrong
token; ADE split by whether/where the first error occurs; |value error| of wrong
tokens in bins (near-misses vs far)."""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
from targets import rollout as roll12
from w1tok import W1Tokenizer

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
T = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_targets.pkl', 'rb'))['targets']
tok = W1Tokenizer()


def detok(ids):
    a = [0.0 if x == 128 else tok.accel_centers[x] for x in ids[:12]]
    k = [0.0 if x == 128 else tok.curv_centers[x] for x in ids[12:]]
    return np.array(a), np.array(k)


for ck in ('best', 'latest'):
    D = pickle.load(open(f'{RES}/w1_dump_overfit256_{ck}_train_f0.0.pkl', 'rb'))
    rows = [(st, D['data'][st]) for st in D['order']]
    print(f'==== checkpoint {ck} (epoch {D["ckpt_epoch"]}), n={len(rows)} ====')
    ade = {m: [] for m in ('floor', 'argmax', 'expect', 'hybrid0.5')}
    first, exact, per_slot, verr = [], [], np.zeros(24), []
    for st, d in rows:
        t = T[st]; P = t['P'][:12]; gt = d['gt_tok']; pr = d['tok']
        a, k = detok(gt); ade['floor'].append(np.linalg.norm(roll12(a, k, t['v0']) - P, axis=1).mean())
        for m in ('argmax', 'expect'):
            v = np.array(d[m]); ade[m].append(np.linalg.norm(roll12(v[:12], v[12:], t['v0']) - P, axis=1).mean())
        v = np.where(np.array(d['p_stop']) > 0.5, np.array(d['argmax']), np.array(d['expect']))
        ade['hybrid0.5'].append(np.linalg.norm(roll12(v[:12], v[12:], t['v0']) - P, axis=1).mean())
        ok = np.array([p == g for p, g in zip(pr, gt)]); per_slot += ok
        exact.append(ok.all()); first.append(int(np.argmin(ok)) if not ok.all() else -1)
        for s in np.where(~ok)[0]:
            if pr[s] != 128 and gt[s] != 128: verr.append(abs(pr[s] - gt[s]))
    ade = {m: np.array(v) for m, v in ade.items()}
    for m, v in ade.items():
        print(f'  ADE@6s {m:10} mean {v.mean():.3f}  median {np.median(v):.3f}  p90 {np.percentile(v,90):.3f}')
    first = np.array(first); exact = np.array(exact)
    print(f'  exact 24-token sequence: {exact.mean():.3f}')
    print('  AR per-slot token accuracy: accel ' + ' '.join(f'{x:.2f}' for x in per_slot[:12] / len(rows))
          + ' | curv ' + ' '.join(f'{x:.2f}' for x in per_slot[12:] / len(rows)))
    h = np.bincount(first[first >= 0], minlength=24)
    print('  first wrong slot histogram (slot:count): ' + ' '.join(f'{s}:{c}' for s, c in enumerate(h) if c))
    for lab, m in (('all tokens right', first == -1), ('first error in accel slots 0-11', (first >= 0) & (first < 12)),
                   ('first error in curv slots 12-23', first >= 12)):
        if m.any():
            print(f'  {lab:32} n={m.sum():3d}  argmax ADE mean {ade["argmax"][m].mean():.3f} '
                  f'median {np.median(ade["argmax"][m]):.3f}   floor {ade["floor"][m].mean():.3f}')
    verr = np.array(verr)
    print(f'  wrong non-STOP tokens: n={len(verr)}, |bin error| 1: {np.mean(verr==1):.2f}, 2-3: '
          f'{np.mean((verr>=2)&(verr<=3)):.2f}, >=4: {np.mean(verr>=4):.2f}')
