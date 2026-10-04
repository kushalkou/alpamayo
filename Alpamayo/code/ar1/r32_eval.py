"""ar1/r32_eval.py -- R3.2: B1 (3 front cams t0, native order, cached) over 3 seeds
(42 = B1, 123 = B1_s123, 2024 = B1_s2024) vs the no-camera VLA 3-seed mean (T9b: A3,
G6C_s123, G6C_s2024). Shared eval (q2_traj; hybrid decode, tau per run on holdout).
Per-seed values, mean +- sd (ddof 1) per metric; paired scene bootstrap of the seed-MEAN
per-sample metrics (B1 mean - no-cam mean); strata; stationary false-go (R3.0a rule).
  python ar1/r32_eval.py"""
import sys
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds

B1 = {42: 'B1', 123: 'B1_s123', 2024: 'B1_s2024'}


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data()
    LEN, ci, _ = Q.seed_tags()
    print('tau: ' + ', '.join(f'{t} {Q.run(t)["tau"]}' for t in list(B1.values()) + list(ci.values())))
    G = np.stack([np.asarray(T[r['sample_token']]['P'])[:6] for r in rva]); gmax = np.linalg.norm(G, axis=2).max(1)
    for blk, m in Q.blocks():
        sub = [r for r, k in zip(rva, m) if k]
        cut = lambda p: {k: v[m] for k, v in Q.strip(p).items()}
        print(f'\n===== {blk} =====')
        for nm, tags in (('no cam', ci), ('B1', B1)):
            per = [cut(Q.run(t)) for t in tags.values()]
            S = [E.summary(p) for p in per]
            for key, lab in (('L2_NoAvg_3s', 'L2@3s NoAvg'), ('L2_TemAvg_3s', 'L2@3s TemAvg'), ('ADE6', 'ADE@6s'),
                             ('ADE6_med', 'ADE@6s med'), ('L2_3s_p95', 'L2@3s p95'), ('Col_TemAvg_3s', 'Col% TemAvg 3s')):
                v = np.array([s[key] for s in S])
                print(f'  {nm:7} {lab:15} ' + ' '.join(f's{sd}={x:.3f}' for sd, x in zip(tags, v))
                      + f' | mean {v.mean():.3f} +- {v.std(ddof=1):.3f}')
        nc = Q.avg([cut(Q.run(t)) for t in ci.values()]); b1 = Q.avg([cut(Q.run(t)) for t in B1.values()])
        print(E.compare(sub, b1, nc, 'B1 3-seed mean - no-cam 3-seed mean'))
        st = C['st'][m]
        for s in ('straight', 'turning', 'stationary'):
            k = st == s
            c = E.scene_boot([r for r, q in zip(sub, k) if q], b1['l2'][k, 5], nc['l2'][k, 5])
            print(f'  {s:10} n={int(k.sum()):4d}  L2@3s no-cam {nc["l2"][k, 5].mean():.3f}  B1 {b1["l2"][k, 5].mean():.3f}'
                  f'  diff {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]')
        stat = (C['st'] == 'stationary') & m; gs = stat & (gmax < 0.5)
        fg = {}
        for nm, tags in (('no cam', ci), ('B1', B1)):
            fg[nm] = [float((np.linalg.norm(preds(load(t, 'val'), rva, 'hybrid', Q.run(t)['tau'])[:, :6], axis=2).max(1)[gs] > 1.0).mean())
                      for t in tags.values()]
        print('  stationary false-go (R3.0a rule): ' + '; '.join(f'{k} ' + ' '.join(f'{x:.3f}' for x in v)
                                                         + f' (mean {np.mean(v):.3f})' for k, v in fg.items()))


if __name__ == '__main__':
    main()
