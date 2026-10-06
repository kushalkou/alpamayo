"""ar1/r6_final.py -- R6 final addendum F2 (2x2 table) and F3 (old M1 false-go slot-1 words).
Per-sample dumps only (F1). Rules in the R6_REPORT.md header.
  python ar1/r6_final.py [--f3]"""
import sys, os, pickle
from collections import Counter
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from r5_a5c1 import coll_rates
from meta_actions import LON

CELLS = (('no camera', 'traj-only', ('A3', 'G6C_s123', 'G6C_s2024')),
         ('no camera', '+meta 2 Hz', ('M1v2a_nocam',)),
         ('cameras', 'traj-only', ('B1', 'B1_s123', 'B1_s2024')),
         ('cameras', '+meta 2 Hz', ('M1v2a', 'M1v2a_s123', 'M1v2a_s2024')))


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); occ = C['occ']
    g = np.linalg.norm(np.stack([np.asarray(T[r['sample_token']]['P'])[:6] for r in rva]), axis=2).max(1)
    stat = C['st'] == 'stationary'; gs = stat & (g < 0.5)
    print(f'GT-stopped stationary val n {gs.sum()}; GT-moving stationary (missed-go base) n {(stat & (g > 1.0)).sum()}')
    if '--f3' in sys.argv:
        cnt = Counter(); n = 0; nfg = 0
        for t in ('M1', 'M1_s123', 'M1_s2024'):
            d = load(t, 'val'); P = preds(d, rva, 'hybrid', Q.run(t)['tau'])
            dm = np.linalg.norm(P[:, :6], axis=2).max(1); fg = gs & (dm > 1.0)
            w = [LON[d['data'][rva[i]['sample_token']]['meta_gen'][0]] for i in np.where(fg)[0]]
            print(f'  {t:9} false-go {fg.sum()}/{gs.sum()} = {fg.sum() / gs.sum():.3f}; slot-1 {dict(Counter(w))}')
            cnt.update(w); n += gs.sum(); nfg += fg.sum()
        print(f'F3 old M1 pooled: false-go {nfg}/{n} = {nfg / n:.3f}; slot-1 word among false-gos: ' +
              ', '.join(f'{k} {cnt[k]} ({cnt[k] / nfg:.3f})' for k in ('maintain', 'gentle_acc')) +
              f', other {nfg - cnt["maintain"] - cnt["gentle_acc"]} ({(nfg - cnt["maintain"] - cnt["gentle_acc"]) / nfg:.3f}) = ' +
              str({k: v for k, v in cnt.items() if k not in ('maintain', 'gentle_acc')}))
        return
    print('F2 2x2 (per-sample dumps; per run, then mean over seeds; * single seed)')
    print(f'  {"cell":24} {"n":>2} {"L2@3s":>6} {"ADE@6s":>6} {"FG":>6} {"MG":>6} {"col aa":>6} {"col yaw":>7}  per-seed L2@3s')
    for row, col, tags in CELLS:
        if not all(os.path.exists(f'{Q.RES}/w1_dump_{t}_val_f0.0.pkl') for t in tags):
            print(f'  {row} / {col}: dumps missing'); continue
        v = []
        for t in tags:
            R = Q.run(t); P = preds(load(t, 'val'), rva, 'hybrid', R['tau'])
            dm = np.linalg.norm(P[:, :6], axis=2).max(1)
            v.append([R['l2'][:, 5].mean(), np.nanmean(R['ade']), (dm[gs] > 1.0).mean(), (dm[stat & (g > 1.0)] < 0.5).mean(),
                      coll_rates(rva, P, occ, 'aa')[0], coll_rates(rva, P, occ, 'yaw')[0]])
        v = np.array(v); m = v.mean(0)
        print(f'  {row + " / " + col:24} {len(tags):2d}{"*" if len(tags) == 1 else " "}{m[0]:6.3f} {m[1]:6.3f} {m[2]:6.3f} '
              f'{m[3]:6.3f} {m[4]:6.2f} {m[5]:7.2f}  ' + ' '.join(f'{x:.3f}' for x in v[:, 0]))


if __name__ == '__main__':
    main()
