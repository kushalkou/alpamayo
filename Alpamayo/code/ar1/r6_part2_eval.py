"""ar1/r6_part2_eval.py -- R6 Part 2 gates GS1-GS5 and per-run diagnostics (header rules).
Strata (pre-registered): all, excl. first frames, excl. WSS, WSS only.
  python ar1/r6_part2_eval.py"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from r33_eval import derive, controls
from r5_a5c1 import coll_rates
from meta_actions import LON, LAT

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
NC = ('A3', 'G6C_s123', 'G6C_s2024'); B1 = ('B1', 'B1_s123', 'B1_s2024'); V2 = ('M1v2a', 'M1v2a_s123', 'M1v2a_s2024')


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); ff = C['ff']; occ = C['occ']
    wss = pickle.load(open(f'{DATA}/r5_wss.pkl', 'rb')); w = np.array([wss[r['scene_name']] for r in rva])
    S = {'all': np.ones(len(rva), bool), 'excl. first frames': ~ff, 'excl. WSS': ~w, 'WSS only': w}
    mean = lambda tags: Q.avg([Q.run(t) for t in tags])
    nc, b1, v2 = mean(NC), mean(B1), mean(V2); v2n = Q.strip(Q.run('M1v2a_nocam')); v42 = Q.strip(Q.run('M1v2a'))
    print('tau: ' + ', '.join(f'{t} {Q.run(t)["tau"]}' for t in V2 + ('M1v2a_nocam',)))
    print('\nL2@3s NoAvg per seed and 3-seed mean +- sd (all 5,119):')
    for nm, tags in (('no cam', NC), ('B1', B1), ('M1-v2a', V2)):
        v = np.array([Q.run(t)['l2'][:, 5].mean() for t in tags])
        print(f'  {nm:7} ' + ' '.join(f'{x:.3f}' for x in v) + f' | {v.mean():.3f} +- {v.std(ddof=1):.3f}')
    print(f'  no-cam M1-v2a s42 {v2n["l2"][:, 5].mean():.3f}')
    G = {}
    print('\nPaired scene bootstrap (10,000), L2@3s NoAvg; ADE@6s in brackets after')
    for pair, A, B in (('M1-v2a 3s - B1 3s', v2, b1), ('M1-v2a 3s - no cam 3s', v2, nc),
                       ('M1-v2a s42 - no-cam M1-v2a s42', v42, v2n)):
        for sn, m in S.items():
            sub = [r for r, k in zip(rva, m) if k]
            c = E.scene_boot(sub, A['l2'][m, 5], B['l2'][m, 5]); c2 = E.scene_boot(sub, A['ade'][m], B['ade'][m])
            G[(pair, sn)] = c
            print(f'  {pair:31} {sn:19} n {m.sum():4d}  {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]   ADE6 {c2[0]:+.3f} [{c2[1]:+.3f},{c2[2]:+.3f}]')
    g = np.linalg.norm(np.stack([np.asarray(T[r['sample_token']]['P'])[:6] for r in rva]), axis=2).max(1)
    stat = C['st'] == 'stationary'
    print('\nPer run: false-go / missed-go (all; WSS; non-WSS), collision 3 s NoAvg veh and TemAvg veh+ped, VAD port (aa) and'
          ' yaw-aware')
    fgs = {}
    for t in NC + B1 + V2 + ('M1v2a_nocam',):
        P = preds(load(t, 'val'), rva, 'hybrid', Q.run(t)['tau']); d = np.linalg.norm(P[:, :6], axis=2).max(1)
        fg = lambda m: (d[m & stat & (g < 0.5)] > 1.0).mean()
        mg = (d[stat & (g > 1.0)] < 0.5).mean(); fgs[t] = fg(np.ones(len(rva), bool))
        line = f'  {t:12} FG {fgs[t]:.3f} / {fg(w):.3f} / {fg(~w):.3f}  MG {mg:.3f}'
        if t in V2 + ('M1v2a_nocam',):
            ca = coll_rates(rva, P, occ, 'aa'); cy = coll_rates(rva, P, occ, 'yaw')
            line += f'  Col No/Tem aa {ca[0]:.2f}/{ca[1]:.2f} yaw {cy[0]:.2f}/{cy[1]:.2f}'
        print(line)
    pooled = np.mean([fgs[t] for t in V2])
    print('\nWords (val n_fut = 12, 2 Hz labels): collapse share, per-class F1 pooled over slots')
    lab = pickle.load(open(f'{DATA}/ar1_meta2hz.pkl', 'rb')); r12 = [r for r in rva if r['n_fut'] == 12]
    y = np.array([lab[r['sample_token']] for r in r12]); maj = np.array([4] * 6 + [6] * 6)
    for t in V2 + ('M1v2a_nocam',):
        dv = load(t, 'val')['data']; gen = np.array([dv[r['sample_token']]['meta_gen'] for r in r12])
        f = []
        for o, nm in ((0, LON), (6, LAT)):
            yy, pp = y[:, o:o + 6].ravel(), gen[:, o:o + 6].ravel()
            for c in range(7):
                n = (yy == c).sum()
                if n < 30:
                    continue
                tp = ((pp == c) & (yy == c)).sum(); P_ = tp / max((pp == c).sum(), 1); R_ = tp / n
                f.append(f'{nm[c][:9]} {2 * P_ * R_ / max(P_ + R_, 1e-9):.2f}')
        dva = load(t, 'val'); ga = np.array([dva['data'][r['sample_token']]['meta_gen'] for r in rva])
        der = np.array([derive(a, k, T[r['sample_token']]['v0']) for (a, k), r in zip(controls(dva, rva, Q.run(t)['tau']), rva)])
        print(f'  {t:12} collapse {(gen == maj).all(1).mean():.3f} (labels {(y == maj).all(1).mean():.3f}); consistency '
              f'{(der == ga).all(1).mean():.3f}; F1: ' + ', '.join(f))
    c1 = G[('M1-v2a 3s - B1 3s', 'all')]; c2a = G[('M1-v2a 3s - B1 3s', 'excl. first frames')]
    c2b = G[('M1-v2a 3s - B1 3s', 'excl. WSS')]; c4 = G[('M1-v2a 3s - no cam 3s', 'all')]
    c5 = G[('M1-v2a s42 - no-cam M1-v2a s42', 'all')]
    print('\nGATES')
    print(f'  GS1 M1-v2a 3s - B1 3s, all: {c1[0]:+.3f} [{c1[1]:+.3f},{c1[2]:+.3f}] -> '
          + ('"meta-action (consistent labels) improves on traj-only"' if c1[2] < 0 else 'condition not met'))
    print(f'  GS2 excl. first frames {c2a[0]:+.3f} [{c2a[1]:+.3f},{c2a[2]:+.3f}] ({"upper < 0" if c2a[2] < 0 else "not < 0"}); '
          f'excl. WSS {c2b[0]:+.3f} [{c2b[1]:+.3f},{c2b[2]:+.3f}] ({"upper < 0" if c2b[2] < 0 else "not < 0"})')
    print(f'  GS3 M1-v2a pooled false-go {pooled:.3f} -> ' + ('"false-go resolved"' if pooled <= 0.10 else 'condition not met'))
    print(f'  GS4 M1-v2a 3s - no cam 3s: {c4[0]:+.3f} [{c4[1]:+.3f},{c4[2]:+.3f}] -> '
          + ('"cameras + words beat no-camera"' if c4[2] < 0 else 'condition not met'))
    v5 = ('"cameras still hurt with words"' if c5[1] > 0 else '"cameras neutral with words"' if c5[1] <= 0 <= c5[2]
          else 'upper < 0 (cameras help with words; outside the two pre-registered labels)')
    print(f'  GS5 M1-v2a s42 - no-cam M1-v2a s42: {c5[0]:+.3f} [{c5[1]:+.3f},{c5[2]:+.3f}] -> {v5}')


if __name__ == '__main__':
    main()
