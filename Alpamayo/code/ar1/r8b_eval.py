"""ar1/r8b_eval.py -- R8 Part B: no-route arms, 3 seeds. Gates GN1-3, GN2-3 and per-run
diagnostics. Rules in the R8_REPORT.md header. Per-sample dumps, holdout tau.
  python ar1/r8b_eval.py"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from r5_a5c1 import coll_rates

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
ARMS = {'M1-v2a': ('M1v2a', 'M1v2a_s123', 'M1v2a_s2024'), 'B1': ('B1', 'B1_s123', 'B1_s2024'),
        'M1-v2a-NR': ('M1v2a_NR', 'M1v2a_NR_s123', 'M1v2a_NR_s2024'), 'B1-NR': ('B1_NR', 'B1_NR_s123', 'B1_NR_s2024')}


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); ff = C['ff']; occ = C['occ']
    tokv = [r['sample_token'] for r in rva]
    g = np.linalg.norm(np.stack([np.asarray(T[t]['P'])[:6] for t in tokv]), axis=2).max(1)
    v0 = np.array([V0[t]['v0_can'] for t in tokv])
    wss = pickle.load(open(f'{DATA}/r5_wss.pkl', 'rb')); w = np.array([wss[r['scene_name']] for r in rva])
    S = {'all': np.ones(len(rva), bool), 'excl. first frames': ~ff, 'excl. WSS': ~w, 'WSS only': w,
         'C2 STOPPED': (v0 <= 0.2) & (g <= 1.0), 'C2 START': (v0 <= 0.2) & (g > 1.0), 'C2 MOVING': v0 > 0.2}
    stat = C['st'] == 'stationary'; gs = stat & (g < 0.5); gm = stat & (g > 1.0)
    print('tau ' + ', '.join(f'{t} {Q.run(t)["tau"]}' for tags in ARMS.values() for t in tags))
    print('\nL2@3s NoAvg per seed | 3-seed mean +- sd')
    for nm, tags in ARMS.items():
        v = np.array([Q.run(t)['l2'][:, 5].mean() for t in tags])
        print(f'  {nm:10} ' + ' '.join(f'{x:.3f}' for x in v) + f' | {v.mean():.3f} +- {v.std(ddof=1):.3f}')
    M = {nm: Q.avg([Q.run(t) for t in tags]) for nm, tags in ARMS.items()}
    G = {}
    print('\nPaired scene bootstrap (10,000), 3-seed means, L2@3s NoAvg; ADE@6s after')
    for pair, a, b in (('GN1-3 M1-v2a-NR - M1-v2a', 'M1-v2a-NR', 'M1-v2a'), ('GN1-3 B1-NR - B1', 'B1-NR', 'B1'),
                       ('GN2-3 M1-v2a-NR - B1-NR', 'M1-v2a-NR', 'B1-NR'), ('ref M1-v2a - B1 (route)', 'M1-v2a', 'B1')):
        for sn, m in S.items():
            sub = [r for r, k in zip(rva, m) if k]
            c = E.scene_boot(sub, M[a]['l2'][m, 5], M[b]['l2'][m, 5]); c2 = E.scene_boot(sub, M[a]['ade'][m], M[b]['ade'][m])
            G[(pair, sn)] = c
            print(f'  {pair:26} {sn:18} n {m.sum():4d}  {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]   '
                  f'ADE6 {c2[0]:+.3f} [{c2[1]:+.3f},{c2[2]:+.3f}]')
    print('\nPer run: C2 L2@3s STOPPED START MOVING | C2 FG MG | R3 FG all / WSS / non-WSS, MG | collision @3 s '
          'NoAvg / TemAvg vehicles, aa | yaw')
    for nm, tags in ARMS.items():
        for t in tags:
            P = preds(load(t, 'val'), rva, 'hybrid', Q.run(t)['tau']); l2 = Q.run(t)['l2'][:, 5]
            dm = np.linalg.norm(P[:, :6], axis=2).max(1)
            fg = lambda m: (dm[m & gs] > 1.0).mean()
            ca = coll_rates(rva, P, occ, 'aa'); cy = coll_rates(rva, P, occ, 'yaw')
            print(f'  {t:16} ' + ' '.join(f'{l2[S[k]].mean():.3f}' for k in ('C2 STOPPED', 'C2 START', 'C2 MOVING')) +
                  f' | {(dm[S["C2 STOPPED"]] > 1.0).mean():.3f} {(dm[S["C2 START"]] <= 1.0).mean():.3f} | '
                  f'{fg(np.ones(len(rva), bool)):.3f} / {fg(w):.3f} / {fg(~w):.3f}, {(dm[gm] < 0.5).mean():.3f} | '
                  f'{ca[0]:.2f}/{ca[1]:.2f} | {cy[0]:.2f}/{cy[1]:.2f}')
    print('\nGATES (3 seeds)')
    for p in ('GN1-3 M1-v2a-NR - M1-v2a', 'GN1-3 B1-NR - B1'):
        c = G[(p, 'all')]; print(f'  {p}: {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]')
    c = G[('GN2-3 M1-v2a-NR - B1-NR', 'all')]
    print(f'  GN2-3 M1-v2a-NR - B1-NR: {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}] -> ' +
          ('"meta helps without route (3 seeds)"' if c[2] < 0 else '"meta hurts without route"' if c[1] > 0 else '"flat"'))


if __name__ == '__main__':
    main()
