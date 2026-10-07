"""ar1/r8_partA.py -- R8 Part A (CPU): A1 selected tau table, A2 tau SENSITIVITY (gate GT1),
A3 command-vs-motion DIAGNOSTIC (gate GL1). Rules in the R8_REPORT.md header.
  python ar1/r8_partA.py"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q, records
from gate_a import load, preds, ade12, TAUS

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
RUNS = ('A3', 'G6C_s123', 'G6C_s2024', 'B1', 'B1_s123', 'B1_s2024', 'B2', 'M1', 'M1_s123', 'M1_s2024',
        'M1v2a', 'M1v2a_s123', 'M1v2a_s2024', 'M1v2a_nocam', 'B1_NR', 'M1v2a_NR')
CMD = ('right', 'left', 'straight')


def c2(recs, V0, T):
    v0 = np.array([V0[r['sample_token']]['v0_can'] for r in recs])
    g = np.linalg.norm(np.stack([np.asarray(T[r['sample_token']]['P'])[:6] for r in recs]), axis=2).max(1)
    return {'STOPPED': (v0 <= 0.2) & (g <= 1.0), 'START': (v0 <= 0.2) & (g > 1.0), 'MOVING': v0 > 0.2}, g


def main():
    C = Q.ctx(); rva = C['rva']; rho = C['rho']; W, V0, T = E.data()
    G = np.stack([np.asarray(T[r['sample_token']]['P'])[:6] for r in rva])
    ST, g = c2(rva, V0, T)
    wss = pickle.load(open(f'{DATA}/r5_wss.pkl', 'rb')); w = np.array([wss[r['scene_name']] for r in rva])
    stat = C['st'] == 'stationary'; gs = stat & (g < 0.5); gm = stat & (g > 1.0)

    print('A1 selected tau (holdout, mean ADE@6s over all n_fut = 12 holdout samples); holdout ADE@6s at 0.3/0.5/0.7/0.9')
    for t in RUNS:
        dh = load(t, 'holdout')
        a = [ade12(rho, preds(dh, rho, 'hybrid', x)) for x in TAUS]
        print(f'  {t:14} tau {Q.run(t)["tau"]}   ' + ' '.join(f'{x:.4f}' for x in a))

    print('\nA2 SENSITIVITY (fixed tau grid; NOT headline). L2@3s all | R3 FG all, WSS | R3 MG | C2 L2@3s '
          'STOPPED START MOVING | C2 FG MG')
    fgw = {}
    for t in ('M1v2a', 'M1v2a_s123', 'M1v2a_s2024', 'M1v2a_NR', 'M1v2a_nocam'):
        dv = load(t, 'val')
        for x in TAUS:
            P = preds(dv, rva, 'hybrid', x); l2 = np.linalg.norm(P[:, 5] - G[:, 5], axis=1)
            dm = np.linalg.norm(P[:, :6], axis=2).max(1)
            fgw[(t, x)] = (dm[gs & w] > 1.0).mean()
            sel = ' *' if x == Q.run(t)['tau'] else '  '
            print(f'  {t:12} tau {x}{sel} {l2.mean():.3f} | {(dm[gs] > 1.0).mean():.3f} {fgw[(t, x)]:.3f} | '
                  f'{(dm[gm] < 0.5).mean():.3f} | ' + ' '.join(f'{l2[m].mean():.3f}' for m in ST.values()) +
                  f' | {(dm[ST["STOPPED"]] > 1.0).mean():.3f} {(dm[ST["START"]] <= 1.0).mean():.3f}')
    print('  (* = holdout-selected tau)')
    v = [fgw[(t, 0.3)] for t in ('M1v2a', 'M1v2a_s123', 'M1v2a_s2024')]
    print(f'GT1 tau 0.3 WSS false-go per seed {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}; max - min {max(v) - min(v):.3f} -> ' +
          ('"seed fragility is a threshold-selection artifact"' if max(v) - min(v) <= 0.05 else '"not explained by tau"'))

    print('\nA3 DIAGNOSTIC command [P] share per C2 stratum (right / left / straight)')
    res = {}
    for split, R in (('train', records.build('train', ['cmd'], 0.0)), ('val', rva)):
        S, _ = c2(R, V0, T); cmd = np.array([r['command'] for r in R])
        for k, m in S.items():
            sh = [(cmd[m] == c).mean() for c in range(3)]
            res[(split, k)] = 1 - sh[2]
            print(f'  {split:5} {k:8} n {m.sum():5d}  right {sh[0]:.3f} left {sh[1]:.3f} straight {sh[2]:.3f}')
    a, b = res[('val', 'START')], res[('val', 'STOPPED')]
    print(f'GL1 val P(cmd != straight | START) {a:.3f} (>= 0.20), P(cmd != straight | STOPPED) {b:.3f} (<= 0.02) -> ' +
          ('"privileged command leaks start information"' if a >= 0.20 and b <= 0.02 else '"no start leak via command"'))
    P = preds(load('M1v2a', 'val'), rva, 'hybrid', Q.run('M1v2a')['tau']); dm = np.linalg.norm(P[:, :6], axis=2).max(1)
    cmd = np.array([r['command'] for r in rva])
    print('  M1-v2a s42 (holdout tau) by command: C2 FG on STOPPED | C2 MG on START')
    for c in range(3):
        s, t = ST['STOPPED'] & (cmd == c), ST['START'] & (cmd == c)
        print(f'    {CMD[c]:8} STOPPED n {s.sum():4d} FG {(dm[s] > 1.0).mean() if s.any() else float("nan"):.3f} | '
              f'START n {t.sum():4d} MG {(dm[t] <= 1.0).mean() if t.any() else float("nan"):.3f}')


if __name__ == '__main__':
    main()
