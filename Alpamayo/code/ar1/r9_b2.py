"""ar1/r9_b2.py -- R9 B2: Table 12 analog with the R5 A5 metric set (token decoder vs flow
expert) for M1-v2a s42, plus A3 (no-cam) and M1 rows; gates GE1-GE3 (M1-v2a). Rules in the
R9_REPORT.md header.
  python ar1/r9_b2.py"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from r5_a5c1 import coll_rates

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); occ = C['occ']; tokv = [r['sample_token'] for r in rva]
    G12 = [np.asarray(T[t]['P']) for t in tokv]; n12 = np.array([r['n_fut'] == 12 for r in rva])
    g = np.array([np.linalg.norm(x[:6], axis=1).max() for x in G12]); v0 = np.array([V0[t]['v0_can'] for t in tokv])
    gstop = (C['st'] == 'stationary') & (g < 0.5)
    ST = {'STOPPED': (v0 <= 0.2) & (g <= 1.0), 'START': (v0 <= 0.2) & (g > 1.0), 'MOVING': v0 > 0.2}

    def per(P):        # per-sample ADE@3s (all), ADE@6s (n_fut 12, else nan), L2@3s
        a3 = np.array([np.linalg.norm(P[i][:6] - G12[i][:6], axis=1).mean() for i in range(len(rva))])
        a6 = np.array([np.linalg.norm(P[i][:12] - G12[i][:12], axis=1).mean() if n12[i] else np.nan for i in range(len(rva))])
        l2 = np.array([np.linalg.norm(P[i][5] - G12[i][5]) for i in range(len(rva))])
        return a3, a6, l2
    hold = lambda P: float((np.array([np.linalg.norm(P[i][:6], axis=1).max() for i in range(len(rva))])[gstop] < 0.5).mean())

    def row(nm, P, extra=''):
        a3, a6, l2 = per(P); ca = coll_rates(rva, P, occ, 'aa'); cy = coll_rates(rva, P, occ, 'yaw')
        print(f'  {nm:28} {a3.mean():6.3f} {np.nanmean(a6):6.3f} {extra:>26} {hold(P):5.3f} {ca[0]:5.2f}/{cy[0]:5.2f} | '
              + ' '.join(f'{l2[m].mean():.3f}' for m in ST.values()))
        return a6

    print('B2 Table 12 analog (val; ADE@3s all 5,119, ADE@6s n_fut = 12; hold = GT-stopped stationary staying < 0.5 m '
          f'in 3 s, n {gstop.sum()}; Col @3 s NoAvg aa/yaw; C2 L2@3s STOPPED START MOVING)')
    print(f'  {"model / readout":28} {"ADE3":>6} {"ADE6":>6} {"minADE6 3/6s  spread 3/6s":>26} {"hold":>5} {"Col aa/yaw":>11} | C2')
    R = {}
    for tag, xp in (('M1v2a', 'ar1_expert_M1v2a.pkl'), ('A3', 'ar1_expert_A3.pkl'), ('M1', 'ar1_expert_M1.pkl')):
        X = pickle.load(open(f'{RES}/{xp}', 'rb')); S = X['val']['traj']
        assert X['val']['tokens'] == tokv
        tok = preds(load(tag, 'val'), rva, 'hybrid', Q.run(tag)['tau'])
        r = {'token': row(f'{tag} token decoder', tok, '(1 sample)')}
        m3 = np.mean([np.linalg.norm(S[i, :, :6] - G12[i][None, :6], axis=2).mean(1).min() for i in range(len(rva))])
        m6 = np.nanmean([np.linalg.norm(S[i] - G12[i][None, :12], axis=2).mean(1).min() if n12[i] else np.nan
                         for i in range(len(rva))])
        sp3 = np.mean([np.mean([np.linalg.norm(S[i, a, 5] - S[i, b, 5]) for a in range(6) for b in range(a + 1, 6)]) for i in range(len(rva))])
        sp6 = np.mean([np.mean([np.linalg.norm(S[i, a, 11] - S[i, b, 11]) for a in range(6) for b in range(a + 1, 6)]) for i in range(len(rva))])
        pk = [per(S[:, k]) for k in range(6)]
        cs = np.array([coll_rates(rva, S[:, k], occ, 'aa') + coll_rates(rva, S[:, k], occ, 'yaw') for k in range(6)]).mean(0)
        print(f'  {tag + " expert single draw":28} {np.mean([p[0].mean() for p in pk]):6.3f} '
              f'{np.mean([np.nanmean(p[1]) for p in pk]):6.3f} {f"{m3:.3f}/{m6:.3f}  {sp3:.2f}/{sp6:.2f}":>26} '
              f'{np.mean([hold(S[:, k]) for k in range(6)]):5.3f} {cs[0]:5.2f}/{cs[2]:5.2f} | '
              + ' '.join(f'{np.mean([p[2][m].mean() for p in pk]):.3f}' for m in ST.values()))
        r['mean6'] = row(f'{tag} expert mean of 6', S.mean(1))
        sel = np.stack([S[i, np.argmin(np.linalg.norm(S[i] - tok[i][None], axis=2).mean(1))] for i in range(len(rva))])
        r['near'] = row(f'{tag} expert nearest to token', sel)
        r['hold_tok'], r['hold_m6'] = hold(tok), hold(S.mean(1))
        R[tag] = r
    print('\nGATES (M1-v2a s42; paired scene bootstrap 10,000 on ADE@6s, n_fut = 12)')
    r = R['M1v2a']
    c1 = E.scene_boot(rva, r['mean6'], r['token']); c3 = E.scene_boot(rva, r['near'], r['token'])
    f = lambda c: f'{c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]'
    print(f'  GE1 mean-of-6 - token {f(c1)} -> ' + ('"expert improves the long horizon"' if c1[2] < 0 else 'condition not met'))
    print(f'  GE2 hold mean-of-6 {r["hold_m6"]:.3f} vs token {r["hold_tok"]:.3f} - 0.05 -> ' +
          ('"expert holds stops"' if r['hold_m6'] >= r['hold_tok'] - 0.05 else '"expert drifts at standstill"'))
    print(f'  GE3 nearest-to-token - token {f(c3)} -> ' +
          ('"expert adds value even when anchored to the token decision"' if c3[2] < 0 else 'condition not met'))
    for tag in ('A3', 'M1'):
        c = E.scene_boot(rva, R[tag]['mean6'], R[tag]['token']); d = E.scene_boot(rva, R[tag]['near'], R[tag]['token'])
        print(f'  (contrast {tag}: mean-of-6 - token {f(c)}; nearest - token {f(d)}; hold {R[tag]["hold_m6"]:.3f} vs {R[tag]["hold_tok"]:.3f})')


if __name__ == '__main__':
    main()
