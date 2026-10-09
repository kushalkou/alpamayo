"""ar1/r10_b2.py -- R10 B2: Table 12 analog for the M1-v2a token decoder vs flow expert, 3 seeds
(R9 B2 metric set), gates GE1-3 / GE2-3 / GE3-3, STOPPED L2@3s CI. Rules in R10_REPORT.md.
  python ar1/r10_b2.py"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from r5_a5c1 import coll_rates

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
TAGS = ('M1v2a', 'M1v2a_s123', 'M1v2a_s2024')


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); occ = C['occ']; tokv = [r['sample_token'] for r in rva]
    G12 = [np.asarray(T[t]['P']) for t in tokv]; n12 = np.array([r['n_fut'] == 12 for r in rva])
    g = np.array([np.linalg.norm(x[:6], axis=1).max() for x in G12]); v0 = np.array([V0[t]['v0_can'] for t in tokv])
    gstop = (C['st'] == 'stationary') & (g < 0.5)
    ST = {'STOPPED': (v0 <= 0.2) & (g <= 1.0), 'START': (v0 <= 0.2) & (g > 1.0), 'MOVING': v0 > 0.2}

    def per(P):
        a3 = np.array([np.linalg.norm(P[i][:6] - G12[i][:6], axis=1).mean() for i in range(len(rva))])
        a6 = np.array([np.linalg.norm(P[i][:12] - G12[i][:12], axis=1).mean() if n12[i] else np.nan for i in range(len(rva))])
        l2 = np.array([np.linalg.norm(P[i][5] - G12[i][5]) for i in range(len(rva))])
        hd = np.array([np.linalg.norm(P[i][:6], axis=1).max() for i in range(len(rva))]) < 0.5
        return {'a3': a3, 'a6': a6, 'l2': l2, 'hold': hd[gstop].mean(),
                'col': coll_rates(rva, P, occ, 'aa')[0], 'coly': coll_rates(rva, P, occ, 'yaw')[0]}

    def line(nm, m, extra=''):
        print(f'  {nm:24} {m["a3"].mean():.3f} {np.nanmean(m["a6"]):.3f} {extra:>21} {m["hold"]:.3f} {m["col"]:.2f}/{m["coly"]:.2f} | '
              + ' '.join(f'{m["l2"][s].mean():.3f}' for s in ST.values()))

    print('R10 B2 Table 12 analog, M1-v2a token vs flow expert (val; ADE@3s all, ADE@6s n_fut = 12; hold n '
          f'{gstop.sum()}; Col @3 s NoAvg aa/yaw; C2 L2@3s STOPPED START MOVING)')
    print(f'  {"run / readout":24} ADE3  ADE6  {"minADE6 3/6 spread3/6":>21} hold  Col      | C2')
    M = {k: [] for k in ('token', 'draw', 'mean6', 'near')}
    for t in TAGS:
        X = pickle.load(open(f'{RES}/ar1_expert_{t}.pkl', 'rb')); S = X['val']['traj']
        assert X['val']['tokens'] == tokv
        print(f'  [{t}] expert best epoch {X["best_epoch"]} (holdout {X["ho_best_med"]:.4f}; evals {len(X["hist"])}; '
              f'{"CAP" if X["best_epoch"] == 100 or len(X["hist"]) == 20 else "patience"}); train {X["train_gpu_h"]:.1f} GPU-h')
        tok = preds(load(t, 'val'), rva, 'hybrid', Q.run(t)['tau'])
        mt = per(tok); M['token'].append(mt); line(f'{t} token', mt, '(1 sample)')
        m3 = np.mean([np.linalg.norm(S[i, :, :6] - G12[i][None, :6], axis=2).mean(1).min() for i in range(len(rva))])
        m6 = np.nanmean([np.linalg.norm(S[i] - G12[i][None, :12], axis=2).mean(1).min() if n12[i] else np.nan for i in range(len(rva))])
        sp = [np.mean([np.mean([np.linalg.norm(S[i, a, h] - S[i, b, h]) for a in range(6) for b in range(a + 1, 6)])
                       for i in range(len(rva))]) for h in (5, 11)]
        pk = [per(S[:, k]) for k in range(6)]
        md = {k: (np.mean([p[k] for p in pk], 0) if k in ('a3', 'a6', 'l2') else float(np.mean([p[k] for p in pk]))) for k in pk[0]}
        M['draw'].append(md); line(f'{t} exp 1 draw', md, f'{m3:.3f}/{m6:.3f} {sp[0]:.2f}/{sp[1]:.2f}')
        mm = per(S.mean(1)); M['mean6'].append(mm); line(f'{t} exp mean6', mm)
        sel = np.stack([S[i, np.argmin(np.linalg.norm(S[i] - tok[i][None], axis=2).mean(1))] for i in range(len(rva))])
        mn = per(sel); M['near'].append(mn); line(f'{t} exp near-tok', mn)
    print('  3-seed means:')
    A = {}
    for k, L in M.items():
        A[k] = {q: (np.mean([x[q] for x in L], 0) if q in ('a3', 'a6', 'l2') else float(np.mean([x[q] for x in L]))) for q in L[0]}
        line(f'3s {k}', A[k])
    f = lambda c: f'{c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]'
    c1 = E.scene_boot(rva, A['mean6']['a6'], A['token']['a6']); c3 = E.scene_boot(rva, A['near']['a6'], A['token']['a6'])
    sub = [r for r, x in zip(rva, ST['STOPPED']) if x]
    cs = E.scene_boot(sub, A['mean6']['l2'][ST['STOPPED']], A['token']['l2'][ST['STOPPED']])
    cs1 = [E.scene_boot(sub, M['mean6'][j]['l2'][ST['STOPPED']], M['token'][j]['l2'][ST['STOPPED']]) for j in range(3)]
    print('\nGATES (3 seeds, paired scene bootstrap 10,000)')
    print(f'  GE1-3 mean-of-6 - token ADE@6s {f(c1)} -> ' + ('"expert improves the long horizon"' if c1[2] < 0 else 'condition not met'))
    print(f'  GE2-3 hold mean-of-6 {A["mean6"]["hold"]:.3f} vs token {A["token"]["hold"]:.3f} (per seed ' +
          ', '.join(f'{M["mean6"][j]["hold"]:.3f}/{M["token"][j]["hold"]:.3f}' for j in range(3)) + ') -> ' +
          ('"expert holds stops"' if A['mean6']['hold'] >= A['token']['hold'] - 0.05 else '"expert drifts at standstill"'))
    print(f'  GE3-3 nearest-to-token - token ADE@6s {f(c3)} -> ' +
          ('"expert adds value even when anchored to the token decision"' if c3[2] < 0 else 'condition not met'))
    print(f'  STOPPED L2@3s mean-of-6 - token {f(cs)}; per seed ' + '; '.join(f(c) for c in cs1))


if __name__ == '__main__':
    main()
