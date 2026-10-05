"""ar1/r33_seeds_a3.py -- R3.3 seeds (M1 s42 / s123 / s2024) + amendment A3 (M1 s42
re-decoded: M1_G greedy constrained = determinism check, M1_T07 valid-word sampling T=0.7).
Shared eval (q2_traj): hybrid decode, tau per dump on holdout; official val; both strata.
Per-seed values, mean +- sd (ddof 1), paired scene bootstrap of the seed-mean per-sample
metrics vs the B1 and no-camera 3-seed means; stationary false-go (R3.0a rule).
A3: word macro-F1 / accuracy vs CAN and 2 Hz (NEW) labels, consistency, majority share,
trajectory metrics, for M1 (original), M1_G and M1_T07.
  python ar1/r33_seeds_a3.py
"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from r33_eval import derive, controls, f1
import finetune_meta as FM

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
M1 = {42: 'M1', 123: 'M1_s123', 2024: 'M1_s2024'}
B1 = {42: 'B1', 123: 'B1_s123', 2024: 'B1_s2024'}


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data()
    LEN, NC, _ = Q.seed_tags()
    print('tau: ' + ', '.join(f'{t} {Q.run(t)["tau"]}' for t in list(M1.values()) + ['M1_G', 'M1_T07']))
    G = np.stack([np.asarray(T[r['sample_token']]['P'])[:6] for r in rva]); gmax = np.linalg.norm(G, axis=2).max(1)
    for blk, m in Q.blocks():
        sub = [r for r, k in zip(rva, m) if k]
        cut = lambda p: {k: v[m] for k, v in Q.strip(p).items()}
        print(f'\n===== {blk} =====')
        for nm, tags in (('no cam', NC), ('B1', B1), ('M1', M1)):
            S = [E.summary(cut(Q.run(t))) for t in tags.values()]
            for key, lab in (('L2_NoAvg_3s', 'L2@3s NoAvg'), ('L2_TemAvg_3s', 'L2@3s TemAvg'), ('ADE6', 'ADE@6s'),
                             ('ADE6_med', 'ADE@6s med'), ('L2_3s_p95', 'L2@3s p95'), ('Col_TemAvg_3s', 'Col% Tem 3s')):
                v = np.array([s[key] for s in S])
                print(f'  {nm:6} {lab:13} ' + ' '.join(f'{x:.3f}' for x in v) + f' | {v.mean():.3f} +- {v.std(ddof=1):.3f}')
        mean = {nm: Q.avg([cut(Q.run(t)) for t in tags.values()]) for nm, tags in (('nc', NC), ('b1', B1), ('m1', M1))}
        print(E.compare(sub, mean['m1'], mean['b1'], 'M1 3-seed - B1 3-seed'))
        print(E.compare(sub, mean['m1'], mean['nc'], 'M1 3-seed - no cam 3-seed'))
        st = C['st'][m]
        print('  strata L2@3s (no cam / B1 / M1, 3-seed means) and M1 - B1:')
        for s in ('straight', 'turning', 'stationary'):
            k = st == s
            c = E.scene_boot([r for r, q in zip(sub, k) if q], mean['m1']['l2'][k, 5], mean['b1']['l2'][k, 5])
            print(f'    {s:10} n={int(k.sum()):4d} ' + ' / '.join(f'{mean[x]["l2"][k, 5].mean():.3f}' for x in ('nc', 'b1', 'm1'))
                  + f'   M1-B1 {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]')
        gs = (C['st'] == 'stationary') & m & (gmax < 0.5)
        fg = [float((np.linalg.norm(preds(load(t, 'val'), rva, 'hybrid', Q.run(t)['tau'])[:, :6], axis=2).max(1)[gs] > 1.0).mean())
              for t in M1.values()]
        print(f'  stationary false-go M1 seeds: ' + ' '.join(f'{x:.3f}' for x in fg) + f' (mean {np.mean(fg):.3f})')

    # A3 + seeds: words
    r12 = [r for r in rva if r['n_fut'] == 12]
    Mt = pickle.load(open(f'{DATA}/ar1_meta.pkl', 'rb')); stt = {'missing': 0}
    can = np.array([sum(FM.meta_labels(Mt, r['sample_token'], stt), []) for r in r12])
    new = np.array([derive(T[r['sample_token']]['acc'][:12], T[r['sample_token']]['cur'][:12], T[r['sample_token']]['v0'])
                    for r in r12])
    maj = np.array([4] * 6 + [6] * 6)
    print('\nWORDS (val n_fut = 12): slot-mean acc / macro-F1 vs CAN and vs NEW (2 Hz) labels; consistency; majority')
    g42 = None
    for tag in ('M1', 'M1_G', 'M1_T07', 'M1_s123', 'M1_s2024'):
        dv = load(tag, 'val')['data']
        gen = np.array([dv[r['sample_token']]['meta_gen'] for r in r12])
        if tag == 'M1': g42 = gen
        der = np.array([derive(a, k, T[r['sample_token']]['v0']) for (a, k), r in
                        zip(controls(load(tag, 'val'), r12, Q.run(tag)['tau']), r12)])
        out = []
        for y in (can, new):
            out.append((np.mean([(gen[:, s] == y[:, s]).mean() for s in range(6)]),
                        np.mean([f1(y[:, s], gen[:, s], 7) for s in range(6)]),
                        np.mean([(gen[:, 6 + s] == y[:, 6 + s]).mean() for s in range(6)]),
                        np.mean([f1(y[:, 6 + s], gen[:, 6 + s], 7) for s in range(6)])))
        c = der == gen
        same = '' if tag != 'M1_G' else f'; identical to M1 words: {(gen == g42).all(1).mean():.3f}'
        print(f'  {tag:9} CAN lon {out[0][0]:.3f}/{out[0][1]:.3f} lat {out[0][2]:.3f}/{out[0][3]:.3f} | NEW lon '
              f'{out[1][0]:.3f}/{out[1][1]:.3f} lat {out[1][2]:.3f}/{out[1][3]:.3f} | consist {c.all(1).mean():.3f} '
              f'| majority {(gen == maj).all(1).mean():.3f} | distinct {len(set(map(tuple, gen)))}{same}')
    print(f'  labels    majority share CAN {(can == maj).all(1).mean():.3f}, NEW {(new == maj).all(1).mean():.3f}')
    print('\nA3 TRAJECTORY (ALL 5,119; EXCL. first frames in brackets)')
    print('  ' + E.HDR)
    for tag in ('M1', 'M1_G', 'M1_T07'):
        print(E.row(tag, Q.strip(Q.run(tag))))
    for a_, b_ in (('M1_G', 'M1'), ('M1_T07', 'M1')):
        print(E.compare(rva, Q.strip(Q.run(a_)), Q.strip(Q.run(b_)), f'{a_} - {b_}'))
    dg, d0 = load('M1_G', 'val')['data'], load('M1', 'val')['data']
    print(f'  M1_G vs M1 trajectory tokens identical: {np.mean([dg[r["sample_token"]]["tok"] == d0[r["sample_token"]]["tok"] for r in rva]):.4f}')


if __name__ == '__main__':
    main()
