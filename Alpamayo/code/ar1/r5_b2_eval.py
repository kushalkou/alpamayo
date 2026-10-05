"""ar1/r5_b2_eval.py -- R5 B2: M1-v2a (2 Hz labels) seed 42 evaluation and gates GB2-GB5
(definitions in the R5_REPORT.md header). Labels for words = 2 Hz (data/ar1_meta2hz.pkl).
  python ar1/r5_b2_eval.py"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from r33_eval import derive, controls
from meta_actions import LON, LAT

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
TAG = 'M1v2a'


def prf(y, p, c):
    tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return P, R, (2 * P * R / max(P + R, 1e-9)), int((y == c).sum())


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); ff = C['ff']
    wss = pickle.load(open(f'{DATA}/r5_wss.pkl', 'rb')); w = np.array([wss[r['scene_name']] for r in rva])
    rows = {'VLA no cam 3-seed [P]': Q.avg([Q.run(t) for t in ('A3', 'G6C_s123', 'G6C_s2024')]),
            'B1 seed 42 [P]': Q.strip(Q.run('B1')), 'M1 seed 42 (10 Hz labels) [P]': Q.strip(Q.run('M1')),
            'M1-v2a seed 42 (2 Hz labels) [P]': Q.strip(Q.run(TAG))}
    print(f'tau (holdout): {TAG} {Q.run(TAG)["tau"]}')
    for blk, m in Q.blocks():
        print(f'\n===== {blk} =====\n' + E.HDR)
        for k, v in rows.items():
            print(E.row(k, {kk: vv[m] for kk, vv in v.items()}))
    print('\nGB4 and strata: M1-v2a - B1 s42 (and - M1 s42), paired scene bootstrap, L2@3s / ADE@6s')
    A, B, M = rows['M1-v2a seed 42 (2 Hz labels) [P]'], rows['B1 seed 42 [P]'], rows['M1 seed 42 (10 Hz labels) [P]']
    for sub_nm, base in (('(a) all', np.ones(len(rva), bool)), ('(b) excl. WSS [POST-HOC]', ~w), ('(c) WSS only [POST-HOC]', w)):
        for ff_nm, fm in (('with first frames', np.ones(len(rva), bool)), ('excl. first frames', ~ff)):
            m = base & fm; sub = [r for r, k in zip(rva, m) if k]
            for nm, X in (('- B1 s42', B), ('- M1 s42', M)):
                c1 = E.scene_boot(sub, A['l2'][m, 5], X['l2'][m, 5]); c2 = E.scene_boot(sub, A['ade'][m], X['ade'][m])
                print(f'  {sub_nm:25} {ff_nm:19} n {m.sum():4d}  M1-v2a {nm}: L2@3s {c1[0]:+.3f} [{c1[1]:+.3f},'
                      f'{c1[2]:+.3f}]  ADE6 {c2[0]:+.3f} [{c2[1]:+.3f},{c2[2]:+.3f}]')
    # words
    lab = pickle.load(open(f'{DATA}/ar1_meta2hz.pkl', 'rb'))
    r12 = [r for r in rva if r['n_fut'] == 12]
    dv = load(TAG, 'val')['data']
    gen = np.array([dv[r['sample_token']]['meta_gen'] for r in r12]); y = np.array([lab[r['sample_token']] for r in r12])
    maj = np.array([4] * 6 + [6] * 6)
    gs, ls = (gen == maj).all(1).mean(), (y == maj).all(1).mean()
    print(f'\nGB2 collapse: generated all-majority share {gs:.3f} vs label share {ls:.3f}: {100 * abs(gs - ls):.1f} points'
          f' (M1 s42 with 10 Hz labels: generated 0.433 vs its label share 0.168)')
    print(f'  distinct generated sequences {len(set(map(tuple, gen)))}, label sequences {len(set(map(tuple, y)))}')
    print('\nGB3 per-class precision / recall / F1, pooled over the 6 slots, val n_fut = 12 (2 Hz labels)')
    bad = []
    for h, o, names in (('lon', 0, LON), ('lat', 6, LAT)):
        yy, pp = y[:, o:o + 6].ravel(), gen[:, o:o + 6].ravel()
        for c in range(7):
            P, R, F, n = prf(yy, pp, c)
            if n == 0 and (pp == c).sum() == 0:
                continue
            flag = ''
            if n >= 30 and F == 0:
                flag = '  <-- F1 = 0 with support >= 30'; bad.append(f'{h}:{names[c]}')
            print(f'  {h} {names[c]:11} support {n:6d}  generated {int((pp == c).sum()):6d}  P {P:.3f}  R {R:.3f}  F1 {F:.3f}{flag}')
        acc = [(gen[:, o + s] == y[:, o + s]).mean() for s in range(6)]
        print(f'  {h} slot accuracy ' + ' '.join(f'{a:.3f}' for a in acc))
    print(f'  GB3 input: classes with support >= 30 and F1 = 0: {bad or "none"}')
    # false-go
    g = np.linalg.norm(np.stack([np.asarray(T[r['sample_token']]['P'])[:6] for r in rva]), axis=2).max(1)
    stat = C['st'] == 'stationary'
    print('\nfalse-go / missed-go (stationary, R3 rules)')
    for t in ('B1', 'M1', TAG):
        d = np.linalg.norm(preds(load(t, 'val'), rva, 'hybrid', Q.run(t)['tau'])[:, :6], axis=2).max(1)
        out = []
        for nm, m in (('all', stat), ('WSS', stat & w), ('non-WSS', stat & ~w)):
            k = m & (g < 0.5); kk = m & (g > 1.0)
            out.append(f'{nm} FG {(d[k] > 1.0).mean():.3f} (n {k.sum()}) MG {(d[kk] < 0.5).mean() if kk.any() else float("nan"):.3f}')
        print(f'  {t:6} ' + ' | '.join(out))
    # consistency
    dva = load(TAG, 'val')
    gen_all = np.array([dva['data'][r['sample_token']]['meta_gen'] for r in rva])
    der = np.array([derive(a, k, T[r['sample_token']]['v0']) for (a, k), r in zip(controls(dva, rva, Q.run(TAG)['tau']), rva)])
    c = (der == gen_all).all(1)
    print(f'\nGB5 consistency (trajectory follows own words, all 12 slots), all val: {c.mean():.3f}; stationary '
          f'{c[stat].mean():.3f}; moving {c[~stat].mean():.3f}  (M1 s42: 0.410)')


if __name__ == '__main__':
    main()
