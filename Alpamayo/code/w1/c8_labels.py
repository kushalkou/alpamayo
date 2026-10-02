"""w1/c8_labels.py -- QUEUE v2 amendment C8: label ambiguity and V8b vs V8a per class (CPU).

lon rule (records.meta_lon): v3 = segment speed 2.5-3.0 s, dv = v3 - v0_can;
stop if v3 < 0.5; accelerate if dv > +1.0; decelerate if dv < -1.0; else maintain.
a. near-threshold: |dv - 1.0| < 0.3 (accel boundary), |dv + 1.0| < 0.3 (decel boundary),
   |v3 - 0.5| < 0.3 (stop boundary); shares on train and val (each boundary, raw, and
   the union). Accuracy / macro-F1 of MLP (3 seeds, mean) / V8a / V8b on all val and on
   val excluding the union.
b. per-class recall and confusion, V8b vs V8a; paired scene bootstrap (2,000 draws) of
   recall(V8b) - recall(V8a) per class; flag CI excluding 0.
c. agreement V8a vs V8b; accuracy of each on the disagreements.
Both strata (ALL 5,119 and EXCL. 140 first frames). Inputs: results/w1_c7.pkl (CLS).
"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import records, evalw1 as E
from c7_bottleneck import macro_f1, NAMES

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
B = 0.3


def near(R):
    v3 = np.array([float(t['s'][5]) for t in R]); dv = v3 - np.array([t['v0'] for t in R])
    m = {'accel |dv-1|<.3': np.abs(dv - 1.0) < B, 'decel |dv+1|<.3': np.abs(dv + 1.0) < B,
         'stop |v3-.5|<.3': np.abs(v3 - 0.5) < B}
    m['union'] = m['accel |dv-1|<.3'] | m['decel |dv+1|<.3'] | m['stop |v3-.5|<.3']
    return m


def recall(y, p, c):
    k = y == c
    return ((p == c) & k).sum() / max(k.sum(), 1)


def main():
    C = pickle.load(open(f'{RES}/w1_c7.pkl', 'rb'))
    CLS, I = C['CLS'], C['info']
    y, scene = I['yva'], I['scene']
    tr = records.build('train', ('cmd',)); va = records.build('val', ('cmd',), n_fut=6)
    assert [t['sample_token'] for t in va] == I['order']
    assert (np.array([t['meta_lon'] for t in va]) == y).all()
    Ntr, Nva = near(tr), near(va)
    recs = E.split_records('val'); W, V0, _ = E.data()
    ff = np.array([not V0[r['sample_token']]['can_ok'] and len(r['past_poses']) == 0 for r in recs])
    P = {'MLP': [CLS[('MLP', s)]['val'].argmax(1) for s in (42, 123, 2024)],
         'V8a': [CLS[('V8a', 42)]['val'].argmax(1)], 'V8b': [CLS[('V8b', 42)]['val'].argmax(1)]}
    out = {}

    print('C8a NEAR-THRESHOLD SHARES (band 0.3 m/s)')
    print(f'  {"boundary":18} {"train":>7} {"val":>7}')
    for k in Ntr:
        print(f'  {k:18} {Ntr[k].mean():7.3f} {Nva[k].mean():7.3f}')
    for blk, m in (('ALL 5,119', np.ones(len(y), bool)), (f'EXCL. FIRST FRAMES ({int((~ff).sum())})', ~ff)):
        print(f'\n===== {blk} =====')
        far = m & ~Nva['union']
        print(f'C8a accuracy / macro-F1, all val (n={m.sum()}) vs excl. near-threshold (n={far.sum()}); '
              f'MLP = mean of 3 seeds')
        for nm, ps in P.items():
            r = [(np.mean([(p[mm] == y[mm]).mean() for p in ps]), np.mean([macro_f1(y[mm], p[mm]) for p in ps]))
                 for mm in (m, far)]
            out[(blk[:3], 'a', nm)] = r
            print(f'  {nm:4}  all: acc {r[0][0]:.3f} F1 {r[0][1]:.3f} | excl. near: acc {r[1][0]:.3f} F1 {r[1][1]:.3f}')

        a, b = P['V8a'][0][m], P['V8b'][0][m]; yy, sc = y[m], scene[m]
        print('\nC8b per-class recall, V8b - V8a, paired scene bootstrap (2,000)')
        us = np.unique(sc); idx = [np.where(sc == s)[0] for s in us]
        rs = np.random.RandomState(0)
        D = []
        for _ in range(2000):
            ii = np.concatenate([idx[j] for j in rs.randint(0, len(us), len(us))])
            D.append([recall(yy[ii], b[ii], c) - recall(yy[ii], a[ii], c) for c in range(4)])
        D = np.array(D)
        for c in range(4):
            lo, hi = np.percentile(D[:, c], [2.5, 97.5])
            d = recall(yy, b, c) - recall(yy, a, c)
            flag = '  <-- V8b better, CI excludes 0' if lo > 0 else ('  <-- V8b worse, CI excludes 0' if hi < 0 else '')
            out[(blk[:3], 'b', c)] = (recall(yy, a, c), recall(yy, b, c), d, lo, hi)
            print(f'  {NAMES[c]:6} n={int((yy == c).sum()):5d}  V8a {recall(yy, a, c):.3f}  V8b {recall(yy, b, c):.3f}  '
                  f'diff {d:+.3f} [{lo:+.3f},{hi:+.3f}]{flag}')
        print('  confusion (rows true, cols pred: stop accel decel maint)     V8a | V8b')
        for i in range(4):
            print(f'    {NAMES[i]:6} ' + ' '.join(f'{((yy == i) & (a == j)).sum():5d}' for j in range(4)) + '  | '
                  + ' '.join(f'{((yy == i) & (b == j)).sum():5d}' for j in range(4)))

        dis = a != b
        out[(blk[:3], 'c')] = (1 - dis.mean(), dis.sum(), (a[dis] == yy[dis]).mean(), (b[dis] == yy[dis]).mean(),
                               ((a[dis] != yy[dis]) & (b[dis] != yy[dis])).mean())
        print(f'\nC8c agreement V8a/V8b: {1 - dis.mean():.3f} ({int((~dis).sum())} of {len(a)}); on the {int(dis.sum())} '
              f'disagreements: V8a correct {(a[dis] == yy[dis]).mean():.3f}, V8b correct {(b[dis] == yy[dis]).mean():.3f}, '
              f'neither {((a[dis] != yy[dis]) & (b[dis] != yy[dis])).mean():.3f}')
        print(f'      accuracy where they agree: {(a[~dis] == yy[~dis]).mean():.3f}')
    out['near'] = {k: (Ntr[k].mean(), Nva[k].mean()) for k in Ntr}
    pickle.dump(out, open(f'{RES}/w1_c8.pkl', 'wb'))


if __name__ == '__main__':
    main()
