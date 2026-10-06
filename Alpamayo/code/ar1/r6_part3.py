"""ar1/r6_part3.py -- R6 Part 3 (CPU): 3a Li et al. protocol rows (R5 A4 functions) and 3b
comfort (first-order: lon accel, lat accel, yaw rate; R5 C1 differencing). Models present
on disk are included (new M1-v2a seeds when their dumps exist).
  python ar1/r6_part3.py"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from r5_a4 import li_metrics
from r5_a5c1 import comfort

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
GROUPS = (('no-cam VLA', ('A3', 'G6C_s123', 'G6C_s2024')), ('B1', ('B1', 'B1_s123', 'B1_s2024')),
          ('M1 (10 Hz words)', ('M1', 'M1_s123', 'M1_s2024')), ('M1-v2a (2 Hz words)', ('M1v2a', 'M1v2a_s123', 'M1v2a_s2024')),
          ('no-cam M1-v2a', ('M1v2a_nocam',)))


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data()
    G = np.stack([np.asarray(r['gt_lidar6']) for r in rva])
    P = {}
    for nm, tags in GROUPS:
        for t in tags:
            if os.path.exists(f'{RES}/w1_dump_{t}_val_f0.0.pkl'):
                P[t] = preds(load(t, 'val'), rva, 'hybrid', Q.run(t)['tau'])
    tok = {t: P[t] for t in ('A3', 'M1')}
    for tag, xp in (('A3', 'ar1_expert_A3.pkl'), ('M1', 'ar1_expert_M1.pkl')):
        S = pickle.load(open(f'{RES}/{xp}', 'rb'))['val']['traj']
        P[f'{tag} expert mean of 6'] = S.mean(1)
        P[f'{tag} expert nearest token'] = np.stack([S[i, np.argmin(np.linalg.norm(S[i] - tok[tag][i][None], axis=2).mean(1))]
                                                     for i in range(len(rva))])
        P[f'{tag} expert draw 0'] = S[:, 0]
    print('3a  Li et al. protocol (R5 A4 implementation), ALL 5,119: L2 TemAvg 1/2/3 s (avg) | Col % 1/2/3 s (avg)')
    M = li_metrics(rva, P, G)
    for nm, tags in GROUPS:
        have = [t for t in tags if t in M]
        if not have:
            continue
        for t in have:
            v = M[t]
            print(f'  {t:28} {v[0]:.2f} {v[1]:.2f} {v[2]:.2f} ({np.mean(v[:3]):.2f}) | {v[3]:.2f} {v[4]:.2f} {v[5]:.2f} ({np.mean(v[3:]):.2f})')
        if len(have) > 1:
            v = np.mean([M[t] for t in have], 0)
            print(f'  {nm + " mean (" + str(len(have)) + ")":28} {v[0]:.2f} {v[1]:.2f} {v[2]:.2f} ({np.mean(v[:3]):.2f}) | '
                  f'{v[3]:.2f} {v[4]:.2f} {v[5]:.2f} ({np.mean(v[3:]):.2f})')
    for k in [k for k in M if 'expert' in k]:
        v = M[k]
        print(f'  {k:28} {v[0]:.2f} {v[1]:.2f} {v[2]:.2f} ({np.mean(v[:3]):.2f}) | {v[3]:.2f} {v[4]:.2f} {v[5]:.2f} ({np.mean(v[3:]):.2f})')
    print('  Li Table 1: GoStraight 0.38 0.79 1.33 (0.83) | 0.15 0.60 2.50; Ego-MLP 0.15 0.32 0.59 (0.35) | 0.00 0.27 0.85')

    n12 = np.array([r['n_fut'] == 12 for r in rva]); idx = np.where(n12)[0]
    print(f'\n3b  comfort (first-order: lon accel, lat accel, yaw rate), val n_fut = 12 (n {len(idx)}); appendix columns:'
          ' all six bounds (2 Hz differences; jerk crude)')
    rows = [('GT', [np.asarray(T[rva[i]['sample_token']]['P']) for i in idx])] + \
           [(k, [P[k][i] for i in idx]) for k in P]
    print(f'  {"model":28} {"first-order":>11} {"lon_acc":>8} {"lat_acc":>8} {"yaw_rate":>8} | {"all six (appx)":>14}')
    for nm, L in rows:
        F = [comfort(L[j], T[rva[i]['sample_token']]['v0']) for j, i in enumerate(idx)]
        fo = np.mean([f['lon_acc'] and f['lat_acc'] and f['yaw_rate'] for f in F])
        print(f'  {nm:28} {fo:11.3f} {np.mean([f["lon_acc"] for f in F]):8.3f} {np.mean([f["lat_acc"] for f in F]):8.3f} '
              f'{np.mean([f["yaw_rate"] for f in F]):8.3f} | {np.mean([f["all"] for f in F]):14.3f}')


if __name__ == '__main__':
    main()
