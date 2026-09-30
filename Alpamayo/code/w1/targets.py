"""w1/targets.py -- WEEK1 STEP 2a/2b/2c: new control targets, percentile tokenizer, floor.

2b FRAME. The trajectory is VAD's evaluation trajectory itself: the LIDAR_TOP sensor
  position at future keyframes 1..n_fut (n_fut <= 12) in the t0 LIDAR frame (x right,
  y forward), built exactly as vad_nuscenes_converter.py builds gt_ego_fut_trajs
  (global sensor pose -> ego at lcf -> lidar at lcf). Its first 6 steps equal
  gt_lidar6. The unicycle rollout runs directly in this frame from (0,0) with heading
  yaw0 = pi/2 (+y = forward), so predictions need no conversion and GT L2 = 0 with no
  offset.
2a TARGETS. Chord construction from the chosen v0 (v0_can, w1/v0_sources.py): segment
  k (0..n-1) from P_k to P_k+1 has speed s_k = |dP_k|/dt and heading theta_k.
    acc[0] = (s_0 - v0)/dt,  acc[k] = (s_k - s_k-1)/dt       (a0 is now nontrivial)
    cur[k] = wrap(theta_k - theta_k-1) / (s_k*dt),  theta_-1 = yaw0 = pi/2
  If s_k*dt <= 0.01 m the heading is held (cur = 0). A rollout of the continuous
  targets reproduces the trajectory to float precision (checked). STOP when s_k < 0.1
  (the tokenizer rule, unchanged; STOP decodes to (0,0)).
TOKENIZER (ii). 64 equal-mass bins per channel on the NEW official-train targets (non-STOP
  steps of n_fut=12 records); centre = MEDIAN of the train values in the bin. (Mean
  centres were tried first: the chord curvature has a heavy low-speed noise tail, which
  dragged the edge-bin means to +-2-3 rad/m; holdout floor 0.785 mean vs 0.440 with
  median centres. Chosen on HOLDOUT, see Alpamayo/w1_bins_variants.log.)
2c. Would-be CE class weights from the new train tokens.
Output: Alpamayo/data/w1_targets.pkl.
"""
import sys, math, pickle, contextlib, io
import numpy as np
from pyquaternion import Quaternion
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
from tokenizer import STOP_TOKEN, STOP_SPEED_THRESHOLD
from tok_floor import Bins, roll_batch
import plan_metrics as PMX

DT = 0.5
YAW0 = math.pi / 2
ROOT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
D = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def lidar_traj(nusc, rec):
    from build_data import lidar_global_pos, to_lidar
    toks = [rec['sample_token']]
    for _ in range(rec['n_fut']):
        toks.append(nusc.get('sample', toks[-1])['next'])
    g = np.stack([lidar_global_pos(nusc, nusc.get('sample', t)) for t in toks])
    return to_lidar(g, rec['lidar_ep0'], rec['lidar_cs'])[:, :2]      # [n+1,2], row0 ~ 0


def controls(P, v0):
    n = len(P) - 1
    d = np.diff(P, axis=0); s = np.linalg.norm(d, axis=1) / DT
    acc = np.empty(n); cur = np.zeros(n)
    acc[0] = (s[0] - v0) / DT; acc[1:] = np.diff(s) / DT
    th_prev = YAW0
    for k in range(n):
        if s[k] * DT > 0.01:
            th = math.atan2(d[k, 1], d[k, 0])
            cur[k] = wrap(th - th_prev) / (s[k] * DT); th_prev = th
    return acc, cur, s


def rollout(acc, cur, v0):
    return roll_batch(np.asarray(acc)[None], np.asarray(cur)[None], v0, YAW0)[0]


def fit_bins(A, K):
    out = []
    for X in (A, K):
        e = np.quantile(X, np.linspace(0, 1, 65)); e[0], e[-1] = -np.inf, np.inf
        e = np.maximum.accumulate(e)
        idx = np.clip(np.searchsorted(e, X, 'right') - 1, 0, 63)
        c = np.array([np.median(X[idx == b]) if (idx == b).any() else np.nan for b in range(64)])
        for b in range(64):
            if np.isnan(c[b]): c[b] = c[b - 1] if b else np.nanmin(c)
        out += [c, e]
    return Bins(out[0], out[2], out[1], out[3])


def tokenize(B, acc, cur, s):
    return [B.tok(acc[j], cur[j], s[j]) for j in range(len(acc))]


def detok_roll(B, toks, v0):
    da, dk = zip(*[B.detok(a, k) for a, k in toks])
    return rollout(da, dk, v0)


def main():
    from nuscenes.nuscenes import NuScenes
    W = pickle.load(open(f'{D}/w1_data.pkl', 'rb'))
    V0 = pickle.load(open(f'{D}/w1_v0.pkl', 'rb'))
    nusc = NuScenes(version='v1.0-trainval', dataroot=ROOT, verbose=False)
    T = {}
    rec_err = []
    for r in W['records']:
        P = lidar_traj(nusc, r)
        v0 = V0[r['sample_token']]['v0_can']
        acc, cur, s = controls(P, v0)
        rec_err.append(np.abs(rollout(acc, cur, v0) - P[1:]).max())
        assert np.abs(P[1:7] - r['gt_lidar6']).max() < 1e-9
        T[r['sample_token']] = {'P': P[1:], 'v0': v0, 'acc': acc, 'cur': cur, 's': s}
    rec_err = np.array(rec_err)
    print(f'continuous-target rollout vs trajectory: max err {rec_err.max():.2e} m '
          f'(n={len(rec_err)}); samples > 1e-6: {(rec_err > 1e-6).sum()}')
    tr = [r for r in W['records'] if r['split'] == 'train' and r['n_fut'] == 12]
    A, K = [], []
    for r in tr:
        t = T[r['sample_token']]; m = t['s'] >= STOP_SPEED_THRESHOLD
        A.append(t['acc'][m]); K.append(t['cur'][m])
    A, K = np.concatenate(A), np.concatenate(K)
    B = fit_bins(A, K)
    print(f'train controls (non-STOP steps n={len(A)}): accel p1/p50/p99 '
          f'{np.percentile(A,1):.2f}/{np.median(A):.3f}/{np.percentile(A,99):.2f}  '
          f'curv p1/p50/p99 {np.percentile(K,1):.4f}/{np.median(K):.5f}/{np.percentile(K,99):.4f}')
    print(f'  bin centre range accel [{B.ac[0]:.2f},{B.ac[-1]:.2f}]  curv [{B.kc[0]:.4f},{B.kc[-1]:.4f}]')
    for r in W['records']:
        t = T[r['sample_token']]
        t['tokens'] = tokenize(B, t['acc'], t['cur'], t['s'])

    # ---- floor ----
    from tok_floor import uniform_bins
    U = uniform_bins()
    print('\nFLOOR on the new targets, true v0_can: rollout of detok(tok(GT))')
    for sp, lab, BB, sel in (('holdout', '(ii) ALL', B, None), ('holdout', '(ii) CAN ok', B, True),
                             ('holdout', '(ii) no CAN', B, False), ('holdout', '(i) uniform ALL', U, None),
                             ('val', '(ii) ALL', B, None), ('val', '(ii) CAN ok', B, True),
                             ('val', '(ii) no CAN', B, False), ('val', '(i) uniform ALL', U, None)):
        R = [r for r in W['records'] if r['split'] == sp and
             (sel is None or V0[r['sample_token']]['can_ok'] == sel)]
        per = {'l2': []}
        ade, fde = [], []
        det = True
        for r in R:
            t = T[r['sample_token']]
            tk = t['tokens'] if BB is B else tokenize(BB, t['acc'], t['cur'], t['s'])
            Pq = detok_roll(BB, tk, t['v0'])
            per['l2'].append(np.linalg.norm(Pq[:6] - t['P'][:6], axis=1))
            if r['n_fut'] == 12:
                e = np.linalg.norm(Pq - t['P'], axis=1); ade.append(e.mean()); fde.append(e[-1])
            if len(ade) <= 300 and tokenize(B, t['acc'], t['cur'], t['s']) != t['tokens']:
                det = False
        l2 = np.stack(per['l2']); ade = np.array(ade)
        print(f'  {sp:8} {lab:16} n={len(R)} (n_fut=12: {len(ade)})  L2 NoAvg 1/2/3s '
              + ' '.join(f'{l2[:, k-1].mean():.3f}' for k in (2, 4, 6))
              + '  TemAvg ' + ' '.join(f'{l2[:, :k].mean():.3f}' for k in (2, 4, 6))
              + f'  ADE@6s {ade.mean():.3f} (med {np.median(ade):.3f}, p95 '
              f'{np.percentile(ade,95):.3f})  FDE@6s {np.mean(fde):.3f}  deterministic {det}')
    # roundtrip exactness: detok twice identical
    t = T[tr[0]['sample_token']]
    assert [B.detok(*x) for x in t['tokens']] == [B.detok(*x) for x in t['tokens']]

    # ---- 2c: would-be inverse-frequency weights ----
    Y = np.array([[64 if a == STOP_TOKEN else a for a, _ in T[r['sample_token']]['tokens']] +
                  [64 if k == STOP_TOKEN else k for _, k in T[r['sample_token']]['tokens']]
                  for r in tr])
    print('\n2c would-be sqrt-inverse-frequency CE weights (train, classes with count > 0)')
    pooled = np.bincount(Y.ravel(), minlength=65).astype(float)
    for lab, cnt in (('pooled all slots', pooled),
                     ('pooled accel slots', np.bincount(Y[:, :12].ravel(), minlength=65).astype(float)),
                     ('pooled curv slots', np.bincount(Y[:, 12:].ravel(), minlength=65).astype(float))):
        c = cnt[cnt > 0]; w = np.sqrt(c.sum() / c)
        print(f'  {lab:20} weight max/min {w.max()/w.min():7.2f}   (incl. STOP; excl. STOP: '
              f'{(np.sqrt(cnt[:64][cnt[:64]>0].sum()/cnt[:64][cnt[:64]>0])).max()/(np.sqrt(cnt[:64][cnt[:64]>0].sum()/cnt[:64][cnt[:64]>0])).min():.2f})')
    print('  per slot max/min (excl. STOP): ' + ' '.join(
        f'{(lambda c: (np.sqrt(c.sum()/c)).max()/(np.sqrt(c.sum()/c)).min())(np.bincount(Y[:, j], minlength=65)[:64][np.bincount(Y[:, j], minlength=65)[:64] > 0]):.1f}'
        for j in range(24)))
    pickle.dump({'targets': T, 'bins': (B.ac, B.kc, B.ae, B.ke), 'yaw0': YAW0},
                open(f'{D}/w1_targets.pkl', 'wb'))
    print(f'\nsaved {D}/w1_targets.pkl')


if __name__ == '__main__':
    main()
