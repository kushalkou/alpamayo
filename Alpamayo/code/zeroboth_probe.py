"""zeroboth_probe.py — resolve the 2.3(c) zeroboth oracle-gain anomaly.

2.3(c) reported oracle gain vs CV = 0.424 on TURNING for zeroboth, yet 2.4 found
zeroboth predicts ZERO turns across all 3,614 samples. If it were truly CV-like,
min(zeroboth, CV) should be ~CV and the gain ~0.

PREDICTION under test: zeroboth matches CV on CURVATURE and diverges on ACCEL,
i.e. the residual is longitudinal only.

Decomposition: roll out the two cross-combinations
  (zeroboth accel + CV curvature)  and  (CV accel + zeroboth curvature)
CV's implied controls are accel=0, curvature=0 (constant speed, straight).
"""
import sys, pickle, math
import numpy as np

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
DT, N = 0.5, 12
HOR = {2: '1s', 4: '2s', 6: '3s', 12: '6s'}
CURV_T = 0.05


def rollout(acc, cur, v0, yaw0):
    x = y = 0.0; yaw = yaw0; v = v0; out = np.empty((N, 2))
    for j in range(N):
        v = max(0.0, v + acc[j] * DT)
        yaw = yaw + v * cur[j] * DT
        x += v * math.cos(yaw) * DT; y += v * math.sin(yaw) * DT
        out[j] = (x, y)
    return out


def cv_traj(v0, yaw0):
    k = np.arange(1, N + 1) * DT * v0
    return np.stack([k * math.cos(yaw0), k * math.sin(yaw0)], 1)


def ade(p, g, s=N):
    return float(np.linalg.norm(p[:s] - g[:s], axis=1).mean())


def main():
    D = pickle.load(open(f'{RES}/dump_test.pkl', 'rb'))
    meta = D['meta']; idx = sorted(meta)
    turn = [i for i in idx if meta[i]['maxcurv'] > CURV_T]
    print(f"[probe] test n={len(idx)}  TURNING n={len(turn)}\n")

    M = 'zeroboth_jul12'
    zb, cv, mixAC, mixCA, gt = {}, {}, {}, {}, {}
    zero = np.zeros(12)
    for i in idx:
        v = np.array(D['data'][M][i]['expect'])          # V1s values
        a, c = v[:12], v[12:]
        v0, yaw0 = meta[i]['v0'], meta[i]['yaw0']
        gt[i] = np.array(meta[i]['gt'])
        zb[i] = rollout(a, c, v0, yaw0)
        cv[i] = cv_traj(v0, yaw0)
        mixAC[i] = rollout(a, zero, v0, yaw0)            # zeroboth accel + CV curvature
        mixCA[i] = rollout(zero, c, v0, yaw0)            # CV accel + zeroboth curvature

    # 1) per-sample L2 between zeroboth and CV trajectories, per horizon
    print("(1) mean per-sample L2 |zeroboth - CV| trajectory distance, TURNING subset")
    print(f"    {'horizon':10} {'mean L2 (m)':>12} {'median':>10} {'p95':>10}")
    for s, lab in HOR.items():
        d = np.array([np.linalg.norm(zb[i][:s] - cv[i][:s], axis=1).mean() for i in turn])
        print(f"    {lab:10} {d.mean():12.4f} {np.median(d):10.4f} {np.percentile(d,95):10.4f}")

    # 2) decomposition
    print("\n(2) decomposition on TURNING -- ADE@6s vs ground truth")
    rows = [('CV (accel=0, curv=0)', cv), ('zeroboth (accel+curv)', zb),
            ('zeroboth ACCEL + CV curv', mixAC), ('CV accel + zeroboth CURV', mixCA)]
    print(f"    {'variant':30} {'ADE6s':>9} {'median':>9} {'p95':>9}")
    for lab, T in rows:
        e = np.array([ade(T[i], gt[i]) for i in turn])
        print(f"    {lab:30} {e.mean():9.3f} {np.median(e):9.3f} {np.percentile(e,95):9.3f}")
    print(f"\n    trajectory divergence from CV (mean L2 @6s, TURNING):")
    for lab, T in rows[1:]:
        d = np.array([np.linalg.norm(T[i] - cv[i], axis=1).mean() for i in turn])
        print(f"      {lab:30} {d.mean():9.4f}")

    # 3) correlation of per-sample ADE
    ez = np.array([ade(zb[i], gt[i]) for i in turn])
    ec = np.array([ade(cv[i], gt[i]) for i in turn])
    print(f"\n(3) per-sample ADE@6s correlation (TURNING): "
          f"pearson r = {np.corrcoef(ez, ec)[0,1]:.4f}, "
          f"spearman = {np.corrcoef(np.argsort(np.argsort(ez)), np.argsort(np.argsort(ec)))[0,1]:.4f}")
    orc = np.minimum(ez, ec)
    print(f"    CV {ec.mean():.3f}  zeroboth {ez.mean():.3f}  oracle min {orc.mean():.3f}  "
          f"gain vs CV {ec.mean()-orc.mean():.3f}")
    print(f"    fraction where zeroboth < CV: {(ez < ec).mean():.4f}")

    # 4) how much curvature does zeroboth actually emit?
    mx = np.array([np.abs(np.array(D['data'][M][i]['expect'])[12:]).max() for i in idx])
    mxa = np.array([np.abs(np.array(D['data'][M][i]['argmax'])[12:]).max() for i in idx])
    print(f"\n(4) zeroboth max|predicted curvature| over all {len(idx)} test samples")
    print(f"    V1s (expectation): mean {mx.mean():.5f}  p50 {np.median(mx):.5f}  "
          f"p95 {np.percentile(mx,95):.5f}  max {mx.max():.5f}  frac>0.05 {(mx>CURV_T).mean():.4f}")
    print(f"    argmax           : mean {mxa.mean():.5f}  p50 {np.median(mxa):.5f}  "
          f"p95 {np.percentile(mxa,95):.5f}  max {mxa.max():.5f}  frac>0.05 {(mxa>CURV_T).mean():.4f}")
    mxacc = np.array([np.abs(np.array(D['data'][M][i]['expect'])[:12]).max() for i in idx])
    print(f"    max|predicted ACCEL|: mean {mxacc.mean():.4f}  p50 {np.median(mxacc):.4f}  "
          f"p95 {np.percentile(mxacc,95):.4f}  max {mxacc.max():.4f}")


if __name__ == '__main__':
    main()
