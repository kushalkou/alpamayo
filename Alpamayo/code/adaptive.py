"""adaptive.py — GATE 3.3 confidence-adaptive and per-horizon shrinkage.

  (A) alpha_i = alpha_0 * exp(-lambda * max|predicted_curv_i|)   per-SAMPLE
      (2.4 showed the model is worst where it predicts large curvature, so shrink
       harder when it commits harder)
  (B) alpha_t = alpha_0 * (1 - t/T)^gamma                        per-HORIZON
      (error compounds with time under double integration)

Both grids are fit on VAL and evaluated on TEST, then compared against the global
alpha* from Gate 3.2 with a paired bootstrap. Reports both; recommends neither.

Vectorised: trajectories are stacked into [n,12,2] arrays so a whole grid point is
one numpy expression rather than n Python-level ADE calls.
"""
import numpy as np
from shrink_lib import load, traj, ade, paired, maxpredcurv, strata, N

MODELS = ['y1_full', 'y1_ego', 'zeroboth_jul12']
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)      # global alpha grid
A0GRID = np.round(np.arange(0, 1.5001, 0.05), 2)      # alpha_0 may exceed 1 (then clipped)
LAMBDAS = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
GAMMAS = np.array([0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0])


def stack(T, idx):
    return np.stack([T[i] for i in idx])


def ade_arr(P, G):
    """P,G [n,12,2] -> per-sample ADE@6s [n]."""
    return np.linalg.norm(P - G, axis=2).mean(1)


def main():
    Dv, mv, iv, Gv, CVv = load('val')
    Dt, mt, it, Gt, CVt = load('test')
    print(f"[3.3] val n={len(iv)}  test n={len(it)}")
    tprof = 1 - np.arange(N) / N

    for m in MODELS:
        Mv = stack({i: traj(Dv, mv, m, i, 'expect') for i in iv}, iv)
        Cv = stack(CVv, iv); GV = stack(Gv, iv)
        Mt = stack({i: traj(Dt, mt, m, i, 'expect') for i in it}, it)
        Ct = stack(CVt, it); GT = stack(Gt, it)
        Kv = np.array([maxpredcurv(Dv, m, i) for i in iv])
        Kt = np.array([maxpredcurv(Dt, m, i) for i in it])

        # global alpha* (Gate 3.2 reference), fit on val
        gcur = np.array([ade_arr(a*Mv + (1-a)*Cv, GV).mean() for a in ALPHAS])
        gstar = float(ALPHAS[int(gcur.argmin())])

        # (A) curvature-adaptive, fit on val
        bestA = (1e9, None, None)
        for lam in LAMBDAS:
            w = np.exp(-lam * Kv)
            for a0 in A0GRID:
                ai = np.clip(a0 * w, 0, 1)[:, None, None]
                e = ade_arr(ai*Mv + (1-ai)*Cv, GV).mean()
                if e < bestA[0]: bestA = (e, float(a0), float(lam))
        _, a0A, lamA = bestA

        # (B) per-horizon decay, fit on val
        bestB = (1e9, None, None)
        for gam in GAMMAS:
            prof = tprof ** gam if gam > 0 else np.ones(N)
            for a0 in A0GRID:
                at = np.clip(a0 * prof, 0, 1)[None, :, None]
                e = ade_arr(at*Mv + (1-at)*Cv, GV).mean()
                if e < bestB[0]: bestB = (e, float(a0), float(gam))
        _, a0B, gamB = bestB

        # TEST
        cv = ade_arr(Ct, GT)
        eg = ade_arr(gstar*Mt + (1-gstar)*Ct, GT)
        aiT = np.clip(a0A * np.exp(-lamA * Kt), 0, 1)[:, None, None]
        eA = ade_arr(aiT*Mt + (1-aiT)*Ct, GT)
        profB = tprof ** gamB if gamB > 0 else np.ones(N)
        atT = np.clip(a0B * profB, 0, 1)[None, :, None]
        eB = ade_arr(atT*Mt + (1-atT)*Ct, GT)

        print(f"\n[{m} / V1s]")
        print(f"  VAL fits: global alpha*={gstar:.2f} | (A) alpha0={a0A:.2f} lambda={lamA:g}"
              f" | (B) alpha0={a0B:.2f} gamma={gamB:g}")
        print(f"  alpha_t profile (B): " + " ".join(f"{v:.2f}" for v in
              np.clip(a0B*profB, 0, 1)))
        print(f"  TEST ADE@6s      {'mean':>8} {'median':>8} {'p95':>8}")
        for lab, e in (('CV', cv), ('global alpha*', eg),
                       ('(A) curv-adaptive', eA), ('(B) per-horizon', eB)):
            print(f"    {lab:18} {e.mean():8.3f} {np.median(e):8.3f} {np.percentile(e,95):8.3f}")
        for lab, e in (('(A) - global', eA), ('(B) - global', eB)):
            d, lo, hi, p = paired(e, eg)
            print(f"    {lab:18} {d:+8.4f}  [{lo:+7.4f},{hi:+7.4f}]  p={p:.4f}")
        for lab, e in (('(A) - CV', eA), ('(B) - CV', eB), ('global - CV', eg)):
            d, lo, hi, p = paired(e, cv)
            print(f"    {lab:18} {d:+8.4f}  [{lo:+7.4f},{hi:+7.4f}]  p={p:.4f}")
        st = strata(mt, it)
        for sname in ('straight', 'turning', 'stationary'):
            sub = [k for k, i in enumerate(it) if st[i] == sname]
            if not sub: continue
            print(f"    [{sname:10} n={len(sub):4d}] CV {cv[sub].mean():6.3f} | "
                  f"global {eg[sub].mean():6.3f} | (A) {eA[sub].mean():6.3f} | "
                  f"(B) {eB[sub].mean():6.3f}")


if __name__ == '__main__':
    main()
