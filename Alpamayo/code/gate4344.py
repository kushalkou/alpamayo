"""gate4344.py — GATE 4.3 win-rate honesty and GATE 4.4 identity check.

4.3 blend at alpha* vs CV per stratum: win rate + Wilson CI, median gap, and the
    gap distribution at p50/p75/p90/p95/p99. The gap is (CV - blend) per sample, so
    POSITIVE = blend better.

4.4 in the same code path that produced the headline: alpha=1 must reproduce the
    unblended numbers exactly and alpha=0 must reproduce CV exactly.
"""
import numpy as np
from shrink_lib import load, traj, ade, wilson, CURV_T

ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)
MODELS = ['y1_full', 'y1_ego', 'zeroboth_jul12']


def ade_arr(P, G):
    return np.linalg.norm(P - G, axis=2).mean(1)


def main():
    Dv, mv, iv, Gv, CVv = load('val')
    Dt, mt, it, Gt, CVt = load('test')
    Cv = np.stack([CVv[i] for i in iv]); GV = np.stack([Gv[i] for i in iv])
    Ct = np.stack([CVt[i] for i in it]); GT = np.stack([Gt[i] for i in it])
    curv = np.array([mt[i]['maxcurv'] for i in it])
    v0 = np.array([mt[i]['v0'] for i in it])
    strata = [('ALL', np.ones(len(it), bool)),
              ('STRAIGHT', curv <= CURV_T),
              ('TURNING', curv > CURV_T),
              ('STATIONARY', np.abs(v0) < 0.5)]
    cv = ade_arr(Ct, GT)

    print("=" * 90)
    print("GATE 4.3 — WIN-RATE HONESTY.  gap = CV - blend per sample; POSITIVE = blend better")
    print("=" * 90)
    for m in MODELS:
        Mv = np.stack([traj(Dv, mv, m, i, 'expect') for i in iv])
        Mt = np.stack([traj(Dt, mt, m, i, 'expect') for i in it])
        cur = np.array([ade_arr(a * Mv + (1 - a) * Cv, GV).mean() for a in ALPHAS])
        a_s = float(ALPHAS[int(cur.argmin())])
        bl = ade_arr(a_s * Mt + (1 - a_s) * Ct, GT)
        print(f"\n[{m} / V1s, alpha*={a_s:.2f}]")
        print(f"  {'stratum':12} {'n':>5} {'win rate':>9} {'Wilson 95%':>17} "
              f"{'mean gap':>9} {'med gap':>8}")
        for sname, sel in strata:
            g = cv[sel] - bl[sel]
            w = int((g > 0).sum()); n = int(sel.sum())
            lo, hi = wilson(w, n)
            print(f"  {sname:12} {n:5d} {w/n:9.4f} [{lo:6.4f},{hi:6.4f}] "
                  f"{g.mean():+9.4f} {np.median(g):+8.4f}")
        print(f"  gap quantiles (CV - blend, m):")
        print(f"  {'stratum':12} {'p50':>9} {'p75':>9} {'p90':>9} {'p95':>9} {'p99':>9}")
        for sname, sel in strata:
            g = cv[sel] - bl[sel]
            print(f"  {sname:12} " + " ".join(f"{np.percentile(g, q):+9.4f}"
                                              for q in (50, 75, 90, 95, 99)))

    print("\n" + "=" * 90)
    print("GATE 4.4 — IDENTITY CHECK (same code path as the headline)")
    print("=" * 90)
    for m in MODELS:
        Mt = np.stack([traj(Dt, mt, m, i, 'expect') for i in it])
        e1 = ade_arr(1.0 * Mt + 0.0 * Ct, GT)
        eu = ade_arr(Mt, GT)
        e0 = ade_arr(0.0 * Mt + 1.0 * Ct, GT)
        print(f"  {m:16} max|ADE(alpha=1) - ADE(unblended)| = "
              f"{np.abs(e1 - eu).max():.3e}   (means {e1.mean():.4f} / {eu.mean():.4f})")
        print(f"  {'':16} max|ADE(alpha=0) - ADE(CV)|        = "
              f"{np.abs(e0 - cv).max():.3e}   (means {e0.mean():.4f} / {cv.mean():.4f})")


if __name__ == '__main__':
    main()
