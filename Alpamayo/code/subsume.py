"""subsume.py — GATE 3.4: does shrinkage subsume the decode fix?

2x2: {argmax, V1s} x {alpha=1, alpha*}, on TEST, per stratum. alpha* is fit on VAL
SEPARATELY for each decode (fitting one decode's alpha and applying it to the other
would confound the comparison).

If blended-argmax reaches blended-V1s, the decode gain and the shrinkage gain are
the same underlying variance reduction and must be described as ONE finding.
"""
import numpy as np
from shrink_lib import load, traj, ade, paired, strata

MODELS = ['y1_full', 'y1_ego', 'zeroboth_jul12']
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)


def main():
    Dv, mv, iv, Gv, CVv = load('val')
    Dt, mt, it, Gt, CVt = load('test')
    print(f"[3.4] val n={len(iv)}  test n={len(it)}")
    st = strata(mt, it)
    cv = np.array([ade(CVt[i], Gt[i]) for i in it])

    for m in MODELS:
        print(f"\n[{m}]  2x2  {{argmax, V1s}} x {{alpha=1, alpha*}}   (alpha* fit on VAL "
              f"per decode)")
        cells = {}
        astar = {}
        for mode, tag in (('argmax', 'argmax'), ('expect', 'V1s')):
            Tv = {i: traj(Dv, mv, m, i, mode) for i in iv}
            Tt = {i: traj(Dt, mt, m, i, mode) for i in it}
            g = np.array([np.mean([ade(a*Tv[i] + (1-a)*CVv[i], Gv[i]) for i in iv])
                          for a in ALPHAS])
            a_s = float(ALPHAS[int(g.argmin())]); astar[tag] = a_s
            cells[(tag, 'alpha=1')] = np.array([ade(Tt[i], Gt[i]) for i in it])
            cells[(tag, 'alpha*')] = np.array([ade(a_s*Tt[i] + (1-a_s)*CVt[i], Gt[i])
                                               for i in it])
        print(f"  VAL alpha*: argmax {astar['argmax']:.2f}   V1s {astar['V1s']:.2f}")
        print(f"  TEST ADE@6s mean (median / p95), n={len(it)}    CV = {cv.mean():.3f} "
              f"({np.median(cv):.3f} / {np.percentile(cv,95):.3f})")
        print(f"    {'cell':22} {'mean':>8} {'median':>8} {'p95':>8}")
        for tag in ('argmax', 'V1s'):
            for al in ('alpha=1', 'alpha*'):
                e = cells[(tag, al)]
                print(f"    {tag+' / '+al:22} {e.mean():8.3f} {np.median(e):8.3f} "
                      f"{np.percentile(e,95):8.3f}")
        d, lo, hi, p = paired(cells[('argmax', 'alpha*')], cells[('V1s', 'alpha*')])
        print(f"    blended-argmax - blended-V1s: {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}"
              f"   {'-> SUBSUMED (n.s.)' if p > 0.05 else '-> NOT subsumed'}")
        d2, lo2, hi2, p2 = paired(cells[('V1s', 'alpha=1')], cells[('argmax', 'alpha=1')])
        print(f"    unblended V1s - argmax:       {d2:+.4f} [{lo2:+.4f},{hi2:+.4f}] p={p2:.4f}")
        for sname in ('straight', 'turning', 'stationary'):
            sub = [k for k, i in enumerate(it) if st[i] == sname]
            if not sub: continue
            print(f"    [{sname:10} n={len(sub):4d}] CV {cv[sub].mean():6.3f} | "
                  f"arg/a=1 {cells[('argmax','alpha=1')][sub].mean():6.3f} | "
                  f"arg/a* {cells[('argmax','alpha*')][sub].mean():6.3f} | "
                  f"V1s/a=1 {cells[('V1s','alpha=1')][sub].mean():6.3f} | "
                  f"V1s/a* {cells[('V1s','alpha*')][sub].mean():6.3f}")


if __name__ == '__main__':
    main()
