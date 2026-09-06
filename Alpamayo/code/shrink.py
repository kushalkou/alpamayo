"""shrink.py — Gate 3.1 (stationary hybrid) and 3.2 (trajectory shrinkage), offline.

Everything is computed from dump_{val,test}.pkl, which holds per-slot argmax and
STOP-aware-expectation values plus p(STOP). No GPU needed.

RULE: alpha and tau are fit on VAL ONLY and applied to TEST.
A blend is a model+CV ENSEMBLE, not the model beating CV. Captioned as such.
"""
import sys, pickle, math, numpy as np

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
DT, N = 0.5, 12
HOR = {2: '1s', 4: '2s', 6: '3s', 12: '6s'}
CURV_T = 0.05
rng = np.random.RandomState(0)


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


def ade(p, g, steps=N):
    return float(np.linalg.norm(p[:steps] - g[:steps], axis=1).mean())


def paired(a, b, n=10000, chunk=2000):
    d = np.asarray(a, float) - np.asarray(b, float); m = len(d)
    out = np.empty(n)
    for s in range(0, n, chunk):
        k = min(chunk, n - s); out[s:s+k] = d[rng.randint(0, m, (k, m))].mean(1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return d.mean(), lo, hi, min(1.0, 2 * min((out <= 0).mean(), (out >= 0).mean()))


def wilson(k, n, z=1.96):
    if n == 0: return float('nan'), float('nan')
    ph = k / n; den = 1 + z*z/n
    c = (ph + z*z/(2*n)) / den
    h = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / den
    return c - h, c + h


def load(split):
    with open(f'{RES}/dump_{split}.pkl', 'rb') as f:
        D = pickle.load(f)
    meta = D['meta']; idx = sorted(meta)
    G = {i: np.array(meta[i]['gt']) for i in idx}
    CVt = {i: cv_traj(meta[i]['v0'], meta[i]['yaw0']) for i in idx}
    return D, meta, idx, G, CVt


def values(D, m, i, mode, tau=None):
    """mode: 'argmax' | 'expect' | 'hybrid' (argmax where p_stop>tau)."""
    r = D['data'][m][i]
    a = np.array(r['argmax']); e = np.array(r['expect'])
    if mode == 'argmax': v = a
    elif mode == 'expect': v = e
    else:
        ps = np.array(r['p_stop']); v = np.where(ps > tau, a, e)
    return v[:12], v[12:]


def traj(D, meta, m, i, mode, tau=None):
    acc, cur = values(D, m, i, mode, tau)
    return rollout(acc, cur, meta[i]['v0'], meta[i]['yaw0'])


def main():
    Dv, mv, iv, Gv, CVv = load('val')
    Dt, mt, it, Gt, CVt = load('test')
    models = list(Dt['data'].keys())
    print(f"val n={len(iv)}  test n={len(it)}  models={models}")

    # cache trajectories
    TR = {}
    for split, (D, meta, idx) in (('val', (Dv, mv, iv)), ('test', (Dt, mt, it))):
        for m in models:
            for mode in ('argmax', 'expect'):
                TR[(split, m, mode, None)] = {i: traj(D, meta, m, i, mode) for i in idx}

    # ── 3.1 STATIONARY HYBRID ────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("GATE 3.1 — STATIONARY HYBRID:  p(STOP) > tau -> argmax, else expectation")
    print("tau selected on VAL, applied to TEST.  CV: val/test reported per split.")
    print("=" * 96)
    stat_v = [i for i in iv if abs(mv[i]['v0']) < 0.5]
    stat_t = [i for i in it if abs(mt[i]['v0']) < 0.5]
    print(f"\nVAL sweep (n_stat={len(stat_v)}, n_nonstat={len(iv)-len(stat_v)}) ADE@6s mean")
    print(f"{'variant':22} {'ALL':>9} {'STATIONARY':>12} {'NON-STAT':>10}")
    def rep(D, meta, idx, stat, mode, tau=None, tr=None):
        T = tr if tr is not None else {i: traj(D, meta, i and i, mode, tau) for i in idx}
        G = {i: np.array(meta[i]['gt']) for i in idx}
        a = np.array([ade(T[i], G[i]) for i in idx])
        s = np.array([ade(T[i], G[i]) for i in stat])
        ns = np.array([ade(T[i], G[i]) for i in idx if i not in set(stat)])
        return a, s, ns
    best_tau = {}
    for m in models:
        print(f"\n[{m}]")
        cvA = np.array([ade(CVv[i], Gv[i]) for i in iv])
        cvS = np.array([ade(CVv[i], Gv[i]) for i in stat_v])
        print(f"{'CV':22} {cvA.mean():9.3f} {cvS.mean():12.3f} "
              f"{np.array([ade(CVv[i],Gv[i]) for i in iv if abs(mv[i]['v0'])>=0.5]).mean():10.3f}")
        rows = [('argmax', TR[('val', m, 'argmax', None)]),
                ('V1s_expect', TR[('val', m, 'expect', None)])]
        for tau in (0.3, 0.5, 0.7, 0.9):
            rows.append((f'hybrid_tau{tau}', {i: traj(Dv, mv, m, i, 'hybrid', tau) for i in iv}))
        scores = {}
        for lab, T in rows:
            a = np.array([ade(T[i], Gv[i]) for i in iv])
            s = np.array([ade(T[i], Gv[i]) for i in stat_v])
            ns = np.array([ade(T[i], Gv[i]) for i in iv if abs(mv[i]['v0']) >= 0.5])
            scores[lab] = a.mean()
            print(f"{lab:22} {a.mean():9.3f} {s.mean():12.3f} {ns.mean():10.3f}")
        cand = {k: v for k, v in scores.items() if k.startswith('hybrid')}
        bt = min(cand, key=cand.get); best_tau[m] = float(bt.replace('hybrid_tau', ''))
        print(f"  -> VAL-selected tau = {best_tau[m]}")

    print(f"\nTEST at the VAL-selected tau — ADE@6s mean (median)")
    print(f"{'model / variant':30} {'ALL':>16} {'STATIONARY':>16} {'NON-STAT':>16}")
    nstat_t = [i for i in it if abs(mt[i]['v0']) >= 0.5]
    cvA = np.array([ade(CVt[i], Gt[i]) for i in it])
    cvS = np.array([ade(CVt[i], Gt[i]) for i in stat_t])
    cvN = np.array([ade(CVt[i], Gt[i]) for i in nstat_t])
    print(f"{'CV baseline':30} {cvA.mean():8.3f} ({np.median(cvA):5.3f}) "
          f"{cvS.mean():8.3f} ({np.median(cvS):5.3f}) {cvN.mean():8.3f} ({np.median(cvN):5.3f})")
    hyb_test = {}
    for m in models:
        rows = [('argmax', TR[('test', m, 'argmax', None)]),
                ('V1s_expect', TR[('test', m, 'expect', None)]),
                (f'hybrid_tau{best_tau[m]}',
                 {i: traj(Dt, mt, m, i, 'hybrid', best_tau[m]) for i in it})]
        hyb_test[m] = rows[2][1]
        for lab, T in rows:
            a = np.array([ade(T[i], Gt[i]) for i in it])
            s = np.array([ade(T[i], Gt[i]) for i in stat_t])
            ns = np.array([ade(T[i], Gt[i]) for i in nstat_t])
            print(f"{m+'/'+lab:30} {a.mean():8.3f} ({np.median(a):5.3f}) "
                  f"{s.mean():8.3f} ({np.median(s):5.3f}) {ns.mean():8.3f} ({np.median(ns):5.3f})")
    print(f"\nPaired bootstrap on TEST (negative => first better), ADE@6s")
    print(f"{'model':18} {'comparison':32} {'delta':>8} {'95% CI':>18} {'p':>8} {'subset':>10}")
    for m in models:
        A = TR[('test', m, 'argmax', None)]; E = TR[('test', m, 'expect', None)]; H = hyb_test[m]
        for sub, sname in ((it, 'ALL'), (stat_t, 'STATION'), (nstat_t, 'NONSTAT')):
            for lab, X, Y in (('hybrid - argmax', H, A), ('hybrid - V1s', H, E)):
                d, lo, hi, p = paired(np.array([ade(X[i], Gt[i]) for i in sub]),
                                      np.array([ade(Y[i], Gt[i]) for i in sub]))
                print(f"{m:18} {lab:32} {d:+8.3f}  [{lo:+6.3f},{hi:+6.3f}] {p:8.4f} {sname:>10}")

    # ── 3.2 GLOBAL TRAJECTORY SHRINKAGE ──────────────────────────────────────
    print("\n" + "=" * 96)
    print("GATE 3.2 — GLOBAL TRAJECTORY SHRINKAGE (model+CV ENSEMBLE, not model beating CV)")
    print("  traj = alpha*model + (1-alpha)*CV, per timestep, in position space")
    print("  alpha FIT ON VAL (n=%d), EVALUATED ON TEST (n=%d)" % (len(iv), len(it)))
    print("=" * 96)
    alphas = np.round(np.arange(0, 1.0001, 0.05), 2)
    strata_t = [('ALL', it),
                ('STRAIGHT', [i for i in it if mt[i]['maxcurv'] <= CURV_T]),
                ('TURNING', [i for i in it if mt[i]['maxcurv'] > CURV_T]),
                ('STATIONARY', stat_t)]
    for m in models:
        for mode in ('expect', 'argmax'):
            Tv = TR[('val', m, mode, None)]; Tt = TR[('test', m, mode, None)]
            curve = []
            for al in alphas:
                e = np.array([ade(al*Tv[i] + (1-al)*CVv[i], Gv[i]) for i in iv])
                curve.append(e.mean())
            curve = np.array(curve); astar = float(alphas[int(curve.argmin())])
            # bootstrap CI on alpha* itself
            per = np.stack([[ade(al*Tv[i] + (1-al)*CVv[i], Gv[i]) for i in iv] for al in alphas])
            bs = []
            for _ in range(2000):
                sel = rng.randint(0, len(iv), len(iv))
                bs.append(alphas[int(per[:, sel].mean(1).argmin())])
            alo, ahi = np.percentile(bs, [2.5, 97.5])
            tag = 'V1s' if mode == 'expect' else 'argmax'
            print(f"\n[{m} / {tag}]  VAL alpha curve (ADE@6s):")
            print("  " + "  ".join(f"{a:.2f}:{c:.3f}" for a, c in zip(alphas[::2], curve[::2])))
            print(f"  alpha* = {astar:.2f}   bootstrap 95% CI [{alo:.2f}, {ahi:.2f}]"
                  f"   (alpha=0 -> {curve[0]:.3f} == CV, alpha=1 -> {curve[-1]:.3f})")
            print(f"  TEST at alpha*={astar:.2f}:")
            print(f"  {'stratum':12} {'n':>6} {'CV':>8} {'alpha=1':>9} {'blend':>8} "
                  f"{'medCV':>8} {'med a=1':>8} {'medblend':>9} {'p95 a=1':>9} {'p95 bl':>8}")
            for sname, sub in strata_t:
                c = np.array([ade(CVt[i], Gt[i]) for i in sub])
                a1 = np.array([ade(Tt[i], Gt[i]) for i in sub])
                bl = np.array([ade(astar*Tt[i] + (1-astar)*CVt[i], Gt[i]) for i in sub])
                print(f"  {sname:12} {len(sub):6d} {c.mean():8.3f} {a1.mean():9.3f} {bl.mean():8.3f} "
                      f"{np.median(c):8.3f} {np.median(a1):8.3f} {np.median(bl):9.3f} "
                      f"{np.percentile(a1,95):9.3f} {np.percentile(bl,95):8.3f}")
            print(f"  paired bootstrap, TEST:")
            for sname, sub in strata_t:
                c = np.array([ade(CVt[i], Gt[i]) for i in sub])
                bl = np.array([ade(astar*Tt[i] + (1-astar)*CVt[i], Gt[i]) for i in sub])
                a1 = np.array([ade(Tt[i], Gt[i]) for i in sub])
                d, lo, hi, p = paired(bl, c)
                w = (bl < c).sum(); wlo, whi = wilson(w, len(sub))
                d2, lo2, hi2, p2 = paired(bl, a1)
                print(f"    {sname:12} blend-CV {d:+7.3f} [{lo:+6.3f},{hi:+6.3f}] p={p:6.4f} | "
                      f"blend-a1 {d2:+7.3f} [{lo2:+6.3f},{hi2:+6.3f}] p={p2:6.4f} | "
                      f"win {w/len(sub):.3f} [{wlo:.3f},{whi:.3f}]")
    # ---- 3.2 addendum: PER-STRATUM alpha --------------------------------------
    print("\n" + "=" * 96)
    print("GATE 3.2b — PER-STRATUM ALPHA (fit on VAL per stratum, applied to TEST)")
    print("  (a) single global alpha*")
    print("  (b) per-stratum alpha* under the ORACLE stratum label  [UPPER BOUND, not a system]")
    print("  (c) per-stratum alpha* under DETECTOR (i) past-curvature  [DEPLOYABLE]")
    print("=" * 96)

    def strat_oracle(meta, i):
        if abs(meta[i]['v0']) < 0.5: return 'stationary'
        return 'turning' if meta[i]['maxcurv'] > CURV_T else 'straight'

    def strat_det(meta, i):
        if abs(meta[i]['v0']) < 0.5: return 'stationary'
        return 'turning' if meta[i]['past_curv'] > CURV_T else 'straight'

    for m in models:
        for mode in ('expect',):
            Tv = TR[('val', m, mode, None)]; Tt = TR[('test', m, mode, None)]
            tag = 'V1s'
            # global alpha* (refit here so this block is self-contained)
            gcurve = np.array([np.mean([ade(al*Tv[i] + (1-al)*CVv[i], Gv[i]) for i in iv])
                               for al in alphas])
            gstar = float(alphas[int(gcurve.argmin())])
            # per-stratum alpha* on VAL, using the ORACLE label on val
            astar_s, ci_s = {}, {}
            for sname in ('straight', 'turning', 'stationary'):
                sub = [i for i in iv if strat_oracle(mv, i) == sname]
                if len(sub) < 20:
                    astar_s[sname] = gstar; ci_s[sname] = (float('nan'),)*2; continue
                per = np.stack([[ade(al*Tv[i] + (1-al)*CVv[i], Gv[i]) for i in sub]
                                for al in alphas])
                astar_s[sname] = float(alphas[int(per.mean(1).argmin())])
                bs = [alphas[int(per[:, rng.randint(0, len(sub), len(sub))].mean(1).argmin())]
                      for _ in range(2000)]
                ci_s[sname] = tuple(np.percentile(bs, [2.5, 97.5]))
            print(f"\n[{m} / {tag}] global alpha* = {gstar:.2f}")
            for sname in ('straight', 'turning', 'stationary'):
                nv = sum(1 for i in iv if strat_oracle(mv, i) == sname)
                print(f"  VAL alpha*[{sname:10}] = {astar_s[sname]:.2f}  "
                      f"95% CI [{ci_s[sname][0]:.2f},{ci_s[sname][1]:.2f}]  (n_val={nv})")
            # TEST under each policy
            def blend(alpha_of, sub):
                return np.array([ade(alpha_of(i)*Tt[i] + (1-alpha_of(i))*CVt[i], Gt[i])
                                 for i in sub])
            pol = {'(a) global': lambda i: gstar,
                   '(b) per-stratum ORACLE': lambda i: astar_s[strat_oracle(mt, i)],
                   '(c) per-stratum DETECTOR(i)': lambda i: astar_s[strat_det(mt, i)]}
            print(f"  TEST ADE@6s (mean / median / p95), n={len(it)}:")
            cvv = np.array([ade(CVt[i], Gt[i]) for i in it])
            print(f"    {'CV baseline':30} {cvv.mean():8.3f} {np.median(cvv):8.3f} "
                  f"{np.percentile(cvv,95):8.3f}")
            res = {}
            for pname, fn in pol.items():
                e = blend(fn, it); res[pname] = e
                print(f"    {pname:30} {e.mean():8.3f} {np.median(e):8.3f} "
                      f"{np.percentile(e,95):8.3f}")
            for pname in ('(b) per-stratum ORACLE', '(c) per-stratum DETECTOR(i)'):
                d, lo, hi, p = paired(res[pname], res['(a) global'])
                print(f"    {pname} - global: {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}")
            d, lo, hi, p = paired(res['(c) per-stratum DETECTOR(i)'], cvv)
            print(f"    (c) - CV: {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}")

    # sanity: alpha=0 must recover CV exactly
    m0 = models[0]
    chk = max(abs(ade(0.0*TR[('test', m0, 'expect', None)][i] + 1.0*CVt[i], Gt[i]) - ade(CVt[i], Gt[i]))
              for i in it[:200])
    print(f"\n[SANITY] max |ADE(alpha=0) - ADE(CV)| over 200 test samples = {chk:.2e} (must be ~0)")


if __name__ == '__main__':
    main()
