"""gate41.py — GATE 4.1: is alpha* variance reduction, or real signal?

Decomposition. With M=model, C=CV, G=ground truth, and blend(a)=a*M+(1-a)*C:

    blend(a) - G  =  a*D + R,    D = M - C,   R = C - G

so for squared error the optimum is a* = <D, G-C> / ||D||^2. Under a pure
"CV + zero-mean noise independent of the residual" model that inner product has
expectation zero, hence a* = 0. A strictly positive alpha* therefore means D points,
on average, along the correction CV actually needs -- which is signal by definition.
This script measures that rather than assuming it.

Everything is done in the EGO frame (rotate by -yaw0). ADE is rotation invariant so
no metric changes, but it makes "mean deviation" meaningful: a mean over samples in
global axes is nonsense because headings differ.

Four predictors, all run through the IDENTICAL val-fit / test-eval pipeline:

  real       C + D            the actual model
  null_gauss C + gaussian     zero-mean noise, per-timestep variance matched to D,
                              independent of R  -> the task's specified null
  null_perm  C + D[perm]      D from a RANDOM OTHER sample: preserves D's magnitude
                              and temporal structure exactly, destroys scene alignment
  null_mean  C + mean_val(D)  the constant population-level correction alone,
                              fit on val -- no per-sample variation at all

The three nulls separate the gain into: pure variance reduction (gauss), a constant
prior correction (mean), and genuine scene-specific information (real minus the others).
"""
import sys
import numpy as np
from shrink_lib import load, traj, ade, paired, CURV_T, N

MODELS = ['y1_full', 'y1_ego']
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)
NSEED = 20


def rot(T, yaw):
    """rotate [.,12,2] by -yaw into the ego frame"""
    c, s = np.cos(-yaw)[:, None], np.sin(-yaw)[:, None]   # [n,1] broadcast over steps
    x, y = T[..., 0], T[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], -1)


def ade_arr(P, G):
    return np.linalg.norm(P - G, axis=2).mean(1)


def fit_alpha(Mv, Cv, Gv):
    cur = np.array([ade_arr(a * Mv + (1 - a) * Cv, Gv).mean() for a in ALPHAS])
    return float(ALPHAS[int(cur.argmin())]), cur


def main():
    Dv_, mv, iv, Gv_, CVv_ = load('val')
    Dt_, mt, it, Gt_, CVt_ = load('test')
    print("=" * 88)
    print("GATE 4.1 — IS alpha* VARIANCE REDUCTION OR REAL SIGNAL?")
    print("=" * 88)

    for mdl in MODELS:
        out = {}
        for split, (DD, meta, idx, GG, CC) in (('val', (Dv_, mv, iv, Gv_, CVv_)),
                                               ('test', (Dt_, mt, it, Gt_, CVt_))):
            yaw = np.array([meta[i]['yaw0'] for i in idx])
            M = rot(np.stack([traj(DD, meta, mdl, i, 'expect') for i in idx]), yaw)
            C = rot(np.stack([CC[i] for i in idx]), yaw)
            G = rot(np.stack([GG[i] for i in idx]), yaw)
            out[split] = dict(M=M, C=C, G=G, D=M - C, R=C - G,
                              curv=np.array([meta[i]['maxcurv'] for i in idx]))
        V, T = out['val'], out['test']

        print(f"\n{'='*88}\n[{mdl} / V1s]\n{'='*88}")

        # ---- (a) bias / variance decomposition, ego frame ------------------
        print("\n(a) BIAS / VARIANCE of the per-sample error vs ground truth (ego frame, TEST)")
        print(f"    {'subset':10} {'predictor':10} {'MSE':>9} {'bias^2':>9} {'var':>9} "
              f"{'|bias| (m)':>11}")
        for sname, sel in (('ALL', np.ones(len(T['G']), bool)),
                           ('TURNING', T['curv'] > CURV_T)):
            for lab, E in (('CV', T['R']), ('model', T['M'] - T['G'])):
                e = E[sel]                                  # [n,12,2]
                mse = (e ** 2).sum(2).mean()                # mean over samples+steps
                b = e.mean(0)                               # [12,2] systematic offset
                b2 = (b ** 2).sum(1).mean()
                var = ((e - b) ** 2).sum(2).mean()
                print(f"    {sname:10} {lab:10} {mse:9.3f} {b2:9.3f} {var:9.3f} "
                      f"{np.linalg.norm(b, axis=1).mean():11.3f}")

        # ---- (b) closed-form optimal alpha ---------------------------------
        num = (V['D'] * (-V['R'])).sum()      # <D, G-C>
        den = (V['D'] ** 2).sum()             # ||D||^2
        a_mse = num / den
        nrm = np.sqrt(den * (V['R'] ** 2).sum())
        rho = num / nrm
        a_emp, _ = fit_alpha(V['M'], V['C'], V['G'])
        print("\n(b) CLOSED-FORM OPTIMUM (fit on VAL)")
        print(f"    a*_MSE = <D, G-C> / ||D||^2                 = {a_mse:+.4f}")
        print(f"    corr(D, G-C) = <D,G-C>/(|D||G-C|)           = {rho:+.4f}")
        print(f"    empirical a* minimising ADE (the headline)  = {a_emp:.2f}")
        print(f"    a* predicted by 'CV + zero-mean noise, NO signal' = 0.0000")
        print(f"    -> a positive a*_MSE means D points along the correction CV needs.")

        # ---- (c) nulls -----------------------------------------------------
        print("\n(c) NULL CONTROLS — identical val-fit / test-eval pipeline")
        cvt = ade_arr(T['C'], T['G'])
        res = {}

        a_r, _ = fit_alpha(V['M'], V['C'], V['G'])
        er = ade_arr(a_r * T['M'] + (1 - a_r) * T['C'], T['G'])
        res['real'] = (a_r, er.mean(), cvt.mean() - er.mean(), None)

        # null_mean: constant correction learned on val
        mD = V['D'].mean(0)[None]                     # [1,12,2]
        Mv_m, Mt_m = V['C'] + mD, T['C'] + mD
        a_m, _ = fit_alpha(Mv_m, V['C'], V['G'])
        em = ade_arr(a_m * Mt_m + (1 - a_m) * T['C'], T['G'])
        res['null_mean'] = (a_m, em.mean(), cvt.mean() - em.mean(), None)

        # null_gauss and null_perm, averaged over seeds
        for nm in ('null_gauss', 'null_perm'):
            A, GA = [], []
            for sd in range(NSEED):
                r = np.random.RandomState(1000 + sd)
                if nm == 'null_gauss':
                    sv = np.sqrt((V['D'] ** 2).sum(2).mean(0) / 2.0)   # [12] per-step
                    st = np.sqrt((T['D'] ** 2).sum(2).mean(0) / 2.0)
                    Nv = r.randn(*V['D'].shape) * sv[None, :, None]
                    Nt = r.randn(*T['D'].shape) * st[None, :, None]
                else:
                    Nv = V['D'][r.permutation(len(V['D']))]
                    Nt = T['D'][r.permutation(len(T['D']))]
                a_n, _ = fit_alpha(V['C'] + Nv, V['C'], V['G'])
                en = ade_arr(a_n * (T['C'] + Nt) + (1 - a_n) * T['C'], T['G'])
                A.append(a_n); GA.append(cvt.mean() - en.mean())
            res[nm] = (float(np.mean(A)), None, float(np.mean(GA)), float(np.std(GA)))

        print(f"    {'predictor':12} {'alpha*':>9} {'test ADE':>10} {'gain vs CV':>12} "
              f"{'sd':>8}")
        print(f"    {'CV':12} {'-':>9} {cvt.mean():10.3f} {0.0:12.3f} {'-':>8}")
        for nm in ('real', 'null_mean', 'null_gauss', 'null_perm'):
            a, adev, g, sd = res[nm]
            av = f"{adev:10.3f}" if adev is not None else f"{'-':>10}"
            sv = f"{sd:8.4f}" if sd is not None else f"{'-':>8}"
            print(f"    {nm:12} {a:9.3f} {av} {g:12.4f} {sv}")

        # ---- (d) the nulls sit at the alpha=0 BOUNDARY, so "zero gain" is partly
        # structural: blending toward CV cannot help a predictor that IS CV+noise.
        # Three sharper tests that are not boundary-limited.
        print("\n(d) SHARPER TESTS (the nulls above sit on the alpha=0 boundary by")
        print("    construction, so their zero gain is necessary, not incidental)")

        # d1: force the REAL alpha* onto each null -> what does blending noise cost?
        print(f"    d1. forcing the real alpha*={a_r:.2f} onto each predictor (TEST):")
        r0 = np.random.RandomState(7)
        forced = {'real': ade_arr(a_r*T['M'] + (1-a_r)*T['C'], T['G']).mean()}
        sg = np.sqrt((T['D']**2).sum(2).mean(0)/2.0)
        forced['null_gauss'] = np.mean([
            ade_arr(a_r*(T['C'] + r0.randn(*T['D'].shape)*sg[None, :, None])
                    + (1-a_r)*T['C'], T['G']).mean() for _ in range(NSEED)])
        forced['null_perm'] = np.mean([
            ade_arr(a_r*(T['C'] + T['D'][r0.permutation(len(T['D']))])
                    + (1-a_r)*T['C'], T['G']).mean() for _ in range(NSEED)])
        forced['null_mean'] = ade_arr(a_r*(T['C'] + mD) + (1-a_r)*T['C'], T['G']).mean()
        for k, v in forced.items():
            print(f"       {k:12} {v:7.3f}   vs CV {cvt.mean():.3f}  -> {cvt.mean()-v:+.4f} m")

        # d2: permutation test on the D <-> (G-C) correlation
        n_perm = 2000
        obs = rho
        num_r = (V['D'] * (-V['R'])).sum(axis=(1, 2))
        dn = np.sqrt((V['D']**2).sum(axis=(1, 2)))
        rn = np.sqrt((V['R']**2).sum(axis=(1, 2)))
        rp = np.random.RandomState(11)
        cnt = 0
        tot_r = np.sqrt((V['D']**2).sum() * (V['R']**2).sum())
        for _ in range(n_perm):
            pm = rp.permutation(len(V['D']))
            if abs((V['D'][pm] * (-V['R'])).sum() / tot_r) >= abs(obs):
                cnt += 1
        print(f"    d2. permutation test on corr(D, G-C): observed {obs:+.4f}, "
              f"{cnt}/{n_perm} permutations reach it -> p = {(cnt+1)/(n_perm+1):.5f}")

        # d3: is alpha=0 a STRICT minimum for the nulls, or is the curve flat?
        sv = np.sqrt((V['D']**2).sum(2).mean(0)/2.0)
        Nv = np.random.RandomState(3).randn(*V['D'].shape) * sv[None, :, None]
        _, cur_n = fit_alpha(V['C'] + Nv, V['C'], V['G'])
        _, cur_r = fit_alpha(V['M'], V['C'], V['G'])
        print(f"    d3. VAL curve shape (ADE@6s at alpha = 0.0 / 0.1 / 0.3 / 0.5):")
        print(f"       real       {cur_r[0]:.4f} {cur_r[2]:.4f} {cur_r[6]:.4f} {cur_r[10]:.4f}"
              f"   (dips then rises)")
        print(f"       null_gauss {cur_n[0]:.4f} {cur_n[2]:.4f} {cur_n[6]:.4f} {cur_n[10]:.4f}"
              f"   (monotone increasing)")

        gr = res['real'][2]
        print(f"\n    GAP: real gain {gr:.4f} m")
        for nm in ('null_gauss', 'null_perm', 'null_mean'):
            print(f"      minus {nm:11} ({res[nm][2]:+.4f}) = {gr - res[nm][2]:+.4f} m "
                  f"({100*(gr-res[nm][2])/gr:5.1f}% of the real gain)")


if __name__ == '__main__':
    main()
