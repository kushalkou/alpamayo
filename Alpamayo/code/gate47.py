"""gate47.py — GATE 4.7 vision verdict (+ 4.2 seed robustness). FINAL ANALYSIS.

4.7a norm matching. Blend gain scales with a basis vector's magnitude as well as its
     direction, so contrasts of different norm are not directly comparable BY COEFFICIENT.
     Note: least squares is scale-equivariant -- scaling column j by 1/s scales beta_j by
     s and leaves fitted values, and therefore the marginal GAIN, unchanged. So the gain
     comparison is already norm-invariant; the COEFFICIENT comparison is not. Both are
     reported, and the invariance is verified numerically rather than asserted.

4.7b contrast families, each as the third basis vector in the identical 3-term fit:
       same-seed modality   full_sX - ego_sX          (3)
       cross-seed modality  full_sX - ego_sY, X != Y  (6)
       same-modality seed   full_sX - full_sY, ego_sX - ego_sY  (6)

     VERDICT RULE (fixed before seeing the numbers):
       modality (both kinds) > seed distribution, CIs separated -> term c is VISION
       modality inside the seed range                           -> ENSEMBLE DIVERSITY
       same-seed works, cross-seed does not                     -> run-specific, INCONCLUSIVE

4.7c generalization of the 3-term blend: val vs test gain for 1-term and 3-term, and
     5-fold CV within val for coefficient stability.

4.2  seed robustness of the 1-term global blend: alpha* and test gain per seed.
"""
import pickle, itertools
import numpy as np
from shrink_lib import RES, rollout, cv_traj, CURV_T

ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)
NBOOT = 2000
FULL = ['full_s42', 'full_s123', 'full_s2024']
EGO = ['ego_s42', 'ego_s123', 'ego_s2024']


def load_all(split):
    with open(f'{RES}/dump_{split}.pkl', 'rb') as f: A = pickle.load(f)
    with open(f'{RES}/dump_{split}_seeds.pkl', 'rb') as f: B = pickle.load(f)
    meta = A['meta']; idx = sorted(meta)
    data = dict(A['data']); data.update(B['data'])
    data['full_s42'] = data['y1_full']; data['ego_s42'] = data['y1_ego']
    T = {}
    for m in data:
        T[m] = np.stack([rollout(np.array(data[m][i]['expect'])[:12],
                                 np.array(data[m][i]['expect'])[12:],
                                 meta[i]['v0'], meta[i]['yaw0']) for i in idx])
    C = np.stack([cv_traj(meta[i]['v0'], meta[i]['yaw0']) for i in idx])
    G = np.stack([np.array(meta[i]['gt']) for i in idx])
    curv = np.array([meta[i]['maxcurv'] for i in idx])
    return T, C, G, curv, idx


def ade_arr(P, G): return np.linalg.norm(P - G, axis=2).mean(1)
def mean_norm(X): return float(np.sqrt((X ** 2).sum(axis=(1, 2))).mean())


def gram(X, Y):
    n = len(X)
    Xf = X.reshape(n, -1, X.shape[-1]); Yf = Y.reshape(n, -1)
    return (np.einsum('nti,ntj->nij', Xf, Xf), np.einsum('nti,nt->ni', Xf, Yf))


def fit(X, Y, boot=True, seed=0):
    GG, H = gram(X, Y)
    coef = np.linalg.solve(GG.sum(0), H.sum(0))
    if not boot: return coef, None
    n = len(X); rng = np.random.RandomState(seed)
    bs = []
    for _ in range(NBOOT):
        s = rng.randint(0, n, n)
        try: bs.append(np.linalg.solve(GG[s].sum(0), H[s].sum(0)))
        except np.linalg.LinAlgError: pass
    return coef, np.percentile(np.array(bs), [2.5, 97.5], axis=0)


def main():
    Tv, Cv, Gv, cvv, iv = load_all('val')
    Tt, Ct, Gt, cvt_, it = load_all('test')
    cvA = ade_arr(Ct, Gt)
    print("=" * 90)
    print(f"GATE 4.7 — VISION VERDICT.  val n={len(iv)}  test n={len(it)}")
    print("=" * 90)

    # ---------------- 4.2 ----------------
    print("\n" + "=" * 90)
    print("GATE 4.2 — SEED ROBUSTNESS of the 1-term global blend (alpha fit on VAL)")
    print("=" * 90)
    print(f"  {'model':14} {'alpha*':>7} {'test ADE':>9} {'gain vs CV':>11}")
    print(f"  {'CV':14} {'-':>7} {cvA.mean():9.3f} {0.0:+11.4f}")
    for fam, seeds in (('full', FULL), ('ego', EGO)):
        A, Gn = [], []
        for m in seeds:
            cur = np.array([ade_arr(a * Tv[m] + (1 - a) * Cv, Gv).mean() for a in ALPHAS])
            a_s = float(ALPHAS[int(cur.argmin())])
            e = ade_arr(a_s * Tt[m] + (1 - a_s) * Ct, Gt).mean()
            A.append(a_s); Gn.append(cvA.mean() - e)
            print(f"  {m:14} {a_s:7.2f} {e:9.3f} {cvA.mean()-e:+11.4f}")
        A, Gn = np.array(A), np.array(Gn)
        print(f"  {'-> '+fam+' mean':14} {A.mean():7.3f} {'':9} {Gn.mean():+11.4f}"
              f"   sd(alpha*)={A.std(ddof=1):.3f}  sd(gain)={Gn.std(ddof=1):.4f}"
              f"  range [{Gn.min():+.4f},{Gn.max():+.4f}]")

    # ---------------- contrasts ----------------
    same_mod = [(f, e) for f, e in zip(FULL, EGO)]
    cross_mod = [(f, e) for f in FULL for e in EGO if f[5:] != e[4:]]
    seed_con = ([(a, b) for a, b in itertools.combinations(FULL, 2)] +
                [(a, b) for a, b in itertools.combinations(EGO, 2)])
    fams = [('same-seed modality', same_mod), ('cross-seed modality', cross_mod),
            ('same-modality seed', seed_con)]

    base_v = [Tv['zeroboth_jul12'] - Cv, Tv['ego_s42'] - Tv['zeroboth_jul12']]
    base_t = [Tt['zeroboth_jul12'] - Ct, Tt['ego_s42'] - Tt['zeroboth_jul12']]

    print("\n" + "=" * 90)
    print("4.7a — BASIS-VECTOR NORMS (mean per-sample Frobenius norm, VAL)")
    print("=" * 90)
    print(f"  {'base 1  zeroboth - CV':34} {mean_norm(base_v[0]):8.4f}")
    print(f"  {'base 2  ego_s42 - zeroboth':34} {mean_norm(base_v[1]):8.4f}")
    for fname, pairs in fams:
        for p, q in pairs:
            print(f"  {fname+'  '+p+' - '+q:34} {mean_norm(Tv[p]-Tv[q]):8.4f}")

    for sub_lab in ('ALL', 'TURNING'):
        sv = np.ones(len(iv), bool) if sub_lab == 'ALL' else (cvv > CURV_T)
        st = np.ones(len(it), bool) if sub_lab == 'ALL' else (cvt_ > CURV_T)
        print("\n" + "=" * 90)
        print(f"4.7b — CONTRAST FAMILIES on {sub_lab} (val n={sv.sum()}, test n={st.sum()})")
        print("  third basis vector varied; identical 3-term fit; marginal gain is")
        print("  scale-invariant, coefficients are shown in NORMALISED units")
        print("=" * 90)
        print(f"  {'family':20} {'contrast':26} {'c_norm':>8} {'95% CI (norm)':>19} "
              f"{'marg gain':>10}")
        res = {}
        for fname, pairs in fams:
            res[fname] = []
            for p, q in pairs:
                Xv3 = Tv[p] - Tv[q]; Xt3 = Tt[p] - Tt[q]
                s = mean_norm(Xv3[sv])
                Xv = np.stack(base_v + [Xv3 / s], -1)[sv]
                Xt = np.stack(base_t + [Xt3 / s], -1)[st]
                Yv = (Gv - Cv)[sv]
                coef, ci = fit(Xv, Yv)
                P2 = Ct[st] + coef[0]*Xt[..., 0] + coef[1]*Xt[..., 1]
                P3 = P2 + coef[2]*Xt[..., 2]
                marg = ade_arr(P2, Gt[st]).mean() - ade_arr(P3, Gt[st]).mean()
                res[fname].append(marg)
                print(f"  {fname:20} {p+' - '+q:26} {coef[2]:+8.4f} "
                      f"[{ci[0,2]:+7.4f},{ci[1,2]:+7.4f}] {marg:+10.4f}")
        sd_ = np.array(res['same-modality seed'])
        print(f"\n  SEED-contrast gain distribution: mean {sd_.mean():+.4f}  "
              f"sd {sd_.std(ddof=1):.4f}  min {sd_.min():+.4f}  max {sd_.max():+.4f}  n={len(sd_)}")
        for fname in ('same-seed modality', 'cross-seed modality'):
            m_ = np.array(res[fname])
            z = (m_.mean() - sd_.mean()) / max(sd_.std(ddof=1), 1e-12)
            inside = (m_.min() >= sd_.min()) and (m_.max() <= sd_.max())
            print(f"  {fname:20} mean {m_.mean():+.4f} sd {m_.std(ddof=1):.4f} "
                  f"range [{m_.min():+.4f},{m_.max():+.4f}]  -> {z:+.2f} SD above seed mean"
                  f"   {'INSIDE seed range' if inside else 'OUTSIDE seed range'}")
        if sub_lab == 'ALL':
            sm = np.array(res['same-seed modality']); cm = np.array(res['cross-seed modality'])
            sep = (min(sm.min(), cm.min()) > sd_.max())
            if sep:
                v = 'VISION -- both modality families clear the entire seed distribution'
            elif sm.mean() > sd_.max() and cm.mean() <= sd_.max():
                v = 'INCONCLUSIVE -- run-specific: same-seed works, cross-seed does not'
            elif max(sm.mean(), cm.mean()) <= sd_.max():
                v = 'ENSEMBLE DIVERSITY -- modality gain sits inside the seed range'
            else:
                v = 'PARTIAL -- modality mean exceeds seed mean but ranges overlap'
            print(f"\n  >>> VERDICT ({sub_lab}): {v}")

    # ---------------- 4.7a invariance check ----------------
    Xr = np.stack(base_v + [Tv['full_s42'] - Tv['ego_s42']], -1)
    s = mean_norm(Tv['full_s42'] - Tv['ego_s42'])
    Xn = np.stack(base_v + [(Tv['full_s42'] - Tv['ego_s42']) / s], -1)
    Y = Gv - Cv
    cr, _ = fit(Xr, Y, boot=False); cn, _ = fit(Xn, Y, boot=False)
    pr = Cv + sum(cr[k]*Xr[..., k] for k in range(3))
    pn = Cv + sum(cn[k]*Xn[..., k] for k in range(3))
    print("\n" + "=" * 90)
    print("4.7a — SCALE-INVARIANCE CHECK (raw vs normalised fit, same contrast)")
    print("=" * 90)
    print(f"  raw coefficients        {np.array2string(cr, precision=4)}")
    print(f"  normalised coefficients {np.array2string(cn, precision=4)}  "
          f"(c_norm = c_raw * {s:.4f} = {cr[2]*s:.4f})")
    print(f"  max |fitted_raw - fitted_normalised| = {np.abs(pr-pn).max():.3e}"
          f"   -> marginal gains are identical either way")

    # ---------------- 4.7c ----------------
    print("\n" + "=" * 90)
    print("4.7c — GENERALIZATION: 1-term vs 3-term blend, and 5-fold CV within VAL")
    print("=" * 90)
    m = 'full_s42'
    cur = np.array([ade_arr(a * Tv[m] + (1 - a) * Cv, Gv).mean() for a in ALPHAS])
    a_s = float(ALPHAS[int(cur.argmin())])
    v1 = ade_arr(Cv, Gv).mean() - ade_arr(a_s*Tv[m] + (1-a_s)*Cv, Gv).mean()
    t1 = cvA.mean() - ade_arr(a_s*Tt[m] + (1-a_s)*Ct, Gt).mean()
    X3v = np.stack(base_v + [Tv['full_s42'] - Tv['ego_s42']], -1)
    X3t = np.stack(base_t + [Tt['full_s42'] - Tt['ego_s42']], -1)
    c3, _ = fit(X3v, Gv - Cv, boot=False)
    v3 = ade_arr(Cv, Gv).mean() - ade_arr(Cv + sum(c3[k]*X3v[..., k] for k in range(3)), Gv).mean()
    t3 = cvA.mean() - ade_arr(Ct + sum(c3[k]*X3t[..., k] for k in range(3)), Gt).mean()
    print(f"  {'blend':12} {'params':>7} {'VAL gain':>10} {'TEST gain':>10} {'degradation':>12}")
    print(f"  {'1-term':12} {1:7d} {v1:+10.4f} {t1:+10.4f} {v1-t1:+12.4f}")
    print(f"  {'3-term':12} {3:7d} {v3:+10.4f} {t3:+10.4f} {v3-t3:+12.4f}")
    rng = np.random.RandomState(0); perm = rng.permutation(len(iv))
    folds = np.array_split(perm, 5)
    CO, HG = [], []
    for k in range(5):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
        ck, _ = fit(X3v[tr], (Gv - Cv)[tr], boot=False)
        CO.append(ck)
        HG.append(ade_arr(Cv[te], Gv[te]).mean() -
                  ade_arr(Cv[te] + sum(ck[j]*X3v[te][..., j] for j in range(3)), Gv[te]).mean())
    CO = np.array(CO); HG = np.array(HG)
    print(f"\n  5-fold CV within VAL — coefficient stability:")
    names = ['a const-accel', 'b ego', 'c vision']
    for j in range(3):
        print(f"    {names[j]:16} mean {CO[:,j].mean():+.4f}  sd {CO[:,j].std(ddof=1):.4f}"
              f"  range [{CO[:,j].min():+.4f},{CO[:,j].max():+.4f}]"
              f"  cv%={100*CO[:,j].std(ddof=1)/abs(CO[:,j].mean()):.1f}")
    print(f"    held-out gain     mean {HG.mean():+.4f}  sd {HG.std(ddof=1):.4f}")


if __name__ == '__main__':
    main()
