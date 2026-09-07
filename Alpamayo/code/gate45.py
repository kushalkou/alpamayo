"""gate45.py — GATE 4.5 nested blend + 4.6 effect-size sanity.

4.5  traj = CV + a*(zeroboth - CV) + b*(y1_ego - zeroboth) + c*(y1_full - y1_ego)

     a = value of a constant-acceleration correction scaled by ego state
         (zeroboth emits ONE constant control sequence, so zeroboth-CV is exactly
          "a fixed accel profile, rolled out from this sample's speed and heading")
     b = value of ACTUAL EGO INPUT beyond that constant
     c = value of VISION beyond ego

     Fit jointly by least squares in position space on VAL, evaluate on TEST.
     "Signal" throughout means correlated with (GT - CV). It does NOT mean perception.

4.6  optimal-shrinkage identity MSE_blend ~= MSE_CV*(1-r^2), and the blend gain
     broken out by horizon (1s/2s/3s/6s).
"""
import numpy as np
from shrink_lib import load, traj, ade, CURV_T, N

ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)
NBOOT = 2000
TERMS = ['a  const-accel (zeroboth-CV)', 'b  ego beyond const (ego-zeroboth)',
         'c  vision beyond ego (full-ego)']


def build(split):
    D, meta, idx, G, CV = load(split)
    Z = np.stack([traj(D, meta, 'zeroboth_jul12', i, 'expect') for i in idx])
    E = np.stack([traj(D, meta, 'y1_ego', i, 'expect') for i in idx])
    F = np.stack([traj(D, meta, 'y1_full', i, 'expect') for i in idx])
    C = np.stack([CV[i] for i in idx])
    GG = np.stack([G[i] for i in idx])
    curv = np.array([meta[i]['maxcurv'] for i in idx])
    X = np.stack([Z - C, E - Z, F - E], -1)      # [n,12,2,3]
    Y = GG - C                                   # [n,12,2]
    return dict(X=X, Y=Y, C=C, G=GG, curv=curv, Z=Z, E=E, F=F)


def gram(d, sel=None):
    X = d['X'] if sel is None else d['X'][sel]
    Y = d['Y'] if sel is None else d['Y'][sel]
    n = len(X)
    Xf = X.reshape(n, -1, 3)          # [n,24,3]
    Yf = Y.reshape(n, -1)             # [n,24]
    GG = np.einsum('nti,ntj->nij', Xf, Xf)     # [n,3,3]
    H = np.einsum('nti,nt->ni', Xf, Yf)        # [n,3]
    return GG, H


def solve(GG, H, idxs=None):
    A = GG.sum(0) if idxs is None else GG[idxs].sum(0)
    b = H.sum(0) if idxs is None else H[idxs].sum(0)
    return np.linalg.solve(A, b)


def ade_of(P, G):
    return np.linalg.norm(P - G, axis=2).mean(1)


def report(vd, td, label):
    GG, H = gram(vd)
    coef = solve(GG, H)
    rng = np.random.RandomState(0)
    n = len(GG)
    bs = np.array([solve(GG, H, rng.randint(0, n, n)) for _ in range(NBOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)

    print(f"\n--- {label} (fit VAL n={len(vd['X'])}, eval TEST n={len(td['X'])}) ---")
    print(f"    {'term':36} {'coef':>8} {'95% CI':>20} {'excludes 0':>11}")
    for k in range(3):
        ex = 'YES' if (lo[k] > 0 or hi[k] < 0) else 'no'
        print(f"    {TERMS[k]:36} {coef[k]:+8.4f}  [{lo[k]:+7.4f},{hi[k]:+7.4f}] {ex:>11}")

    cv = ade_of(td['C'], td['G'])
    print(f"\n    marginal TEST ADE@6s as terms are added (CV = {cv.mean():.4f}):")
    prev = cv.mean()
    P = td['C'].copy()
    names = ['+ a (const-accel)', '+ b (ego)', '+ c (vision)']
    for k in range(3):
        P = P + coef[k] * td['X'][..., k]
        e = ade_of(P, td['G']).mean()
        print(f"      {names[k]:22} {e:8.4f}   marginal {prev - e:+.4f} m   "
              f"cumulative {cv.mean() - e:+.4f} m")
        prev = e
    return coef, lo, hi


def main():
    V, T = build('val'), build('test')
    print("=" * 88)
    print("GATE 4.5 — NESTED BLEND: where does the correction come from?")
    print("  traj = CV + a*(zeroboth-CV) + b*(ego-zeroboth) + c*(full-ego)")
    print("  'signal' = correlated with (GT - CV). NOT perception.")
    print("=" * 88)
    report(V, T, 'ALL')
    vt = V['curv'] > CURV_T
    tt = T['curv'] > CURV_T
    Vt = {k: (v[vt] if isinstance(v, np.ndarray) and len(v) == len(V['curv']) else v)
          for k, v in V.items()}
    Tt = {k: (v[tt] if isinstance(v, np.ndarray) and len(v) == len(T['curv']) else v)
          for k, v in T.items()}
    report(Vt, Tt, 'TURNING only')

    # ---------------- 4.6 ----------------
    print("\n" + "=" * 88)
    print("GATE 4.6 — EFFECT-SIZE SANITY")
    print("=" * 88)
    for mdl, key in (('y1_full', 'F'), ('y1_ego', 'E')):
        M = V[key]; C = V['C']; G = V['G']
        Dv = M - C; Rv = C - G
        r = (Dv * (-Rv)).sum() / np.sqrt((Dv ** 2).sum() * (Rv ** 2).sum())
        a_mse = (Dv * (-Rv)).sum() / (Dv ** 2).sum()
        cur = np.array([ade_of(a * M + (1 - a) * C, G).mean() for a in ALPHAS])
        a_ade = float(ALPHAS[int(cur.argmin())])
        Mt, Ct, Gt = T[key], T['C'], T['G']
        mse_cv = ((Ct - Gt) ** 2).sum(2).mean()
        mse_a = (((a_ade * Mt + (1 - a_ade) * Ct) - Gt) ** 2).sum(2).mean()
        mse_m = (((a_mse * Mt + (1 - a_mse) * Ct) - Gt) ** 2).sum(2).mean()
        pred = mse_cv * (1 - r ** 2)
        print(f"\n[{mdl}]  r = corr(D, G-C) = {r:+.4f}  (fit on VAL)")
        print(f"    predicted MSE_blend = MSE_CV*(1-r^2) = {mse_cv:.3f}*{1-r**2:.4f} "
              f"= {pred:.3f}")
        print(f"    observed  MSE_blend at ADE-optimal alpha={a_ade:.2f}   = {mse_a:.3f}")
        print(f"    observed  MSE_blend at MSE-optimal alpha={a_mse:.4f} = {mse_m:.3f}")
        print(f"    MSE_CV = {mse_cv:.3f}   reduction at MSE-opt = "
              f"{100*(mse_cv-mse_m)/mse_cv:.2f}%  (predicted {100*r**2:.2f}%)")
        print(f"    blend gain vs CV by horizon (TEST, ADE mean, m):")
        print(f"      {'horizon':10} {'CV':>8} {'blend':>8} {'gain':>8} {'gain %':>8}")
        for st, lab in ((2, '1s'), (4, '2s'), (6, '3s'), (12, '6s')):
            c = np.linalg.norm(Ct[:, :st] - Gt[:, :st], axis=2).mean(1).mean()
            b = np.linalg.norm((a_ade * Mt + (1 - a_ade) * Ct)[:, :st] - Gt[:, :st],
                               axis=2).mean(1).mean()
            print(f"      {lab:10} {c:8.4f} {b:8.4f} {c-b:+8.4f} {100*(c-b)/c:7.2f}%")


if __name__ == '__main__':
    main()
