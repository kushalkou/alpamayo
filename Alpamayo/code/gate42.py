"""gate42.py — GATE 4.2 seed robustness + the SEED CONTROL for Gate 4.5's vision term.

The 4.5 result put term c = (full - ego) at +0.2546 with a CI excluding 0. But those
are two SEPARATELY TRAINED checkpoints, so that difference carries seed/run variation as
well as the vision modality. This script asks whether a same-modality, different-seed
contrast does just as well -- if it does, term c is ensemble diversity, not vision.

Design: fit the SAME 3-term blend, varying only the third basis vector, on VAL; evaluate
on TEST. Report the coefficient, its bootstrap CI, and the marginal ADE gain.

    traj = CV + a*(zeroboth - CV) + b*(ego_s42 - zeroboth) + c*(CONTRAST)

  MODALITY contrasts (vision vs no vision, necessarily across runs):
      full_s42 - ego_s42,  full_s123 - ego_s123,  full_s2024 - ego_s2024
  SEED contrasts (same modality, different seed -- the null for "vision"):
      full_s123 - full_s42, full_s2024 - full_s42,
      ego_s123  - ego_s42,  ego_s2024  - ego_s42

If mean(marginal gain | MODALITY) is not clearly above mean(marginal gain | SEED),
the vision claim collapses into ensemble diversity.
"""
import pickle
import numpy as np
from shrink_lib import RES, rollout, cv_traj, ade, paired, CURV_T

ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)
NBOOT = 2000


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


def fit3(Xv, Yv):
    n = len(Xv)
    Xf = Xv.reshape(n, -1, Xv.shape[-1]); Yf = Yv.reshape(n, -1)
    GG = np.einsum('nti,ntj->nij', Xf, Xf); H = np.einsum('nti,nt->ni', Xf, Yf)
    coef = np.linalg.solve(GG.sum(0), H.sum(0))
    rng = np.random.RandomState(0)
    bs = np.array([np.linalg.solve(GG[s].sum(0), H[s].sum(0))
                   for s in (rng.randint(0, n, n) for _ in range(NBOOT))])
    return coef, np.percentile(bs, [2.5, 97.5], axis=0)


def main():
    Tv, Cv, Gv, curv_v, iv = load_all('val')
    Tt, Ct, Gt, curv_t, it = load_all('test')
    cv_t = ade_arr(Ct, Gt)
    print("=" * 90)
    print(f"GATE 4.2 — SEED ROBUSTNESS   val n={len(iv)}  test n={len(it)}")
    print("  single-model shrinkage: alpha fit on VAL, evaluated on TEST")
    print("=" * 90)
    print(f"  {'model':14} {'alpha*':>7} {'test ADE':>9} {'gain vs CV':>11}")
    print(f"  {'CV':14} {'-':>7} {cv_t.mean():9.3f} {0.0:11.3f}")
    summ = {}
    for fam, seeds in (('full', ['full_s42', 'full_s123', 'full_s2024']),
                       ('ego', ['ego_s42', 'ego_s123', 'ego_s2024'])):
        rows = []
        for m in seeds:
            cur = np.array([ade_arr(a * Tv[m] + (1 - a) * Cv, Gv).mean() for a in ALPHAS])
            a_s = float(ALPHAS[int(cur.argmin())])
            e = ade_arr(a_s * Tt[m] + (1 - a_s) * Ct, Gt).mean()
            rows.append((a_s, e, cv_t.mean() - e))
            print(f"  {m:14} {a_s:7.2f} {e:9.3f} {cv_t.mean()-e:+11.4f}")
        A = np.array([r[0] for r in rows]); Gn = np.array([r[2] for r in rows])
        summ[fam] = (A, Gn)
        print(f"  {'-> '+fam+' mean':14} {A.mean():7.3f} {'':9} {Gn.mean():+11.4f}"
              f"   sd(alpha*)={A.std(ddof=1):.3f}  sd(gain)={Gn.std(ddof=1):.4f}")

    print("\n" + "=" * 90)
    print("SEED CONTROL for Gate 4.5 term c  —  is 'vision beyond ego' just ensemble diversity?")
    print("  traj = CV + a*(zeroboth-CV) + b*(ego_s42-zeroboth) + c*(CONTRAST)")
    print("=" * 90)
    base_v = [Tv['zeroboth_jul12'] - Cv, Tv['ego_s42'] - Tv['zeroboth_jul12']]
    base_t = [Tt['zeroboth_jul12'] - Ct, Tt['ego_s42'] - Tt['zeroboth_jul12']]
    contrasts = [
        ('MODALITY', 'full_s42  - ego_s42',   'full_s42', 'ego_s42'),
        ('MODALITY', 'full_s123 - ego_s123',  'full_s123', 'ego_s123'),
        ('MODALITY', 'full_s2024- ego_s2024', 'full_s2024', 'ego_s2024'),
        ('SEED',     'full_s123 - full_s42',  'full_s123', 'full_s42'),
        ('SEED',     'full_s2024- full_s42',  'full_s2024', 'full_s42'),
        ('SEED',     'ego_s123  - ego_s42',   'ego_s123', 'ego_s42'),
        ('SEED',     'ego_s2024 - ego_s42',   'ego_s2024', 'ego_s42'),
    ]
    for sub, lab_sub in ((np.ones(len(it), bool), 'ALL'), (curv_t > CURV_T, 'TURNING')):
        subv = np.ones(len(iv), bool) if lab_sub == 'ALL' else (curv_v > CURV_T)
        print(f"\n-- {lab_sub} (val n={subv.sum()}, test n={sub.sum()}) --")
        print(f"  {'kind':9} {'contrast':22} {'c':>8} {'95% CI':>19} "
              f"{'marg gain':>10} {'excl 0':>7}")
        res = {'MODALITY': [], 'SEED': []}
        for kind, name, p, q in contrasts:
            Xv = np.stack(base_v + [Tv[p] - Tv[q]], -1)[subv]
            Xt = np.stack(base_t + [Tt[p] - Tt[q]], -1)[sub]
            Yv = (Gv - Cv)[subv]
            coef, ci = fit3(Xv, Yv)
            P2 = Ct[sub] + coef[0]*Xt[..., 0] + coef[1]*Xt[..., 1]
            P3 = P2 + coef[2]*Xt[..., 2]
            marg = ade_arr(P2, Gt[sub]).mean() - ade_arr(P3, Gt[sub]).mean()
            ex = 'YES' if (ci[0, 2] > 0 or ci[1, 2] < 0) else 'no'
            res[kind].append(marg)
            print(f"  {kind:9} {name:22} {coef[2]:+8.4f} "
                  f"[{ci[0,2]:+7.4f},{ci[1,2]:+7.4f}] {marg:+10.4f} {ex:>7}")
        m_, s_ = np.array(res['MODALITY']), np.array(res['SEED'])
        print(f"  mean marginal gain:  MODALITY {m_.mean():+.4f} (sd {m_.std(ddof=1):.4f}, "
              f"n={len(m_)})   SEED {s_.mean():+.4f} (sd {s_.std(ddof=1):.4f}, n={len(s_)})")
        print(f"  difference MODALITY - SEED = {m_.mean()-s_.mean():+.4f} m")
        print(f"  VERDICT: " + ("vision adds beyond seed diversity"
                                if m_.mean() > s_.mean() + max(s_.std(ddof=1), 1e-9)
                                else "NOT separable from ensemble diversity"))


if __name__ == '__main__':
    main()
