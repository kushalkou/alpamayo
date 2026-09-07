"""gate47b.py — GATE 4.7 verdict, deconfounded.

The contrast families in 4.7b are confounded by SPAN STRUCTURE, not modality. Base is
span{zeroboth-CV, ego_s42-CV}. A contrast (P - Q) whose Q is ALREADY in that span adds
exactly one new model direction P; a contrast where neither endpoint is in the span adds
only the particular difference direction P-Q, which is generally not the useful one. That
is why every contrast containing ego_s42 gains and the rest do not -- regardless of family.

Clean formulation: add ONE model at a time as the third basis vector and measure the
marginal gain. Then "full model added" vs "ego model added" is an apples-to-apples test of
vision, with the span structure held fixed.

Robustness: repeat with the base built on ego_s123 and on ego_s2024, so the result does not
depend on which ego seed happens to sit in the base.
"""
import pickle
import numpy as np
from shrink_lib import RES, rollout, cv_traj, CURV_T

NBOOT = 2000
FULL = ['full_s42', 'full_s123', 'full_s2024']
EGO = ['ego_s42', 'ego_s123', 'ego_s2024']


def load_all(split):
    with open(f'{RES}/dump_{split}.pkl', 'rb') as f: A = pickle.load(f)
    with open(f'{RES}/dump_{split}_seeds.pkl', 'rb') as f: B = pickle.load(f)
    meta = A['meta']; idx = sorted(meta)
    data = dict(A['data']); data.update(B['data'])
    data['full_s42'] = data['y1_full']; data['ego_s42'] = data['y1_ego']
    T = {m: np.stack([rollout(np.array(data[m][i]['expect'])[:12],
                              np.array(data[m][i]['expect'])[12:],
                              meta[i]['v0'], meta[i]['yaw0']) for i in idx]) for m in data}
    C = np.stack([cv_traj(meta[i]['v0'], meta[i]['yaw0']) for i in idx])
    G = np.stack([np.array(meta[i]['gt']) for i in idx])
    curv = np.array([meta[i]['maxcurv'] for i in idx])
    return T, C, G, curv, idx


def ade_arr(P, G): return np.linalg.norm(P - G, axis=2).mean(1)


def fit(X, Y, boot=True, seed=0):
    n = len(X)
    Xf = X.reshape(n, -1, X.shape[-1]); Yf = Y.reshape(n, -1)
    GG = np.einsum('nti,ntj->nij', Xf, Xf); H = np.einsum('nti,nt->ni', Xf, Yf)
    coef = np.linalg.solve(GG.sum(0), H.sum(0))
    if not boot: return coef, None
    rng = np.random.RandomState(seed); bs = []
    for _ in range(NBOOT):
        s = rng.randint(0, n, n)
        try: bs.append(np.linalg.solve(GG[s].sum(0), H[s].sum(0)))
        except np.linalg.LinAlgError: pass
    return coef, np.array(bs)


def main():
    Tv, Cv, Gv, cvv, iv = load_all('val')
    Tt, Ct, Gt, cvt_, it = load_all('test')
    print("=" * 90)
    print("GATE 4.7 (deconfounded) — marginal gain of ADDING ONE MODEL to the base span")
    print("  base = CV + a*(zeroboth-CV) + b*(EGOBASE - zeroboth);  3rd vector = (M - EGOBASE)")
    print("  so the 3rd term contributes exactly one new model direction M, for every M.")
    print("=" * 90)

    for sub_lab in ('ALL', 'TURNING'):
        sv = np.ones(len(iv), bool) if sub_lab == 'ALL' else (cvv > CURV_T)
        st = np.ones(len(it), bool) if sub_lab == 'ALL' else (cvt_ > CURV_T)
        print(f"\n{'='*90}\n{sub_lab}  (val n={sv.sum()}, test n={st.sum()})\n{'='*90}")
        for base_ego in EGO:
            others = [m for m in FULL + EGO if m != base_ego]
            bv = [Tv['zeroboth_jul12'] - Cv, Tv[base_ego] - Tv['zeroboth_jul12']]
            bt = [Tt['zeroboth_jul12'] - Ct, Tt[base_ego] - Tt['zeroboth_jul12']]
            P2v = None
            print(f"\n  base ego = {base_ego}")
            print(f"    {'model added':14} {'kind':6} {'coef':>8} {'95% CI':>19} {'marg gain':>10}")
            res = {'FULL': [], 'EGO': []}
            for M in others:
                X3v = Tv[M] - Tv[base_ego]; X3t = Tt[M] - Tt[base_ego]
                Xv = np.stack(bv + [X3v], -1)[sv]; Xt = np.stack(bt + [X3t], -1)[st]
                coef, bs = fit(Xv, (Gv - Cv)[sv])
                ci = np.percentile(bs, [2.5, 97.5], axis=0)
                P2 = Ct[st] + coef[0]*Xt[..., 0] + coef[1]*Xt[..., 1]
                P3 = P2 + coef[2]*Xt[..., 2]
                marg = ade_arr(P2, Gt[st]).mean() - ade_arr(P3, Gt[st]).mean()
                kind = 'FULL' if M.startswith('full') else 'EGO'
                res[kind].append(marg)
                print(f"    {M:14} {kind:6} {coef[2]:+8.4f} "
                      f"[{ci[0,2]:+7.4f},{ci[1,2]:+7.4f}] {marg:+10.4f}")
            f_, e_ = np.array(res['FULL']), np.array(res['EGO'])
            print(f"    -> adding a FULL-VISION model: mean {f_.mean():+.4f} "
                  f"(n={len(f_)}, range [{f_.min():+.4f},{f_.max():+.4f}])")
            print(f"    -> adding an EGO-ONLY   model: mean {e_.mean():+.4f} "
                  f"(n={len(e_)}, range [{e_.min():+.4f},{e_.max():+.4f}])")
            print(f"    -> FULL - EGO = {f_.mean()-e_.mean():+.4f} m"
                  f"   {'(full higher)' if f_.mean() > e_.mean() else '(ego higher)'}")


if __name__ == '__main__':
    main()
