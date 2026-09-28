"""leak/causal_eval.py -- OVERNIGHT2 item 6 analysis: causal-feature retrain vs CV.

Everything on the causal subset (>= 2 past poses), paired with CV and the old
(leaky-input) checkpoints on exactly the same samples.
  - decodes: argmax / V1s (STOP-aware expectation) / tau-hybrid (tau fit on val)
  - shrinkage: alpha fit on val subset, test eval, paired bootstrap vs CV
  - nulls (gate41 definitions, reused): null_mean, null_gauss, null_perm (20 seeds),
    forced real alpha*, permutation test on corr(D, G-C)
  - per-slot argmax token accuracy (leak fingerprint must be gone)
Usage: python leak/causal_eval.py causal_ego [causal_full]
"""
import sys, pickle, contextlib, io
import numpy as np
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE)
import shrink_lib as SL
from gate41 import rot, ade_arr, fit_alpha, NSEED
from tokenizer import TrajectoryTokenizer, STOP_TOKEN

RES = SL.RES
TAUS = (0.3, 0.5, 0.7, 0.9)


def load(split, tag):
    F = pickle.load(open(f'{RES}/dump_{split}.pkl', 'rb'))
    C = pickle.load(open(f'{RES}/dump_{split}_{tag}.pkl', 'rb'))
    idx = sorted(C['meta'])
    D = {'meta': F['meta'], 'data': dict(F['data'])}
    D['data'].update(C['data'])
    return D, idx


def arrs(D, idx, m, mode, tau=None):
    meta = D['meta']
    yaw = np.array([meta[i]['yaw0'] for i in idx])
    M = np.stack([SL.traj(D, meta, m, i, mode, tau) for i in idx])
    C = np.stack([SL.cv_traj(meta[i]['v0'], meta[i]['yaw0']) for i in idx])
    G = np.stack([np.array(meta[i]['gt']) for i in idx])
    return rot(M, yaw), rot(C, yaw), rot(G, yaw)


def ci(a, b):
    d, lo, hi, p = SL.paired(a, b)
    return f'{d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.4f}'


def main():
    names = sys.argv[1:] or ['causal_ego']
    with contextlib.redirect_stdout(io.StringIO()):
        tok = TrajectoryTokenizer()
    for cm in names:
        tag = cm
        Dv, iv = load('val', tag); Dt, it = load('test', tag)
        print('=' * 90)
        print(f'[{cm}] causal subset: val n={len(iv)}, test n={len(it)}. ADE@6s mean (median).')
        print('=' * 90)
        refs = [cm, 'y1_ego'] + (['y1_full'] if cm == 'causal_full' else [])
        _, Ct, Gt = arrs(Dt, it, cm, 'argmax')
        cv = ade_arr(Ct, Gt)
        print(f'  CV                        {cv.mean():.3f} ({np.median(cv):.3f})')
        E = {}
        for m in refs:
            for mode in ('argmax', 'expect'):
                Mt, _, _ = arrs(Dt, it, m, mode)
                E[(m, mode)] = ade_arr(Mt, Gt)
            # tau on val
            best = None
            for tau in TAUS:
                Mv, Cv, Gv = arrs(Dv, iv, m, 'hybrid', tau)
                s = ade_arr(Mv, Gv).mean()
                if best is None or s < best[1]: best = (tau, s)
            Mt, _, _ = arrs(Dt, it, m, 'hybrid', best[0])
            E[(m, 'hybrid')] = ade_arr(Mt, Gt)
            for mode in ('argmax', 'expect', 'hybrid'):
                e = E[(m, mode)]
                lab = mode if mode != 'hybrid' else f'hybrid tau={best[0]}'
                print(f'  {m:10} {lab:14} {e.mean():.3f} ({np.median(e):.3f})   vs CV {ci(e, cv)}')

        print('\n  SHRINKAGE (V1s): alpha fit on val subset, applied to test subset')
        BL = {}
        for m in refs:
            Mv, Cv, Gv = arrs(Dv, iv, m, 'expect'); Mt, _, _ = arrs(Dt, it, m, 'expect')
            a, cur = fit_alpha(Mv, Cv, Gv)
            bs = []
            rs = np.random.RandomState(0)
            per = np.stack([ade_arr(al * Mv + (1 - al) * Cv, Gv) for al in np.round(np.arange(0, 1.0001, .05), 2)])
            for _ in range(2000):
                bs.append(np.round(np.arange(0, 1.0001, .05), 2)[int(per[:, rs.randint(0, len(iv), len(iv))].mean(1).argmin())])
            bl = ade_arr(a * Mt + (1 - a) * Ct, Gt); BL[m] = bl
            print(f'  {m:10} alpha*={a:.2f} (boot CI [{np.percentile(bs,2.5):.2f},'
                  f'{np.percentile(bs,97.5):.2f}])  blend {bl.mean():.3f} ({np.median(bl):.3f})'
                  f'   blend-CV {ci(bl, cv)}   win {np.mean(bl < cv):.3f}')
        for m in refs[1:]:
            print(f'  blend[{cm}] - blend[{m}]  {ci(BL[cm], BL[m])}')
        # strata
        mt = Dt['meta']
        for sname, sel in (('STRAIGHT', [k for k, i in enumerate(it) if mt[i]['maxcurv'] <= SL.CURV_T]),
                           ('TURNING', [k for k, i in enumerate(it) if mt[i]['maxcurv'] > SL.CURV_T])):
            sel = np.array(sel)
            print(f'  {sname:9} n={len(sel):5d}  blend[{cm}]-CV {ci(BL[cm][sel], cv[sel])}')

        # nulls (gate41 definitions) for the causal model
        print('\n  NULLS (gate41 definitions), V1s, val-fit alpha / test eval')
        Mv, Cv, Gv = arrs(Dv, iv, cm, 'expect'); Mt, _, _ = arrs(Dt, it, cm, 'expect')
        V = dict(M=Mv, C=Cv, G=Gv, D=Mv - Cv, R=Cv - Gv); T = dict(M=Mt, C=Ct, G=Gt, D=Mt - Ct)
        a_r, _ = fit_alpha(V['M'], V['C'], V['G'])
        real = cv.mean() - ade_arr(a_r * T['M'] + (1 - a_r) * T['C'], T['G']).mean()
        mD = V['D'].mean(0)[None]
        a_m, _ = fit_alpha(V['C'] + mD, V['C'], V['G'])
        g_m = cv.mean() - ade_arr(a_m * (T['C'] + mD) + (1 - a_m) * T['C'], T['G']).mean()
        f_m = cv.mean() - ade_arr(a_r * (T['C'] + mD) + (1 - a_r) * T['C'], T['G']).mean()
        out = {'real': (a_r, real, real), 'null_mean': (a_m, g_m, f_m)}
        for nm in ('null_gauss', 'null_perm'):
            A, GA, FA = [], [], []
            for sd in range(NSEED):
                r = np.random.RandomState(1000 + sd)
                if nm == 'null_gauss':
                    sv = np.sqrt((V['D'] ** 2).sum(2).mean(0) / 2.0)
                    st = np.sqrt((T['D'] ** 2).sum(2).mean(0) / 2.0)
                    Nv = r.randn(*V['D'].shape) * sv[None, :, None]
                    Nt = r.randn(*T['D'].shape) * st[None, :, None]
                else:
                    Nv = V['D'][r.permutation(len(V['D']))]; Nt = T['D'][r.permutation(len(T['D']))]
                a_n, _ = fit_alpha(V['C'] + Nv, V['C'], V['G'])
                A.append(a_n)
                GA.append(cv.mean() - ade_arr(a_n * (T['C'] + Nt) + (1 - a_n) * T['C'], T['G']).mean())
                FA.append(cv.mean() - ade_arr(a_r * (T['C'] + Nt) + (1 - a_r) * T['C'], T['G']).mean())
            out[nm] = (np.mean(A), np.mean(GA), np.mean(FA))
        print(f'    {"predictor":11} {"alpha*":>7} {"gain vs CV":>11} {"gain at forced real a*":>23}')
        for nm, (a, g, f) in out.items():
            print(f'    {nm:11} {a:7.3f} {g:+11.4f} {f:+23.4f}')
        num = (V['D'] * (-V['R'])).sum(); tot = np.sqrt((V['D'] ** 2).sum() * (V['R'] ** 2).sum())
        obs = num / tot; rp = np.random.RandomState(11); cnt = 0
        for _ in range(2000):
            if abs((V['D'][rp.permutation(len(V['D']))] * (-V['R'])).sum() / tot) >= abs(obs): cnt += 1
        print(f'    corr(D, G-C) on val = {obs:+.4f}; permutation p = {(cnt+1)/2001:.5f}')

        # per-slot accuracy
        C = pickle.load(open(f'{RES}/dump_test_{tag}.pkl', 'rb'))['ce'][cm]

        def v2t(v, s):
            if v == 0.0: return STOP_TOKEN
            c = tok.accel_centers if s < 12 else tok.curv_centers
            return int(np.argmin(np.abs(c - v)))
        acc = {m: np.mean([[v2t(Dt['data'][m][i]['argmax'][s], s) == C[i]['gt'][s]
                            for s in range(24)] for i in it], 0) for m in refs}
        print('\n  per-slot argmax token accuracy (test subset): slot ' +
              '  '.join(f'{m:>10}' for m in refs))
        for s in range(6):
            print(f'    a{s:<2}  ' + '  '.join(f'{acc[m][s]:10.3f}' for m in refs))
        print(f'    k0   ' + '  '.join(f'{acc[m][12]:10.3f}' for m in refs))
        ce = np.mean([-np.array(C[i]['lp65'][1:12]) for i in it])
        print(f'  free-running CE accel slots 1-11 ({cm}): {ce:.4f}')


if __name__ == '__main__':
    main()
