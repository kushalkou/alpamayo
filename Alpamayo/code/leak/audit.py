"""leak/audit.py -- OVERNIGHT2 item 1: nuScenes ego-state leak audit (CPU only).

Read-only w.r.t. frozen code/results: imports shrink_lib / dataset / tokenizer
unchanged, reads dump_{val,test}.pkl and the cached split, writes nothing but stdout.

1a  numeric check of the two identities the code implies:
      ego[3,3] (current accel)    == future_accelerations[1]          (GT accel slot 1)
      ego[3,2] / v0 (yaw rate/v)  == future_curvatures[0]             (GT curv slot 12)
1b  per-slot argmax token accuracy on test (y1_ego, y1_full, zeroboth)
1c  leaked-feature predictor (no learning) through the shrinkage pipeline, and the
    same predictor built from CAUSAL (past-pose) accel / yaw-rate
1d  paired re-evaluation on the causal subset (>= 2 past poses)
1e  CV seeded with backward speed vs forward v0
"""
import sys, math, pickle, contextlib, io
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
import shrink_lib as SL
from dataset import compute_ego_state, pose_to_xyyaw
from tokenizer import TrajectoryTokenizer, STOP_TOKEN

DT = 0.5
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)
MODELS = ('y1_ego', 'y1_full', 'zeroboth_jul12')
SP = pickle.load(open(f'{SL.RES}/leak_split.pkl', 'rb'))
with contextlib.redirect_stdout(io.StringIO()):
    TOK = TrajectoryTokenizer()


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def curv_from(yr, v0):
    return yr / v0 if v0 * DT > 0.01 else 0.0          # same ds>0.01 guard as extractor


def features(t):
    """leaked (current row of compute_ego_state) and causal (past-pose) quantities."""
    e = compute_ego_state(t).numpy()
    v0 = float(t['future_speeds'][0])
    P = [pose_to_xyyaw(p) for p in list(t.get('past_poses', [])) + [t['current_pose']]]
    f = {'v0': v0, 'yaw0': float(e[3, 1]), 'a_leak': float(e[3, 3]),
         'yr_leak': float(e[3, 2]), 'n_hist': len(t.get('past_poses', []))}
    if len(P) >= 2:
        s0 = math.hypot(P[-1][0] - P[-2][0], P[-1][1] - P[-2][1]) / DT
        f['v_back'] = s0
        f['yr_c'] = wrap(P[-1][2] - P[-2][2]) / DT
    if len(P) >= 3:
        s1 = math.hypot(P[-2][0] - P[-3][0], P[-2][1] - P[-3][1]) / DT
        f['a_c'] = (f['v_back'] - s1) / DT
    return f


def pred_traj(f, a, yr, hold):
    k = curv_from(yr, f['v0'])
    acc = np.zeros(12); cur = np.zeros(12)
    if hold:
        acc[1:] = a; cur[:] = k
    else:
        acc[1] = a; cur[0] = k
    return SL.rollout(acc, cur, f['v0'], f['yaw0'])


def blend_eval(Pv, Pt, CVv, CVt, Gv, Gt, iv, it):
    """alpha on val subset iv, test on it. returns dict of arrays + alpha."""
    curve = [np.mean([SL.ade(a * Pv[i] + (1 - a) * CVv[i], Gv[i]) for i in iv]) for a in ALPHAS]
    a = float(ALPHAS[int(np.argmin(curve))])
    cv = np.array([SL.ade(CVt[i], Gt[i]) for i in it])
    st = np.array([SL.ade(Pt[i], Gt[i]) for i in it])
    bl = np.array([SL.ade(a * Pt[i] + (1 - a) * CVt[i], Gt[i]) for i in it])
    return {'alpha': a, 'cv': cv, 'stand': st, 'blend': bl}


def row(name, r):
    d, lo, hi, p = SL.paired(r['blend'], r['cv'])
    return (f'  {name:30} a*={r["alpha"]:.2f}  CV {r["cv"].mean():.3f}  stand '
            f'{r["stand"].mean():.3f}  blend {r["blend"].mean():.3f}  '
            f'gain {-d:+.3f} [{-hi:+.3f},{-lo:+.3f}] p={p:.4f}')


def main():
    Dv, mv, iv, Gv, CVv = SL.load('val')
    Dt, mt, it, Gt, CVt = SL.load('test')
    F = {'val': {i: features(SP['val'][i]) for i in iv},
         'test': {i: features(SP['test'][i]) for i in it}}

    # ---- 1a numeric identities ----
    print('=' * 90); print('1a  identity check on TEST (n=%d)' % len(it)); print('=' * 90)
    da, dk, nh = [], [], []
    for i in it:
        t = SP['test'][i]; f = F['test'][i]
        da.append(abs(f['a_leak'] - float(t['future_accelerations'][1])))
        gk = float(t['future_curvatures'][0]); ck = curv_from(f['yr_leak'], f['v0'])
        dk.append(abs(ck - gk))
        nh.append(f['n_hist'])
    da, dk, nh = np.array(da), np.array(dk), np.array(nh)
    print(f'  |ego[3,3] - future_accelerations[1]|  max {da.max():.2e}  median {np.median(da):.2e}')
    print(f'  |ego[3,2]/v0 - future_curvatures[0]|  max {dk.max():.2e}  median {np.median(dk):.2e}'
          f'  (frac < 1e-6: {(dk < 1e-6).mean():.4f})')
    print('  n_hist (real past poses) distribution: ' +
          '  '.join(f'{k}:{(nh == k).sum()}' for k in range(5)))

    # ---- 1b per-slot token accuracy ----
    print('=' * 90); print('1b  per-slot argmax token accuracy, TEST (AR decode, from dump)'); print('=' * 90)
    gt_tok = {}
    for i in it:
        tk = TOK.tokenize(SP['test'][i])
        gt_tok[i] = np.array([a for a, _ in tk] + [k for _, k in tk])

    def val2tok(v, slot):
        if v == 0.0: return STOP_TOKEN
        c = TOK.accel_centers if slot < 12 else TOK.curv_centers
        return int(np.argmin(np.abs(c - v)))
    acc = {}
    for m in MODELS:
        hit = np.zeros(24)
        for i in it:
            av = Dt['data'][m][i]['argmax']
            hit += np.array([val2tok(av[s], s) == gt_tok[i][s] for s in range(24)])
        acc[m] = hit / len(it)
    print('  slot  ' + '  '.join(f'{m:>14}' for m in MODELS))
    for s in range(24):
        lab = f'a{s}' if s < 12 else f'k{s-12}'
        print(f'  {s:2d} {lab:>3} ' + '  '.join(f'{acc[m][s]:14.3f}' for m in MODELS))

    # ---- 1c / 1d predictors ----
    Pr = {}
    for split, idx in (('val', iv), ('test', it)):
        for hold in (False, True):
            tg = 'hold' if hold else 'once'
            Pr[(split, f'leak_{tg}')] = {i: pred_traj(F[split][i], F[split][i]['a_leak'],
                                         F[split][i]['yr_leak'], hold) for i in idx}
            Pr[(split, f'causal_{tg}')] = {i: pred_traj(F[split][i], F[split][i].get('a_c', 0.0),
                                           F[split][i].get('yr_c', 0.0), hold)
                                           for i in idx}
            Pr[(split, f'leak_acc_only_{tg}')] = {i: pred_traj(F[split][i], F[split][i]['a_leak'],
                                                  0.0, hold) for i in idx}
            Pr[(split, f'leak_yaw_only_{tg}')] = {i: pred_traj(F[split][i], 0.0,
                                                  F[split][i]['yr_leak'], hold) for i in idx}
        for m in MODELS:
            D, meta = (Dv, mv) if split == 'val' else (Dt, mt)
            Pr[(split, m)] = {i: SL.traj(D, meta, m, i, 'expect') for i in idx}

    names = ['leak_once', 'leak_hold', 'leak_acc_only_once', 'leak_acc_only_hold',
             'leak_yaw_only_once', 'leak_yaw_only_hold', 'causal_once', 'causal_hold'] + list(MODELS)

    print('=' * 90)
    print('1c  FULL test set: predictor blended with CV (alpha fit on full val). gain = CV - blend')
    print('=' * 90)
    for n in names:
        if n.startswith('causal'): continue
        print(row(n, blend_eval(Pr[('val', n)], Pr[('test', n)], CVv, CVt, Gv, Gt, iv, it)))

    sv = [i for i in iv if F['val'][i]['n_hist'] >= 2]
    st = [i for i in it if F['test'][i]['n_hist'] >= 2]
    print('=' * 90)
    print(f'1c/1d  CAUSAL SUBSET (>= 2 past poses): val n={len(sv)}/{len(iv)}, '
          f'test n={len(st)}/{len(it)}. alpha refit on val subset. Paired vs CV.')
    print('=' * 90)
    R = {}
    for n in names:
        R[n] = blend_eval(Pr[('val', n)], Pr[('test', n)], CVv, CVt, Gv, Gt, sv, st)
        print(row(n, R[n]))
    print('  paired differences on the subset (negative => first better):')
    for a, b in (('leak_hold', 'causal_hold'), ('leak_once', 'causal_once'),
                 ('y1_ego', 'leak_hold'), ('y1_ego', 'causal_hold')):
        d, lo, hi, p = SL.paired(R[a]['blend'], R[b]['blend'])
        print(f'    blend[{a}] - blend[{b}]  {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.4f}')

    # ---- 1e ----
    s1 = [i for i in it if F['test'][i]['n_hist'] >= 1]
    cvf = np.array([SL.ade(CVt[i], Gt[i]) for i in s1])
    cvb = np.array([SL.ade(SL.cv_traj(F['test'][i]['v_back'], F['test'][i]['yaw0']), Gt[i])
                    for i in s1])
    d, lo, hi, p = SL.paired(cvb, cvf)
    print('=' * 90)
    print(f'1e  CV seeded with backward speed |p0-p-1|/dt vs forward v0, test n={len(s1)}: '
          f'{cvb.mean():.3f} vs {cvf.mean():.3f}  diff {d:+.3f} [{lo:+.3f},{hi:+.3f}]')


if __name__ == '__main__':
    main()
