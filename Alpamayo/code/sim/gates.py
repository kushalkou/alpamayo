"""sim/gates.py -- G1..G4, pilot sweep, and calibration hook for the sim benchmark.

Evaluation reuses shrink_lib UNCHANGED (rollout, cv_traj, ade, paired, load, traj);
only its RES directory is pointed at the sim cell's dump dir.
Blend alpha is fit on VAL (grid 0..1 step .05) and evaluated on TEST.
Model is NON-AUTOREGRESSIVE (simplification vs nuScenes).

Usage: python gates.py g1 | g2 | g3 | pilot    (g4 is reported with g3)
"""
import sys, json, numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/sim')
import shrink_lib as SL
import model as M

ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)


def ades(D, meta, idx, G, m, mode):
    return np.array([SL.ade(SL.traj(D, meta, m, i, mode), G[i]) for i in idx])


def eval_cell(od):
    SL.RES = od
    Dv, mv, iv, Gv, CVv = SL.load('val')
    Dt, mt, it, Gt, CVt = SL.load('test')
    r = {'n_test': len(it)}
    r['cv'] = np.array([SL.ade(CVt[i], Gt[i]) for i in it])
    for m in ('mlp', 'meanctl'):
        for mode in ('expect', 'argmax'):
            r[f'{m}_{mode}'] = ades(Dt, mt, it, Gt, m, mode)
    # blend: fit alpha on val, apply on test (mlp / expectation)
    Tv = {i: SL.traj(Dv, mv, 'mlp', i, 'expect') for i in iv}
    Tt = {i: SL.traj(Dt, mt, 'mlp', i, 'expect') for i in it}
    curve = [np.mean([SL.ade(a * Tv[i] + (1 - a) * CVv[i], Gv[i]) for i in iv]) for a in ALPHAS]
    a_star = float(ALPHAS[int(np.argmin(curve))])
    r['alpha'] = a_star
    r['blend'] = np.array([SL.ade(a_star * Tt[i] + (1 - a_star) * CVt[i], Gt[i]) for i in it])
    r['alpha0_check'] = max(abs(SL.ade(0.0 * Tt[i] + 1.0 * CVt[i], Gt[i]) - SL.ade(CVt[i], Gt[i]))
                            for i in it)
    with open(f'{od}/ce.json') as f:
        r['ce'] = json.load(f)
    return r


def line(lab, a, b):
    d, lo, hi, p = SL.paired(a, b)
    return f'  {lab:34} {d:+8.3f}  [{lo:+7.3f},{hi:+7.3f}]  p={p:.4f}'


def summary(r):
    s = [f'  n_test={r["n_test"]}   ADE@6s mean (median)']
    for k in ('cv', 'meanctl_expect', 'mlp_expect', 'mlp_argmax', 'blend'):
        s.append(f'    {k:18} {r[k].mean():7.3f} ({np.median(r[k]):6.3f})')
    s.append(f'    alpha* (val) = {r["alpha"]:.2f}   alpha=0 identity max|dADE| = '
             f'{r["alpha0_check"]:.1e}')
    return '\n'.join(s)


def info_gain(r, split='test'):
    c = r['ce']['ce'][split]
    cm, cu = np.mean(c['model'][1:12]), np.mean(c['unigram'][1:12])
    return 1 - cm / cu, cm, cu


def g1():
    od = M.run(0.0, 0.0, seed=0, train_seed=0)
    r = eval_cell(od)
    print('=' * 90)
    print('G1  rho=0, p=0  (observation independent of the future; no turns)')
    print('    model = NON-AUTOREGRESSIVE MLP (simplification)')
    print('=' * 90)
    print(summary(r))
    print('  paired bootstrap on TEST (negative => first better), ADE@6s')
    print(line('mlp(expect) - CV', r['mlp_expect'], r['cv']))
    print(line('CV+meanctl - CV', r['meanctl_expect'], r['cv']))
    print(line('mlp(expect) - CV+meanctl', r['mlp_expect'], r['meanctl_expect']))
    print(line('blend - CV+meanctl', r['blend'], r['meanctl_expect']))
    ig, cm, cu = info_gain(r)
    print(f'  accel slots 1-11: CE_model {cm:.4f}  CE_unigram {cu:.4f}  info gain {ig:.4f}')
    return r


if __name__ == '__main__':
    {'g1': g1}[sys.argv[1]]()
