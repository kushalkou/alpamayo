"""leak/locate.py -- OVERNIGHT2 item 5: locate nuScenes on the sim's (p, rho) axes.
Compute and report only.

p  : sim test fraction with max|GT curv| > 0.05 at each p (3 seeds) vs nuScenes 17.7%
     (638/3614, same definition, same threshold).
L1 : perception gap on accel slots 1-11, (CE_no_obs - CE_obs) / CE_no_obs.
     sim: MLP with observation zeroed vs present (seed 0, every map cell).
     nuScenes: y1_ego (no vision) vs y1_full, FREE-RUNNING CE (argmax fed back),
     65-way STOP-aware support, from results/dump_test_ce.pkl.
L2 : 1 - CE_model / CE_unigram_given_v0bin, accel slots 1-11, same fixed v0 bins
     (sim/model.py V0_EDGES), add-one smoothing, unigram fit on each TRAIN split.
Also given on slots 2-11 because nuScenes slot 1 is the leaked slot (item 1).
"""
import sys, json, pickle, contextlib, io
import numpy as np
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE)
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
PS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75)
RHOS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
V0_EDGES = np.array([0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14])   # == sim/model.py
NUS_TURN = 638 / 3614


def interp_x(xs, ys, y0):
    """x where piecewise-linear y(x) first crosses y0 (ys assumed ~monotone)."""
    for k in range(len(xs) - 1):
        a, b = ys[k], ys[k + 1]
        if (a - y0) * (b - y0) <= 0 and a != b:
            return xs[k] + (y0 - a) / (b - a) * (xs[k + 1] - xs[k])
    return float('nan')


def sim_side():
    mp = json.load(open(f'{RES}/sim/map_rows.json'))
    loc = json.load(open(f'{RES}/sim/locate_rows.json'))
    tf = {p: np.mean([r['info']['turn_frac_test'] for r in mp if r['p'] == p]) for p in PS}
    p_hat = interp_x(PS, [tf[p] for p in PS], NUS_TURN)
    print('p LOCATOR: sim turn fraction (max|curv|>0.05) by p, 3-seed mean')
    print('  ' + '  '.join(f'p={p:g}:{tf[p]:.3f}' for p in PS))
    print(f'  nuScenes {NUS_TURN:.3f} -> p_hat = {p_hat:.3f} (linear interpolation)')

    def ce(r, sl):
        with open(f'{RES}/sim/p{r["p"]:.2f}_rho{r["rho"]:.2f}_s0_t0'
                  + ('_noobs' if r['zero_obs'] else '') + '/ce.json') as f:
            c = json.load(f)['ce']['test']
        return np.mean(c['model'][sl]), np.mean(c['unigram_v0'][sl])

    L1, L2 = {}, {}
    for sl, lab in ((slice(1, 12), '1-11'), (slice(2, 12), '2-11')):
        for p in PS:
            for rho in RHOS:
                ob = [r for r in loc if r['p'] == p and r['rho'] == rho and not r['zero_obs']][0]
                no = [r for r in loc if r['p'] == p and r['rho'] == rho and r['zero_obs']][0]
                cm, cuv = ce(ob, sl); cn, _ = ce(no, sl)
                L1[(lab, p, rho)] = (cn - cm) / cn
                L2[(lab, p, rho)] = 1 - cm / cuv
    for name, T in (('L1 perception gap (CE_noobs-CE_obs)/CE_noobs', L1),
                    ('L2 v0-conditioned floor 1 - CE_obs/CE_unigram|v0', L2)):
        for lab in ('1-11', '2-11'):
            print(f'\nSIM {name}, accel slots {lab}, seed 0')
            print('  p \\ rho ' + ''.join(f'{r:>8g}' for r in RHOS))
            for p in PS:
                print(f'  {p:7.2f} ' + ''.join(f'{T[(lab, p, r)]:8.4f}' for r in RHOS))
    return p_hat, L1, L2


def nus_side():
    from tokenizer import TrajectoryTokenizer
    with contextlib.redirect_stdout(io.StringIO()):
        tok = TrajectoryTokenizer()
    D = pickle.load(open(f'{RES}/dump_test_ce.pkl', 'rb'))
    C = D['ce']; idx = sorted(C['y1_ego'])
    # train unigram given v0 bin (65-way, accel half)
    import inference as INF
    from dataset import build_scene_split
    with contextlib.redirect_stdout(io.StringIO()):
        allt = pickle.load(open(INF.TRAJECTORIES_PATH, 'rb'))
        tr, _, _ = build_scene_split(allt, INF.NUSCENES_ROOT)
    cnt = np.ones((len(V0_EDGES) + 1, 65))
    for t in tr:
        b = np.searchsorted(V0_EDGES, float(t['future_speeds'][0]), side='right')
        for a, _ in tok.tokenize(t):
            cnt[b, 64 if a == 128 else a] += 1
    uv = cnt / cnt.sum(1, keepdims=True)
    te = pickle.load(open(f'{RES}/leak_split.pkl', 'rb'))['test']
    out = {}
    for sl, lab in ((slice(1, 12), '1-11'), (slice(2, 12), '2-11')):
        ce_m = {m: np.mean([-np.array(C[m][i]['lp65'][sl]) for i in idx]) for m in C}
        cu = []
        for i in idx:
            b = np.searchsorted(V0_EDGES, float(te[i]['future_speeds'][0]), side='right')
            g = C['y1_ego'][i]['gt'][sl]
            cu.append(np.mean([-np.log(uv[b, 64 if x == 128 else x]) for x in g]))
        cu = float(np.mean(cu))
        L1 = (ce_m['y1_ego'] - ce_m['y1_full']) / ce_m['y1_ego']
        out[lab] = dict(ce_ego=ce_m['y1_ego'], ce_full=ce_m['y1_full'], ce_uv=cu, L1=L1,
                        L2_full=1 - ce_m['y1_full'] / cu, L2_ego=1 - ce_m['y1_ego'] / cu)
        o = out[lab]
        print(f'\nnuScenes accel slots {lab} (test n={len(idx)}, free-running, 65-way):')
        print(f'  CE y1_ego {o["ce_ego"]:.4f}  CE y1_full {o["ce_full"]:.4f}  '
              f'CE unigram|v0 {cu:.4f}')
        print(f'  L1 = {L1:+.4f}   L2(full) = {o["L2_full"]:+.4f}   L2(ego) = {o["L2_ego"]:+.4f}')
    return out


def main():
    p_hat, L1s, L2s = sim_side()
    import os
    if not os.path.exists(f'{RES}/dump_test_ce.pkl'):
        print('\n[nuScenes side pending: dump_test_ce.pkl not yet written]'); return
    N = nus_side()
    print('\nIMPLIED rho (interpolated along rho at the two p rows bracketing p_hat):')
    lo = max(p for p in PS if p <= p_hat); hi = min(p for p in PS if p >= p_hat)
    for lab in ('1-11', '2-11'):
        for nm, T, v in (('L1', L1s, N[lab]['L1']), ('L2', L2s, N[lab]['L2_full'])):
            r = [interp_x(RHOS, [T[(lab, p, x)] for x in RHOS], v) for p in (lo, hi)]
            print(f'  slots {lab} {nm}: nuScenes {v:+.4f} -> rho at p={lo:g}: {r[0]:.3f}, '
                  f'at p={hi:g}: {r[1]:.3f}')


if __name__ == '__main__':
    main()
