"""sim/gates.py -- G1..G4, pilot, full map, sensitivity, sim-side locators.

Evaluation reuses shrink_lib UNCHANGED (rollout, cv_traj, ade, paired, load, traj);
only its RES directory is pointed at the sim cell's dump dir.
Blend alpha is fit on VAL (grid 0..1 step .05) and evaluated on TEST.
Every quantity is reported against BOTH plain CV and CV + v0-conditioned mean control
('v0mean': train mean control within the sample's v0 bin, rolled out from v0).
Model is NON-AUTOREGRESSIVE (simplification vs nuScenes).

Winner rule per cell: MODEL if model < CV with CI excluding 0; else BLEND if
blend < CV with CI excluding 0; else CV.

Usage: python gates.py g1 | g2 | g3 | pilot | map | sens | locate
"""
import os, sys, json, numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/sim')
import shrink_lib as SL
import model as M
import generate as GEN

ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)
AR0 = (GEN.A_LO, GEN.A_HI)


def ades(D, meta, idx, G, m, mode):
    return np.array([SL.ade(SL.traj(D, meta, m, i, mode), G[i]) for i in idx])


def ci(a, b):
    d, lo, hi, p = SL.paired(a, b)
    return [float(d), float(lo), float(hi), float(p)]


def eval_cell(od):
    SL.RES = od
    Dv, mv, iv, Gv, CVv = SL.load('val')
    Dt, mt, it, Gt, CVt = SL.load('test')
    r = {'n_test': len(it)}
    r['cv'] = np.array([SL.ade(CVt[i], Gt[i]) for i in it])
    for m in ('mlp', 'meanctl', 'v0mean'):
        r[f'{m}_expect'] = ades(Dt, mt, it, Gt, m, 'expect')
    r['mlp_argmax'] = ades(Dt, mt, it, Gt, 'mlp', 'argmax')
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


def summarize(r):
    """JSON-able row: means, medians, CIs vs CV and vs v0mean, G4, winner, CE stats."""
    cv, vm = r['cv'], r['v0mean_expect']
    row = {'n_test': r['n_test'], 'alpha': r['alpha'], 'alpha0_check': r['alpha0_check']}
    for k in ('cv', 'meanctl_expect', 'v0mean_expect', 'mlp_expect', 'mlp_argmax', 'blend'):
        row[k] = float(r[k].mean()); row[k + '_med'] = float(np.median(r[k]))
    for k in ('meanctl_expect', 'v0mean_expect', 'mlp_expect', 'mlp_argmax', 'blend'):
        row[k + '_vs_cv'] = ci(r[k], cv)
    for k in ('mlp_expect', 'blend'):
        row[k + '_vs_v0mean'] = ci(r[k], vm)
    row['expect_vs_argmax'] = ci(r['mlp_expect'], r['mlp_argmax'])
    m, b = row['mlp_expect_vs_cv'], row['blend_vs_cv']
    row['winner'] = 'MODEL' if m[2] < 0 else ('BLEND' if b[2] < 0 else 'CV')
    c = r['ce']['ce']['test']
    cm, cu, cuv = (float(np.mean(c[k][1:12])) for k in ('model', 'unigram', 'unigram_v0'))
    row.update(ce_model=cm, ce_uni=cu, ce_uni_v0=cuv, ig=1 - cm / cu, ig_v0=1 - cm / cuv)
    row['info'] = r['ce']['info']
    return row


def run_cell(job):
    """job = (p, rho, seed, gpu, zero_obs, arange). Cached in <od>/row.json."""
    p, rho, seed, gpu, zero_obs, arange = job
    od = (f'{M.SIMRES}/{GEN.cell_name(p, rho, seed, arange)}_t{seed}'
          + ('_noobs' if zero_obs else ''))
    rj = f'{od}/row.json'
    if os.path.exists(rj):
        return json.load(open(rj))
    M.run(p, rho, seed=seed, train_seed=seed, device=f'cuda:{gpu}', zero_obs=zero_obs,
          arange=arange)
    row = summarize(eval_cell(od))
    row.update(p=p, rho=rho, seed=seed, zero_obs=zero_obs, arange=list(arange))
    json.dump(row, open(rj, 'w'), indent=1, default=str)
    return row


def run_many(cells, workers=16):
    """cells = list of (p, rho, seed, zero_obs, arange). Parallel over 8 GPUs."""
    import multiprocessing as mp
    jobs = [(p, rho, s, k % 8, z, tuple(a)) for k, (p, rho, s, z, a) in enumerate(cells)]
    ctx = mp.get_context('spawn')
    with ctx.Pool(min(workers, len(jobs))) as pool:
        return pool.map(run_cell, jobs, chunksize=1)


def fci(c, neg_good=True):
    return f'{c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]'


def table(rows, extra=()):
    h = (f'  {"p":>4} {"rho":>4} {"s":>2} {"CV":>6} {"v0mean":>6} {"model":>6} {"blend":>6} '
         f'{"a*":>4}  {"model-CV":>24}  {"blend-CV":>24}  {"model-v0mean":>24}  '
         f'{"blend-v0mean":>24}  winner')
    out = [h]
    for r in rows:
        out.append(f'  {r["p"]:4.2f} {r["rho"]:4.2f} {r["seed"]:2d} {r["cv"]:6.3f} '
                   f'{r["v0mean_expect"]:6.3f} {r["mlp_expect"]:6.3f} {r["blend"]:6.3f} '
                   f'{r["alpha"]:4.2f}  {fci(r["mlp_expect_vs_cv"]):>24}  '
                   f'{fci(r["blend_vs_cv"]):>24}  {fci(r["mlp_expect_vs_v0mean"]):>24}  '
                   f'{fci(r["blend_vs_v0mean"]):>24}  {r["winner"]}')
    return '\n'.join(out)


def g1():
    r = run_many([(0.0, 0.0, 0, False, AR0)], 1)[0]
    print('G1  rho=0, p=0\n' + table([r]))


def g2():
    r = run_many([(0.5, 1.0, 0, False, AR0)], 1)[0]
    print('=' * 90); print('G2  rho=1, p=0.5  (model NON-AUTOREGRESSIVE)'); print('=' * 90)
    print(table([r]))
    m = r['mlp_expect_vs_cv']
    print(f'  G2 {"PASS" if m[2] < 0 else "FAIL"}: model - CV = {fci(m)}, margin '
          f'{-m[0]:.3f} m ({-m[0] / r["cv"] * 100:.1f}% of CV)')
    return r


def g3():
    rhos = (0.0, 0.25, 0.5, 0.75, 1.0)
    rows = run_many([(0.5, rho, 0, False, AR0) for rho in rhos])
    print('=' * 90); print('G3  p=0.5, rho sweep  (model NON-AUTOREGRESSIVE)'); print('=' * 90)
    print(table(rows))
    gm = [-r['mlp_expect_vs_cv'][0] for r in rows]
    gb = [-r['blend_vs_cv'][0] for r in rows]
    gv = [-r['mlp_expect_vs_v0mean'][0] for r in rows]
    mono = lambda g: all(b > a for a, b in zip(g, g[1:]))
    print('  gap CV - model   : ' + '  '.join(f'{g:+.3f}' for g in gm) +
          f'   monotone: {mono(gm)}')
    print('  gap CV - blend   : ' + '  '.join(f'{g:+.3f}' for g in gb) +
          f'   monotone: {mono(gb)}')
    print('  gap v0mean-model : ' + '  '.join(f'{g:+.3f}' for g in gv) +
          f'   monotone: {mono(gv)}')
    print('=' * 90); print('G4  expectation vs argmax at every G3 point'); print('=' * 90)
    ok = True
    for r in rows:
        c = r['expect_vs_argmax']; ok &= c[2] < 0
        print(f'  rho={r["rho"]:.2f}  argmax {r["mlp_argmax"]:.3f}  expect {r["mlp_expect"]:.3f}'
              f'  expect-argmax {fci(c)}  gain {-c[0]:.3f}')
    print(f'  G4 {"PASS" if ok else "FAIL"}')
    return rows


def pilot():
    cells = [(p, rho, 0, False, AR0) for p in (0.0, 0.2, 0.5) for rho in (0.0, 0.5, 1.0)]
    rows = run_many(cells)
    print('PILOT  (1 seed)\n' + table(rows))
    return rows


PS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75)
RHOS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def fullmap():
    cells = [(p, rho, s, False, AR0) for s in (0, 1, 2) for p in PS for rho in RHOS]
    rows = run_many(cells)
    json.dump(rows, open(f'{M.SIMRES}/map_rows.json', 'w'), default=str)
    print('FULL MAP rows: %d' % len(rows))
    return rows


def sens():
    rows = []
    for ar in ((0.1, 0.5), (0.2, 1.0), (0.5, 2.0)):
        rows += run_many([(p, rho, 0, False, ar) for p in (0.0, 0.2, 0.5)
                          for rho in (0.0, 0.5, 1.0)])
    json.dump(rows, open(f'{M.SIMRES}/sens_rows.json', 'w'), default=str)
    for ar in ((0.1, 0.5), (0.2, 1.0), (0.5, 2.0)):
        print(f'ACCEL U{ar}\n' + table([r for r in rows if tuple(r['arange']) == ar]))
    return rows


def locate():
    """sim side of L1/L2 over the full map grid, seed 0: obs present vs zeroed."""
    cells = [(p, rho, 0, z, AR0) for p in PS for rho in RHOS for z in (False, True)]
    rows = run_many(cells)
    json.dump(rows, open(f'{M.SIMRES}/locate_rows.json', 'w'), default=str)
    return rows


if __name__ == '__main__':
    {'g1': g1, 'g2': g2, 'g3': g3, 'pilot': pilot, 'map': fullmap, 'sens': sens,
     'locate': locate}[sys.argv[1]]()
