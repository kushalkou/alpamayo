"""w1/evalw1.py -- WEEK1 shared evaluation on OFFICIAL VAL (all 5,119 samples).

Every predictor is a [N,12,2] trajectory in VAD's evaluation frame (t0 LIDAR frame),
index-aligned with records.build('val', n_fut=6) (and 'holdout' for blend fitting).
Metrics (plan_metrics.py, ported VAD engine): L2 NoAvg / TemAvg at 1/2/3 s and box
collision NoAvg (vehicles) / TemAvg (vehicles + pedestrians) on steps 1..6; ADE@6s /
FDE@6s on the n_fut = 12 subset. Mean / median / p95.
Paired SCENE-LEVEL bootstrap: resample the 150 val scenes with replacement (10,000
draws), statistic = mean per-sample difference over the drawn scenes; 95% CI;
two-sided p.
Baselines, all causal, from v0_can / a_can / yr_can (w1_v0.pkl):
  CV        straight at v0
  KIN       the kinematic reference rule (accel + yaw rate over the first 0.5 s, then held)
  ORACLE-KIN  (amendment 4) CV + the oracle meta-action: constant accel = the median
            train accel of the sample's lon class, constant curvature = the median
            train curvature of its lat (command) class. Both are medians over train
            samples of the per-sample mean control over the first 6 steps.
Strata (same definitions as before): stationary v0 < 0.5; else turning if max|GT
curvature| > 0.05 (the CAM-based curvature, as before); else straight.
"""
import sys, math, pickle, collections
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import plan_metrics as PMX
import records
from targets import rollout as roll12

D = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
DT = 0.5
_W = None


def data():
    global _W
    if _W is None:
        W = pickle.load(open(f'{D}/w1_data.pkl', 'rb'))
        V0 = pickle.load(open(f'{D}/w1_v0.pkl', 'rb'))
        T = pickle.load(open(f'{D}/w1_targets.pkl', 'rb'))['targets']
        _W = (W, V0, T)
    return _W


def split_records(split):
    W, V0, T = data()
    return [r for r in W['records'] if r['split'] == split]


def cv(recs):
    W, V0, T = data()
    out = []
    for r in recs:
        v = V0[r['sample_token']]['v0_can']
        out.append(roll12(np.zeros(12), np.zeros(12), v))
    return np.stack(out)


def kin(recs):
    W, V0, T = data()
    out = []
    for r in recs:
        d = V0[r['sample_token']]
        acc = np.zeros(12); cur = np.zeros(12); acc[0] = d['a_can']
        v1 = max(0.0, d['v0_can'] + d['a_can'] * DT)
        if v1 > 0.5: cur[0] = d['yr_can'] / v1
        out.append(roll12(acc, cur, d['v0_can']))
    return np.stack(out)


def oracle_kin_table():
    W, V0, T = data()
    tr = records.build('train', ('cmd', 'lon'))
    A = collections.defaultdict(list); K = collections.defaultdict(list)
    for t in tr:
        A[t['meta_lon']].append(float(np.mean(t['acc'][:6])))
        K[t['command']].append(float(np.mean(t['cur'][:6])))
    return ({c: float(np.median(v)) for c, v in A.items()},
            {c: float(np.median(v)) for c, v in K.items()})


def oracle_kin(recs, table, lon_labels=None, lat_labels=None):
    W, V0, T = data()
    a_tab, k_tab = table
    out = []
    for j, r in enumerate(recs):
        t = T[r['sample_token']]
        lon = lon_labels[j] if lon_labels is not None else records.meta_lon(t['P'], t['s'], t['v0'])
        lat = lat_labels[j] if lat_labels is not None else r['command']
        out.append(roll12(np.full(12, a_tab[lon]), np.full(12, k_tab[lat]), t['v0']))
    return np.stack(out)


def per_sample(recs, pred, occ=None):
    """per-sample arrays: l2[n,6], col_veh[n,6], col_vp[n,6], ade/fde (nan if n_fut<12)."""
    W, V0, T = data()
    per = collections.defaultdict(list)
    for j, r in enumerate(recs):
        o = occ[j] if occ is not None else PMX.occupancies(r)
        m = PMX.sample_metrics(r, pred[j][:6], o)
        for k, v in m.items(): per[k].append(v)
        P = T[r['sample_token']]['P']
        if r['n_fut'] == 12:
            e = np.linalg.norm(pred[j] - P[:12], axis=1); per['ade'].append(e.mean()); per['fde'].append(e[-1])
        else:
            per['ade'].append(np.nan); per['fde'].append(np.nan)
    return {k: np.array(v) if k in ('ade', 'fde') else np.stack(v) for k, v in per.items()}


def summary(per):
    a = PMX.aggregate(per)
    ade = per['ade'][~np.isnan(per['ade'])]; fde = per['fde'][~np.isnan(per['fde'])]
    a.update(ADE6=float(ade.mean()), ADE6_med=float(np.median(ade)), ADE6_p95=float(np.percentile(ade, 95)),
             FDE6=float(fde.mean()),
             L2_3s_med=float(np.median(per['l2'][:, 5])), L2_3s_p95=float(np.percentile(per['l2'][:, 5], 95)),
             L2T_3s_med=float(np.median(per['l2'].mean(1))), L2T_3s_p95=float(np.percentile(per['l2'].mean(1), 95)))
    return a


def row(name, per):
    a = summary(per)
    return (f'  {name:22} ' + ' '.join(f'{a[f"L2_NoAvg_{s}s"]:.3f}' for s in (1, 2, 3)) + ' | '
            + ' '.join(f'{a[f"L2_TemAvg_{s}s"]:.3f}' for s in (1, 2, 3)) + ' | '
            + ' '.join(f'{a[f"Col_NoAvg_{s}s"]:.2f}' for s in (1, 2, 3)) + ' | '
            + ' '.join(f'{a[f"Col_TemAvg_{s}s"]:.2f}' for s in (1, 2, 3)) + ' | '
            + f'{a["ADE6"]:.3f} {a["ADE6_med"]:.3f} {a["ADE6_p95"]:.3f} {a["FDE6"]:.3f} | '
            + f'{a["L2_3s_med"]:.3f} {a["L2_3s_p95"]:.3f}')


HDR = ('  method                 L2 NoAvg 1/2/3s   | L2 TemAvg 1/2/3s  | Col% NoAvg     | '
       'Col% TemAvg    | ADE6 mean/med/p95 FDE6 | L2@3s med/p95')


def scene_boot(recs, a, b, n=10000, seed=0):
    """paired scene-level bootstrap of mean(a - b); nan entries ignored."""
    sc = np.array([r['scene_name'] for r in recs]); d = np.asarray(a, float) - np.asarray(b, float)
    ok = ~np.isnan(d); sc, d = sc[ok], d[ok]
    us = np.unique(sc); idx = {s: np.where(sc == s)[0] for s in us}
    S = np.array([d[idx[s]].sum() for s in us]); C = np.array([len(idx[s]) for s in us])
    rs = np.random.RandomState(seed)
    draws = rs.randint(0, len(us), (n, len(us)))
    bm = S[draws].sum(1) / C[draws].sum(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    p = min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean()))
    return float(d.mean()), float(lo), float(hi), float(p)


def fmt_ci(c):
    return f'{c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}] p={c[3]:.4f}'


def compare(recs, pa, pb, label):
    """paired scene bootstrap on L2@3s NoAvg, L2 TemAvg@3s, ADE@6s."""
    return (f'  {label:34} L2@3s {fmt_ci(scene_boot(recs, pa["l2"][:, 5], pb["l2"][:, 5]))} | '
            f'L2T@3s {fmt_ci(scene_boot(recs, pa["l2"].mean(1), pb["l2"].mean(1)))} | '
            f'ADE6 {fmt_ci(scene_boot(recs, pa["ade"], pb["ade"]))}')


def strata(recs):
    W, V0, T = data()
    s = []
    for r in recs:
        if V0[r['sample_token']]['v0_can'] < 0.5: s.append('stationary')
        elif np.abs(np.array(r['future_curvatures'])).max() > 0.05: s.append('turning')
        else: s.append('straight')
    return np.array(s)


def fit_alpha(P_ho, C_ho, recs_ho):
    """alpha on HOLDOUT by mean ADE@6s over n_fut=12 samples (grid .05)."""
    W, V0, T = data()
    m = np.array([r['n_fut'] == 12 for r in recs_ho])
    G = np.stack([T[r['sample_token']]['P'][:12] for r, k in zip(recs_ho, m) if k])
    A = np.round(np.arange(0, 1.0001, 0.05), 2)
    cur = [np.linalg.norm(a * P_ho[m] + (1 - a) * C_ho[m] - G, axis=2).mean() for a in A]
    return float(A[int(np.argmin(cur))])
