"""w1/gate3.py -- WEEK1 gate 3 (metric sanity) + items 1/2/4 statistics.

GATE 3 on OFFICIAL VAL (GT and CV only -- no model, nothing tuned):
  (a) GT lidar trajectory fed as the prediction: L2 must be exactly 0 at every step.
  (b) GT collision: the literature metric masks GT-colliding timesteps, so the masked
      rate is 0 by construction; the UNMASKED GT box-collision rate must be small and
      nonzero (annotation noise).
  (c) CV evaluated identically (all four numbers).
Also: our CAM-ego GT converted to the lidar frame (conversion floor, both origins,
chosen on HOLDOUT); CV with causal v0 (item 1); split counts (item 2); command
distribution (item 4).
Subsets: ALL = n_fut >= 6 (literature 3 s protocol); CAUSAL = also >= 2 past poses.
"""
import sys, pickle, math, collections
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import plan_metrics as PMX
from dataset import pose_to_xyyaw

DT = 0.5
W = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_data.pkl', 'rb'))
R = W['records']


def causal(r):
    return len(r['past_poses']) >= 2


def cv_pred(r, v0):
    x0, y0, yaw0 = pose_to_xyyaw(r['current_pose'])
    k = np.arange(1, 7) * DT * v0
    xy = np.stack([k * math.cos(yaw0), k * math.sin(yaw0)], 1)
    return xy, np.full(6, yaw0)


def own_gt(r):
    x0, y0, _ = pose_to_xyyaw(r['current_pose'])
    return (np.array(r['future_positions'])[:6] - np.array([x0, y0]),
            np.array(r['future_yaws'])[:6])


def back_v0(r):
    p = [pose_to_xyyaw(x) for x in r['past_poses'][-1:] + [r['current_pose']]]
    return math.hypot(p[1][0] - p[0][0], p[1][1] - p[0][1]) / DT


def run(recs, preds):
    per = collections.defaultdict(list)
    for r, p in zip(recs, preds):
        m = PMX.sample_metrics(r, p, r['_occ'])
        for k, v in m.items(): per[k].append(v)
    return {k: np.stack(v) for k, v in per.items()}


def fmt(name, per):
    a = PMX.aggregate(per)
    l2n = per['l2'][:, 5]; l2t = per['l2'].mean(1)
    return (f'  {name:28} ' + ' '.join(f'{a[f"L2_NoAvg_{s}s"]:.3f}' for s in (1, 2, 3)) + ' | '
            + ' '.join(f'{a[f"L2_TemAvg_{s}s"]:.3f}' for s in (1, 2, 3)) + ' | '
            + ' '.join(f'{a[f"Col_NoAvg_{s}s"]:.2f}' for s in (1, 2, 3)) + ' | '
            + ' '.join(f'{a[f"Col_TemAvg_{s}s"]:.2f}' for s in (1, 2, 3))
            + f' || L2@3s med {np.median(l2n):.3f} p95 {np.percentile(l2n,95):.3f}'
            + f'; TemAvg3s med {np.median(l2t):.3f} p95 {np.percentile(l2t,95):.3f}')


HDR = ('  method                       L2 NoAvg 1/2/3s   | L2 TemAvg 1/2/3s  | '
       'Col% NoAvg(veh)  | Col% TemAvg(veh+ped)')


def main():
    # ---- item 2 counts ----
    print('ITEM 2: official split (records = samples with >= 6 future keyframes)')
    for sp in ('train', 'holdout', 'val'):
        S = [r for r in R if r['split'] == sp]
        sc = len({r['scene_name'] for r in S})
        print(f'  {sp:8} scenes {sc:4d}  n_fut>=6 {len(S):6d}  n_fut=12 {sum(r["n_fut"]==12 for r in S):6d}'
              f'  +causal(>=2 past) {sum(causal(r) for r in S):6d}'
              f'  n_fut=12 & causal {sum(causal(r) and r["n_fut"]==12 for r in S):6d}')
    # ---- item 4 command ----
    print('\nITEM 4: command distribution (0 right / 1 left / 2 straight), n_fut>=6')
    for sp in ('train', 'holdout', 'val'):
        c = np.bincount([r['command'] for r in R if r['split'] == sp], minlength=3)
        print(f'  {sp:8} right {c[0]:5d} ({c[0]/c.sum():.3f})  left {c[1]:5d} ({c[1]/c.sum():.3f})'
              f'  straight {c[2]:5d} ({c[2]/c.sum():.3f})')
    # ---- timing ----
    dts = np.array([(r['lidar_ts'] - r['cam_ts']) / 1e3 for r in R])
    print(f'\nLIDAR_TOP vs CAM_FRONT keyframe timestamp (ms): mean {dts.mean():.1f} '
          f'median {np.median(dts):.1f} p95 {np.percentile(np.abs(dts),95):.1f}')

    # ---- conversion floor, choose origin on holdout ----
    ho = [r for r in R if r['split'] == 'holdout' and causal(r)]
    fl = {}
    for org in ('cam', 'lidar'):
        e = [np.linalg.norm(PMX.pred_to_lidar(r, *own_gt(r), origin=org) - r['gt_lidar6'], axis=1)
             for r in ho]
        fl[org] = np.stack(e)
        print(f'  conversion floor HOLDOUT origin={org:5}: L2@1/2/3s '
              + ' '.join(f'{fl[org][:, k].mean():.4f}' for k in (1, 3, 5)))
    org = min(fl, key=lambda o: fl[o][:, 5].mean())
    print(f'  -> origin chosen on holdout: {org}')

    for sub in ('ALL', 'CAUSAL'):
        V = [r for r in R if r['split'] == 'val' and (sub == 'ALL' or causal(r))]
        for r in V: r['_occ'] = PMX.occupancies(r)
        print(f'\n{"="*90}\nGATE 3 on OFFICIAL VAL, subset {sub} (n={len(V)})\n{"="*90}')
        print(HDR)
        gt = run(V, [r['gt_lidar6'] for r in V])
        print(fmt('GT (lidar, literature)', gt))
        print(f'  (a) GT L2 max over all samples/steps = {gt["l2"].max():.3e}  '
              f'-> {"PASS" if gt["l2"].max() == 0 else "FAIL"}')
        raw = {k: np.stack([PMX.gt_raw_collision(r, r['_occ'])[k] for r in V]) for k in ('veh', 'vp')}
        print(f'  (b) GT masked collision (metric) max = {max(gt["col_veh"].max(), gt["col_vp"].max()):.0f}'
              f' (0 by construction); UNMASKED GT box collision % per step 1..6:')
        for k in ('veh', 'vp'):
            print(f'      {k:4} ' + ' '.join(f'{100*raw[k][:, t].mean():.2f}' for t in range(6))
                  + f'   any-step {100*raw[k].any(1).mean():.2f}%')
        own = run(V, [PMX.pred_to_lidar(r, *own_gt(r), origin=org) for r in V])
        print(fmt(f'own CAM GT -> lidar ({org})', own))
        cv = run(V, [PMX.pred_to_lidar(r, *cv_pred(r, float(r['future_speeds'][0])), origin=org)
                     for r in V])
        print(fmt('CV (fwd v0, decode seed)', cv))
        if sub == 'CAUSAL':
            cvb = run(V, [PMX.pred_to_lidar(r, *cv_pred(r, back_v0(r)), origin=org) for r in V])
            print(fmt('CV (causal backward v0)', cvb))
        for r in V: r.pop('_occ')


if __name__ == '__main__':
    main()
