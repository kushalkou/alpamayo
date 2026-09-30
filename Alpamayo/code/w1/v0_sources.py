"""w1/v0_sources.py -- WEEK1 STEP 1b/1c: causal v0 (and accel / yaw rate) sources.

Per record (all 29,049), t0 = sample['timestamp'] (the LIDAR keyframe time, which is
also the evaluation origin chosen on holdout):
  v0_fwd   = future_speeds[0] = |p1 - p0|/dt                 (NOT causal; side number)
  v0_back  = |p0 - p-1|/dt ; a_back = (v0_back - |p-1 - p-2|/dt)/dt ;
             yr_back = wrap(yaw0 - yaw-1)/dt                  (poses, CAM ego pose)
  v0_bext  = v0_back + 0.25 * a_back  (extrapolate the half-step lag to t0)
  v0_can, a_can, yr_can = CAN 'pose' message (50 Hz): vel[0], accel[0],
             rotation_rate[2], taken from the LATEST message with utime <= t0 (causal;
             the nearest message can be <= 10 ms in the future). Offsets reported.
  Fallbacks (counted): no past pose -> v0_back = a_back = yr_back = 0; one past pose ->
  a_back = 0. CAN-blacklisted scene or no message at/before t0 -> CAN fields fall back
  to the BACKWARD POSE DIFFERENCES v0_back / a_back / yr_back (flag can_ok = False).
Evaluation (official val, ALL 5,119; lidar origin; same plan_metrics code as gate 3):
  CV(v0)            straight, constant v0
  KIN(v0, a, yr)    the kinematic reference rule: the measured accel and yaw rate act
                    over the first 0.5 s step, then speed and heading are held (the
                    'causal_once' rule of OVERNIGHT2 item 1c, with a0 now applied at
                    step 0 because v0 is instantaneous).
Output: Alpamayo/data/w1_v0.pkl {sample_token: {...}}; tables to stdout.
"""
import sys, math, pickle, collections, bisect
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import plan_metrics as PMX
from dataset import pose_to_xyyaw

DT = 0.5
ROOT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
OUT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_v0.pkl'


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def can_index(ut, t0):
    """index of the LAST message with utime <= t0 (never a later one); -1 if none."""
    return bisect.bisect_right(ut, t0) - 1 if len(ut) else -1


def pose_feats(r):
    P = [pose_to_xyyaw(p) for p in list(r['past_poses']) + [r['current_pose']]]
    d = {'n_past': len(P) - 1}
    if len(P) >= 2:
        d['v0_back'] = math.hypot(P[-1][0] - P[-2][0], P[-1][1] - P[-2][1]) / DT
        d['yr_back'] = wrap(P[-1][2] - P[-2][2]) / DT
    else:
        d['v0_back'] = 0.0; d['yr_back'] = 0.0
    if len(P) >= 3:
        v1 = math.hypot(P[-2][0] - P[-3][0], P[-2][1] - P[-3][1]) / DT
        d['a_back'] = (d['v0_back'] - v1) / DT
    else:
        d['a_back'] = 0.0
    d['v0_bext'] = max(0.0, d['v0_back'] + 0.25 * d['a_back'])
    return d


def build():
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    W = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_data.pkl', 'rb'))
    nusc = NuScenes(version='v1.0-trainval', dataroot=ROOT, verbose=False)
    can = NuScenesCanBus(dataroot=ROOT)
    black = {f'scene-{i:04d}' for i in can.can_blacklist}
    cache = {}
    out = {}
    for r in W['records']:
        t0 = nusc.get('sample', r['sample_token'])['timestamp']
        d = pose_feats(r); d['v0_fwd'] = float(r['future_speeds'][0]); d['t0'] = t0
        sc = r['scene_name']
        if sc not in cache:
            try:
                m = [] if sc in black else can.get_messages(sc, 'pose')
            except Exception:
                m = []
            cache[sc] = (np.array([x['utime'] for x in m]), m)
        ut, m = cache[sc]
        k = can_index(ut, t0)
        if k >= 0:
            x = m[k]
            d.update(can_ok=True, v0_can=float(x['vel'][0]), a_can=float(x['accel'][0]),
                     yr_can=float(x['rotation_rate'][2]), can_dt_ms=(t0 - ut[k]) / 1e3)
            j = int(np.argmin(np.abs(ut - t0)))
            d['can_nearest_dt_ms'] = (ut[j] - t0) / 1e3
        else:
            d.update(can_ok=False, v0_can=d['v0_back'], a_can=d['a_back'], yr_can=d['yr_back'],
                     can_dt_ms=float('nan'), can_nearest_dt_ms=float('nan'))
        out[r['sample_token']] = d
    pickle.dump(out, open(OUT, 'wb'))
    return W, out


def kin_pred(v0, a, yr, yaw0):
    acc = np.zeros(12); cur = np.zeros(12); acc[0] = a
    v1 = max(0.0, v0 + a * DT)
    if v1 > 0.5: cur[0] = yr / v1
    P, Y = PMX.rollout_xy_yaw(acc, cur, v0, yaw0)
    return P[:6], Y[:6]


def cv_pred(v0, yaw0):
    k = np.arange(1, 7) * DT * v0
    return np.stack([k * math.cos(yaw0), k * math.sin(yaw0)], 1), np.full(6, yaw0)


def evaluate(W, V0):
    V = [r for r in W['records'] if r['split'] == 'val']
    for r in V: r['_occ'] = PMX.occupancies(r)
    dd = [V0[r['sample_token']] for r in V]
    print(f'official val n={len(V)}; n_past=0: {sum(d["n_past"]==0 for d in dd)}, '
          f'n_past=1: {sum(d["n_past"]==1 for d in dd)}; CAN ok: {sum(d["can_ok"] for d in dd)}')
    off = np.array([d['can_dt_ms'] for d in dd if d['can_ok']])
    near = np.array([d['can_nearest_dt_ms'] for d in dd if d['can_ok']])
    print(f'CAN causal lookup offset t0 - utime (ms): mean {off.mean():.2f} max {off.max():.2f}; '
          f'nearest-message offset: max |dt| {np.abs(near).max():.2f} ms '
          f'(future side max {near.max():.2f} ms)')
    print('CAN offset t0 - utime (ms) distribution, val: p50 %.2f p90 %.2f p99 %.2f max %.2f'
          % tuple(np.percentile(off, [50, 90, 99, 100])))
    for sp in ('train', 'holdout', 'val'):
        R = [r for r in W['records'] if r['split'] == sp]
        sc = sorted({r['scene_name'] for r in R})
        bad = sorted({r['scene_name'] for r in R if not V0[r['sample_token']]['can_ok']})
        n_ok = sum(V0[r['sample_token']]['can_ok'] for r in R)
        print(f'CAN coverage {sp:8}: samples {n_ok}/{len(R)}  scenes {len(sc)-len(bad)}/{len(sc)}'
              f'  scenes without CAN: {" ".join(bad) if bad else "none"}')
    variants = [('v0_fwd (side)', 'v0_fwd', 'a_back', 'yr_back'),
                ('v0_back', 'v0_back', 'a_back', 'yr_back'),
                ('v0_back+0.25a', 'v0_bext', 'a_back', 'yr_back'),
                ('v0_can', 'v0_can', 'a_can', 'yr_can')]
    print('\n' + ' ' * 22 + 'L2 NoAvg 1/2/3s   | L2 TemAvg 1/2/3s  | Col% NoAvg 1/2/3 | Col% TemAvg 1/2/3'
          ' || L2 NoAvg@3s median / p95')
    res = {}
    for lab, vk, ak, yk in variants:
        for rule in ('CV', 'KIN'):
            per = collections.defaultdict(list)
            for r, d in zip(V, dd):
                yaw0 = pose_to_xyyaw(r['current_pose'])[2]
                xy, yw = (cv_pred(d[vk], yaw0) if rule == 'CV'
                          else kin_pred(d[vk], d[ak], d[yk], yaw0))
                m = PMX.sample_metrics(r, PMX.pred_to_lidar(r, xy, yw, origin='lidar'), r['_occ'])
                for k, v in m.items(): per[k].append(v)
            per = {k: np.stack(v) for k, v in per.items()}
            a = PMX.aggregate(per); res[(lab, rule)] = per
            print(f'  {rule:3} {lab:16} ' + ' '.join(f'{a[f"L2_NoAvg_{s}s"]:.3f}' for s in (1, 2, 3))
                  + ' | ' + ' '.join(f'{a[f"L2_TemAvg_{s}s"]:.3f}' for s in (1, 2, 3))
                  + ' | ' + ' '.join(f'{a[f"Col_NoAvg_{s}s"]:.2f}' for s in (1, 2, 3))
                  + ' | ' + ' '.join(f'{a[f"Col_TemAvg_{s}s"]:.2f}' for s in (1, 2, 3))
                  + f' || {np.median(per["l2"][:, 5]):.3f} / {np.percentile(per["l2"][:, 5], 95):.3f}')
    return V, res


if __name__ == '__main__':
    W, V0 = build()
    evaluate(W, V0)
