"""ar1/meta_actions.py -- R1.2: AR1 Table 5 atomic meta-actions at 10 Hz (CPU).

AR1 (paper, Table 5 + text): lon {gentle accelerate, strong accelerate, gentle
decelerate, strong decelerate, maintain speed, stop, reverse}; lat {steer left, steer
right, sharp steer left, sharp steer right, reverse left, reverse right, go straight};
"automatically labeled at 10Hz". NO numeric thresholds are given in the paper ("not
specified"); every threshold below is OURS.

Signals (highest rate on disk; every value at tick t uses data <= t only):
  source 1  CAN 'pose' (~50 Hz): v(t) = vel[0] of the latest message <= t (signed
            longitudinal speed); a(t) = (v(t) - v(t - 0.5)) / 0.5; yaw rate w(t) = mean
            rotation_rate[2] over messages in (t - 0.5, t] (z up, + = left).
  source 2  fallback when CAN does not cover the tick (blacklisted scenes, gaps > 0.1 s):
            LIDAR_TOP ego poses (~20 Hz, sweep metadata; no image files needed):
            v(t) = signed displacement along the heading over (t - 0.2, t] / dt,
            a as above, w(t) = wrap(yaw(t) - yaw(t - 0.5)) / 0.5.
  curvature k(t) = w(t) / v(t) if |v| >= 1.0 m/s, else 0 (undefined at crawl speed).
Thresholds (ours):
  lon  reverse      v < -0.2 m/s
       stop         |v| <= 0.2 m/s
       strong acc   a >= +2.0 m/s^2        gentle acc  +0.5 <= a < +2.0
       strong dec   a <= -2.0 m/s^2        gentle dec  -2.0 < a <= -0.5
       maintain     |a| < 0.5 m/s^2
       (2.0 m/s^2 ~ 0.2 g, a common comfort bound; 0.5 m/s^2 separates intent from
       CAN speed noise.)
  lat  stop         -> go straight (no lateral motion)
       reverse      |k| >= 0.01 -> reverse left / right, else go straight
       forward      |k| < 0.01 1/m (radius > 100 m) go straight
                    0.01 <= |k| < 0.1 (radius 10-100 m) steer left / right
                    |k| >= 0.1 (radius < 10 m) sharp steer left / right
Ticks: per w1 record, t0 + 0.1 j, j = 0..60 (t0 = sample timestamp; 6.0 s future, the
nuScenes horizon). Ticks beyond the source coverage get -1.
Output: Alpamayo/data/ar1_meta.pkl {sample_token: {'lon': int8[61], 'lat': int8[61],
'src': int8[61] (1 CAN, 2 pose, 0 none), 'v','a','k': float32[61]}}; counts to stdout.
"""
import json, math, pickle, bisect, collections
import numpy as np

ROOT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
LON = ('gentle_acc', 'strong_acc', 'gentle_dec', 'strong_dec', 'maintain', 'stop', 'reverse')
LAT = ('steer_L', 'steer_R', 'sharp_L', 'sharp_R', 'reverse_L', 'reverse_R', 'straight')
NT, DT, WIN = 61, 0.1, 0.5
GAP = 0.1                                   # max staleness of a source sample (s)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def lon_cls(v, a):
    if v < -0.2: return 6
    if abs(v) <= 0.2: return 5
    if a >= 2.0: return 1
    if a >= 0.5: return 0
    if a <= -2.0: return 3
    if a <= -0.5: return 2
    return 4


def lat_cls(lon, k):
    if lon == 5: return 6
    if lon == 6:
        return 6 if abs(k) < 0.01 else (4 if k > 0 else 5)
    if abs(k) < 0.01: return 6
    if abs(k) < 0.1: return 0 if k > 0 else 1
    return 2 if k > 0 else 3


class CanSrc:
    def __init__(self, msgs):
        self.t = np.array([m['utime'] for m in msgs], dtype=np.int64) / 1e6
        self.v = np.array([m['vel'][0] for m in msgs])
        self.w = np.array([m['rotation_rate'][2] for m in msgs])

    def at(self, t):
        i = bisect.bisect_right(self.t, t) - 1; j = bisect.bisect_right(self.t, t - WIN) - 1
        if i < 0 or j < 0 or t - self.t[i] > GAP or (t - WIN) - self.t[j] > GAP:
            return None
        lo = bisect.bisect_right(self.t, t - WIN)
        return self.v[i], (self.v[i] - self.v[j]) / WIN, float(self.w[lo:i + 1].mean())


class PoseSrc:
    def __init__(self, poses):                  # [(t, x, y, yaw)] sorted
        self.t = np.array([p[0] for p in poses]); self.P = poses

    def _i(self, t):
        i = bisect.bisect_right(self.t, t) - 1
        return i if i >= 0 and t - self.t[i] <= GAP else None

    def _v(self, t):
        i, j = self._i(t), self._i(t - 0.2)
        if i is None or j is None or i == j:
            return None
        _, x1, y1, h1 = self.P[i]; _, x0, y0, _ = self.P[j]
        return ((x1 - x0) * math.cos(h1) + (y1 - y0) * math.sin(h1)) / (self.t[i] - self.t[j])

    def at(self, t):
        v, v0 = self._v(t), self._v(t - WIN)
        i, j = self._i(t), self._i(t - WIN)
        if v is None or v0 is None or i is None or j is None:
            return None
        return v, (v - v0) / WIN, wrap(self.P[i][3] - self.P[j][3]) / (self.t[i] - self.t[j])


def main():
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    W = pickle.load(open(f'{DATA}/w1_data.pkl', 'rb'))['records']
    S = {s['token']: s for s in json.load(open(f'{ROOT}/v1.0-trainval/sample.json'))}
    scenes = {r['scene_name'] for r in W}
    can = NuScenesCanBus(dataroot=ROOT)
    black = {f'scene-{i:04d}' for i in can.can_blacklist}
    CS = {}
    for sc in scenes:
        try:
            m = [] if sc in black else can.get_messages(sc, 'pose')
        except Exception:
            m = []
        CS[sc] = CanSrc(m) if len(m) else None
    print('loading sample_data / ego_pose ...', flush=True)
    s2scene = {}
    sc_tok = {s['token']: s['name'] for s in json.load(open(f'{ROOT}/v1.0-trainval/scene.json'))}
    for s in S.values():
        s2scene[s['token']] = sc_tok[s['scene_token']]
    want = {}
    for d in json.load(open(f'{ROOT}/v1.0-trainval/sample_data.json')):
        if '/LIDAR_TOP/' in d['filename']:
            want[d['ego_pose_token']] = s2scene[d['sample_token']]
    PS = collections.defaultdict(list)
    for e in json.load(open(f'{ROOT}/v1.0-trainval/ego_pose.json')):
        sc = want.get(e['token'])
        if sc in scenes:
            PS[sc].append((e['timestamp'] / 1e6, e['translation'][0], e['translation'][1],
                           yaw_of(e['rotation'])))
    PS = {sc: PoseSrc(sorted(v)) for sc, v in PS.items()}

    out, agree = {}, collections.Counter()
    for r in W:
        st = r['sample_token']; sc = r['scene_name']; t0 = S[st]['timestamp'] / 1e6
        L = {k: np.full(NT, -1, np.int8) for k in ('lon', 'lat', 'src')}
        F = {k: np.full(NT, np.nan, np.float32) for k in ('v', 'a', 'k')}
        for j in range(NT):
            t = t0 + DT * j
            x = CS[sc].at(t) if CS[sc] is not None else None
            src = 1
            y = PS[sc].at(t) if sc in PS else None
            if x is None:
                x, src = y, 2
            if x is None:
                L['src'][j] = 0
                continue
            v, a, w = x
            k = w / v if abs(v) >= 1.0 else 0.0
            lo = lon_cls(v, a)
            L['lon'][j], L['lat'][j], L['src'][j] = lo, lat_cls(lo, k), src
            F['v'][j], F['a'][j], F['k'][j] = v, a, k
            if src == 1 and y is not None and j % 10 == 0:      # cross-check sources
                ky = y[2] / y[0] if abs(y[0]) >= 1.0 else 0.0
                ly = lon_cls(y[0], y[1])
                agree['lon'] += int(ly == lo); agree['lat'] += int(lat_cls(ly, ky) == L['lat'][j])
                agree['n'] += 1
        out[st] = {**L, **F, 'split': r['split'], 'command': r['command']}
    pickle.dump(out, open(f'{DATA}/ar1_meta.pkl', 'wb'))
    report(out, agree)


def report(out, agree):
    sp = ('train', 'holdout', 'val')
    print(f'\nsource cross-check (CAN vs pose-derived, every 10th tick where both exist, '
          f'n={agree["n"]}): lon agree {agree["lon"] / agree["n"]:.3f}, '
          f'lat agree {agree["lat"] / agree["n"]:.3f}')
    for s in sp:
        R = [o for o in out.values() if o['split'] == s]
        src = np.concatenate([o['src'] for o in R])
        print(f'{s:8} records {len(R):6d}  ticks {src.size:8d}: CAN {np.mean(src == 1):.3f}  '
              f'pose {np.mean(src == 2):.3f}  none {np.mean(src == 0):.3f}')
    for nm, names in (('lon', LON), ('lat', LAT)):
        print(f'\n{nm.upper()} class counts, all valid 10 Hz ticks in [t0, t0+6.0 s] (share):')
        print(f'  {"class":11}' + ''.join(f'{s:>17}' for s in sp))
        C = {s: np.concatenate([o[nm] for o in out.values() if o['split'] == s]) for s in sp}
        for c, n in enumerate(names):
            print(f'  {n:11}' + ''.join(f'{int((C[s] == c).sum()):9d} ({np.mean(C[s][C[s] >= 0] == c):.3f})'
                                        for s in sp))
        print(f'\n{nm.upper()} at the t0 tick only (one per keyframe):')
        C0 = {s: np.array([o[nm][0] for o in out.values() if o['split'] == s]) for s in sp}
        for c, n in enumerate(names):
            print(f'  {n:11}' + ''.join(f'{int((C0[s] == c).sum()):9d} ({np.mean(C0[s][C0[s] >= 0] == c):.3f})'
                                        for s in sp))
    # sign sanity: lat at t0+3s vs the VAD command (0 right, 1 left, 2 straight)
    print('\nsign check, val: share of LEFT-type vs RIGHT-type lat labels over [t0, t0+6 s] by command')
    for cmd, cn in ((1, 'cmd LEFT'), (0, 'cmd RIGHT'), (2, 'cmd STRAIGHT')):
        x = np.concatenate([o['lat'] for o in out.values() if o['split'] == 'val' and o['command'] == cmd])
        x = x[x >= 0]
        print(f'  {cn:13} n_ticks {x.size:7d}  left {np.isin(x, (0, 2)).mean():.3f}  '
              f'right {np.isin(x, (1, 3)).mean():.3f}  straight {np.mean(x == 6):.3f}')


if __name__ == '__main__':
    main()
