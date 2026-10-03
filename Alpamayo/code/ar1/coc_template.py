"""ar1/coc_template.py -- R1.4 PROTOTYPE: template Chain-of-Causation (CoC) traces (CPU).

Follows AR1 Tables 1-3 in structure: (1) driving decision, closed set (Table 1),
labelled from the FUTURE (as AR1 labels the decision from the realised behaviour);
(2) critical components (Table 2) from the HISTORY WINDOW ONLY (keyframe annotations at
t0 .. t0-2.0 s, CAN <= t0); (3) a composed one-sentence trace linking them.
Only for human review in phase R1 -- no training data is written.

Critical components (all causal):
  ego        speed now (CAN <= t0) and 1.0 s ago (2 Hz poses; records.ego_state_w1),
             turn-signal state (CAN vehicle_monitor, last message <= t0)
  objects    annotations at t0 with visibility >= 40% and >= 1 lidar point; position in
             the t0 ego frame (x forward, y left); velocity = displacement from the same
             instance's annotation 1.0 s earlier (0.5 s if that is missing); motion
             status moving / stationary (0.5 m/s). nuScenes attribute labels
             (moving / parked) are NOT used: they are annotated with the whole clip in
             view.
  in-path    corridor along the ego's CURRENT curvature (CAN yaw rate / speed at t0):
             0 < x < 50 m, |y - k x^2 / 2| - w_obj / 2 < 1.5 m; nearest = lead.
  construction  >= 3 cones / barriers within 30 m ahead, |y| < 6 m
  routing    VAD command (left / right / straight) -- PRIVILEGED (derived from the GT
             future); kept only as a placeholder for a non-privileged route.
  NOT AVAILABLE in nuScenes (left out, see R1.4 plan): traffic-light state, stop /
  yield signs and lines, lane count / line types / lane membership (the map expansion
  is not on disk), road grade / speed bumps, weather from sensors.
Driving decision (future, 6 s; meta-actions from ar1/meta_actions.py at 10 Hz):
  lon  Yield               decel or stop ahead AND a VRU in path within 25 m (or a
                           crossing / oncoming vehicle in path within 25 m)
       Lead following      in-path vehicle (stationary or same direction) < 40 m AND (decel / stop ahead, or the
                           lead is moving with |dv| < 2 m/s, or both are stationary)
       Stop (static)       ego stops ahead, or stays stopped, with no in-path lead
       Speed adaptation    decel ahead and heading change > 30 deg (slowing for a turn)
       Set speed tracking  otherwise
       (gap-searching, acceleration for passing: not derivable without lanes; never
       emitted)
  lat  Turn L/R            heading change > 30 deg over the available future
       Lane change L/R     final lateral offset 2.5-5 m, heading change < 15 deg
       None                ego stationary for the whole window
       Lane keeping        otherwise
       (merge / split, nudges, pull-over, abort: need lane geometry; never emitted)
Output: AR1_COC_EXAMPLES.txt (20 traces, train split, stratified, seed 0) and the
decision distribution per split to stdout.
"""
import sys, json, math, pickle, bisect, collections, textwrap
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import records

ROOT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
OUT = '/home/dgx1user/Alpamayo-Kushal/AR1_COC_EXAMPLES.txt'
CMD = {0: 'turn right', 1: 'turn left', 2: 'go straight'}


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_ego(p, e):
    """global xy -> t0 ego frame (x forward, y left); e = (x, y, yaw)."""
    dx, dy = p[0] - e[0], p[1] - e[1]
    c, s = math.cos(e[2]), math.sin(e[2])
    return dx * c + dy * s, -dx * s + dy * c


def kind(cat):
    if cat.startswith('human.pedestrian'): return 'pedestrian'
    if cat in ('vehicle.bicycle', 'vehicle.motorcycle'): return cat.split('.')[1]
    if cat.startswith('vehicle.bus'): return 'bus'
    if cat.startswith('vehicle.'): return cat.split('.')[1].replace('construction', 'construction vehicle')
    if cat in ('movable_object.trafficcone', 'movable_object.barrier'): return 'cone/barrier'
    return None


def load():
    J = lambda n: json.load(open(f'{ROOT}/v1.0-trainval/{n}.json'))
    cat = {c['token']: c['name'] for c in J('category')}
    inst = {i['token']: cat[i['category_token']] for i in J('instance')}
    S = {s['token']: s['timestamp'] for s in J('sample')}
    print('loading sample_annotation.json ...', flush=True)
    A = {a['token']: a for a in J('sample_annotation')}
    by_s = collections.defaultdict(list)
    for a in A.values():
        by_s[a['sample_token']].append(a['token'])
    return inst, S, A, by_s


def objects(r, inst, S, A, by_s, k_path):
    e = (r['lidar_ep0']['translation'][0], r['lidar_ep0']['translation'][1],
         yaw_of(r['lidar_ep0']['rotation']))
    t0 = S[r['sample_token']]
    out = []
    for tok in by_s[r['sample_token']]:
        a = A[tok]
        kd = kind(inst[a['instance_token']])
        if kd is None or int(a['visibility_token']) < 2 or a['num_lidar_pts'] < 1:
            continue
        x, y = to_ego(a['translation'], e)
        prev, p = None, a
        for _ in range(2):                      # walk back up to 1.0 s
            if not p['prev']:
                break
            p = A[p['prev']]
            if (t0 - S[p['sample_token']]) / 1e6 <= 1.05:
                prev = p
        vx = vy = None
        if prev is not None:
            dt = (t0 - S[prev['sample_token']]) / 1e6
            vx = (a['translation'][0] - prev['translation'][0]) / dt
            vy = (a['translation'][1] - prev['translation'][1]) / dt
        sp = math.hypot(vx, vy) if vx is not None else None
        vl = None
        if sp is not None:                      # velocity in the ego frame
            c, s = math.cos(e[2]), math.sin(e[2])
            vl = (vx * c + vy * s, -vx * s + vy * c)
        yc = k_path * x * x / 2
        inp = 0 < x < 50 and abs(y - yc) - a['size'][0] / 2 < 1.5
        out.append({'kind': kd, 'x': x, 'y': y, 'speed': sp, 'vel': vl, 'in_path': inp,
                    'moving': sp is not None and sp > 0.5})
    return out


def rel_dir(o):
    if not o['moving'] or o['vel'] is None:
        return 'stationary'
    ang = math.degrees(math.atan2(o['vel'][1], o['vel'][0]))
    if abs(ang) < 30: return 'moving the same way'
    if abs(ang) > 150: return 'oncoming'
    return 'crossing ' + ('to the left' if ang > 0 else 'to the right')


def future(r, M):
    """decision features from the GT future and the 10 Hz meta-actions."""
    e = (r['lidar_ep0']['translation'][0], r['lidar_ep0']['translation'][1],
         yaw_of(r['lidar_ep0']['rotation']))
    fp = np.asarray(r['future_positions'])
    xf, yf = to_ego(fp[-1], e)
    dpsi = math.degrees(math.atan2(math.sin(r['future_yaws'][-1] - e[2]),
                                   math.cos(r['future_yaws'][-1] - e[2])))
    lon = M['lon'][1:]; ok = lon >= 0
    dec = np.isin(lon[:30], (2, 3)).mean() >= 0.3 if ok[:30].any() else False
    stop_f = bool((lon == 5).any())
    stay = bool((lon[ok] == 5).mean() > 0.5) if ok.any() else False
    vmin = float(np.nanmin(M['v'])) if np.isfinite(M['v']).any() else float('nan')
    vend = float(M['v'][np.where(np.isfinite(M['v']))[0][-1]]) if np.isfinite(M['v']).any() else float('nan')
    return {'dpsi': dpsi, 'y_end': yf, 'x_end': xf, 'dec': dec, 'stop': stop_f, 'stay': stay,
            'vmin': vmin, 'vend': vend, 'n_fut': len(fp)}


def decide(ego, objs, F):
    v0 = ego['v0']
    veh = [o for o in objs if o['in_path'] and o['kind'] not in ('pedestrian', 'bicycle', 'cone/barrier')
           and o['x'] < 40]
    lead = min((o for o in veh if rel_dir(o) in ('stationary', 'moving the same way')),
               key=lambda o: o['x'], default=None)
    vru = min((o for o in objs if o['in_path'] and o['kind'] in ('pedestrian', 'bicycle') and o['x'] < 25),
              key=lambda o: o['x'], default=None)
    if vru is None:                         # crossing / oncoming vehicle in the path
        vru = min((o for o in veh if o is not lead and o['x'] < 25 and
                   rel_dir(o) not in ('stationary', 'moving the same way')),
                  key=lambda o: o['x'], default=None)
    slow = F['dec'] or F['stop']
    if slow and vru is not None:
        lon, cause = 'Yield (agent right-of-way)', vru
    elif lead is not None and (slow or (lead['moving'] and lead['speed'] is not None
                                        and abs(lead['speed'] - v0) < 2) or (v0 < 0.2 and not lead['moving'])):
        lon, cause = 'Lead obstacle following', lead
    elif F['stop'] or (v0 < 0.2 and F['stay']):
        lon, cause = 'Stop for static constraints', None
    elif F['dec'] and abs(F['dpsi']) > 30:
        lon, cause = 'Speed adaptation (road events)', None
    else:
        lon, cause = 'Set speed tracking', None
    if abs(F['dpsi']) > 30:
        lat = 'Turn ' + ('left' if F['dpsi'] > 0 else 'right')
    elif 2.5 <= abs(F['y_end']) <= 5 and abs(F['dpsi']) < 15:
        lat = 'Lane change ' + ('left' if F['y_end'] > 0 else 'right')
    elif v0 < 0.2 and F['stay']:
        lat = 'None'
    else:
        lat = 'Lane keeping & centering'
    return lon, lat, cause, lead, vru


def describe(o):
    sp = f'{o["speed"]:.1f} m/s' if o['speed'] is not None else 'speed unknown (first frame)'
    return f'{o["kind"]} {o["x"]:.0f} m ahead, {o["y"]:+.1f} m lateral, {rel_dir(o)} ({sp})'


def compose(lon, lat, cause, ego, objs, F, cmd, cons):
    act = {'Yield (agent right-of-way)': 'slow down and yield',
           'Lead obstacle following': 'keep a safe gap',
           'Stop for static constraints': 'stop and hold',
           'Speed adaptation (road events)': 'slow down',
           'Set speed tracking': 'proceed at the target speed'}[lon]
    if lon.startswith('Lead') and cause is not None and not cause['moving']:
        act = 'stay stopped' if v0_of(ego) < 0.2 else ('stop behind it' if F['stop'] else 'slow down')
    why = ''
    if cause is not None:
        why = f' because the {cause["kind"]} {cause["x"]:.0f} m ahead in the ego path is {rel_dir(cause)}'
        if lon.startswith('Yield'):
            why += ' and has right of way'
    elif lon.startswith('Stop'):
        why = (' with no lead vehicle in the path; the reason (signal, sign or queue) is '
               'not in the annotations')
    elif lon.startswith('Speed'):
        why = f' for the upcoming {"left" if F["dpsi"] > 0 else "right"} turn'
    elif cons:
        why = ', passing the construction zone'
    else:
        why = ' because the path ahead is clear'
    lt = {'None': '', 'Lane keeping & centering': ', keeping the lane'}.get(lat)
    if lt is None:
        d = lat.split()[-1]
        lt = f', turning {d}' if lat.startswith('Turn') else f', changing lanes to the {d}'
    return f'{act[0].upper()}{act[1:]}{lt}{why}.'


def v0_of(ego):
    return ego['v0']


def main():
    inst, S, A, by_s = load()
    M = pickle.load(open(f'{DATA}/ar1_meta.pkl', 'rb'))
    W = {r['sample_token']: r for r in pickle.load(open(f'{DATA}/w1_data.pkl', 'rb'))['records']}
    V0 = pickle.load(open(f'{DATA}/w1_v0.pkl', 'rb'))
    H = pickle.load(open(f'{DATA}/ar1_hist.pkl', 'rb'))
    sig = {}
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    can = NuScenesCanBus(dataroot=ROOT)
    rows = {}
    for st, r in W.items():
        if r['n_fut'] < 12:
            continue
        sc = r['scene_name']
        if sc not in sig:
            try:
                m = can.get_messages(sc, 'vehicle_monitor')
            except Exception:
                m = []
            sig[sc] = (np.array([x['utime'] for x in m]), m)
        ut, m = sig[sc]; i = bisect.bisect_right(ut, S[st]) - 1
        ts = 'unknown (no CAN)' if i < 0 else ('left' if m[i]['left_signal'] else
                                               'right' if m[i]['right_signal'] else 'off')
        e4 = records.ego_state_w1(r['past_poses'], r['current_pose'], V0[st])
        v0 = float(V0[st]['v0_can']); yr = float(V0[st]['yr_can'])
        ego = {'v0': v0, 'v_1s': float(e4[1, 0]), 'signal': ts,
               'k': yr / v0 if v0 >= 1.0 else 0.0}
        objs = objects(r, inst, S, A, by_s, ego['k'])
        F = future(r, M[st])
        lon, lat, cause, lead, vru = decide(ego, objs, F)
        cons = sum(o['kind'] == 'cone/barrier' and 0 < o['x'] < 30 and abs(o['y']) < 6 for o in objs) >= 3
        rows[st] = dict(r=r, ego=ego, objs=objs, F=F, lon=lon, lat=lat, cause=cause, lead=lead,
                        vru=vru, cons=cons, split=r['split'])
    for s in ('train', 'holdout', 'val'):
        R = [x for x in rows.values() if x['split'] == s]
        print(f'\n{s} (n_fut = 12, n = {len(R)}) decision distribution:')
        for k in ('lon', 'lat'):
            c = collections.Counter(x[k] for x in R)
            print('  ' + '; '.join(f'{n} {v} ({v / len(R):.3f})' for n, v in c.most_common()))
    # 20 examples: train, stratified by lon decision, then lat turn / lane change, seed 0
    rs = np.random.RandomState(0)
    tr = [st for st, x in rows.items() if x['split'] == 'train']
    quota = [('Lead obstacle following', None, 4), ('Stop for static constraints', None, 3),
             ('Yield (agent right-of-way)', None, 3), ('Speed adaptation (road events)', None, 2),
             ('Set speed tracking', 'Lane keeping & centering', 3), (None, 'Turn', 3),
             (None, 'Lane change', 2)]
    pick = []
    for lo, la, n in quota:
        pool = [st for st in tr if (lo is None or rows[st]['lon'] == lo)
                and (la is None or rows[st]['lat'].startswith(la)) and st not in pick]
        pick += list(rs.choice(pool, min(n, len(pool)), replace=False))
    with open(OUT, 'w') as f:
        f.write('AR1_COC_EXAMPLES -- 20 template CoC traces for human review (phase R1.4)\n'
                'Generator: Alpamayo/code/ar1/coc_template.py (rules in its header). Train split,\n'
                'stratified by decision, seed 0. Inputs = history window only (t0-2 s .. t0);\n'
                'the DECISION is labelled from the future 6 s; [future] lines are shown only so a\n'
                'reviewer can check the decision and are not part of the trace. Ego frame: x forward,\n'
                'y left. Routing is the VAD command = PRIVILEGED (from the GT future).\n'
                'Review questions per trace: decision correct? cause correct? anything missing?\n')
        for n, st in enumerate(pick, 1):
            x = rows[st]; r = x['r']; e = x['ego']; F = x['F']
            near = sorted([o for o in x['objs'] if o['kind'] != 'cone/barrier' and 0 < o['x'] < 40
                           and abs(o['y']) < 12], key=lambda o: math.hypot(o['x'], o['y']))[:3]
            f.write('\n' + '-' * 89 + '\n')
            f.write(f'#{n:02d}  {r["scene_name"]}  sample {st}\n')
            f.write(f'     CAM_FRONT t0: {H[st]["cams"]["CAM_FRONT"][-1][0].split("/")[-1]}\n')
            f.write(f'  critical components (history only):\n')
            f.write(f'     ego: {e["v0"]:.1f} m/s now, {e["v_1s"]:.1f} m/s 1 s ago; turn signal {e["signal"]}\n')
            f.write(f'     lead in path: {describe(x["lead"]) if x["lead"] else "none within 40 m"}\n')
            f.write(f'     yield cand.: {describe(x["vru"]) if x["vru"] else "none within 25 m"}\n')
            for o in near:
                if o is not x['lead'] and o is not x['vru']:
                    f.write(f'     nearby: {describe(o)}\n')
            f.write(f'     construction zone: {"yes" if x["cons"] else "no"}\n')
            f.write(f'     routing [privileged]: {CMD[r["command"]]}\n')
            f.write(f'     not observable: traffic-light state, signs, lane lines\n')
            f.write(f'  driving decision: lon = {x["lon"]}; lat = {x["lat"]}\n')
            f.write(textwrap.fill(compose(x['lon'], x['lat'], x['cause'], e, x['objs'], F,
                                          CMD[r['command']], x['cons']), 89,
                                  initial_indent='  trace: ', subsequent_indent='         ') + '\n')
            f.write(f'  [future] min / end speed {F["vmin"]:.1f} / {F["vend"]:.1f} m/s, heading '
                    f'{F["dpsi"]:+.0f} deg, end x {F["x_end"]:.0f} m y {F["y_end"]:+.1f} m\n')
    print(f'\nwrote {OUT} ({len(pick)} traces)')


if __name__ == '__main__':
    main()
