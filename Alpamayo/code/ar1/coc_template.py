"""ar1/coc_template.py -- R2.6: template Chain-of-Causation (CoC) traces, v2 (map-aware).

v1 (R1.4, commit 62d63cc) had no map. v2 adds the nuScenes map expansion v1.3 (lanes,
lane connectors, stop lines with their type, crosswalks, intersection road segments).
Structure follows AR1 Tables 1-3: (1) driving decision, closed set (Table 1), labelled
from the FUTURE 6 s; (2) critical components (Table 2) from the HISTORY WINDOW ONLY
(keyframe annotations t0 .. t0-1 s, CAN <= t0) plus the static map (a prior, not
future information); (3) one composed trace. For human review only; writes no training
data.

Map helpers (all at t0 from the ego pose):
  ego lane    lane / lane connector within 3 m whose arcline heading is within 45 deg
              of the ego heading, nearest by lateral distance
  path lanes  ego lane + successors (outgoing graph) to depth 3 = every branch the ego
              could take (the chosen branch would need the route, which is privileged)
  in path     (v3, R3.4b) object ahead (0 < x < 50 m) whose centre is within 1.5 m + half
              its width of the ego's REALIZED 6 s path (GT future positions; used for the
              label only, like AR1's labeller that sees the future). If the ego covers
              < 5 m: centre inside the ego lane or its direct successors and |y| < 2.5 m.
  crosswalk   a crosswalk ahead only counts if it intersects the realized path.
  lanes here  lanes / connectors within 8 m, heading within 30 deg of the ego
  stop line   nearest stop line ahead (0 < x < 40 m, |y| < 6 m) with its type
              (STOP_SIGN, TRAFFIC_LIGHT, PED_CROSSING, YIELD, TURN_STOP)
  crosswalk   nearest crosswalk centroid ahead (0 < x < 25 m, |y| < 8 m)
  turn        (v3) heading change accumulated INSIDE intersection polygons > 30 deg; a
              bend within a lane is lane keeping.
  intersection  the ego, or a point on its future path, lies on an intersection road
              segment (future use is only for the decision, never for the components)
Objects: annotations at t0, visibility >= 40%, >= 1 lidar point; velocity from the same
  instance 1.0 s (else 0.5 s) earlier; moving if speed > 0.5 m/s; relative motion from
  the velocity angle in the ego frame. nuScenes moving / parked attributes are NOT used.
Leads / yield candidates:
  lead     nearest in-path object that is stationary (vehicles only) or moving the same
           way (any class, so a pedestrian or cyclist ahead in the lane can be a lead)
  yield    nearest in-path pedestrian / cyclist that is crossing, oncoming or
           stationary, or in-path vehicle that is crossing / oncoming, within 25 m;
           else a pedestrian standing on the crosswalk ahead within 25 m
Decision (future; meta-actions from ar1/meta_actions.py):
  lat  Turn L/R        heading change > 30 deg, or > 20 deg through an intersection
       Lane change L/R final lane not reachable from the ego lane (depth 4), heading
                       change < 30 deg, lateral offset > 2 m
       None            ego stationary for the whole window
       Lane keeping    otherwise
  lon  Yield           decel / stop ahead and a yield candidate
       Passing         lane change with a slower or stationary in-path lead
                       (Acceleration for passing / overtaking if the ego speeds up,
                       else set speed with the lead as the stated cause of the change)
       Lead following  lead within 40 m and (decel / stop ahead, |dv| < 2 m/s, or both
                       stationary)
       Stop (static)   ego stops ahead or stays stopped; cause = stop line type /
                       crosswalk / unknown
       Speed adaptation decel ahead and a turn
       Set speed       otherwise
  Multi-phase: if the ego stops and later moves above 1 m/s, the trace says
  "..., then proceed".
Output: AR1_COC_EXAMPLES.txt -- the SAME 20 samples as v1 (tokens from commit 62d63cc)
for a direct comparison. --all also prints the decision distribution per split.
"""
import sys, json, math, pickle, bisect, collections, textwrap, subprocess
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import records

ROOT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
OUT = '/home/dgx1user/Alpamayo-Kushal/AR1_COC_EXAMPLES.txt'
CMD = {0: 'turn right', 1: 'turn left', 2: 'go straight'}
STOP_T = {'STOP_SIGN': 'stop-sign', 'TRAFFIC_LIGHT': 'traffic-light', 'PED_CROSSING': 'crosswalk',
          'YIELD': 'yield', 'TURN_STOP': 'turn'}
VRU = ('pedestrian', 'bicycle')


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_ego(p, e):
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


class Map:
    """R3.4a: the devkit's get_records_in_radius / record_on_point scan every record and
    rebuild its shapely polygon on each call (0.32 s per call, ~5 s per sample). Here each
    polygon is built ONCE per map and indexed with an STRtree."""
    LAYERS = ('lane', 'lane_connector', 'road_segment', 'stop_line', 'ped_crossing')

    def __init__(self, name):
        from nuscenes.map_expansion.map_api import NuScenesMap
        from nuscenes.map_expansion import arcline_path_utils as ap
        from shapely.strtree import STRtree
        self.m = NuScenesMap(dataroot=ROOT, map_name=name); self.ap = ap; self._arc = {}
        self.tok, self.poly, self.tree = {}, {}, {}
        for L in self.LAYERS:
            recs = [r for r in getattr(self.m, L) if r.get('polygon_token')]
            if L == 'road_segment':
                recs = [r for r in recs if r['is_intersection']]
            self.tok[L] = [r['token'] for r in recs]
            self.poly[L] = [self.m.extract_polygon(r['polygon_token']) for r in recs]
            self.tree[L] = STRtree(self.poly[L])
        self.rec = {L: {r['token']: r for r in getattr(self.m, L)} for L in ('stop_line', 'ped_crossing')}
        self.lpoly = {t: p for L in ('lane', 'lane_connector') for t, p in zip(self.tok[L], self.poly[L])}

    def in_radius(self, x, y, r, layers):
        from shapely.geometry import box
        q = box(x - r, y - r, x + r, y + r)
        return [self.tok[L][i] for L in layers for i in self.tree[L].query(q, predicate='intersects')]

    def arc(self, tok):
        if tok not in self._arc:
            self._arc[tok] = self.m.get_arcline_path(tok)
        return self._arc[tok]

    def proj(self, tok, x, y):
        """(lateral distance, lane heading at the projection)."""
        p, _ = self.ap.project_pose_to_lane((x, y, 0.0), self.arc(tok))
        return math.hypot(p[0] - x, p[1] - y), p[2]

    def lane_at(self, x, y, yaw, r=3.0, dh=math.pi / 4):
        best = None
        for tok in self.in_radius(x, y, r, ('lane', 'lane_connector')):
            d, h = self.proj(tok, x, y)
            if abs(wrap(h - yaw)) < dh and (best is None or d < best[0]):
                best = (d, tok)
        return best[1] if best else None

    def succ(self, tok, depth):
        out, front = {tok}, [tok]
        for _ in range(depth):
            front = [o for t in front for o in self.m.get_outgoing_lane_ids(t) if o not in out]
            out.update(front)
        return out

    def lanes_here(self, x, y, yaw):
        n = 0
        for tok in self.in_radius(x, y, 8, ('lane', 'lane_connector')):
            d, h = self.proj(tok, x, y)
            n += int(d < 8 and abs(wrap(h - yaw)) < math.pi / 6)
        return n

    def on_intersection(self, x, y):
        from shapely.geometry import Point
        return len(self.tree['road_segment'].query(Point(x, y), predicate='intersects')) > 0

    def in_lanes(self, x, y, lanes):
        from shapely.geometry import Point
        pt = Point(x, y)
        return any(self.lpoly[t].contains(pt) for t in lanes if t in self.lpoly)

    def ahead(self, layer, e, xmax, ymax):
        best = None
        for tok in self.in_radius(e[0], e[1], xmax + 10, (layer,)):
            poly = self.poly[layer][self.tok[layer].index(tok)]
            x, y = to_ego((poly.centroid.x, poly.centroid.y), e)
            if 0 < x < xmax and abs(y) < ymax and (best is None or x < best[0]):
                best = (x, y, self.rec[layer][tok], poly)
        return best


def load():
    J = lambda n: json.load(open(f'{ROOT}/v1.0-trainval/{n}.json'))
    cat = {c['token']: c['name'] for c in J('category')}
    inst = {i['token']: cat[i['category_token']] for i in J('instance')}
    S = {s['token']: s['timestamp'] for s in J('sample')}
    logs = {l['token']: l['location'] for l in J('log')}
    loc = {s['name']: logs[s['log_token']] for s in J('scene')}
    print('loading sample_annotation.json ...', flush=True)
    A = {a['token']: a for a in J('sample_annotation')}
    by_s = collections.defaultdict(list)
    for a in A.values():
        by_s[a['sample_token']].append(a['token'])
    return inst, S, A, by_s, loc


def ego_pose(r):
    return (r['lidar_ep0']['translation'][0], r['lidar_ep0']['translation'][1],
            yaw_of(r['lidar_ep0']['rotation']))


def objects(r, inst, S, A, by_s, M, path):
    e = ego_pose(r); t0 = S[r['sample_token']]; out = []
    for tok in by_s[r['sample_token']]:
        a = A[tok]
        kd = kind(inst[a['instance_token']])
        if kd is None or int(a['visibility_token']) < 2 or a['num_lidar_pts'] < 1:
            continue
        x, y = to_ego(a['translation'], e)
        prev, p = None, a
        for _ in range(2):
            if not p['prev']:
                break
            p = A[p['prev']]
            if (t0 - S[p['sample_token']]) / 1e6 <= 1.05:
                prev = p
        sp = vl = None
        if prev is not None:
            dt = (t0 - S[prev['sample_token']]) / 1e6
            vx = (a['translation'][0] - prev['translation'][0]) / dt
            vy = (a['translation'][1] - prev['translation'][1]) / dt
            sp = math.hypot(vx, vy); c, s = math.cos(e[2]), math.sin(e[2])
            vl = (vx * c + vy * s, -vx * s + vy * c)
        if isinstance(path, tuple):                 # realized future path (LineString, 'line')
            from shapely.geometry import Point as _P
            inp = bool(0 < x < 50 and path[0].distance(_P(a['translation'][0], a['translation'][1]))
                       < 1.5 + a['size'][0] / 2)
        else:                                       # ego (almost) stationary: lanes, |y| < 2.5 m
            inp = bool(0 < x < 50 and abs(y) < 2.5 and path and
                       M.in_lanes(a['translation'][0], a['translation'][1], path))
        out.append({'kind': kd, 'x': x, 'y': y, 'speed': sp, 'vel': vl, 'in_path': inp,
                    'moving': sp is not None and sp > 0.5, 'g': a['translation'][:2],
                    'ann': {'translation': a['translation'], 'size': a['size'], 'rotation': a['rotation']}})
    return out


def rel_dir(o):
    if not o['moving'] or o['vel'] is None:
        return 'stationary'
    ang = math.degrees(math.atan2(o['vel'][1], o['vel'][0]))
    if abs(ang) < 30: return 'moving the same way'
    if abs(ang) > 150: return 'oncoming'
    return 'crossing ' + ('to the left' if ang > 0 else 'to the right')


def future(r, Mt, M, lane0):
    e = ego_pose(r)
    fp = np.asarray(r['future_positions']); fy = np.asarray(r['future_yaws'])
    xf, yf = to_ego(fp[-1], e)
    dpsi = math.degrees(wrap(fy[-1] - e[2]))
    lon = Mt['lon'][1:]; ok = lon >= 0; v = Mt['v'][1:]
    dec = bool(np.isin(lon[:30], (2, 3)).mean() >= 0.3) if ok[:30].any() else False
    stop_f = bool((lon == 5).any())
    stay = bool((lon[ok] == 5).mean() > 0.5) if ok.any() else False
    go_after = False
    if stop_f:
        j = int(np.argmax(lon == 5))
        go_after = bool(np.nanmax(np.r_[v[j:], -1]) > 1.0)
    fin = v[np.isfinite(v)]
    lane_f = M.lane_at(fp[-1][0], fp[-1][1], fy[-1]) if lane0 else None
    lc = (lane0 is not None and lane_f is not None and lane_f not in M.succ(lane0, 4)
          and abs(dpsi) < 30 and abs(yf) > 2.0)
    ins = [M.on_intersection(p[0], p[1]) for p in fp]
    inter = any(ins)
    yy = np.r_[e[2], fy]
    dpsi_int = math.degrees(sum(wrap(yy[k + 1] - yy[k]) for k in range(len(fp)) if ins[k]))
    return {'dpsi': dpsi, 'y_end': yf, 'x_end': xf, 'dec': dec, 'stop': stop_f, 'stay': stay,
            'go_after': go_after, 'vmin': float(fin.min()) if len(fin) else float('nan'),
            'vend': float(fin[-1]) if len(fin) else float('nan'), 'lc': lc, 'inter': inter,
            'speed_up': bool(len(fin) and fin[-1] > Mt['v'][0] + 0.5), 'dpsi_int': dpsi_int}


def decide(ego, objs, F, xw):
    from shapely.geometry import Point
    v0 = ego['v0']
    lead = min((o for o in objs if o['in_path'] and o['x'] < 40 and o['kind'] != 'cone/barrier' and
                ((o['kind'] not in VRU and not o['moving']) or rel_dir(o) == 'moving the same way')),
               key=lambda o: o['x'], default=None)
    yc = min((o for o in objs if o['in_path'] and o['x'] < 25 and o is not lead and o['kind'] != 'cone/barrier'
              and (o['kind'] in VRU or rel_dir(o) not in ('stationary', 'moving the same way'))),
             key=lambda o: o['x'], default=None)
    if yc is None and xw is not None:                       # pedestrian on the crosswalk ahead
        yc = min((o for o in objs if o['kind'] == 'pedestrian' and 0 < o['x'] < 25 and
                  xw[3].contains(Point(*o['g']))), key=lambda o: o['x'], default=None)
    slow = F['dec'] or F['stop']
    if abs(F['dpsi_int']) > 30:                      # R3.4b: turns happen inside intersections
        lat = 'Turn ' + ('left' if F['dpsi_int'] > 0 else 'right')
    elif F['lc']:
        lat = 'Lane change ' + ('left' if F['y_end'] > 0 else 'right')
    elif v0 < 0.2 and F['stay']:
        lat = 'None'
    else:
        lat = 'Lane keeping & centering'
    slower = lead is not None and (not lead['moving'] or (lead['speed'] or 0) < v0 - 0.5)
    if slow and yc is not None:
        lon, cause = 'Yield (agent right-of-way)', yc
    elif lat.startswith('Lane change') and slower:
        lon = 'Acceleration for passing/overtaking' if F['speed_up'] else 'Set speed tracking'
        cause = lead
    elif lead is not None and (slow or (lead['moving'] and lead['speed'] is not None and abs(lead['speed'] - v0) < 2)
                               or (v0 < 0.2 and not lead['moving'])):
        lon, cause = 'Lead obstacle following', lead
    elif F['stop'] or (v0 < 0.2 and F['stay']):
        lon, cause = 'Stop for static constraints', None
    elif F['dec'] and lat.startswith('Turn'):
        lon, cause = 'Speed adaptation (road events)', None
    else:
        lon, cause = 'Set speed tracking', None
    return lon, lat, cause, lead, yc


def describe(o):
    sp = f'{o["speed"]:.1f} m/s' if o['speed'] is not None else 'speed unknown'
    return f'{o["kind"]} {o["x"]:.0f} m ahead, {o["y"]:+.1f} m lat., {rel_dir(o)} ({sp})'


def compose(lon, lat, cause, ego, F, sl, xw, cons):
    if lon.startswith('Yield'):
        act = 'slow down and yield'
    elif lon.startswith('Acceleration'):
        act = 'speed up'
    elif lon.startswith('Lead'):
        act = ('stay stopped' if ego['v0'] < 0.2 else 'stop behind it' if F['stop'] else 'slow down') \
            if not cause['moving'] else 'keep a safe gap'
    elif lon.startswith('Stop'):
        act = 'stay stopped' if ego['v0'] < 0.2 else 'stop'
    elif lon.startswith('Speed'):
        act = 'slow down'
    else:
        act = 'keep the current speed'
    if F['stop'] and F['go_after'] and lon.startswith(('Stop', 'Yield', 'Lead')):
        act += ', then proceed'
    d = lat.split()[-1]
    lt = {'None': '', 'Lane keeping & centering': ', keeping the lane'}.get(
        lat, f', turning {d}' if lat.startswith('Turn') else f', changing lanes to the {d}')
    if cause is not None and lat.startswith('Lane change') and not lon.startswith(('Yield', 'Lead')):
        why = f' to pass the {cause["kind"]} {cause["x"]:.0f} m ahead, which is {rel_dir(cause)}'
    elif cause is not None:
        why = f' because the {cause["kind"]} {cause["x"]:.0f} m ahead in the ego path is {rel_dir(cause)}'
        if lon.startswith('Yield'):
            why += ' and has right of way'
    elif lon.startswith('Stop'):
        if sl is not None:
            why = f' at the {STOP_T.get(sl[2]["stop_line_type"], "marked")} stop line {sl[0]:.0f} m ahead'
            if sl[2]['stop_line_type'] == 'TRAFFIC_LIGHT':
                why += ' (light state not annotated)'
        elif xw is not None:
            why = f' before the crosswalk {xw[0]:.0f} m ahead'
        else:
            why = '; no lead, stop line or crosswalk explains it in the map or annotations'
    elif lon.startswith('Speed'):
        why = f' for the {d} turn' + (' at the intersection' if F['inter'] else '')
    elif cons:
        why = ' through the construction zone'
    else:
        why = ' because the lane ahead is clear'
    return f'{act[0].upper()}{act[1:]}{lt}{why}.'


_G = {}


def _init_globals():
    inst, S, A, by_s, loc = load()
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    _G.update(inst=inst, S=S, A=A, by_s=by_s, loc=loc,
              Mt=pickle.load(open(f'{DATA}/ar1_meta.pkl', 'rb')),
              W={r['sample_token']: r for r in pickle.load(open(f'{DATA}/w1_data.pkl', 'rb'))['records']},
              V0=pickle.load(open(f'{DATA}/w1_v0.pkl', 'rb')), can=NuScenesCanBus(dataroot=ROOT), maps={})


def _scene_rows(job):
    """one scene (worker process; maps built lazily per process)."""
    sc, toks = job
    G = _G; S = G['S']; out = {}
    mn = G['loc'][sc]
    if mn not in G['maps']:
        G['maps'][mn] = Map(mn)
    M = G['maps'][mn]
    try:
        m = G['can'].get_messages(sc, 'vehicle_monitor')
    except Exception:
        m = []
    ut = np.array([x['utime'] for x in m])
    for st in toks:
        r = G['W'][st]; V0 = G['V0']
        i = bisect.bisect_right(ut, S[st]) - 1
        ts = 'unknown (no CAN)' if i < 0 else ('left' if m[i]['left_signal'] else
                                               'right' if m[i]['right_signal'] else 'off')
        e4 = records.ego_state_w1(r['past_poses'], r['current_pose'], V0[st])
        ego = {'v0': float(V0[st]['v0_can']), 'v_1s': float(e4[1, 0]), 'signal': ts}
        e = ego_pose(r)
        lane0 = M.lane_at(e[0], e[1], e[2])
        from shapely.geometry import LineString
        fp = np.asarray(r['future_positions'])[:, :2]
        line = LineString(np.vstack([[e[0], e[1]], fp]))
        if line.length >= 5.0:                      # R3.4b: in path = near the realized path
            path = (line, 'line')
        else:
            path = M.succ(lane0, 1) if lane0 else set()
        objs = objects(r, G['inst'], S, G['A'], G['by_s'], M, path)
        sl = M.ahead('stop_line', e, 40, 6)
        xw = M.ahead('ped_crossing', e, 25, 8)
        if xw is not None and isinstance(path, tuple) and not xw[3].intersects(path[0]):
            xw = None                               # crosswalk not on the ego's path
        F = future(r, G['Mt'][st], M, lane0)
        lon, lat, cause, lead, yc = decide(ego, objs, F, xw)
        cons = sum(o['kind'] == 'cone/barrier' and 0 < o['x'] < 30 and abs(o['y']) < 6 for o in objs) >= 3
        near = sorted([o for o in objs if o['kind'] != 'cone/barrier' and 0 < o['x'] < 40 and abs(o['y']) < 12
                       and o is not lead and o is not yc], key=lambda o: math.hypot(o['x'], o['y']))[:3]
        out[st] = dict(scene=sc, command=r['command'], n_fut=r['n_fut'], ego=ego, F=F, lon=lon, lat=lat,
                       cause=cause, lead=lead, yc=yc, near=near, cons=cons, split=r['split'],
                       sl=sl[:3] if sl else None, xw=xw[:3] if xw else None, lane0=lane0,
                       nl=M.lanes_here(e[0], e[1], e[2]), inter0=M.on_intersection(e[0], e[1]))
    return out


def build_rows(want=None, procs=40, min_fut=12):
    import multiprocessing as mp
    if not _G:
        _init_globals()
    jobs = collections.defaultdict(list)
    for st, r in _G['W'].items():
        if r['n_fut'] >= min_fut and (want is None or st in want):
            jobs[r['scene_name']].append(st)
    rows = {}
    with mp.get_context('fork').Pool(procs) as pool:
        for k, part in enumerate(pool.imap_unordered(_scene_rows, sorted(jobs.items()), chunksize=1)):
            rows.update(part)
            if (k + 1) % 100 == 0:
                print(f'  scenes {k + 1}/{len(jobs)}', flush=True)
    return rows


def components_text(x):
    e = x['ego']; sl = x['sl']
    c = [f'ego {e["v0"]:.1f} m/s (1 s ago {e["v_1s"]:.1f}), turn signal {e["signal"]}',
         f'{x["nl"]} lane(s){", in an intersection" if x["inter0"] else ""}']
    if sl:
        c.append(f'{STOP_T.get(sl[2]["stop_line_type"], "marked")} stop line {sl[0]:.0f} m ahead')
    if x['xw']:
        c.append(f'crosswalk {x["xw"][0]:.0f} m ahead')
    if x['lead']:
        c.append('lead ' + describe(x['lead']))
    if x['yc']:
        c.append('yield candidate ' + describe(x['yc']))
    if x['cons']:
        c.append('construction zone')
    c.append(f'route: {CMD[x["command"]]}')
    return '; '.join(c)


def coc_text(x):
    return (f'Driving decision: {x["lon"]}; {x["lat"]}. Critical components: {components_text(x)}. '
            f'Reasoning: {compose(x["lon"], x["lat"], x["cause"], x["ego"], x["F"], x["sl"], x["xw"], x["cons"])}')


def write_examples(rows, picks):
    H = pickle.load(open(f'{DATA}/ar1_hist.pkl', 'rb'))
    with open(OUT, 'w') as f:
        f.write('AR1_COC_EXAMPLES v2 -- 20 template CoC traces for human review (phase R2.6)\n'
                'Generator: Alpamayo/code/ar1/coc_template.py v2 (map-aware; rules in its header).\n'
                'The SAME 20 train samples as v1 (commit 62d63cc), same numbering. Components use\n'
                'the history window (t0-1 s .. t0) and the static map only; the DECISION is\n'
                'labelled from the future 6 s; [future] lines are for the reviewer and are not\n'
                'part of the trace. Ego frame: x forward, y left. Routing = VAD command [P].\n'
                'Changes vs v1: AR1_R2_REPORT.md, R2.6 (one line per example).\n')
        for n, st in picks:
            x = rows[st]; e = x['ego']; F = x['F']; near = x['near']
            f.write('\n' + '-' * 89 + '\n')
            f.write(f'#{n}  {x["scene"]}  sample {st}\n')
            f.write(f'     CAM_FRONT t0: {H[st]["cams"]["CAM_FRONT"][-1][0].split("/")[-1]}\n')
            f.write('  critical components (history + static map):\n')
            f.write(f'     ego: {e["v0"]:.1f} m/s now, {e["v_1s"]:.1f} m/s 1 s ago; turn signal {e["signal"]}\n')
            f.write(f'     lane: {"mapped" if x["lane0"] else "none within 3 m"}; {x["nl"]} lane(s) in the '
                    f'ego direction; {"in" if x["inter0"] else "not in"} an intersection\n')
            sl = x['sl']
            f.write('     stop line ahead: ' + (f'{STOP_T.get(sl[2]["stop_line_type"], sl[2]["stop_line_type"])} '
                                                f'line {sl[0]:.0f} m ahead' if sl else 'none within 40 m') + '\n')
            f.write('     crosswalk ahead: ' + (f'{x["xw"][0]:.0f} m' if x['xw'] else 'none within 25 m') + '\n')
            f.write(f'     lead in path: {describe(x["lead"]) if x["lead"] else "none within 40 m"}\n')
            f.write(f'     yield cand.: {describe(x["yc"]) if x["yc"] else "none within 25 m"}\n')
            for o in near:
                f.write(f'     nearby: {describe(o)}\n')
            f.write(f'     construction zone: {"yes" if x["cons"] else "no"}\n')
            f.write(f'     routing [privileged]: {CMD[x["command"]]}\n')
            f.write('     not observable: traffic-light state\n')
            f.write(f'  driving decision: lon = {x["lon"]}; lat = {x["lat"]}\n')
            f.write(textwrap.fill(compose(x['lon'], x['lat'], x['cause'], e, F, x['sl'], x['xw'], x['cons']),
                                  89, initial_indent='  trace: ', subsequent_indent='         ') + '\n')
            f.write(f'  [future] min / end speed {F["vmin"]:.1f} / {F["vend"]:.1f} m/s, heading '
                    f'{F["dpsi"]:+.0f} deg, end x {F["x_end"]:.0f} m y {F["y_end"]:+.1f} m'
                    f'{", via intersection" if F["inter"] else ""}\n')


AUDIT = '/home/dgx1user/Alpamayo-Kushal/AR1_COC_AUDIT.txt'


def write_audit(rows, n=100, seed=0):
    """100 random VAL traces, stratified by lon decision (equal share per class present,
    capped by availability; leftovers filled from the remaining pool), seed 0."""
    H = pickle.load(open(f'{DATA}/ar1_hist.pkl', 'rb'))
    rs = np.random.RandomState(seed)
    val = sorted(st for st, x in rows.items() if x['split'] == 'val')
    by = collections.defaultdict(list)
    for st in val:
        by[rows[st]['lon']].append(st)
    cls = sorted(by); q = n // len(cls); pick = []
    for c in cls:
        pick += list(rs.choice(by[c], min(q, len(by[c])), replace=False))
    rest = [st for st in val if st not in set(pick)]
    pick += list(rs.choice(rest, n - len(pick), replace=False))
    pick = sorted(pick, key=lambda st: (rows[st]['lon'], rows[st]['lat'], st))
    with open(AUDIT, 'w') as f:
        f.write('AR1_COC_AUDIT -- 100 random VAL CoC traces for human audit (phase R3.4d)\n'
                'Generator: Alpamayo/code/ar1/coc_template.py v3 (STRtree map index; yield / lead only\n'
                'for agents on the ego path or on a crosswalk the path crosses; turn only for heading\n'
                'change inside map intersection polygons). Stratified by lon decision, seed 0:\n'
                + ''.join(f'  {c}: {sum(rows[st]["lon"] == c for st in pick)} of {len(by[c])}\n' for c in cls) +
                'Causes = critical components (history t0-1 s .. t0 + static map). The decision is\n'
                'labelled from the future 6 s; the [future] line is for the auditor only.\n'
                'Image paths are relative to Alpamayo/nuscenes/. Route = VAD command [P].\n')
        for k, st in enumerate(pick, 1):
            x = rows[st]; F = x['F']
            img = H[st]['cams']['CAM_FRONT'][-1][0].split('nuscenes/')[1]
            f.write('\n' + '-' * 89 + '\n')
            f.write(f'A{k:03d}  {x["scene"]}  sample {st}\n')
            f.write(f'  image:\n    {img}\n')
            f.write(f'  decision: lon = {x["lon"]}; lat = {x["lat"]}\n')
            f.write(textwrap.fill(components_text(x), 89, initial_indent='  causes: ',
                                  subsequent_indent='          ') + '\n')
            f.write(textwrap.fill(compose(x['lon'], x['lat'], x['cause'], x['ego'], F, x['sl'], x['xw'], x['cons']),
                                  89, initial_indent='  trace: ', subsequent_indent='         ') + '\n')
            f.write(f'  [future] min / end speed {F["vmin"]:.1f} / {F["vend"]:.1f} m/s; heading {F["dpsi"]:+.0f} deg '
                    f'({F["dpsi_int"]:+.0f} in intersections);\n           end x {F["x_end"]:.0f} m, y {F["y_end"]:+.1f} m\n')


def main():
    import time
    t0 = time.time()
    if '--all' in sys.argv:
        rows = build_rows(None, min_fut=6)
        print(f'built {len(rows)} rows in {time.time() - t0:.0f} s', flush=True)
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained('/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason')
        out = {}
        for st, x in rows.items():
            tr = compose(x['lon'], x['lat'], x['cause'], x['ego'], x['F'], x['sl'], x['xw'], x['cons'])
            cc = coc_text(x)
            out[st] = {'split': x['split'], 'n_fut': x['n_fut'], 'lon': x['lon'], 'lat': x['lat'], 'trace': tr,
                       'coc': cc, 'n_tok_trace': len(tk(tr)['input_ids']), 'n_tok_coc': len(tk(cc)['input_ids'])}
        pickle.dump(out, open(f'{DATA}/ar1_coc.pkl', 'wb'))
        for s in ('train', 'holdout', 'val'):
            R = [x for x in out.values() if x['split'] == s]
            R12 = [x for x in R if x['n_fut'] == 12]
            print(f'\n{s}: n = {len(R)} (n_fut >= 6; {len(R12)} with n_fut = 12)')
            for k in ('lon', 'lat'):
                c = collections.Counter(x[k] for x in R)
                print(f'  {k}: ' + '; '.join(f'{n} {v} ({v / len(R):.3f})' for n, v in c.most_common()))
            for k in ('n_tok_trace', 'n_tok_coc'):
                v = np.array([x[k] for x in R])
                print(f'  {k}: mean {v.mean():.1f} median {np.median(v):.0f} p95 {np.percentile(v, 95):.0f} '
                      f'max {v.max()}')
        write_audit(rows)
        print(f'wrote {DATA}/ar1_coc.pkl and {AUDIT}; total {time.time() - t0:.0f} s')
        return
    v1 = subprocess.run(['git', '-C', '/home/dgx1user/Alpamayo-Kushal', 'show', '62d63cc:AR1_COC_EXAMPLES.txt'],
                        capture_output=True, text=True, check=True).stdout
    picks = [(l.split()[0][1:], l.split()[3]) for l in v1.splitlines() if l.startswith('#')]
    rows = build_rows({st for _, st in picks})
    write_examples(rows, picks)
    print(f'wrote {OUT}; {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
