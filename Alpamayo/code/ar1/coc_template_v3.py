"""ar1/coc_template_v3.py -- R11 Part B: CoC v3 = v2 (ar1/coc_template.py, speed_src 2hz, UNCHANGED,
imported) + the audit fixes, each switchable for the per-fix count (B5):
  B1 'route'  : "route: <cmd>" removed from the critical components (cmd stays a model input).
  B2 'bumper' : agents measured from the front bumper (x_f = x - 3.528 m: yaw-aware-metric ego box,
                4.084 m long, centre 0.5 m ahead of the LiDAR, LiDAR 0.986 m ahead of the rear axle);
                lead / yield candidates need x_f > 0 and a centre within 0.925 + 0.5 + half its width of
                the realized GT bumper path (straight corridor |y| <= bound if the path is < 5 m);
                v2's crosswalk-pedestrian yield branch (no corridor test) is disabled;
                distances in the text are from the bumper (map items too, floored at 0);
                stationary whole horizon (v0 < 0.2, GT max displacement <= 0.5 m) -> "stay stopped".
  B3 'turn'   : turn if |dpsi inside intersections| > 30 deg OR |dpsi| > 45 deg.
  B4 'nudge'  : "In-lane nudge left/right" (AR1 Table 1): not a turn / lane change; GT lateral offset
                (ego frame at t0) reaches >= 0.5 m and ends <= 0.5 x that max; a stationary vehicle /
                object outside the corridor but within 3 m of it, on the side away from the nudge,
                0 < x_f <= 30 m, is the cause.
  python ar1/coc_template_v3.py            -> data/ar1_coc_2hz_v3.pkl (+ per-fix change counts)
"""
import sys, math, pickle, time, collections
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import coc_template as CT
from shapely.geometry import LineString, Point

DATA = CT.DATA
BUMP, HW_EGO, MARGIN = 3.528, 0.925, 0.5
FIX = {'route': True, 'bumper': True, 'turn': True, 'nudge': True}


def corridor_bound(a):
    return HW_EGO + MARGIN + a['size'][0] / 2


def bumper_path(r, e):
    fp = np.asarray(r['future_positions'])[:, :2]; fy = np.asarray(r['future_yaws'])
    pts = [(e[0] + BUMP * math.cos(e[2]), e[1] + BUMP * math.sin(e[2]))]
    pts += [(p[0] + BUMP * math.cos(y), p[1] + BUMP * math.sin(y)) for p, y in zip(fp, fy)]
    return LineString(pts)


def objects_v3(r, inst, S, A, by_s, M, e):
    """v2 objects() (velocities, kinds, filters) with bumper distances and the v3 corridor."""
    line = bumper_path(r, e); straight = line.length < 5.0
    out = CT.objects(r, inst, S, A, by_s, M, set())             # in_path recomputed below
    for o in out:
        a = o['ann']; o['x_rear'] = o['x']; o['x'] = o['x'] - BUMP
        b = corridor_bound(a)
        if straight:
            d = abs(o['y'])
        else:
            d = line.distance(Point(a['translation'][0], a['translation'][1]))
        o['d_path'] = d; o['bound'] = b
        o['in_path'] = bool(0 < o['x'] <= 50 and d <= b)
    return out


def nudge(r, e, objs, F):
    fp = np.asarray(r['future_positions'])[:, :2]
    if len(fp) < 2:
        return None
    y = np.array([CT.to_ego(p, e)[1] for p in fp]); k = int(np.argmax(np.abs(y))); m = abs(y[k])
    if m < 0.5 or abs(y[-1]) > 0.5 * m:
        return None
    side = 'left' if y[k] > 0 else 'right'
    cand = [o for o in objs if not o['moving'] and o['kind'] not in CT.VRU and 0 < o['x'] <= 30 and not o['in_path']
            and o['d_path'] <= o['bound'] + 3 and (o['y'] < 0 if side == 'left' else o['y'] > 0)]
    if not cand:
        return None
    return side, min(cand, key=lambda o: o['x'])


def decide_v3(ego, objs, F, xw, nd):
    # B2: candidates must be in the corridor, so v2's 'pedestrian on the crosswalk ahead' branch
    # (any lateral offset) is disabled by passing xw=None; xw still names a crosswalk stop.
    lon, lat, cause, lead, yc = CT.decide(ego, objs, F, None if FIX['bumper'] else xw)
    if FIX['turn'] and not lat.startswith('Turn') and abs(F['dpsi']) > 45:
        lat = 'Turn ' + ('left' if F['dpsi'] > 0 else 'right')
        if lon == 'Set speed tracking' and F['dec']:
            lon = 'Speed adaptation (road events)'     # same v2 rule as for intersection turns
    if FIX['nudge'] and nd is not None and not lat.startswith(('Turn', 'Lane change')):
        lat = f'In-lane nudge {nd[0]}'
    return lon, lat, cause, lead, yc


def compose_v3(lon, lat, cause, ego, F, sl, xw, cons, still, nd):
    tr = CT.compose(lon, lat, cause, ego, F, sl, xw, cons)
    if lat.startswith('In-lane nudge'):
        d = lat.split()[-1]; o = nd[1]
        tr = tr.replace(', keeping the lane', '')
        tr = tr[:-1] + f', nudging {d} within the lane around the stationary {o["kind"]} {o["x"]:.0f} m ahead.'
    if FIX['bumper'] and still:
        for a in ('Slow down and yield', 'Slow down', 'Stop behind it', 'Stop', 'Keep a safe gap', 'Keep the current speed'):
            if tr.startswith(a):
                tr = 'Stay stopped' + (' and yield' if a.endswith('yield') else '') + tr[len(a):]; break
    return tr


def components_v3(x):
    e = x['ego']; sl = x['sl']; f = (lambda d: max(0.0, d - BUMP)) if FIX['bumper'] else (lambda d: d)
    c = [f'ego {e["v0"]:.1f} m/s (1 s ago {e["v_1s"]:.1f}), turn signal {e["signal"]}',
         f'{x["nl"]} lane(s){", in an intersection" if x["inter0"] else ""}']
    if sl:
        c.append(f'{CT.STOP_T.get(sl[2]["stop_line_type"], "marked")} stop line {f(sl[0]):.0f} m ahead')
    if x['xw']:
        c.append(f'crosswalk {f(x["xw"][0]):.0f} m ahead')
    if x['lead']:
        c.append('lead ' + CT.describe(x['lead']))
    if x['yc']:
        c.append('yield candidate ' + CT.describe(x['yc']))
    if x.get('nd'):
        c.append('nudge around ' + CT.describe(x['nd'][1]))
    if x['cons']:
        c.append('construction zone')
    if not FIX['route']:
        c.append(f'route: {CT.CMD[x["command"]]}')
    return '; '.join(c)


def _scene_rows_v3(job):
    import bisect
    sc, toks = job
    G = CT._G; S = G['S']; out = {}
    mn = G['loc'][sc]
    if mn not in G['maps']:
        G['maps'][mn] = CT.Map(mn)
    M = G['maps'][mn]
    try:
        m = G['can'].get_messages(sc, 'vehicle_monitor')
    except Exception:
        m = []
    ut = np.array([q['utime'] for q in m])
    import records
    for st in toks:
        r = G['W'][st]; V0 = G['V0']
        i = bisect.bisect_right(ut, S[st]) - 1
        ts = 'unknown (no CAN)' if i < 0 else ('left' if m[i]['left_signal'] else 'right' if m[i]['right_signal'] else 'off')
        e4 = records.ego_state_w1(r['past_poses'], r['current_pose'], V0[st])
        ego = {'v0': float(V0[st]['v0_can']), 'v_1s': float(e4[1, 0]), 'signal': ts}
        e = CT.ego_pose(r); lane0 = M.lane_at(e[0], e[1], e[2])
        fp = np.asarray(r['future_positions'])[:, :2]
        if FIX['bumper']:
            objs = objects_v3(r, G['inst'], S, G['A'], G['by_s'], M, e)
        else:
            line = LineString(np.vstack([[e[0], e[1]], fp]))
            path = (line, 'line') if line.length >= 5.0 else (M.succ(lane0, 1) if lane0 else set())
            objs = CT.objects(r, G['inst'], S, G['A'], G['by_s'], M, path)
            for o in objs:
                if o['in_path'] or True:
                    a = o['ann']; o['bound'] = corridor_bound(a)
                    o['d_path'] = (LineString(np.vstack([[e[0], e[1]], fp])).distance(Point(*a['translation'][:2]))
                                   if len(fp) else abs(o['y']))
        sl = M.ahead('stop_line', e, 40, 6)
        xw = M.ahead('ped_crossing', e, 25, 8)
        if xw is not None:
            line = LineString(np.vstack([[e[0], e[1]], fp]))
            if line.length >= 5.0 and not xw[3].intersects(line):
                xw = None
        F = CT.future(r, CT.mt2(st), M, lane0)
        nd = nudge(r, e, objs, F) if FIX['nudge'] else None
        lon, lat, cause, lead, yc = decide_v3(ego, objs, F, xw, nd)
        if not lat.startswith('In-lane nudge'):
            nd = None
        cons = sum(o['kind'] == 'cone/barrier' and 0 < o['x'] < 30 and abs(o['y']) < 6 for o in objs) >= 3
        g = np.linalg.norm(fp - np.asarray(e[:2])[None], axis=1).max() if len(fp) else 0.0
        still = ego['v0'] < 0.2 and g <= 0.5
        x = dict(scene=sc, command=r['command'], n_fut=r['n_fut'], ego=ego, F=F, lon=lon, lat=lat, cause=cause,
                 lead=lead, yc=yc, nd=nd, cons=cons, split=r['split'], sl=sl[:3] if sl else None,
                 xw=xw[:3] if xw else None, lane0=lane0, nl=M.lanes_here(e[0], e[1], e[2]),
                 inter0=M.on_intersection(e[0], e[1]), still=still)
        x['trace'] = compose_v3(lon, lat, cause, ego, F, x['sl'], x['xw'], cons, still, nd)
        x['causes'] = components_v3(x)
        x['coc'] = f'Driving decision: {lon}; {lat}. Critical components: {x["causes"]}. Reasoning: {x["trace"]}'
        out[st] = x
    return out


def build_rows_v3(want=None, procs=40):
    import multiprocessing as mp
    jobs = collections.defaultdict(list)
    for st, r in CT._G['W'].items():
        if r['n_fut'] >= 6 and (want is None or st in want):
            jobs[r['scene_name']].append(st)
    rows = {}
    with mp.get_context('fork').Pool(procs) as pool:
        for part in pool.imap_unordered(_scene_rows_v3, sorted(jobs.items()), chunksize=1):
            rows.update(part)
    return rows


def main():
    t0 = time.time()
    CT._init_globals(); CT._G['speed_src'] = '2hz'
    v2 = pickle.load(open(f'{DATA}/ar1_coc_2hz.pkl', 'rb'))
    strip = lambda s: s.replace('; route: go straight', '').replace('; route: turn left', '').replace('; route: turn right', '')
    print(f'route strings in v2 causes: {collections.Counter(c.split("route: ")[-1] for c in (x["causes"] for x in v2.values()))}')
    res = {}
    for name, fx in (('B1 route only', {'route': True, 'bumper': False, 'turn': False, 'nudge': False}),
                     ('B2 bumper only', {'route': False, 'bumper': True, 'turn': False, 'nudge': False}),
                     ('B3 turn only', {'route': False, 'bumper': False, 'turn': True, 'nudge': False}),
                     ('B4 nudge only', {'route': False, 'bumper': False, 'turn': False, 'nudge': True}),
                     ('v3 all fixes', {'route': True, 'bumper': True, 'turn': True, 'nudge': True})):
        FIX.update(fx); rows = build_rows_v3(); res[name] = rows
        assert len(rows) == len(v2)
        dec = np.mean([(rows[t]['lon'], rows[t]['lat']) != (v2[t]['lon'], v2[t]['lat']) for t in v2])
        cau = np.mean([rows[t]['causes'] != v2[t]['causes'] for t in v2])
        cau_s = np.mean([rows[t]['causes'] != strip(v2[t]['causes']) for t in v2])
        trc = np.mean([rows[t]['trace'] != v2[t]['trace'] for t in v2])
        print(f'{name:16} decision changed {dec:.4f}; causes changed {cau:.4f} (vs v2 minus route {cau_s:.4f}); '
              f'trace changed {trc:.4f}  [{time.time() - t0:.0f} s]', flush=True)
    if 'B1 route only' in res:     # sanity: route-only reproduces v2 exactly up to the route clause
        r1 = res['B1 route only']
        assert all(r1[t]['causes'] == strip(v2[t]['causes']) and r1[t]['trace'] == v2[t]['trace'] for t in v2), 'B1 != v2 - route'
        print('check: B1-only rows == v2 with the route clause deleted (decisions, causes, traces): OK')
    v3 = res['v3 all fixes']
    out = {t: {'split': x['split'], 'n_fut': x['n_fut'], 'lon': x['lon'], 'lat': x['lat'], 'trace': x['trace'],
               'coc': x['coc'], 'causes': x['causes']} for t, x in v3.items()}
    pickle.dump(out, open(f'{DATA}/ar1_coc_2hz_v3.pkl', 'wb'))
    pickle.dump(v3, open(f'{DATA}/r11_rows_v3.pkl', 'wb'))
    print(f'wrote {DATA}/ar1_coc_2hz_v3.pkl ({len(out)}; v2 kept) and r11_rows_v3.pkl; {time.time() - t0:.0f} s')
    for k in ('lon', 'lat'):
        a, b = collections.Counter(x[k] for x in v2.values()), collections.Counter(x[k] for x in out.values())
        print(f'{k} distribution v2 -> v3: ' + '; '.join(f'{c} {a[c] / len(v2):.4f} -> {b[c] / len(out):.4f}'
                                                         for c in sorted(set(a) | set(b), key=lambda c: -b[c])))


if __name__ == '__main__':
    main()
