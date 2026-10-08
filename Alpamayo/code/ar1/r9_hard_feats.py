"""ar1/r9_hard_feats.py -- R9 A2: per-sample HARD-set features for val (all 5,119), from
ar1/coc_template.py functions (objects, future, Map). Definitions in the R9_REPORT.md header.
Output data/r9_hard_val.pkl {sample_token: {dpsi, agent20, inter0, sl20}}. CPU.
  python ar1/r9_hard_feats.py"""
import sys, pickle, time, collections
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import coc_template as CT
import evalw1 as E

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'


def scene(job):
    sc, toks = job
    G = CT._G; S = G['S']; out = {}
    mn = G['loc'][sc]
    if mn not in G['maps']:
        G['maps'][mn] = CT.Map(mn)
    M = G['maps'][mn]
    from shapely.geometry import LineString
    for st in toks:
        r = G['W'][st]; e = CT.ego_pose(r)
        lane0 = M.lane_at(e[0], e[1], e[2])
        fp = np.asarray(r['future_positions'])[:, :2]
        line = LineString(np.vstack([[e[0], e[1]], fp]))
        path = (line, 'line') if line.length >= 5.0 else (M.succ(lane0, 1) if lane0 else set())
        objs = CT.objects(r, G['inst'], S, G['A'], G['by_s'], M, path)
        agent20 = any(o['in_path'] and 0 < o['x'] < 20 and o['kind'] != 'cone/barrier' for o in objs)
        fy = np.asarray(r['future_yaws'])
        dpsi = float(np.degrees(CT.wrap(fy[-1] - e[2])))
        sl = M.ahead('stop_line', e, 20, 6)
        out[st] = {'dpsi': dpsi, 'agent20': bool(agent20), 'inter0': bool(M.on_intersection(e[0], e[1])),
                   'sl20': sl is not None}
    return out


def main():
    import multiprocessing as mp
    CT._init_globals()
    want = {r['sample_token'] for r in E.split_records('val')}
    jobs = collections.defaultdict(list)
    for st in want:
        jobs[CT._G['W'][st]['scene_name']].append(st)
    t0 = time.time(); rows = {}
    with mp.get_context('fork').Pool(24) as pool:
        for part in pool.imap_unordered(scene, sorted(jobs.items()), chunksize=1):
            rows.update(part)
    assert len(rows) == len(want)
    pickle.dump(rows, open(f'{DATA}/r9_hard_val.pkl', 'wb'))
    print(f'{len(rows)} val samples, {len(jobs)} scenes, {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
