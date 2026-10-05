"""ar1/r5_trackA.py -- R5 track A parts A1, A2, A3, A6, A7 (CPU; rules in R5_REPORT.md header).
  python ar1/r5_trackA.py [a2 a1 a3 a6 a7]"""
import sys, json, math, pickle, bisect, collections
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q, records
from gate_a import load, preds
import finetune_meta as FM

ROOT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
NC = ('A3', 'G6C_s123', 'G6C_s2024'); B1 = ('B1', 'B1_s123', 'B1_s2024'); M1 = ('M1', 'M1_s123', 'M1_s2024')
_W, V0, T = E.data()
SPLIT = {r['scene_name']: r['split'] for r in _W['records']}


def wss_scenes():
    """scene -> True if every consecutive keyframe pair has LIDAR ego-pose speed < 0.5 m/s."""
    J = lambda n: json.load(open(f'{ROOT}/v1.0-trainval/{n}.json'))
    sc = {s['token']: s for s in J('scene')}; S = {s['token']: s for s in J('sample')}
    lid = {}
    for d in J('sample_data'):
        if d['is_key_frame'] and d['filename'].startswith('samples/LIDAR_TOP/'):
            lid[d['sample_token']] = d['ego_pose_token']
    EP = {e['token']: e['translation'] for e in J('ego_pose')}
    out, desc = {}, {}
    for s in sc.values():
        tok, P, t = s['first_sample_token'], [], []
        while tok:
            P.append(EP[lid[tok]][:2]); t.append(S[tok]['timestamp'] / 1e6); tok = S[tok]['next']
        P = np.array(P); t = np.array(t)
        v = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.diff(t)
        out[s['name']] = bool((v < 0.5).all()); desc[s['name']] = s['description']
    return out, desc


def gmax(recs):
    return np.linalg.norm(np.stack([np.asarray(T[r['sample_token']]['P'])[:6] for r in recs]), axis=2).max(1)


def near_flags(recs):
    import coc_template as CT
    from shapely.geometry import Point
    CT._init_globals(); G = CT._G; out = []
    for r in recs:
        W = G['W'][r['sample_token']]; mn = G['loc'][W['scene_name']]
        if mn not in G['maps']:
            G['maps'][mn] = CT.Map(mn)
        M = G['maps'][mn]; e = CT.ego_pose(W); pt = Point(e[0], e[1])
        out.append(any(M.poly[L][M.tok[L].index(t)].distance(pt) <= 10
                       for L in ('stop_line', 'ped_crossing', 'road_segment') for t in M.in_radius(e[0], e[1], 10, (L,))))
    return np.array(out)


def a2(wss, desc):
    print('A2  WSS scenes (every keyframe pair < 0.5 m/s) and "parking" scenes per split')
    for sp in ('train', 'holdout', 'val'):
        scs = sorted(s for s in SPLIT if SPLIT[s] == sp)
        w = [s for s in scs if wss[s]]
        n_w = sum(1 for r in _W['records'] if r['split'] == sp and wss[r['scene_name']])
        n_all = sum(1 for r in _W['records'] if r['split'] == sp)
        park = [s for s in scs if 'parking' in desc[s].lower()]
        print(f'  {sp:8} scenes {len(scs):4d}  WSS scenes {len(w):3d}  WSS samples {n_w:5d} of {n_all:5d}')
        print(f'           WSS: {", ".join(w) if w else "none"}')
        print(f'           "parking" ({len(park)}): ' + ', '.join(f'{s}{"*" if wss[s] else ""}' for s in park)
              + '   (* = WSS)')
    print('  stationary samples (v0_can < 0.5), GT-go rate within 3 s, near vs away (10 m of stop line /'
          ' crosswalk / intersection)')
    for sp in ('train', 'val'):
        R = [r for r in E.split_records(sp) if V0[r['sample_token']]['v0_can'] < 0.5 and r['n_fut'] >= 6]
        g = gmax(R) > 1.0; nr = near_flags(R); wv = np.array([wss[r['scene_name']] for r in R])
        print(f'  {sp:5} n {len(R):5d}  near n {nr.sum():5d} GT-go {g[nr].mean():.3f} | away n {(~nr).sum():5d} '
              f'GT-go {g[~nr].mean():.3f} | in WSS scenes n {wv.sum():4d} GT-go {g[wv].mean() if wv.any() else float("nan"):.3f}')
    ntr = sum(wss[s] for s in SPLIT if SPLIT[s] == 'train')
    print(f'  GA2 input: train WSS scenes = {ntr}')


def a1(wss):
    print('\nA1  R4 shuffle (B1 s42, val stationary): false-go of GT-stopped recipients by donor x recipient type')
    keep = set(pickle.load(open(f'{DATA}/r4_val_stationary.pkl', 'rb')))
    R = [t for t in records.build('val', ['cmd'], 0.0, n_fut=6) if t['sample_token'] in keep]
    perm = np.random.RandomState(99).permutation(len(R))
    recs = {r['sample_token']: r for r in E.split_records('val')}
    Rr = [recs[t['sample_token']] for t in R]
    d = load('B1_statshuf', 'val')
    P = np.linalg.norm(preds(d, Rr, 'hybrid', Q.run('B1')['tau'])[:, :6], axis=2).max(1)
    P0 = np.linalg.norm(preds(load('B1', 'val'), Rr, 'hybrid', Q.run('B1')['tau'])[:, :6], axis=2).max(1)
    g = gmax(Rr); gs = g < 0.5
    rw = np.array([wss[r['scene_name']] for r in Rr]); dw = rw[perm]
    print('  recipient \\ donor        WSS donor            non-WSS donor        (unshuffled false-go)')
    for nm, m in (('WSS recipient', rw), ('non-WSS recipient', ~rw), ('all recipients', np.ones(len(Rr), bool))):
        c = []
        for dm in (dw, ~dw):
            k = gs & m & dm
            c.append(f'{(P[k] > 1.0).mean():.3f} (n {k.sum():3d})' if k.any() else '-')
        print(f'  {nm:22} {c[0]:20} {c[1]:20} {(P0[gs & m] > 1.0).mean():.3f}')
    k1, k2 = gs & dw, gs & ~dw
    diff = (P[k1] > 1.0).mean() - (P[k2] > 1.0).mean()
    print(f'  GA1 input: false-go(WSS donor) - false-go(non-WSS donor), all GT-stopped recipients = {diff * 100:+.1f} points')


def a3(wss):
    print('\nA3  paired scene bootstrap, L2@3s NoAvg and ADE@6s; (b) and (c) are POST-HOC')
    C = Q.ctx(); rva = C['rva']; ff = C['ff']; w = np.array([wss[r['scene_name']] for r in rva])
    cut = lambda p, m: {k: v[m] for k, v in Q.strip(p).items()}
    mean = lambda tags: Q.avg([Q.run(t) for t in tags])
    pairs = (('B1 - no cam (3-seed means)', mean(B1), mean(NC)), ('M1 - B1 (3-seed means)', mean(M1), mean(B1)),
             ('B2 - B1 (seed 42)', Q.strip(Q.run('B2')), Q.strip(Q.run('B1'))))
    res = {}
    for sub_nm, base in (('(a) all', np.ones(len(rva), bool)), ('(b) excl. WSS scenes [POST-HOC]', ~w),
                         ('(c) WSS scenes only [POST-HOC]', w)):
        for ff_nm, fm in (('with first frames', np.ones(len(rva), bool)), ('excl. first frames', ~ff)):
            m = base & fm
            sub = [r for r, k in zip(rva, m) if k]
            print(f'  {sub_nm}, {ff_nm}: n {m.sum()} ({len(set(r["scene_name"] for r in sub))} scenes)')
            for nm, a_, b_ in pairs:
                A, Bm = cut(a_, m), cut(b_, m)
                c1 = E.scene_boot(sub, A['l2'][:, 5], Bm['l2'][:, 5]); c2 = E.scene_boot(sub, A['ade'], Bm['ade'])
                res[(sub_nm[:3], ff_nm[:4], nm[:2])] = c1
                print(f'    {nm:28} L2@3s {c1[0]:+.3f} [{c1[1]:+.3f},{c1[2]:+.3f}]  ADE6 {c2[0]:+.3f} [{c2[1]:+.3f},{c2[2]:+.3f}]')
    c = res[('(b)', 'with', 'B1')]
    print(f'  GA3 input: (b) with first frames, B1 - no cam L2@3s {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]')


def a6():
    print('\nA6  M1 (old 10 Hz labels), stationary val, first speed word (slot t0+1 s): generated (cols) vs GT CAN (rows),'
          ' seeds pooled')
    from meta_actions import LON
    C = Q.ctx(); R = [r for r, s in zip(C['rva'], C['st'] == 'stationary') if s]
    Mt = pickle.load(open(f'{DATA}/ar1_meta.pkl', 'rb')); st = {'missing': 0}
    gt = np.array([FM.meta_labels(Mt, r['sample_token'], st)[0][0] for r in R])
    gen = np.concatenate([[load(t, 'val')['data'][r['sample_token']]['meta_gen'][0] for r in R] for t in M1])
    gt3 = np.tile(gt, 3)
    cls = [c for c in range(7) if (gt3 == c).any() or (gen == c).any()]
    print('    GT \\ gen      ' + ' '.join(f'{LON[c][:9]:>10}' for c in cls) + '   total')
    for a in cls:
        print(f'    {LON[a][:12]:12} ' + ' '.join(f'{int(((gt3 == a) & (gen == b)).sum()):10d}' for b in cls)
              + f'   {int((gt3 == a).sum()):5d}')
    print(f'    accuracy {np.mean(gen == gt3):.3f} over {len(gen)} sample-seeds')


def a7():
    print('\nA7  CoC "turn signal" cause: source and coverage')
    import inspect, coc_template as CT
    src = inspect.getsource(CT._scene_rows)
    ln = [l.strip() for l in src.split('\n') if 'vehicle_monitor' in l or 'left_signal' in l]
    print('  code (ar1/coc_template.py _scene_rows): ' + ' || '.join(ln))
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    can = NuScenesCanBus(dataroot=ROOT)
    S = {s['token']: s['timestamp'] for s in json.load(open(f'{ROOT}/v1.0-trainval/sample.json'))}
    cache = {}
    for sp in ('train', 'val'):
        R = E.split_records(sp); cov = on = l = rr = 0; dts = []
        for r in R:
            sc = r['scene_name']
            if sc not in cache:
                try:
                    m = can.get_messages(sc, 'vehicle_monitor')
                except Exception:
                    m = []
                cache[sc] = (np.array([x['utime'] for x in m]), m)
            ut, m = cache[sc]; i = bisect.bisect_right(ut, S[r['sample_token']]) - 1
            if i >= 0:
                cov += 1; dts.append((S[r['sample_token']] - ut[i]) / 1e6)
                l += m[i]['left_signal']; rr += m[i]['right_signal']
        print(f'  {sp:5} samples {len(R)}: message <= t0 available {cov / len(R):.3f}; signal on: left '
              f'{l / max(cov, 1):.3f}, right {rr / max(cov, 1):.3f}; message age median {np.median(dts):.2f} s, '
              f'p95 {np.percentile(dts, 95):.2f} s (vehicle_monitor is 2 Hz)')


if __name__ == '__main__':
    parts = sys.argv[1:] or ['a2', 'a1', 'a3', 'a6', 'a7']
    wss, desc = wss_scenes() if {'a1', 'a2', 'a3'} & set(parts) else (None, None)
    if wss is not None:
        pickle.dump(wss, open(f'{DATA}/r5_wss.pkl', 'wb'))
    for p in parts:
        {'a2': lambda: a2(wss, desc), 'a1': lambda: a1(wss), 'a3': lambda: a3(wss), 'a6': a6, 'a7': a7}[p]()
