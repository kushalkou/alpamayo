"""ar1/r4_diag.py -- R4-DIAG analysis (CPU; definitions fixed in R4_DIAG.md header).
  python ar1/r4_diag.py  > Alpamayo/r4_diag.txt"""
import sys, math, pickle, bisect, collections
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from r33_eval import derive, controls
from meta_actions import LON, LAT
import finetune_meta as FM

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
ROOT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
NC = {42: 'A3', 123: 'G6C_s123', 2024: 'G6C_s2024'}
B1 = {42: 'B1', 123: 'B1_s123', 2024: 'B1_s2024'}
M1 = {42: 'M1', 123: 'M1_s123', 2024: 'M1_s2024'}
_, V0, T = E.data()


def dmax(P):
    return np.linalg.norm(P[:, :6], axis=2).max(1)


def gt_d(recs):
    return dmax(np.stack([np.asarray(T[r['sample_token']]['P'])[:6] for r in recs]))


def pred_d(tag, recs, split='val', tau_tag=None):
    return dmax(preds(load(tag, split), recs, 'hybrid', Q.run(tau_tag or tag)['tau']))


def rates(g, p):
    gs, gm = g < 0.5, g > 1.0
    return len(g), gm.mean(), (p[gs] > 1.0).mean(), (p[gm] < 0.5).mean(), int(gs.sum()), int(gm.sum())


def ci(recs, x):
    c = E.scene_boot(recs, x, np.zeros(len(x)))
    return f'{c[0]:.3f} [{c[1]:.3f},{c[2]:.3f}]'


def stop_time(recs):
    """seconds stopped before t0 (speed <= 0.5 m/s), CAN pose; None if no source."""
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    import json
    can = NuScenesCanBus(dataroot=ROOT)
    S = {s['token']: s['timestamp'] for s in json.load(open(f'{ROOT}/v1.0-trainval/sample.json'))}
    cache, out, src = {}, [], collections.Counter()
    for r in recs:
        sc = r['scene_name']
        if sc not in cache:
            try:
                m = can.get_messages(sc, 'pose')
            except Exception:
                m = []
            cache[sc] = (np.array([x['utime'] for x in m]), np.array([x['vel'][0] for x in m]))
        ut, v = cache[sc]; t0 = S[r['sample_token']]
        i = bisect.bisect_right(ut, t0) - 1
        if i < 0:
            out.append(None); src['none'] += 1; continue
        mv = np.where(np.abs(v[:i + 1]) > 0.5)[0]
        out.append((t0 - (ut[mv[-1]] if len(mv) else ut[0])) / 1e6); src['can'] += 1
    return out, src


def main():
    C = Q.ctx(); rva = C['rva']
    stat = C['st'] == 'stationary'; ff = C['ff']
    R = [r for r, s in zip(rva, stat) if s]
    g = gt_d(R); gs = g < 0.5
    PD = {(nm, sd): pred_d(t, R) for nm, D in (('no cam', NC), ('B1', B1), ('M1', M1)) for sd, t in D.items()}
    ffs = ff[stat]
    print('PART 1a  val stationary: n, GT-go rate, false-go, missed-go (ALL); false-go excl. first frames')
    for (nm, sd), p in PD.items():
        n, go, fg, mg, ns, ng = rates(g, p)
        n2, _, fg2, _, _, _ = rates(g[~ffs], p[~ffs])
        print(f'  {nm:6} s{sd:<5} n {n} (stopped {ns}, goes {ng})  GT-go {go:.3f}  false-go {fg:.3f}  '
              f'missed-go {mg:.3f} | excl.ff n {n2} false-go {fg2:.3f}')
    for nm in ('no cam', 'B1', 'M1'):
        fgs = [rates(g, PD[(nm, sd)])[2] for sd in (42, 123, 2024)]
        mgs = [rates(g, PD[(nm, sd)])[3] for sd in (42, 123, 2024)]
        print(f'  {nm:6} pooled  false-go {np.mean(fgs):.3f}  missed-go {np.mean(mgs):.3f}')

    print('\nPART 1b  TRAIN 50-scene subset, stationary (314); tau from each model\'s holdout')
    tr_all = {r['sample_token']: r for r in E.split_records('train')}
    pooled = {}
    for nm, D in (('no cam', NC), ('B1', B1)):
        v = []
        for sd, t in D.items():
            d = load(f'{t}_tr50', 'train')['data']
            Rt = [tr_all[s] for s in sorted(d)]
            gt_ = gt_d(Rt); p = dmax(preds({'data': d}, Rt, 'hybrid', Q.run(t)['tau']))
            n, go, fg, mg, ns, ng = rates(gt_, p); v.append(fg)
            print(f'  {nm:6} s{sd:<5} n {n} (stopped {ns}, goes {ng})  GT-go {go:.3f}  false-go {fg:.3f}  missed-go {mg:.3f}')
        pooled[nm] = np.mean(v)
        print(f'  {nm:6} pooled  false-go {pooled[nm]:.3f}')
    for nm, t, tt in (('no cam s42', 'A3_trsub', 'A3'), ('B1 s42', 'B1_trsub', 'B1')):
        d = load(t, 'train')['data']; Rt = [tr_all[s] for s in sorted(d) if V0[s]['v0_can'] < 0.5]
        n, go, fg, mg, ns, ng = rates(gt_d(Rt), dmax(preds({'data': d}, Rt, 'hybrid', Q.run(tt)['tau'])))
        print(f'  side: R3.0b random-1000 train dump, {nm}: stationary n {n} (stopped {ns}) false-go {fg:.3f}')
    b1v = np.mean([rates(g, PD[('B1', sd)])[2] for sd in (42, 123, 2024)])
    print(f'  G1 inputs: B1 train false-go {pooled["B1"]:.3f}, B1 val false-go {b1v:.3f}')

    # per-sample pooled false-go (GT-stopped samples)
    Rs = [r for r, k in zip(R, gs) if k]
    FGb = np.stack([PD[('B1', sd)][gs] > 1.0 for sd in (42, 123, 2024)]).astype(float)
    pfg = FGb.mean(0)
    print(f'\nPART 1c  GT-stopped val stationary samples {len(Rs)}; pooled B1 false-gos {int(FGb.sum())}')
    sc = np.array([r['scene_name'] for r in Rs])
    cnt = collections.Counter()
    for s_, k in zip(sc, FGb.sum(0)):
        cnt[s_] += k
    scenes = sorted(set(sc), key=lambda s_: (-cnt[s_], s_))
    k10 = math.ceil(0.1 * len(scenes)); top = set(scenes[:k10])
    share = sum(cnt[s_] for s_ in top) / max(FGb.sum(), 1)
    base = np.isin(sc, list(top)).mean()
    print(f'  scenes with GT-stopped samples {len(scenes)}; top 10% = {k10} scenes hold {share:.3f} of false-gos '
          f'(they hold {base:.3f} of GT-stopped samples); scenes with >= 1 false-go: '
          f'{sum(cnt[s_] > 0 for s_ in scenes)}')

    print('\nPART 1d  B1 pooled false-go by factor (GT-stopped val stationary; scene bootstrap CI)')
    import coc_template as CT
    CT._init_globals(); G = CT._G
    from shapely.geometry import Point
    lead, near = [], {'stop line': [], 'crosswalk': [], 'intersection': [], 'any': []}
    for r in Rs:
        W = G['W'][r['sample_token']]; mn = G['loc'][W['scene_name']]
        if mn not in G['maps']:
            G['maps'][mn] = CT.Map(mn)
        M = G['maps'][mn]; e = CT.ego_pose(W)
        lane0 = M.lane_at(e[0], e[1], e[2])
        path = M.succ(lane0, 1) if lane0 else set()
        objs = CT.objects(W, G['inst'], G['S'], G['A'], G['by_s'], M, path)
        lead.append(any(o['in_path'] and o['kind'] != 'cone/barrier' and 0 < o['x'] < 20 for o in objs))
        pt = Point(e[0], e[1]); fl = {}
        for nm_, L in (('stop line', 'stop_line'), ('crosswalk', 'ped_crossing'), ('intersection', 'road_segment')):
            fl[nm_] = any(M.poly[L][M.tok[L].index(t)].distance(pt) <= 10 for t in M.in_radius(e[0], e[1], 10, (L,)))
            near[nm_].append(fl[nm_])
        near['any'].append(any(fl.values()))
    lead = np.array(lead)
    st_t, src = stop_time(Rs)
    cmd = np.array([r['command'] for r in Rs]); ffr = np.array([ff[[x['sample_token'] for x in rva].index(r['sample_token'])] for r in Rs])
    def line(nm_, mask):
        sub = [r for r, k in zip(Rs, mask) if k]
        return f'    {nm_:34} n {int(mask.sum()):4d}  false-go {ci(sub, pfg[mask]) if mask.sum() else "-"}'
    print(f'  all GT-stopped: n {len(Rs)} false-go {ci(Rs, pfg)}')
    print(line('in-path lead < 20 m: yes', lead)); print(line('in-path lead < 20 m: no', ~lead))
    for k, v in near.items():
        v = np.array(v)
        print(line(f'within 10 m of {k}: yes', v)); print(line(f'within 10 m of {k}: no', ~v))
    for c_, n_ in ((1, 'left'), (0, 'right'), (2, 'straight')):
        print(line(f'nav command {n_} [P]', cmd == c_))
    print(line('first frame: yes', ffr)); print(line('first frame: no', ~ffr))
    stt = np.array([np.nan if x is None else x for x in st_t])
    for lo, hi, n_ in ((0, 2, '0-2 s'), (2, 5, '2-5 s'), (5, 1e9, '> 5 s')):
        print(line(f'stopped before t0 {n_}', (stt >= lo) & (stt < hi)))
    print(line('stopped before t0 unknown (no CAN)', np.isnan(stt)) + f'   (sources {dict(src)})')

    print('\nPART 1e  overlap of B1 false-go sets (GT-stopped val stationary)')
    sets = {sd: {r['sample_token'] for r, k in zip(Rs, FGb[i]) if k} for i, sd in enumerate((42, 123, 2024))}
    for a, b in ((42, 123), (42, 2024), (123, 2024)):
        print(f'  s{a} vs s{b}: |A| {len(sets[a])} |B| {len(sets[b])} Jaccard {len(sets[a] & sets[b]) / len(sets[a] | sets[b]):.3f}')
    I = sets[42] & sets[123] & sets[2024]; U = sets[42] | sets[123] | sets[2024]
    print(f'  3-way: intersection {len(I)}, union {len(U)}, Jaccard {len(I) / len(U):.3f}')

    print('\nPART 2a  B1 s42 camera shuffle among the 993 val stationary samples (perm seed 99)')
    d_sh = load('B1_statshuf', 'val')['data']; assert set(d_sh) == {r['sample_token'] for r in R}
    P0 = preds(load('B1', 'val'), R, 'hybrid', Q.run('B1')['tau']); P1 = preds({'data': d_sh}, R, 'hybrid', Q.run('B1')['tau'])
    Pa = preds(load('B1_camshuf', 'val'), R, 'hybrid', Q.run('B1')['tau'])
    Gp = np.stack([np.asarray(T[r['sample_token']]['P'])[5] for r in R])
    for nm_, P in (('unshuffled', P0), ('shuffled among stationary', P1), ('side: all-val shuffle (R2.4)', Pa)):
        n, go, fg, mg, _, _ = rates(g, dmax(P))
        print(f'  {nm_:30} false-go {fg:.3f}  missed-go {mg:.3f}  L2@3s mean {np.linalg.norm(P[:, 5] - Gp, axis=1).mean():.3f}')
    f0, f1_ = rates(g, dmax(P0))[2], rates(g, dmax(P1))[2]
    print(f'  G3 input: |shuffled - real| = {abs(f1_ - f0) * 100:.1f} points')

    print('\nPART 3a  meta-action labels: 10 Hz CAN vs 2 Hz GT-trajectory derivation (n_fut = 12)')
    Mt = pickle.load(open(f'{DATA}/ar1_meta.pkl', 'rb')); st_ = {'missing': 0}
    maj = np.array([4] * 6 + [6] * 6)
    for split, recs in (('val', [r for r in rva if r['n_fut'] == 12]),
                        ('train', [r for r in E.split_records('train') if r['n_fut'] == 12])):
        can = np.array([sum(FM.meta_labels(Mt, r['sample_token'], st_), []) for r in recs])
        new = np.array([derive(T[r['sample_token']]['acc'][:12], T[r['sample_token']]['cur'][:12],
                               T[r['sample_token']]['v0']) for r in recs])
        ag = (can == new).mean(0)
        print(f'  {split} n {len(recs)} per-slot agreement lon ' + ' '.join(f'{x:.3f}' for x in ag[:6]) + ' | lat '
              + ' '.join(f'{x:.3f}' for x in ag[6:]) + f' | min {ag.min():.3f}, mean {ag.mean():.3f}')
        for h, o, nmz in (('lon', 0, LON), ('lat', 6, LAT)):
            a_ = np.bincount(can[:, o:o + 6].ravel(), minlength=7) / can[:, o:o + 6].size
            b_ = np.bincount(new[:, o:o + 6].ravel(), minlength=7) / new[:, o:o + 6].size
            print(f'    {h} class shares  CAN ' + ' '.join(f'{nmz[c]} {a_[c]:.3f}' for c in range(7)))
            print(f'    {h} class shares  2Hz ' + ' '.join(f'{nmz[c]} {b_[c]:.3f}' for c in range(7)))
        print(f'    all-majority share CAN {(can == maj).all(1).mean():.3f}, 2 Hz {(new == maj).all(1).mean():.3f}; '
              f'GT-vs-GT consistency (GT-trajectory-derived vs labels) CAN {(new == can).all(1).mean():.3f}, 2 Hz '
              f'{(new == new).all(1).mean():.3f}')

    print('\nPART 3b  M1 (seeds pooled), stationary val: generated words; L2@3s by slot-1 accel word == GT')
    recs_s = R
    can_s = np.array([sum(FM.meta_labels(Mt, r['sample_token'], st_), []) for r in recs_s])
    new_s = np.array([derive(T[r['sample_token']]['acc'][:12], T[r['sample_token']]['cur'][:12], T[r['sample_token']]['v0'])
                      if r['n_fut'] == 12 else [-9] * 12 for r in recs_s])
    gens, l2s, okc, okn = [], [], [], []
    for sd, t in M1.items():
        dv = load(t, 'val')['data']
        gen = np.array([dv[r['sample_token']]['meta_gen'] for r in recs_s]); gens.append(gen)
        P = preds(load(t, 'val'), recs_s, 'hybrid', Q.run(t)['tau'])
        l2s.append(np.linalg.norm(P[:, 5] - Gp, axis=1)); okc.append(gen[:, 0] == can_s[:, 0]); okn.append(gen[:, 0] == new_s[:, 0])
    gen = np.concatenate(gens); l2 = np.concatenate(l2s); okc = np.concatenate(okc); okn = np.concatenate(okn)
    v12 = np.tile(new_s[:, 0] >= 0, 3)
    for o, nmz, h in ((0, LON, 'lon'), (6, LAT, 'lat')):
        g1 = np.bincount(gen[:, o], minlength=7) / len(gen); ga = np.bincount(gen[:, o:o + 6].ravel(), minlength=7) / gen[:, o:o + 6].size
        lab = np.bincount(can_s[:, o:o + 6].ravel(), minlength=7) / can_s[:, o:o + 6].size
        print(f'  {h} slot t+1 generated: ' + ' '.join(f'{nmz[c]} {g1[c]:.3f}' for c in range(7) if g1[c] > 0))
        print(f'  {h} all slots generated: ' + ' '.join(f'{nmz[c]} {ga[c]:.3f}' for c in range(7) if ga[c] > 0)
              + ' | CAN labels: ' + ' '.join(f'{nmz[c]} {lab[c]:.3f}' for c in range(7) if lab[c] > 0))
    print(f'  slot-1 accel word == CAN label: share {okc.mean():.3f}; L2@3s mean match {l2[okc].mean():.3f} (n {okc.sum()}) '
          f'vs mismatch {l2[~okc].mean():.3f} (n {(~okc).sum()})')
    print(f'  side, vs 2 Hz label (n_fut = 12): share {okn[v12].mean():.3f}; L2@3s match {l2[v12 & okn].mean():.3f} vs '
          f'mismatch {l2[v12 & ~okn].mean():.3f}')

    print('\nPART 3c  consistency (trajectory-derived words == generated words, all 12 slots)')
    allc = []
    for sd, t in M1.items():
        dv = load(t, 'val')['data']
        gen = np.array([dv[r['sample_token']]['meta_gen'] for r in rva])
        der = np.array([derive(a, k, T[r['sample_token']]['v0']) for (a, k), r in
                        zip(controls(load(t, 'val'), rva, Q.run(t)['tau']), rva)])
        c = (der == gen).all(1); allc.append(c)
        print(f'  M1 s{sd:<5} all val {c.mean():.3f} | stationary {c[stat].mean():.3f} | moving {c[~stat].mean():.3f}')
    c = np.concatenate(allc); s3 = np.tile(stat, 3)
    print(f'  pooled    all val {c.mean():.3f} | stationary {c[s3].mean():.3f} | moving {c[~s3].mean():.3f}  (G5 input)')


if __name__ == '__main__':
    main()
