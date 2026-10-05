"""ar1/r5_a5c1.py -- R5 A5 (expert re-eval, no training) and C1 (comfort) (CPU; rules in
the R5_REPORT.md header).
Collision: numpy re-implementation of the VAD-port box check (plan_metrics.evaluate_coll:
ego box 4.084 x 1.85 m, +0.5 m forward offset, 0.5 m grid, masked where the GT box
collides). mode 'aa' = axis-aligned (current metric; checked against the stored per-sample
values), mode 'yaw' = the same box rotated by the heading of each predicted segment
(heading of p_t - p_t-1, held if the step is < 0.1 m; t0 heading = forward). The GT mask
uses the axis-aligned GT box in both modes.
  python ar1/r5_a5c1.py [a5] [c1] [--add TAG ...]
"""
import sys, os, math, pickle
import numpy as np
from skimage.draw import polygon
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import evalw1 as E, q2_traj as Q, plan_metrics as PMX
from gate_a import load, preds

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
PM = PMX.PM
BX, DX, BEV = PM.bx.numpy(), PM.dx.numpy(), PM.bev_dimension
_W, V0, T = E.data()


def footprint(psi):
    """pixel offsets (rows along forward, cols along lateral) of the ego box with heading psi
    (lidar frame: x right, y forward; psi = pi/2 is forward)."""
    f = np.array([-PM.H / 2 + 0.5, PM.H / 2 + 0.5, PM.H / 2 + 0.5, -PM.H / 2 + 0.5])
    l = np.array([PM.W / 2, PM.W / 2, -PM.W / 2, -PM.W / 2])
    x = f * math.cos(psi) + l * math.sin(psi); y = f * math.sin(psi) - l * math.cos(psi)
    pts = np.stack([y, x], 1)                       # VAD pre-swap order (forward, lateral)
    pts = (pts - BX) / DX
    pts[:, [0, 1]] = pts[:, [1, 0]]
    rr, cc = polygon(pts[:, 1], pts[:, 0])
    return np.stack([rr, cc], 1)


FP_AA = footprint(math.pi / 2)


def box_coll(traj6, seg, mode):
    out = np.zeros(6, bool); psi = math.pi / 2; prev = np.zeros(2)
    for t in range(6):
        d = traj6[t] - prev
        if mode == 'yaw' and np.hypot(*d) >= 0.1:
            psi = math.atan2(d[1], d[0])
        prev = traj6[t]
        rc = FP_AA if mode == 'aa' else footprint(psi)
        p = np.array([traj6[t, 1], traj6[t, 0]]) / DX + rc
        r = np.clip((BEV[0] - p[:, 0]).astype(np.int32), 0, BEV[0] - 1)
        c = np.clip(p[:, 1].astype(np.int32), 0, BEV[1] - 1)
        out[t] = seg[t, r, c].any()
    return out


def coll_rates(recs, P, occ, mode):
    """P [N,>=6,2] -> (Col NoAvg 3 s veh %, Col TemAvg 3 s veh+ped %), VAD masking."""
    nv, tv = [], []
    for j, r in enumerate(recs):
        gt = np.asarray(r['gt_lidar6'], dtype=np.float32)
        res = {}
        for k in ('veh', 'vp'):
            seg = occ[j][k].numpy()
            g = box_coll(gt, seg, 'aa')
            res[k] = np.where(g, 0, box_coll(np.asarray(P[j][:6], dtype=np.float32), seg, mode))
        nv.append(res['veh'][5]); tv.append(res['vp'].mean())
    return 100 * float(np.mean(nv)), 100 * float(np.mean(tv))


def comfort(P, v0):
    """P [12,2] lidar frame from t0, v0 -> dict of per-bound pass flags + all (R5 header C1)."""
    dt = 0.5
    p = np.vstack([[0.0, 0.0], P[:12]])
    v = np.diff(p, axis=0) / dt                                   # v_1..v_12
    s = np.linalg.norm(v, axis=1)
    h = np.zeros(13); h[0] = math.pi / 2
    for k in range(1, 13):
        h[k] = math.atan2(v[k - 1, 1], v[k - 1, 0]) if s[k - 1] > 0.1 else h[k - 1]
    sp = np.r_[v0, s]
    a = np.diff(sp) / dt                                          # a_1..a_12
    r = np.array([math.atan2(math.sin(h[k] - h[k - 1]), math.cos(h[k] - h[k - 1])) for k in range(1, 13)]) / dt
    lat = s * r
    ya = np.diff(r) / dt
    lj = np.diff(a) / dt
    V = np.vstack([[v0 * math.cos(h[0]), v0 * math.sin(h[0])], v])   # v_0..v_12
    A = np.diff(V, axis=0) / dt                                      # A_1..A_12
    j = np.linalg.norm(np.diff(A[1:], axis=0), axis=1) / dt          # uses A_2..A_12 -> jerk k = 3..12
    f = {'lon_acc': bool(((a >= -4.05) & (a <= 2.40)).all()), 'lat_acc': bool((np.abs(lat) <= 4.89).all()),
         'yaw_rate': bool((np.abs(r) <= 0.95).all()), 'yaw_acc': bool((np.abs(ya) <= 1.93).all()),
         'lon_jerk': bool((np.abs(lj) <= 4.13).all()), 'jerk': bool((np.abs(j) <= 8.37).all())}
    f['all'] = all(f.values())
    return f


def comfort_rates(recs, P):
    F = [comfort(P[i], T[r['sample_token']]['v0']) for i, r in enumerate(recs)]
    return {k: float(np.mean([x[k] for x in F])) for k in F[0]}


def main():
    C = Q.ctx(); rva = C['rva']; occ = C['occ']
    G12 = [np.asarray(T[r['sample_token']]['P']) for r in rva]
    n12 = np.array([r['n_fut'] == 12 for r in rva])
    stat = C['st'] == 'stationary'
    gstop = np.array([np.linalg.norm(g[:6], axis=1).max() < 0.5 for g in G12]) & stat
    tok = {t: preds(load(t, 'val'), rva, 'hybrid', Q.run(t)['tau']) for t in ('A3', 'M1')}
    parts = sys.argv[1:] or ['a5', 'c1']
    if 'a5' in parts:
        # sanity: numpy axis-aligned collision == stored VAD-port per-sample values (A3 token)
        st_ = C['cache'][('A3', 'A3', 0.0)]
        mine = coll_rates(rva, tok['A3'], occ, 'aa')
        ref = (100 * st_['col_veh'][:, 5].mean(), 100 * st_['col_vp'][:, :6].mean())
        print(f'collision check, A3 token: numpy aa {mine[0]:.3f} / {mine[1]:.3f} vs stored {ref[0]:.3f} / {ref[1]:.3f}')
        print('\nA5  experts vs token decoders, official val (ADE@6s on n_fut = 12; ADE@3s on all 5,119)')
        hdr = (f'  {"model / readout":30} {"ADE@3s":>7} {"ADE@6s":>7} {"minADE6 3s/6s":>14} {"spread 3s/6s":>13} '
               f'{"hold":>6} {"Col NoAvg3 aa/yaw":>18} {"Col TemAvg3 aa/yaw":>19}')
        print(hdr)
        def ade(P, h):
            e = np.array([np.linalg.norm(P[i][:h] - G12[i][:h], axis=1).mean() if (h == 6 or n12[i]) else np.nan
                          for i in range(len(rva))])
            return float(np.nanmean(e))
        def hold(P):
            d = np.array([np.linalg.norm(P[i][:6], axis=1).max() for i in range(len(rva))])
            return float((d[gstop] < 0.5).mean())
        def row(nm, P, extra=''):
            ca = coll_rates(rva, P, occ, 'aa'); cy = coll_rates(rva, P, occ, 'yaw')
            print(f'  {nm:30} {ade(P, 6):7.3f} {ade(P, 12):7.3f} {extra:>28} {hold(P):6.3f} '
                  f'{ca[0]:8.2f} / {cy[0]:5.2f}  {ca[1]:8.2f} / {cy[1]:5.2f}')
        for tag, xp in (('A3', 'ar1_expert_A3.pkl'), ('M1', 'ar1_expert_M1.pkl')):
            X = pickle.load(open(f'{RES}/{xp}', 'rb')); S = X['val']['traj']
            assert X['val']['tokens'] == [r['sample_token'] for r in rva]
            row(f'{tag} token decoder', tok[tag], '(1 sample)')
            m3 = np.mean([np.linalg.norm(S[i, :, :6] - G12[i][None, :6], axis=2).mean(1).min() for i in range(len(rva))])
            m6 = np.nanmean([np.linalg.norm(S[i] - G12[i][None, :12], axis=2).mean(1).min() if n12[i] else np.nan
                             for i in range(len(rva))])
            sp = [np.mean([np.linalg.norm(S[i, a, h] - S[i, b, h]) for a in range(6) for b in range(a + 1, 6)])
                  for i in range(len(rva)) for h in (5,)]
            sp6 = [np.mean([np.linalg.norm(S[i, a, 11] - S[i, b, 11]) for a in range(6) for b in range(a + 1, 6)])
                   for i in range(len(rva))]
            ex = f'{m3:.3f} / {m6:.3f}  {np.mean(sp):.2f} / {np.mean(sp6):.2f}'
            # single draw: average the per-draw metrics
            a3 = np.mean([ade(S[:, k], 6) for k in range(6)]); a6 = np.mean([ade(S[:, k], 12) for k in range(6)])
            hd = np.mean([hold(S[:, k]) for k in range(6)])
            cs = np.array([coll_rates(rva, S[:, k], occ, 'aa') + coll_rates(rva, S[:, k], occ, 'yaw') for k in range(6)]).mean(0)
            print(f'  {tag + " expert, single draw (mean of 6)":30} {a3:7.3f} {a6:7.3f} {ex:>28} {hd:6.3f} '
                  f'{cs[0]:8.2f} / {cs[2]:5.2f}  {cs[1]:8.2f} / {cs[3]:5.2f}')
            row(f'{tag} expert, mean of 6', S.mean(1), '')
            sel = np.stack([S[i, np.argmin(np.linalg.norm(S[i] - tok[tag][i][None], axis=2).mean(1))] for i in range(len(rva))])
            row(f'{tag} expert, nearest to token', sel, '')
        print(f'  hold = share of {int(gstop.sum())} GT-stopped stationary samples whose prediction stays < 0.5 m in 3 s; '
              f'spread = mean pairwise L2 between the 6 draws at 3 s / 6 s')
    if 'c1' in parts:
        rec12 = [r for r, k in zip(rva, n12) if k]
        sub = lambda P: [P[i] for i in range(len(rva)) if n12[i]]
        cache_p = f'{RES}/r5_comfort.pkl'
        cache = pickle.load(open(cache_p, 'rb')) if os.path.exists(cache_p) else {}
        rows = [('GT', [G12[i] for i in range(len(rva)) if n12[i]])]
        for t in ('A3', 'G6C_s123', 'G6C_s2024', 'B1', 'B1_s123', 'B1_s2024', 'M1', 'M1_s123', 'M1_s2024', 'B2') + \
                tuple(a for a in sys.argv if a.startswith('M1v2a')):
            if os.path.exists(f'{RES}/w1_dump_{t}_val_f0.0.pkl'):
                rows.append((t, sub(preds(load(t, 'val'), rva, 'hybrid', Q.run(t)['tau']))))
        for tag, xp in (('A3', 'ar1_expert_A3.pkl'), ('M1', 'ar1_expert_M1.pkl')):
            S = pickle.load(open(f'{RES}/{xp}', 'rb'))['val']['traj']
            for k in range(6):
                rows.append((f'{tag} expert draw {k}', sub(S[:, k])))
            rows.append((f'{tag} expert mean of 6', sub(S.mean(1))))
            sel = np.stack([S[i, np.argmin(np.linalg.norm(S[i] - tok[tag][i][None], axis=2).mean(1))] for i in range(len(rva))])
            rows.append((f'{tag} expert nearest to token', sub(sel)))
        print(f'\nC1  comfort (nuPlan bounds, 2 Hz finite differences; val n_fut = 12, n {len(rec12)}): share within bounds')
        keys = ('all', 'lon_acc', 'lat_acc', 'yaw_rate', 'yaw_acc', 'lon_jerk', 'jerk')
        print(f'  {"model":30} ' + ' '.join(f'{k:>8}' for k in keys))
        res = {}
        for nm, P in rows:
            res[nm] = cache.get(nm) or comfort_rates(rec12, P)
        cache.update(res); pickle.dump(cache, open(cache_p, 'wb'))
        draws = {}
        for nm, v in res.items():
            if ' draw ' in nm:
                draws.setdefault(nm.split(' draw')[0], []).append(v); continue
            print(f'  {nm:30} ' + ' '.join(f'{v[k]:8.3f}' for k in keys))
        for nm, vs in draws.items():
            print(f'  {nm + " single draw (mean)":30} ' + ' '.join(f'{np.mean([v[k] for v in vs]):8.3f}' for k in keys))


if __name__ == '__main__':
    main()
