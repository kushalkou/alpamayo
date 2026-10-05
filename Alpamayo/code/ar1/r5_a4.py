"""ar1/r5_a4.py -- R5 A4: baselines under the Li et al. CVPR 2024 (BEV-Planner) protocol.

Definitions taken from github.com/NVlabs/BEV-Planner (read 2026-10-05):
  planner_head/metric_stp3.py: grid 0.1 m over [-50, 50] m; ego box 4.084 x 1.85 m with
    forward offset 0.5 + 0.985793 m ("distance between the LiDAR and the IMU(ego)"); box
    rotated by the heading of each predicted segment, heading reset to straight when the
    segment is < 1 m; a step is ignored when the GT box collides there.
  planner_head/naive_planner.py: at t = 1, 2, 3 s, L2 = mean over steps <= t (compute_L2,
    i.e. TemAvg); collision = max over steps <= t of the box collision per sample (then
    averaged over samples); evaluate_coll(..., ignore_gt=False).
  Paper: GoStraight = "continues straight at the current velocity"; Ego-MLP = ego velocity,
    acceleration, yaw (rate), driving command -> trajectory.
Ours: the same box / grid / rotation / masking implemented on OUR frame (t0 LIDAR frame
  trajectories and GT boxes from w1_data agents). Differences that remain: their
  trajectories and occupancy live in the ego (IMU) frame of their converter; occupancy
  classes are ours (vehicle; vehicle + human). Their exact converter: UNVERIFIED that it is
  byte-identical to VAD's; "published converter" here = VAD's converter features
  (w1/vad_feats.py, the T4 reproduction).
Rows: GoStraight (v0 from our causal CAN / from the VAD converter), Ego-MLP 3 seeds (our
  causal features / VAD converter features; w1/egomlp_ab.py recipe, CPU, seeded,
  regenerated because predictions were not saved; checked against T4).
Also our VAD-port TemAvg / NoAvg numbers for the same predictions.
  python ar1/r5_a4.py
"""
import sys, math, pickle
import numpy as np, torch, cv2
from skimage.draw import polygon
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import records, evalw1 as E, plan_metrics as PMX
from targets import rollout as roll12
from ego_mlp import train_one
import egomlp_ab as AB

RES, DX, LO, N = 0.1, 0.1, -50.0, 1000
H, Wd = 4.084, 1.85


def occ01(rec, classes):
    seg = np.zeros((6, N, N), np.uint8)
    for a in rec['agents']:
        if a['cls'] not in classes:
            continue
        for t, b in enumerate(a['fut']):
            if b is None:
                continue
            x, y, yaw, l, w = b
            R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            c = R @ np.array([[l / 2, -l / 2, -l / 2, l / 2], [w / 2, w / 2, -w / 2, -w / 2]]) + np.array([[x], [y]])
            col = (c[0] - LO) / DX; row = (-c[1] - LO) / DX            # lidar2cv: y up -> row down
            cv2.fillPoly(seg[t], [np.round(np.stack([col, row], 1)).astype(np.int32)], 1)
    return seg


def box_px(p, psi):
    f = np.array([-H / 2, H / 2, H / 2, -H / 2]) + 0.5 + 0.985793
    l = np.array([Wd / 2, Wd / 2, -Wd / 2, -Wd / 2])
    x = p[0] + f * math.cos(psi) + l * math.sin(psi); y = p[1] + f * math.sin(psi) - l * math.cos(psi)
    rr, cc = polygon((-y - LO) / DX, (x - LO) / DX, (N, N))
    return rr, cc


def coll_steps(traj6, seg):
    out = np.zeros(6, bool); prev = np.zeros(2)
    for t in range(6):
        d = traj6[t] - prev; prev = traj6[t]
        psi = math.atan2(d[1], d[0]) if np.hypot(*d) >= 1.0 else math.pi / 2
        rr, cc = box_px(traj6[t], psi)
        out[t] = seg[t, rr, cc].any() if len(rr) else False
    return out


def li_metrics(recs, preds_by_name, G):
    """per model: L2 TemAvg 1/2/3 s, Li collision (veh+human) 1/2/3 s (%)."""
    names = list(preds_by_name)
    col = {n: np.zeros((len(recs), 3)) for n in names}
    for j, r in enumerate(recs):
        seg = occ01(r, ('vehicle', 'human'))
        g = coll_steps(G[j][:6], seg)
        for n in names:
            c = coll_steps(preds_by_name[n][j][:6], seg) & ~g
            col[n][j] = [c[:2].any(), c[:4].any(), c[:6].any()]
    out = {}
    for n in names:
        e = np.linalg.norm(preds_by_name[n][:, :6] - G[:, :6], axis=2)
        out[n] = [e[:, :k].mean() for k in (2, 4, 6)] + list(100 * col[n].mean(0))
    return out


def main():
    torch.set_num_threads(16)
    tr = records.build('train', ('cmd',)); ho = records.build('holdout', ('cmd',), n_fut=6)
    va = records.build('val', ('cmd',), n_fut=6)
    recs = E.split_records('val'); assert [r['sample_token'] for r in recs] == [t['sample_token'] for t in va]
    W, V0, T = E.data(); VF = AB.VF
    G = np.stack([np.asarray(r['gt_lidar6']) for r in recs])
    P = {'GoStraight, v0 ours (causal CAN)': E.cv(recs),
         'GoStraight, v0 VAD converter': np.stack([roll12(np.zeros(12), np.zeros(12), VF[r['sample_token']][7]) for r in recs])}
    Ytr = np.stack([t['future_positions'] for t in tr]).reshape(len(tr), -1).astype(np.float32)
    Yho = np.stack([np.pad(t['future_positions'], ((0, 12 - len(t['future_positions'])), (0, 0))) for t in ho]).reshape(len(ho), -1)
    mho = np.array([t['n_fut'] == 12 for t in ho])
    for src, F, nm in (('a', AB.fa, 'ours (causal)'), ('b', AB.fb, 'VAD converter')):
        Xtr = F(tr); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        for seed in (42, 123, 2024):
            net, best, bep = train_one(seed, (Xtr - mu) / sd, Ytr, (F(ho) - mu) / sd, Yho, mho, 'cpu')
            with torch.no_grad():
                P[f'Ego-MLP + cmd, {nm}, s{seed}'] = net(torch.tensor((F(va) - mu) / sd)).view(-1, 12, 2).numpy()
            print(f'  [mlp {src}] seed {seed}: holdout ADE@6s {best:.3f} (epoch {bep})', flush=True)
    occ = [PMX.occupancies(r) for r in recs]
    print('\nVAD-port metric (0.5 m grid, axis-aligned box, ALL 5,119): L2 TemAvg 1/2/3 | L2 NoAvg 1/2/3 | '
          'Col TemAvg 1/2/3 | Col NoAvg 1/2/3')
    for n, p in P.items():
        a = E.summary(E.per_sample(recs, p, occ))
        print(f'  {n:40} ' + ' '.join(f'{a[f"L2_TemAvg_{s}s"]:.3f}' for s in (1, 2, 3)) + ' | '
              + ' '.join(f'{a[f"L2_NoAvg_{s}s"]:.3f}' for s in (1, 2, 3)) + ' | '
              + ' '.join(f'{a[f"Col_TemAvg_{s}s"]:.2f}' for s in (1, 2, 3)) + ' | '
              + ' '.join(f'{a[f"Col_NoAvg_{s}s"]:.2f}' for s in (1, 2, 3)))
    print('\nLi et al. protocol (0.1 m grid, yaw-aware box +0.986 m, max-over-horizon collision, L2 TemAvg; ALL 5,119)')
    M = li_metrics(recs, {n: p for n, p in P.items()}, G)
    LI = {'GoStraight': [0.38, 0.79, 1.33, 0.15, 0.60, 2.50], 'Ego-MLP': [0.15, 0.32, 0.59, 0.00, 0.27, 0.85]}
    print(f'  {"model":40} L2 1s   2s   3s   avg  | Col 1s  2s   3s   avg  | delta vs Li Table 1 (L2 avg; Col avg)')
    for n, v in M.items():
        ref = LI['GoStraight'] if n.startswith('GoStraight') else LI['Ego-MLP']
        print(f'  {n:40} {v[0]:.2f} {v[1]:.2f} {v[2]:.2f} {np.mean(v[:3]):.2f} | {v[3]:.2f} {v[4]:.2f} {v[5]:.2f} '
              f'{np.mean(v[3:]):.2f} | {np.mean(v[:3]) - np.mean(ref[:3]):+.2f}; {np.mean(v[3:]) - np.mean(ref[3:]):+.2f}')
    for nm in ('ours (causal)', 'VAD converter'):
        v = np.mean([M[f'Ego-MLP + cmd, {nm}, s{s}'] for s in (42, 123, 2024)], 0)
        print(f'  {"Ego-MLP + cmd, " + nm + ", 3-seed mean":40} {v[0]:.2f} {v[1]:.2f} {v[2]:.2f} {np.mean(v[:3]):.2f} | '
              f'{v[3]:.2f} {v[4]:.2f} {v[5]:.2f} {np.mean(v[3:]):.2f} | {np.mean(v[:3]) - 0.35:+.2f}; '
              f'{np.mean(v[3:]) - np.mean(LI["Ego-MLP"][3:]):+.2f}')
    print('  Li Table 1 (as quoted from the arXiv HTML): GoStraight L2 0.38 0.79 1.33 (0.83), Col 0.15 0.60 2.50; '
          'Ego-MLP L2 0.15 0.32 0.59 (0.35), Col 0.00 0.27 0.85')


if __name__ == '__main__':
    main()
