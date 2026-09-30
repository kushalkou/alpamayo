"""w1/records.py -- WEEK1 STEP 3: model-ready trajectory dicts for the official split.

Each dict is shaped so the UNCHANGED dataset / finetune / ar_eval code does the right
thing through small patches (see finetune_w1.py):
  cam_paths                       6 camera images (unchanged)
  w1_ego      float32 [R,4]       rows 0-3: causal ego state (ego_state_w1 below);
                                  row 4.. : extra tokens [class,0,0,0] (command /
                                  meta-action), consumed by extra_tokens.install()
  w1_tokens   12 (a,k) pairs      tokenizer (ii) tokens of the lidar-frame targets
  future_positions [12,2]         lidar-frame GT trajectory (t0 LIDAR frame)
  current_pose.translation = 0    so ar_eval's gt_local = future_positions
  future_speeds[0] = v0_can       ar_eval seeds the rollout with this
  future_curvatures               the OLD CAM-based curvatures, used ONLY by the
                                  turn-weighted sampler ("old weights")
  command, meta_lon               true labels (for eval/strata); rows carry the
                                  possibly-flipped ones

EGO FEATURES (causal; STEP 1c): for the last 4 pose indices i <= 0 (CAM ego poses):
  speed_i = |p_i - p_i-1|/dt, yaw_i - yaw_0 + pi/2 (relative; current row = pi/2, the
  rollout heading), yaw_rate_i = wrap(yaw_i - yaw_i-1)/dt,
  accel_i = (speed_i - speed_i-1)/dt. Unavailable values = 0 (causal fallback).
  The CURRENT row is replaced by CAN (v0_can, pi/2, yr_can, a_can) when a CAN message
  <= t0 exists; otherwise it keeps the backward pose values (already the v0_can
  fallback). Rows are left-padded by repeating the earliest row.
META-ACTION (A2, privileged oracle): lon from the GT lidar trajectory:
  v3 = s_5 (segment speed over 2.5-3.0 s), dv = v3 - v0_can;
  stop if v3 < 0.5; else accelerate if dv > +1.0; decelerate if dv < -1.0; else maintain.
  lat = command. Flips: with probability f each label is replaced by a uniformly
  random DIFFERENT class (fixed per sample, seeded).
"""
import math, pickle
import numpy as np, torch
from dataset import pose_to_xyyaw

DT = 0.5
YAW0 = math.pi / 2
D = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
LON = ('stop', 'accelerate', 'decelerate', 'maintain')
N_CLS = {'cmd': 3, 'lon': 4, 'lat': 3}


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def ego_state_w1(past_poses, current_pose, can):
    """past_poses/current_pose: pose dicts; can: w1_v0 entry (t0-causal CAN values)."""
    P = [pose_to_xyyaw(p) for p in list(past_poses) + [current_pose]]
    m = len(P); y0 = P[-1][2]
    sp = [0.0] * m; yr = [0.0] * m; ac = [0.0] * m
    for i in range(1, m):
        sp[i] = math.hypot(P[i][0] - P[i-1][0], P[i][1] - P[i-1][1]) / DT
        yr[i] = wrap(P[i][2] - P[i-1][2]) / DT
    for i in range(2, m):
        ac[i] = (sp[i] - sp[i-1]) / DT
    if m >= 3:
        ac[1] = ac[2]
    rows = [[sp[i], wrap(P[i][2] - y0) + YAW0, yr[i], ac[i]] for i in range(m)]
    if m >= 2:
        rows = rows[1:]                    # row 0 has no backward difference
    if can.get('can_ok', False):
        rows[-1] = [can['v0_can'], YAW0, can['yr_can'], can['a_can']]
    while len(rows) < 4:
        rows = [rows[0]] + rows
    return torch.tensor(rows[-4:], dtype=torch.float32)


def meta_lon(P, s, v0):
    v3 = float(s[5]); dv = v3 - v0
    if v3 < 0.5: return 0
    if dv > 1.0: return 1
    if dv < -1.0: return 2
    return 3


def flip(label, n, f, rs):
    if rs.rand() < f:
        return int(rs.choice([c for c in range(n) if c != label]))
    return label


def build(split, extra=('cmd',), flip_rate=0.0, flip_seed=0, n_fut=12):
    """extra: ordered subset of ('cmd', 'lon', 'lat') -> rows 4, 5, ...
    flip_rate applies to 'lon' and 'lat' only (meta-action), never to 'cmd'."""
    W = pickle.load(open(f'{D}/w1_data.pkl', 'rb'))
    V0 = pickle.load(open(f'{D}/w1_v0.pkl', 'rb'))
    T = pickle.load(open(f'{D}/w1_targets.pkl', 'rb'))['targets']
    rs = np.random.RandomState(flip_seed)
    out = []
    for r in W['records']:
        if r['split'] != split or r['n_fut'] < n_fut:
            continue
        st = r['sample_token']; t = T[st]; can = V0[st]
        ego = ego_state_w1(r['past_poses'], r['current_pose'], can)
        lab = {'cmd': r['command'], 'lat': r['command'], 'lon': meta_lon(t['P'], t['s'], t['v0'])}
        shown = {k: (flip(v, N_CLS[k], flip_rate, rs) if k in ('lon', 'lat') else v)
                 for k, v in lab.items()}
        rows = [torch.tensor([[float(shown[k]), 0, 0, 0]]) for k in extra]
        out.append({
            'sample_token': st, 'scene_name': r['scene_name'], 'cam_paths': r['cam_paths'],
            'w1_ego': torch.cat([ego] + rows, 0) if rows else ego,
            'w1_tokens': [tuple(int(x) for x in p) for p in t['tokens'][:12]],
            'future_positions': np.array(t['P'][:12]),
            'future_speeds': np.concatenate([[t['v0']], t['s'][1:]]),
            'future_curvatures': np.array(r['future_curvatures']),
            'current_pose': {'translation': np.zeros(3), 'rotation': np.array([1.0, 0, 0, 0])},
            'past_poses': [], 'command': r['command'], 'meta_lon': lab['lon'],
            'shown': shown, 'n_fut': r['n_fut'], 'can_ok': can['can_ok'],
            'v0': t['v0'], 's': t['s'], 'acc': t['acc'], 'cur': t['cur'],
        })
    return out
