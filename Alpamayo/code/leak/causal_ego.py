"""leak/causal_ego.py -- CAUSAL replacement for dataset.compute_ego_state (OVERNIGHT2 1d/6).

Reads ONLY past poses and the current pose. Rows = the last 4 pose indices i (current
last) with, for pose i:
    speed_i    = |p_i - p_{i-1}| / dt
    yaw_i      = yaw of pose i
    yaw_rate_i = wrap(yaw_i - yaw_{i-1}) / dt
    accel_i    = (speed_i - speed_{i-1}) / dt
The earliest row whose accel is undefined (needs p_{i-2}) copies the next row's accel.
Short histories are left-padded by repeating the earliest row (the original
convention). Samples need >= MIN_PAST past poses (p_-2, p_-1, p_0) -> use keep().

Used ONLY behind the causal flag by finetune_causal.py / dump_ce.py via
monkeypatching; the frozen dataset.py is not edited.
"""
import math
import numpy as np, torch
from dataset import pose_to_xyyaw

DT = 0.5
MIN_PAST = 2


def keep(traj):
    return len(traj.get('past_poses', [])) >= MIN_PAST


def causal_ego_state(traj):
    P = [pose_to_xyyaw(p) for p in list(traj.get('past_poses', [])) + [traj['current_pose']]]
    m = len(P)
    if m < MIN_PAST + 1:
        raise ValueError('causal_ego_state needs >= %d past poses' % MIN_PAST)
    sp = [None] + [math.hypot(P[i][0] - P[i-1][0], P[i][1] - P[i-1][1]) / DT for i in range(1, m)]
    yr = [None] + [math.atan2(math.sin(P[i][2] - P[i-1][2]), math.cos(P[i][2] - P[i-1][2])) / DT
                   for i in range(1, m)]
    ac = [None, None] + [(sp[i] - sp[i-1]) / DT for i in range(2, m)]
    ac[1] = ac[2]
    rows = [[sp[i], P[i][2], yr[i], ac[i]] for i in range(1, m)]
    while len(rows) < 4:
        rows = [rows[0]] + rows
    return torch.tensor(rows[-4:], dtype=torch.float32)


def patch():
    """Monkeypatch dataset.compute_ego_state (NuScenesVLADataset looks it up at call time)."""
    import dataset
    dataset.compute_ego_state = causal_ego_state
