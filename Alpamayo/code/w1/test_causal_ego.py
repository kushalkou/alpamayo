"""w1/test_causal_ego.py -- WEEK1 item 1: causal ego features read no future pose.

Run:  python -m pytest -q w1/test_causal_ego.py   (or python w1/test_causal_ego.py)

1. WHITELIST: the feature function gets a record that raises on ANY key other than
   'past_poses' / 'current_pose'. So no future_*, gt_*, n_fut, ... can be read.
2. PERTURBATION: scrambling every future field leaves the features bit-identical.
3. VALUES: a hand-built constant-acceleration left turn gives the analytic answer.
4. DETECTOR SANITY: the frozen dataset.compute_ego_state FAILS test 1 (it reads
   future_positions / future_yaws / future_speeds) -- proving the test can see a leak.
"""
import sys, copy, math
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/leak')
from causal_ego import causal_ego_state
from pyquaternion import Quaternion

ALLOWED = {'past_poses', 'current_pose'}


class Guarded(dict):
    def __getitem__(self, k):
        if k not in ALLOWED:
            raise AssertionError(f'feature function read forbidden key {k!r}')
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        if k not in ALLOWED:
            raise AssertionError(f'feature function read forbidden key {k!r}')
        return dict.get(self, k, default)


def pose(x, y, yaw):
    q = Quaternion(axis=[0, 0, 1], angle=yaw)
    return {'translation': np.array([x, y, 0.0]), 'rotation': np.array(q.elements)}


def make_traj(n_past=4, v=8.0, a=1.0, r=0.1, dt=0.5):
    """poses at t = -n_past..12 with speed v+a*t and yaw rate r (analytic integration
    on a fine grid). Returns a full record with past, current and future fields."""
    ts = np.arange(-n_past, 13) * dt
    fine = np.linspace(ts[0], ts[-1], 20001)
    sp = v + a * fine; yaw = r * fine
    h = fine[1] - fine[0]
    x = np.concatenate([[0], np.cumsum(sp[:-1] * np.cos(yaw[:-1]) * h)])
    y = np.concatenate([[0], np.cumsum(sp[:-1] * np.sin(yaw[:-1]) * h)])
    idx = [int(round((t - ts[0]) / h)) for t in ts]
    P = [pose(x[i], y[i], yaw[i]) for i in idx]
    cur = n_past
    return {
        'past_poses': P[:cur], 'current_pose': P[cur],
        'future_positions': np.array([p['translation'][:2] for p in P[cur + 1:]]),
        'future_yaws': np.array([r * t for t in ts[cur + 1:]]),
        'future_speeds': np.array([v + a * t for t in ts[cur + 1:]]),
        'future_accelerations': np.full(12, a), 'future_curvatures': np.full(12, 0.0),
        'n_fut': 12, 'gt_lidar6': np.zeros((6, 2)), 'command': 2,
    }


def test_whitelist():
    for n_past in (2, 3, 4):
        causal_ego_state(Guarded(make_traj(n_past)))


def test_perturbation():
    t = make_traj(4)
    base = causal_ego_state(t).numpy()
    rs = np.random.RandomState(0)
    t2 = copy.deepcopy(t)
    for k in list(t2):
        if k.startswith('future') or k in ('gt_lidar6',):
            t2[k] = rs.randn(*np.shape(t2[k])) * 100
    assert np.array_equal(base, causal_ego_state(t2).numpy())


def test_values():
    dt, v, a, r = 0.5, 8.0, 1.0, 0.1
    e = causal_ego_state(make_traj(4, v, a, r, dt)).numpy()
    # current row: backward speed over [-dt,0] = mean speed on that interval
    #   (chord ~ arc for small yaw change); yaw = 0; yaw_rate = r; accel = a
    assert abs(e[3, 0] - (v - a * dt / 2)) < 1e-2, e[3, 0]
    assert abs(e[3, 1] - 0.0) < 1e-9
    assert abs(e[3, 2] - r) < 1e-6
    assert abs(e[3, 3] - a) < 1e-2, e[3, 3]
    assert abs(e[2, 1] - (-r * dt)) < 1e-6          # yaw at t-1


def test_detector_catches_frozen_leak():
    from dataset import compute_ego_state
    try:
        compute_ego_state(Guarded(make_traj(4)))
    except AssertionError as ex:
        assert 'future' in str(ex)
        return
    raise AssertionError('frozen compute_ego_state was NOT caught reading the future')


if __name__ == '__main__':
    for f in (test_whitelist, test_perturbation, test_values, test_detector_catches_frozen_leak):
        f(); print(f'PASS {f.__name__}')
