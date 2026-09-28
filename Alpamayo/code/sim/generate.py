"""sim/generate.py -- synthetic trajectory benchmark with two dials (p, rho).

Each sample: 4 history rows + 12 future steps, in the SAME conventions as nuScenes
(extract_trajectories.py / shrink_lib.rollout):
  - v0 = speed over the first future segment (== nuScenes future_speeds[0])
  - future_accelerations[0] == 0 always (nuScenes: accels[0] = accels[1] = 0)
  - accel[j] = (s[j] - s[j-1]) / dt, the unicycle rollout then reproduces s[j] exactly
  - tokenized with the unchanged TrajectoryTokenizer (64+64 bins + STOP, speed < 0.1)

Latent intent z = (lon in {brake, hold, accel}, lat in {left, straight, right},
                   accel magnitude A, curvature magnitude K, turn onset t_on)
  lon  ~ uniform over 3 classes
  lat  : P(lat != straight) = p, left/right equally likely
  A    ~ U(A_LO, A_HI) m/s^2                       [HAND-SET -- no nuScenes source]
  K    ~ nuScenes TURNING subset (max|GT curv| > 0.05, dump_test.pkl), drawn from the
         turning samples in the same v0 decile band as this sample's v0, so a
         15 m/s car never gets a 0.4 rad/m turn.
  t_on ~ U{-4..8}: first plateau segment. t_on < 0 => turn already in progress,
         visible in the history yaw-rate.
DIAL rho: with prob rho the observation vector encodes z exactly; otherwise it
  encodes an independent z' from the same prior (same v0), so a fake observation is
  not distinguishable by its marginal statistics.
Future controls = f(z) + iid Gaussian noise (SIG_A on accel command, SIG_K on curvature).

History (ego features, 4 rows x [speed, yaw, yaw_rate, accel]) is built from PAST
poses only. Note: nuScenes compute_ego_state's current row uses future_speeds[1] and
future_yaws[0]; the sim deliberately does NOT reproduce that (it would leak slot 1).

Usage: python generate.py --p 0.5 --rho 1.0 --seed 0
"""
import os, sys, math, pickle, argparse
import numpy as np

sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from shrink_lib import rollout, DT, N

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
OUT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/sim'
SPLITS = (('train', 16763), ('val', 3572), ('test', 3614))   # match nuScenes
N_HIST = 4
A_LO, A_HI = 0.2, 1.0        # accel magnitude, m/s^2 (CV ADE ~ nuScenes scale at p=.2)
SIG_A = 0.3                  # aleatoric accel noise, m/s^2 per step
SIG_K = 0.005                # aleatoric curvature noise, rad/m per step
T_ON = np.arange(-4, 9)      # turn plateau onset (segment index; <0 = in progress)
PLATEAU = 6                  # plateau length, segments (3 s)
CURV_T = 0.05
OBS_DIM = 9                  # onehot lon 3 | onehot lat 3 | A | K | t_on


def nuscenes_priors():
    """v0 samples (full test set) and the turning-subset curvature magnitudes."""
    with open(f'{RES}/dump_test.pkl', 'rb') as f:
        meta = pickle.load(f)['meta']
    idx = sorted(meta)
    v0 = np.array([meta[i]['v0'] for i in idx])
    mc = np.array([meta[i]['maxcurv'] for i in idx])
    turn = mc > CURV_T
    tv, tk = v0[turn], mc[turn]
    edges = np.quantile(tv, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    bands = [tk[(tv >= edges[b]) & (tv < edges[b + 1])] for b in range(10)]
    return v0, edges, bands


def draw_z(rs, p, v0, edges, bands, arange=(A_LO, A_HI)):
    lon = int(rs.randint(3))                                  # 0 brake 1 hold 2 accel
    if rs.rand() < p:
        lat = 0 if rs.rand() < 0.5 else 2                     # 0 left 1 straight 2 right
    else:
        lat = 1
    A = float(rs.uniform(*arange))
    b = int(np.searchsorted(edges, v0, side='right') - 1)
    K = float(rs.choice(bands[min(max(b, 0), 9)]))
    t_on = int(rs.choice(T_ON))
    return (lon, lat, A, K, t_on)


def encode(z, arange=(A_LO, A_HI)):
    lon, lat, A, K, t_on = z
    o = np.zeros(OBS_DIM, np.float32)
    o[lon] = 1; o[3 + lat] = 1
    o[6] = (A - arange[0]) / (arange[1] - arange[0])
    o[7] = K / 0.5
    o[8] = (t_on + 4) / 12.0
    return o


def curv_profile(z):
    """Noise-free curvature on segments -N_HIST .. N-1 (index 0 = first future)."""
    lon, lat, A, K, t_on = z
    seg = np.arange(-N_HIST, N)
    prof = np.zeros(len(seg))
    if lat == 1:
        return seg, prof
    sgn = 1.0 if lat == 0 else -1.0                           # left = +yaw
    for n, s in enumerate(seg):
        if t_on <= s < t_on + PLATEAU:
            prof[n] = K
        elif s == t_on - 1 or s == t_on + PLATEAU:
            prof[n] = 0.5 * K                                  # 1-segment ramps
    return seg, sgn * prof


def make_sample(rs, p, rho, v0, edges, bands, tok, arange=(A_LO, A_HI)):
    z = draw_z(rs, p, v0, edges, bands, arange)
    visible = rs.rand() < rho
    zo = z if visible else draw_z(rs, p, v0, edges, bands, arange)
    obs = encode(zo, arange)
    lon, lat, A, K, t_on = z
    seg, kprof = curv_profile(z)
    k_noisy = kprof + rs.randn(len(seg)) * SIG_K
    yaw0 = float(rs.uniform(-math.pi, math.pi))

    # ---- future: speeds s[0..11], s[0] = v0 ----
    a_cmd = {0: -A, 1: 0.0, 2: A}[lon]
    s = np.empty(N); s[0] = v0
    for j in range(1, N):
        s[j] = max(0.0, s[j - 1] + (a_cmd + rs.randn() * SIG_A) * DT)
    acc = np.zeros(N); acc[1:] = np.diff(s) / DT
    cur = k_noisy[N_HIST:].copy()
    gt = rollout(acc, cur, v0, yaw0)

    # ---- history: constant speed v0, curvature profile on past segments ----
    # poses -4..0; segment i (i=-4..-1) runs pose i -> i+1 at speed v0.
    kh = k_noisy[:N_HIST]
    yaws = np.empty(N_HIST + 1); yaws[-1] = yaw0
    for i in range(N_HIST - 1, -1, -1):
        yaws[i] = yaws[i + 1] - v0 * kh[i] * DT
    # rows t-3..t: [speed, yaw, yaw_rate, accel]; rows t-3..t-1 use the forward past
    # segment, current row uses v0 and the backward yaw-rate (no future leak).
    ego = np.zeros((N_HIST, 4), np.float32)
    for r in range(N_HIST):
        pi = r + 1                                             # pose index -3..0 -> 1..4
        ego[r, 0] = v0
        ego[r, 1] = yaws[pi]
        ego[r, 2] = (v0 * kh[pi] if pi < N_HIST else v0 * kh[N_HIST - 1])
        ego[r, 3] = 0.0
    past_curv = float(np.max(np.abs(ego[:, 2] / max(v0, 1e-3))))

    toks = [tok.tokenize_step(a, k, sp) for a, k, sp in zip(acc, cur, s)]
    tokens = [t[0] for t in toks] + [t[1] for t in toks]      # [a0..a11, k0..k11]
    return {
        'ego': ego, 'obs': obs, 'tokens': np.array(tokens, np.int64),
        'acc': acc, 'cur': cur, 'speeds': s,
        'meta': {'gt': gt.tolist(), 'v0': float(v0), 'yaw0': yaw0,
                 'maxcurv': float(np.abs(cur).max()), 'past_curv': past_curv,
                 'n_hist': N_HIST, 'z': z, 'visible': bool(visible)},
    }


def generate(p, rho, seed, arange=(A_LO, A_HI)):
    import contextlib, io
    from tokenizer import TrajectoryTokenizer
    with contextlib.redirect_stdout(io.StringIO()):
        tok = TrajectoryTokenizer()
    v0_pool, edges, bands = nuscenes_priors()
    out = {}
    for k, (split, n) in enumerate(SPLITS):
        rs = np.random.RandomState([seed, k, int(round(p * 1000)), int(round(rho * 1000))])
        v0s = rs.choice(v0_pool, n, replace=True)
        S = [make_sample(rs, p, rho, float(v), edges, bands, tok, arange) for v in v0s]
        out[split] = {
            'ego': np.stack([x['ego'] for x in S]), 'obs': np.stack([x['obs'] for x in S]),
            'tokens': np.stack([x['tokens'] for x in S]),
            'acc': np.stack([x['acc'] for x in S]), 'cur': np.stack([x['cur'] for x in S]),
            'meta': {i: x['meta'] for i, x in enumerate(S)},
        }
    out['config'] = dict(p=p, rho=rho, seed=seed, A=tuple(arange), SIG_A=SIG_A, SIG_K=SIG_K,
                         T_ON=(int(T_ON[0]), int(T_ON[-1])), PLATEAU=PLATEAU)
    return out


def cell_name(p, rho, seed, arange=(A_LO, A_HI)):
    base = f'p{p:.2f}_rho{rho:.2f}_s{seed}'
    if tuple(arange) != (A_LO, A_HI):
        base += f'_A{arange[0]:g}-{arange[1]:g}'
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--rho', type=float, required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    D = generate(a.p, a.rho, a.seed)
    os.makedirs(OUT, exist_ok=True)
    path = f'{OUT}/{cell_name(a.p, a.rho, a.seed)}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(D, f)
    print(f'[gen] {path}  ' + '  '.join(f'{s}={len(D[s]["meta"])}' for s, _ in SPLITS))


if __name__ == '__main__':
    main()
