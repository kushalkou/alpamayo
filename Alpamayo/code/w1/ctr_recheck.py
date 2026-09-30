"""w1/ctr_recheck.py -- STEP 1a: the old 'constant turn rate 3.014' (v2_characterize.py
CTR_const_yawrate) used yr = compute_ego_state(traj)[3,2] = wrap(future_yaws[0]-yaw0)/dt,
a FUTURE yaw. Recompute with a causal yaw rate wrap(yaw0 - yaw_-1)/dt (0 if no past
pose), same loop, same custom test set (n=3614), same v0 = future_speeds[0]."""
import sys, math, pickle, numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
import shrink_lib as SL
from dataset import compute_ego_state, pose_to_xyyaw
te = pickle.load(open(f'{SL.RES}/leak_split.pkl', 'rb'))['test']
out = {}
for name in ('CV', 'CTR_leaky', 'CTR_causal'):
    A = []
    for t in te:
        e = compute_ego_state(t); v0 = float(t['future_speeds'][0]); yaw0 = float(e[3, 1])
        if name == 'CTR_leaky': yr = float(e[3, 2])
        elif name == 'CTR_causal':
            P = [pose_to_xyyaw(p) for p in list(t['past_poses'])[-1:] + [t['current_pose']]]
            yr = (math.atan2(math.sin(P[-1][2] - P[0][2]), math.cos(P[-1][2] - P[0][2])) / 0.5
                  if len(P) == 2 else 0.0)
        else: yr = 0.0
        x = y = 0.0; yaw = yaw0; Pp = []
        for k in range(12):
            yaw += yr * 0.5; x += v0 * math.cos(yaw) * 0.5; y += v0 * math.sin(yaw) * 0.5
            Pp.append((x, y))
        g = np.array(t['future_positions'])[:12] - np.array(t['current_pose']['translation'][:2])
        A.append(np.linalg.norm(np.array(Pp) - g, axis=1).mean())
    out[name] = np.array(A)
    print(f'{name:11} ADE@6s mean {out[name].mean():.3f}  median {np.median(out[name]):.3f}')
d = SL.paired(out['CTR_causal'], out['CV'])
print(f'CTR_causal - CV {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] p={d[3]:.4f}')
