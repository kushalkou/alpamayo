"""oracle_floor.py — GATE 2.3: how much can sub-bin interpolation EVER buy?

The 0.885m roundtrip ADE is a HARD-TOKEN floor: it assumes decoding to bin
CENTERS. A soft decode is not restricted to bin centers, so it can go below that.

(i)  centers floor  : GT -> tokenize -> bin CENTER -> unicycle           [= 0.885m]
(ii) oracle sub-bin : GT -> tokenize -> the true continuous value, clipped to the
                      assigned bin's range -> unicycle. This is the best any decode
                      could do given the same bin assignment.
The gap between them bounds the total headroom available to interpolation.
"""
import sys, pickle, numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from tokenizer import TrajectoryTokenizer, STOP_TOKEN
from inference import unicycle_rollout, N_STEPS, HORIZONS
import inference as INF
from dataset import build_scene_split

tok = TrajectoryTokenizer()
ab, cb = tok.accel_bins, tok.curv_bins
ac, cc = tok.accel_centers, tok.curv_centers

with open(INF.TRAJECTORIES_PATH, 'rb') as f: allt = pickle.load(f)
_, _, test = build_scene_split(allt, INF.NUSCENES_ROOT)
print(f"[oracle] test n={len(test)}")

res = {k: {h: [] for h in HORIZONS.values()} for k in ('centers', 'oracle')}
for t in test:
    toks = tok.tokenize(t)
    a_true = np.asarray(t['future_accelerations'][:N_STEPS], dtype=float)
    k_true = np.asarray(t['future_curvatures'][:N_STEPS],   dtype=float)
    a_c, k_c, a_o, k_o = [], [], [], []
    for j, (at, kt) in enumerate(toks[:N_STEPS]):
        if at == STOP_TOKEN:
            a_c.append(0.0); k_c.append(0.0); a_o.append(0.0); k_o.append(0.0); continue
        a_c.append(ac[at]); k_c.append(cc[kt])
        # oracle: the true value, clipped into its own bin (best sub-bin decode)
        a_o.append(float(np.clip(a_true[j], ab[at], ab[at+1])))
        k_o.append(float(np.clip(k_true[j], cb[kt], cb[kt+1])))
    v0 = float(t['future_speeds'][0])
    yaw0 = float(np.arctan2(2*(t['current_pose']['rotation'][0]*t['current_pose']['rotation'][3] +
                               t['current_pose']['rotation'][1]*t['current_pose']['rotation'][2]),
                            1 - 2*(t['current_pose']['rotation'][2]**2 +
                                   t['current_pose']['rotation'][3]**2)))
    gt = np.array(t['future_positions'])[:N_STEPS] - np.array(t['current_pose']['translation'][:2])
    for key, (aa, kk) in (('centers', (a_c, k_c)), ('oracle', (a_o, k_o))):
        pred, _ = unicycle_rollout(np.array(aa), np.array(kk), v0, yaw0)
        for st, lab in HORIZONS.items():
            res[key][lab].append(float(np.linalg.norm(pred[:st]-gt[:st], axis=1).mean()))

print(f"\n=== GATE 2.3 ORACLE SOFT FLOOR (full test set, n={len(test)}) ===")
print(f"{'decode':28} {'ADE1s':>8} {'ADE2s':>8} {'ADE3s':>8} {'ADE6s':>8} {'med6s':>8}")
for key, lab in (('centers', 'bin centers (hard floor)'), ('oracle', 'oracle sub-bin (soft floor)')):
    r = res[key]
    print(f"{lab:28} " + " ".join(f"{np.mean(r[h]):8.3f}" for h in ['1s','2s','3s','6s']) +
          f" {np.median(r['6s']):8.3f}")
g = np.mean(res['centers']['6s']) - np.mean(res['oracle']['6s'])
print(f"\nTOTAL INTERPOLATION HEADROOM @6s = {g:.3f} m "
      f"({np.mean(res['centers']['6s']):.3f} -> {np.mean(res['oracle']['6s']):.3f})")
