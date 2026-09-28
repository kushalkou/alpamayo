# OVERNIGHT 2 REPORT -- nuScenes leak audit, sim gates, sim map

(MORNING SUMMARY is filled in at the end of the run; see bottom of this section.)

MORNING SUMMARY
  [pending]

---

## 1. nuScenes LEAK AUDIT

Code: Alpamayo/code/leak/{cache_split,audit}.py (new files; nothing frozen touched).
Log: Alpamayo/overnight2_1_leak.log. CPU only, 40 s.

### 1a. compute_ego_state, verbatim (Alpamayo/code/dataset.py:55-111, body)

    dt = 0.5
    past = list(traj.get('past_poses', []))
    seq_poses = past + [traj['current_pose']]          # chronological, ends at current
    xyy = [pose_to_xyyaw(p) for p in seq_poses]        # [(x,y,yaw), ...]
    fx = float(traj['future_positions'][0][0])
    fy = float(traj['future_positions'][0][1])
    fyaws = traj.get('future_yaws', None)
    fyaw = float(fyaws[0]) if fyaws is not None and len(fyaws) > 0 else xyy[-1][2]
    ext_x   = [p[0] for p in xyy] + [fx]
    ext_y   = [p[1] for p in xyy] + [fy]
    ext_yaw = [p[2] for p in xyy] + [fyaw]
    m = len(xyy)                                        # real poses; current index = m-1
    def ang(a):
        return float(np.arctan2(np.sin(a), np.cos(a)))
    fwd_speed, fwd_yawrate = [], []
    for i in range(m):
        d = float(np.hypot(ext_x[i+1]-ext_x[i], ext_y[i+1]-ext_y[i]))
        fwd_speed.append(d / dt)                        # forward diff => true velocity at i
        fwd_yawrate.append(ang(ext_yaw[i+1]-ext_yaw[i]) / dt)
    fs = traj.get('future_speeds', None)
    states = []
    for i in range(m):
        if i < m - 1:
            accel = (fwd_speed[i+1] - fwd_speed[i]) / dt
        else:                                           # current row: use next future speed
            accel = ((float(fs[1]) - fwd_speed[i]) / dt) if (fs is not None and len(fs) > 1) else 0.0
        states.append([fwd_speed[i], ext_yaw[i], fwd_yawrate[i], accel])
    while len(states) < 4:
        states = [states[0]] + states
    states = states[-4:]
    return torch.tensor(states, dtype=torch.float32)   # [4, 4]

Pose indices: current = 0, past = -1..-3, future_positions[k] = pose k+1, and
future_speeds[k] = |p_{k+1} - p_k| / dt (extract_trajectories.py). Full history (m=4):

    row  feature   reads poses           >= current?   > current (FUTURE)?
    t-3  speed     -3, -2                 no            no
    t-3  yaw       -3                     no            no
    t-3  yaw_rate  -3, -2                 no            no
    t-3  accel     -3, -2, -1             no            no
    t-2  speed     -2, -1                 no            no
    t-2  yaw       -2                     no            no
    t-2  yaw_rate  -2, -1                 no            no
    t-2  accel     -2, -1, 0              YES (0)       no
    t-1  speed     -1, 0                  YES (0)       no
    t-1  yaw       -1                     no            no
    t-1  yaw_rate  -1, 0                  YES (0)       no
    t-1  accel     -1, 0, +1              YES           YES  (fwd_speed[cur] = |p1-p0|)
    t    speed     0, +1                  YES           YES  (== future_speeds[0] == v0)
    t    yaw       0                      YES (0)       no
    t    yaw_rate  0, +1 (future_yaws[0]) YES           YES
    t    accel     0, +1, +2              YES           YES  (future_speeds[1])

Padding: with m < 4 real poses the missing rows are COPIES of the earliest real row.
n_hist on test: 0:128  1:128  2:128  3:128  4:3102. For the 128 samples with no past,
all four rows are copies of the current (future-reading) row.

Four features read future poses. Two of them reproduce ground-truth TARGETS exactly
(numeric check over all 3,614 test samples):

    ego[t, accel]          == future_accelerations[1]   (GT accel, slot 1)   max err 3.4e-07
    ego[t, yaw_rate] / v0  == future_curvatures[0]      (GT curv,  slot 12)  max err 1.5e-08

The current-row speed equals v0, which CV is also seeded with. It is a shared input,
not a differential leak (see 1e). The t-1 accel reads v0 and so carries no more than v0.

### 1b. Per-slot argmax token accuracy, TEST n=3614 (AR decode, from dump_test.pkl)

    slot       y1_ego   y1_full  zeroboth
     0  a0      0.998    0.996    0.844     (a0 == 0 by construction; trivial)
     1  a1      0.774    0.676    0.138     <-- leak fingerprint
     2  a2      0.305    0.284    0.138
     3  a3      0.272    0.258    0.139
     4  a4      0.252    0.250    0.139
     5  a5      0.250    0.249    0.141
     6  a6      0.243    0.238    0.141
     7  a7      0.233    0.241    0.142
     8  a8      0.228    0.229    0.143
     9  a9      0.219    0.227    0.143
    10  a10     0.221    0.221    0.143
    11  a11     0.213    0.219    0.143
    12  k0      0.693    0.683    0.611     no spike: k0 ~= k1
    13  k1      0.693    0.695    0.612
    14  k2      0.691    0.698    0.614
    15  k3      0.693    0.693    0.614
    16  k4      0.688    0.691    0.614
    17  k5      0.685    0.688    0.615
    18  k6      0.675    0.680    0.614
    19  k7      0.659    0.677    0.614
    20  k8      0.656    0.665    0.612
    21  k9      0.644    0.661    0.610
    22  k10     0.632    0.657    0.608
    23  k11     0.626    0.653    0.607

Accel slot 1 is 2.5x its neighbour (0.774 vs 0.305). The zero-input model, which
cannot see the leak, is flat at 0.138. The models read the leaked accel. Curvature
slot 12 shows no fingerprint: most curvature tokens sit in the straight bin, and
neighbouring slots are already about 0.69 accurate.

### 1c. Leaked-feature predictor (no learning), shrinkage pipeline

"once" = leaked value on its own slot only (accel slot 1, curv slot 0), then hold.
"hold" = held constant over the horizon. alpha fit on val, evaluated on test.
gain = CV - blend, with a paired bootstrap 95% CI.

FULL test set (n=3614, CV 3.062):

    predictor              a*    standalone  blend   gain     95% CI
    leak_once             1.00     2.413     2.413  +0.649  [+0.610,+0.686]
    leak_hold             0.40     3.731     2.423  +0.639  [+0.584,+0.695]
    leak accel only once  1.00     2.665     2.665  +0.397  [+0.363,+0.432]
    leak yaw   only hold  0.65     2.799     2.667  +0.395  [+0.358,+0.433]
    y1_ego   (headline)   0.25     3.633     2.894  +0.168  [+0.142,+0.194]
    y1_full               0.30     3.588     2.910  +0.152  [+0.123,+0.181]
    zeroboth              0.10     3.266     3.048  +0.014  [+0.010,+0.018]

CAUSAL SUBSET (>= 2 past poses; test n=3358/3614, val n=3318/3572; alpha refit on
the val subset; CV 3.059). Causal accel = backward second difference over poses
-2,-1,0. Causal yaw rate = wrap(yaw_0 - yaw_-1)/dt. Rollout seed v0 unchanged.

    predictor              a*    standalone  blend   gain     95% CI
    leak_once             1.00     2.408     2.408  +0.652  [+0.612,+0.692]
    leak_hold             0.40     3.735     2.415  +0.644  [+0.586,+0.702]
    causal_once           1.00     2.876     2.876  +0.184  [+0.138,+0.228]
    causal_hold           0.25     4.766     2.918  +0.141  [+0.096,+0.184]
    y1_ego                0.25     3.626     2.895  +0.164  [+0.137,+0.191]
    y1_full               0.30     3.592     2.914  +0.145  [+0.115,+0.176]
    zeroboth              0.05     3.264     3.052  +0.008  [+0.006,+0.010]

    blend[y1_ego] - blend[causal_hold]   -0.023  [-0.068,+0.021]  p=0.32
    blend[leak_hold] - blend[causal_hold] -0.503  [-0.560,-0.446]

DECISION RULE (stated in advance): the leaked version gains >= 0.08 m, as the
"contaminated" branch requires (+0.64). But the causal version does NOT gain ~0: it
gains +0.14 to +0.18. So neither branch applies as written. Plainly:

  - The leak is REAL and LARGE. A hand-coded rule that just reads the leaked features
    beats the headline blend by 0.48 m. The models demonstrably read it (1b).
  - A NO-LEARNING CAUSAL rule (CV + last observed accel and yaw rate) matches the
    headline gain: 0.14-0.18 vs 0.164, difference n.s. on the same subset.
  - So the headline gain is attainable without the leak. Whether the trained models
    reach it WITHOUT the leak is unknown until the causal retrain (item 6).
  - Finding 3 (r=+0.34 deviation/correction correlation) and "correction, not
    replacement" are AT RISK: an input that equals a target slot is exactly what
    would produce such a correlation.

### 1d. Causal subset

Required frames: poses -2, -1, 0 (for accel), so samples with n_hist < 2 are excluded:
test n = 3358 of 3614, val n = 3318 of 3572. Every 1c row above on the subset is
paired on these exact samples, including CV and the old checkpoints.

### 1e. Side number (report only)

CV seeded with backward speed |p0 - p-1|/dt: 3.538 vs forward v0 3.061
(test n=3486 with >= 1 past pose; diff +0.477, CI [+0.436,+0.519]). The decode seed
v0 itself reads pose +1. It is shared by CV and every model, so it is not a
differential leak. It is still not deployable as-is: a real vehicle cannot measure
the next 0.5 s of displacement.

---

## 2. SIM G2, G3, G4

Model is a NON-AUTOREGRESSIVE MLP (simplification). Every comparison is given against
plain CV and CV + v0-conditioned mean control ("v0mean": train mean control within the
sample's v0 bin, 15 fixed bins; see model.py V0_EDGES). Test n=3614, ADE@6s, paired
bootstrap. Free parameter recorded: accel magnitude U(0.2,1.0) m/s^2, hand-set to put
CV ADE on nuScenes scale (see item 4 for sensitivity).
Logs: Alpamayo/overnight2_g{2,3,3_seeds}.log.

G1 re-run with the extended harness (unchanged data): v0mean 2.587, which is WORSE than
CV (2.545) standalone. Blend - CV = -0.004 [-0.007,-0.001]. Under the winner rule the
(0,0) cell is therefore labelled BLEND on a 0.004 m v0-only effect (see G1 diagnosis).

### G2 -- rho=1, p=0.5: PASS

    CV 7.069   v0mean 7.177   model 1.113   blend 1.113 (a*=1.00)
    model - CV       -5.957 [-6.274,-5.651]    margin 5.957 m (84.3% of CV)
    model - v0mean   -6.065 [-6.381,-5.762]

The model sits near the 0.94 m token floor. That is expected, not suspicious: at
rho=1 the future is fully visible up to the aleatoric noise.

### G3 -- p=0.5, rho sweep: PASS on the 3-seed mean (seed 0 alone: not monotone)

Seed 0:

    rho    CV     v0mean  model  blend  a*   model-CV                 model-v0mean
    0.00  6.945  7.027  4.917  4.891 0.95  -2.029 [-2.263,-1.796]   -2.110 [-2.342,-1.885]
    0.25  6.931  7.040  4.915  4.876 0.95  -2.016 [-2.263,-1.773]   -2.126 [-2.366,-1.895]
    0.50  6.784  6.853  4.141  4.131 0.95  -2.643 [-2.907,-2.384]   -2.712 [-2.975,-2.451]
    0.75  6.745  6.831  2.948  2.948 1.00  -3.797 [-4.066,-3.532]   -3.883 [-4.155,-3.614]
    1.00  7.069  7.177  1.113  1.113 1.00  -5.957 [-6.274,-5.651]   -6.065 [-6.381,-5.762]

Seed 0 breaks monotonicity vs CV at rho 0 -> .25 (+2.029 -> +2.016, 0.013 m). Each rho
is an independently drawn dataset, and the per-cell CI half-width is ~0.23 m. Seeds 1
and 2 (data seed = train seed):

    rho   CV-model per seed (0,1,2)   mean    rel.   v0mean-model  info gain
    0.00  +2.029 +1.848 +2.313       +2.063  0.297    +2.143        0.0855
    0.25  +2.016 +2.295 +2.076       +2.129  0.306    +2.228        0.1048
    0.50  +2.643 +2.678 +2.612       +2.644  0.378    +2.728        0.1578
    0.75  +3.797 +4.127 +3.875       +3.933  0.569    +4.016        0.2406
    1.00  +5.957 +5.712 +5.865       +5.845  0.839    +5.944        0.3914

The 3-seed mean is strictly monotone vs CV, vs v0mean, in relative terms, and in
accel info gain. The seed-0 dip is sampling noise: the seed-to-seed sd at rho=0 is
about 0.23 m, 17x the dip. Also note the model beats CV by ~2 m even at rho=0,
because p=0.5 turns are partly visible in history (onset < 0 for 4/13 of turns), and
CV never turns.

### G4 -- expectation beats argmax at every G3 point: PASS (seed 0)

    rho    argmax  expect   expect-argmax
    0.00   5.025   4.917   -0.109 [-0.149,-0.069]
    0.25   5.075   4.915   -0.160 [-0.214,-0.105]
    0.50   4.309   4.141   -0.168 [-0.233,-0.101]
    0.75   3.076   2.948   -0.128 [-0.173,-0.081]
    1.00   1.481   1.113   -0.369 [-0.394,-0.343]

G2-G4 pass -> full map launched (tmux session "sim").
