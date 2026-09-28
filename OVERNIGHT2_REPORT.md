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

---

## 3. SIM PILOT and FULL MAP

Code: sim/gates.py (pilot, map), sim/map_report.py. Table: Alpamayo/viz/sim_map.csv
(108 rows = 36 cells x 3 seeds, every quantity with a paired-bootstrap CI vs CV and vs
v0mean). Figure: Alpamayo/viz/sim_phase_map.png. Logs: overnight2_map*.log.
Seeds: data seed = train seed in {0,1,2}. Model = NON-AUTOREGRESSIVE MLP.

### Pilot (seed 0), ADE@6s mean; CIs are paired bootstrap on test (n=3614)

    p    rho   CV     v0mean model  blend  a*   model-CV                 blend-CV                 winner
    0.00 0.00  2.545  2.587  2.792  2.540  0.10 +0.248 [+0.222,+0.273]  -0.004 [-0.007,-0.001]  BLEND
    0.00 0.50  2.513  2.528  2.366  2.333  0.65 -0.148 [-0.196,-0.099]  -0.180 [-0.210,-0.149]  MODEL
    0.00 1.00  2.476  2.519  1.175  1.158  0.95 -1.301 [-1.363,-1.237]  -1.318 [-1.380,-1.258]  MODEL
    0.20 0.00  4.320  4.364  3.595  3.559  0.90 -0.725 [-0.900,-0.559]  -0.761 [-0.927,-0.601]  MODEL
    0.20 0.50  4.302  4.354  3.259  3.243  0.90 -1.042 [-1.207,-0.882]  -1.059 [-1.217,-0.902]  MODEL
    0.20 1.00  4.292  4.339  1.215  1.215  1.00 -3.077 [-3.285,-2.876]  -3.077 [-3.281,-2.876]  MODEL
    0.50 0.00  6.945  7.027  4.917  4.891  0.95 -2.029 [-2.263,-1.796]  -2.055 [-2.290,-1.827]  MODEL
    0.50 0.50  6.784  6.853  4.141  4.131  0.95 -2.643 [-2.907,-2.384]  -2.654 [-2.918,-2.401]  MODEL
    0.50 1.00  7.069  7.177  1.113  1.113  1.00 -5.957 [-6.274,-5.651]  -5.957 [-6.274,-5.648]  MODEL

(model-v0mean and blend-v0mean columns are in the log and CSV. v0mean is worse than
CV in every pilot cell.)

### Full map, 3 seeds: majority winner (k/3), mean alpha*, mean ADE@6s

    p \ rho   0          0.1        0.25       0.5        0.75       1
    0.00      BLEND 3/3  BLEND 3/3  BLEND 3/3  MODEL 3/3  MODEL 3/3  MODEL 3/3
              a*=.08     a*=.10     a*=.25     a*=.68     a*=1.0     a*=.95
    0.10      MODEL 2/3  MODEL 3/3  MODEL 3/3  MODEL 3/3  MODEL 3/3  MODEL 3/3
              a*=.72     a*=.75     a*=.77     a*=.90     a*=.95     a*=.95
    0.20      MODEL 3/3  (all MODEL 3/3, a* .82-1.00)
    0.35      MODEL 3/3  (all MODEL 3/3, a* .90-1.00)
    0.50      MODEL 3/3  (all MODEL 3/3, a* .92-1.00)
    0.75      MODEL 3/3  (all MODEL 3/3, a* .95-1.00)

MAP SHAPE: CV never wins. BLEND owns only the p=0, rho <= 0.25 corner (small alpha*,
0.08-0.25). Every cell with any turns, or with rho >= 0.5, is MODEL, with alpha* -> 1.

WHY the model wins at rho=0 once p>0 (decomposition, seed 0,
overnight2_map_decomp.log):

    p=0.20 rho=0              n     CV      model   model-CV              share of gain
      turn in progress       223   17.169   4.359  -12.810 [-14.8,-10.9]   1.09
      turn, onset >= 0       509    8.960   7.870   -1.091 [ -1.5, -0.7]   0.21
      straight              2882    2.506   2.780   +0.274 [+0.24,+0.31]  -0.30
    p=0.50 rho=0: shares 0.89 / 0.19 / -0.08, straight model-CV +0.326

The rho=0 advantage is almost entirely turns ALREADY IN PROGRESS at t=0
(onset < 0, 4/13 of turns, plus the onset=0 ramp segment, which falls in the history).
Those are visible in the history yaw rate, and CV never turns. This is a second
predictability channel that rho does not control. On STRAIGHT samples the model
LOSES to CV (+0.27 to +0.33 m), which is the nuScenes pattern. The p axis therefore
mixes "turn fraction" with "fraction of the future visible from history". Flagged for
the planner: to make rho the only predictability dial, restrict onset to >= 1, or
report p through a history-invisible turn fraction.

---

## 4. SENSITIVITY -- accel magnitude (seed 0, pilot subgrid)

    U(0.1,0.5):  p=0: BLEND BLEND MODEL | p=.2: MODEL x3 | p=.5: MODEL x3
    U(0.2,1.0):  p=0: BLEND MODEL MODEL | p=.2: MODEL x3 | p=.5: MODEL x3
    U(0.5,2.0):  p=0: BLEND MODEL MODEL | p=.2: MODEL x3 | p=.5: MODEL x3
    (columns rho = 0, .5, 1; full numbers in overnight2_sens_summary.log)
    CV ADE at (p=.2, rho=0): 3.45 / 4.32 / 6.18 for the three ranges.

ANSWER: the winner map keeps its shape (BLEND only in the p=0 low-rho corner, MODEL
elsewhere). The accel magnitude moves the absolute ADE (CV 3.45 -> 6.18 at p=.2) and
shifts the p=0 BLEND/MODEL boundary slightly: smaller accel -> BLEND extends to rho=0.5.

---

## 5. LOCATING nuScenes (computed, not interpreted)

Code: Alpamayo/code/leak/locate.py; nuScenes CE from leak/dump_ce.py (free-running,
argmax fed back, 65-way STOP-aware support; its y1_ego expect values reproduce the
frozen dump bit-exactly). Logs: overnight2_locate_{sim,nus}.log.

### p

    sim turn fraction (max|GT curv| > 0.05), 3-seed mean:
      p=0: 0.000  p=.1: 0.098  p=.2: 0.200  p=.35: 0.350  p=.5: 0.500  p=.75: 0.748
    nuScenes 638/3614 = 0.177  ->  p_hat = 0.177 (between the p=.1 and p=.2 rows)

### rho

Sim (seed 0, accel slots 1-11; slots 2-11 within 0.002 everywhere):

    L1 (CE_noobs - CE_obs)/CE_noobs        L2 1 - CE_obs/CE_unigram|v0bin
    p\rho  0     .1    .25   .5    .75   1  |  0     .1    .25   .5    .75   1
    0.10 -.004 .000  .017  .070  .163  .332 | .010  .011  .029  .082  .174  .340
    0.20 -.005 -.002 .012  .069  .162  .328 | .008  .009  .023  .081  .172  .335
    (all p rows are within ~0.01 of these; full grid in the log)

nuScenes (test n=3614, free-running CE):

    accel slots   CE y1_ego  CE y1_full  CE unigram|v0   L1       L2(full)  L2(ego)
    1-11           3.408      3.473       2.368          -0.019   -0.467    -0.439
    2-11           3.690      3.739       2.374          -0.013   -0.575    -0.555

IMPLIED rho: NEITHER locator lands on the sim's scale. Both nuScenes values sit below
the sim's rho=0 value (L1 -0.019 vs sim min -0.005; L2 -0.47 vs sim min +0.005), so
the implied rho is "below 0" for both. They agree in sign only.

Facts relevant to reading this (not interpretation):
  - nuScenes free-running CE is ABOVE the v0 unigram: the models are worse than a
    speed-conditioned token histogram under their own rollouts. Per-slot CE grows
    with the horizon (y1_ego a1..a11: 0.59 2.36 3.06 3.55 3.66 3.87 4.03 4.03 4.10
    4.13 4.12), and 14% of GT tokens in slots 2-11 get probability < 1e-3.
  - Slot 1 (0.59) is the leaked slot.
  - The sim MLP is NON-autoregressive and trained with unweighted CE. The nuScenes
    models are AR (free-running here) and trained with sqrt inverse-frequency
    weighted CE. The two CE scales are therefore not like-for-like; the locator gap
    may be a property of the decoder and loss, not of rho.
  - The earlier nuScenes "2.24 vs 2.33" was teacher-forced and is not comparable.
