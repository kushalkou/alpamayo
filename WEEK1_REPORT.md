# WEEK 1 REPORT -- "Where does the signal come from? An information-controlled audit
# of a token-based driving VLA."

Synthetic (p, rho) benchmark: DROPPED and archived (commit 2e45c04, Alpamayo/code/sim/
kept, ARCHIVED.md added). No sim jobs were running.

---

## 0. OVERNIGHT2 OUTCOME (items 1 and 6 ran; 6b full-vision also finished)

The ego-state leak is CONFIRMED. compute_ego_state's current row holds the ground-truth
accel of slot 1 (max err 3.4e-7) and the GT curvature of slot 12 (yaw_rate/v0, max err
1.5e-8) on all 3,614 test samples, and the models read it: slot-1 accuracy is 0.774 vs
0.305 at slot 2. A no-learning predictor that just uses the leaked features gains
+0.649 m over CV [+0.610,+0.686] on the full test set. The same predictor with causal
(past-pose) features gains +0.184 [+0.138,+0.228]. Retrained with causal features
(Y1 recipe, seed 42, custom-split test subset with >= 2 past poses, n=3358, CV 3.059),
the ego-only model + CV blend STILL beats CV, but only by 0.014 m (3.045, CI
[-0.024,-0.003], p=0.011, alpha*=0.10), versus 0.164 m for the leaky model on the same
samples: the leak is 91% of the headline gain. The causal full-vision blend gives the
same 0.014 m ([-0.025,-0.002], p=0.018), and full minus ego is -0.000 overall and
-0.014 [-0.050,+0.021] on turning. All three nulls select alpha=0. The slot-1
fingerprint is gone (0.27-0.28). Single seed each; all on the custom split, so NOT
comparable to the literature. Details: OVERNIGHT2_REPORT.md sections 1, 6 and 6b.

---

## 1. CAUSAL EGO FEATURES (default for all new runs)

Definition (Alpamayo/code/leak/causal_ego.py, used unchanged): rows = the last 4 pose
indices i <= 0, with speed_i = |p_i - p_{i-1}|/dt, yaw_i, yaw_rate_i =
wrap(yaw_i - yaw_{i-1})/dt, accel_i = (speed_i - speed_{i-1})/dt. The earliest row's
undefined accel copies its neighbour; short histories are left-padded (original
convention). Needs >= 2 past poses; others are EXCLUDED.

Unit test, Alpamayo/code/w1/test_causal_ego.py -- 4/4 PASS:
  - whitelist: the input record raises on ANY key except past_poses / current_pose;
  - perturbation: scrambling every future_* field leaves the features bit-identical;
  - analytic values on a constant-accel turn;
  - detector sanity: the frozen dataset.compute_ego_state FAILS the whitelist
    (reads the future), so the test can see a leak.

Exclusions (>= 2 past poses) on the official split: train 20913/22213,
holdout 1617/1717, val 4819/5119 (samples with >= 6 future keyframes).

CV with the rollout seed v0 also computed causally (|p0 - p-1|/dt), official val
causal subset n=4819, same metric code as gate 3:

    CV seed           L2 NoAvg 1/2/3s     L2 TemAvg 1/2/3s    Col% NoAvg  Col% TemAvg
    forward v0        0.313 1.088 2.269   0.199 0.533 1.005   0.10 0.37 1.51  0.11 0.22 0.59
    causal v0         0.539 1.462 2.775   0.388 0.796 1.339   0.15 0.50 2.57  0.15 0.25 0.84

The decode seed v0 = |p1 - p0|/dt reads the NEXT pose. It is kept as is, per
instructions, but it is worth +0.33 m of TemAvg L2 at 3 s to CV alone.

## 2. OFFICIAL SPLIT

nuScenes devkit create_splits_scenes(): train 700 / val 150 scenes. 50 official-train
scenes held out for AR-val-ADE selection (RandomState(42).choice over sorted scene
names; list stored in Alpamayo/data/w1_data.pkl). Builder: Alpamayo/code/w1/build_data.py
(29,049 records = every sample with >= 6 future keyframes; 34,149 - 850 x 6).

    split     scenes  n_fut>=6  n_fut=12  +causal   n_fut=12 & causal (trainable)
    train       650    22213     18313     20913     17013
    holdout      50     1717      1417      1617      1317
    val (rep.)  150     5119      4219      4819      3919

Official val: 5119 samples with a full 3 s future (the literature planning count).
Training and holdout use n_fut=12 (12 target steps). Reporting uses the n_fut >= 6
causal subset (4819), plus n_fut=12 for ADE/FDE@6s. The existing checkpoints were
trained on the custom split: any number of theirs on official val is CONTAMINATED and
none is reported here.

## 3. STANDARD PLANNING METRICS -- GATE 3 PASS

Code: Alpamayo/code/w1/plan_metrics.py. Ported, not rewritten:
  - VAD projects/mmdet3d_plugin/VAD/planner/metric_stp3.py ("same as stp3"):
    _get_poly_region_in_image, evaluate_single_coll and evaluate_coll are copied
    verbatim. BEV 200x200 at 0.5 m. Ego box W=1.85 m x L=4.084 m, shifted +0.5 m
    forward, axis-aligned with the t0 heading (as in the source).
  - VAD.py compute_planner_metric_stp3: prefix (TemAvg) aggregation,
    occupancy = vehicle OR pedestrian.
  - UniAD planning_metrics.py: per-timestep (NoAvg) L2, same ego box and masking;
    vehicles only. UniAD's own occupancy orientation is not reproduced: both
    conventions use VAD's engine and differ only in class set and aggregation.
  - Reported collision = obj_box_col (ego-box overlap), counted only where the GT ego
    box does not itself collide -- exactly as in both sources.
  - Occupancy is rebuilt here, because VAD's mmdet3d pipeline is not runnable: agents
    present at t0 (vehicle.* / human.pedestrian.*, >= 1 lidar/radar point), boxed at
    their annotated pose at each future keyframe, in the t0 LIDAR frame.
  - Frame: LIDAR_TOP positions in the t0 LIDAR frame, as VAD's gt_ego_fut_trajs.
    Predictions (CAM-ego origin + heading) are converted with the lidar lever arm. The
    origin (CAM or LIDAR ego pose at t0; the keyframes differ by 36 ms) was chosen on
    HOLDOUT by the conversion floor: LIDAR origin, 0.071 m at 3 s (CAM origin: 0.184).

GATE 3, official val (n=5119; the causal subset n=4819 gives the same picture):

    method                  L2 NoAvg 1/2/3s   L2 TemAvg 1/2/3s   Col% NoAvg(veh)  Col% TemAvg(v+p)
    GT (literature)         0.000 0.000 0.000 0.000 0.000 0.000  0.00 0.00 0.00   0.00 0.00 0.00
    own CAM GT -> lidar     0.054 0.065 0.076 0.052 0.057 0.062  0.02 0.02 0.02   0.03 0.02 0.02
    CV (forward v0)         0.313 1.089 2.274 0.199 0.534 1.007  0.10 0.35 1.43   0.11 0.21 0.56
    CV L2 at 3 s: median 1.802, p95 6.410 (NoAvg); median 0.792, p95 2.868 (TemAvg)

  (a) GT L2 = 0.000e+00 at every sample and step: PASS.
  (b) Masked GT collision = 0 by construction. Unmasked GT box collision, per step
      1..6: vehicles 0.53 0.43 0.43 0.49 0.41 0.37%; veh+ped 1.11 0.96 0.94 0.96 0.86
      0.82% -- small and nonzero: PASS.
  (c) CV evaluated with identical code: PASS.
  "own CAM GT -> lidar" = the floor from our frame/timing convention. It is part of
  every model number (0.06-0.08 m).
  FLAG, not interpreted: CV with the forward v0 scores TemAvg 0.20/0.53/1.01, which is in
  the range of published end-to-end planners (VAD-Base is reported at about
  0.41/0.70/1.05 from memory -- verify). With a causal v0 it is 0.39/0.80/1.34.

ADE@6s / FDE@6s stay as secondary metrics (n_fut=12 subset); they are computed per
model in item 7.

## 4. NAVIGATION COMMAND

Rule, quoted from VAD tools/data_converter/vad_nuscenes_converter.py (ego_fut_trajs =
LIDAR_TOP positions at future steps 0..6 in the t0 LIDAR frame, CUMULATIVE -- this
runs before the per-step offset conversion):

    # drive command according to final fut step offset from lcf
    if ego_fut_trajs[-1][0] >= 2:
        command = np.array([1, 0, 0])  # Turn Right
    elif ego_fut_trajs[-1][0] <= -2:
        command = np.array([0, 1, 0])  # Turn Left
    else:
        command = np.array([0, 0, 1])  # Go Straight

i.e. lateral displacement (x = right) of the lidar at 3 s, threshold +-2 m.

    split     right          left           straight
    train     1791 (0.081)   1376 (0.062)   19046 (0.857)
    holdout     61 (0.036)     97 (0.056)    1559 (0.908)
    val        397 (0.078)    286 (0.056)    4436 (0.867)

NOTE: this command is DERIVED FROM THE GT FUTURE. That is the literature convention,
and it is a future-derived input.
Injection: Alpamayo/code/w1/command.py. ONE learned embedding (3 x 3584, fp32,
N(0, 0.02) init) appended after the 4 ego tokens (context 1540 -> 1541). It rides in a
5th ego_state row, so training, AR validation and dumps need no edits. It is stored as
ego_encoder.cmd_embed, so the existing checkpoint saver keeps it. Unit test
w1/test_command.py PASS (shape, token identity, saved in state_dict, receives grad).

## 5. TOKENIZER FLOOR

Code: Alpamayo/code/w1/tok_floor.py. Floor = ADE@6s of rollout(detok(tok(GT
controls))) with the true v0/yaw0. Bins fit on official TRAIN (n=18313); the decision
is made on HOLDOUT (n=1417). Val is not touched. Method check: variant (i) on the old
custom test reproduces the frozen 1.348 / 0.889 exactly.

    variant                              holdout ADE@6s  median   p95    vs (i)   exact+determ.
    (i)   uniform (current)                 1.086         0.550   3.626    --       yes
    (ii)  percentile, 64/channel            0.279         0.186   0.893  -74.3%     yes
    (iii) uniform + error-feedback          0.067         0.045   0.196  -93.8%     yes
    (iv)  percentile + error-feedback       0.024         0.009   0.053  -97.8%     yes

DECISION (rule as written): adopt (iv), which cuts the floor by 97.8% and is exact
and deterministic.

FLAG before item 6 -- error-feedback changes what a token MEANS (300 holdout samples):

    variant   mean|dtoken| accel/curv   |decoded accel - GT accel|   final heading err
    (i)          2.55 / 0.17               0.077 m/s^2                 4.36 deg
    (ii)        14.83 / 6.08               0.025                       0.75
    (iii)        3.40 / 2.06               0.382                       1.71
    (iv)        14.85 / 11.21              0.154                       0.97

(iii) and (iv) reach their floors by choosing COMPENSATING tokens: the decoded accel
is 6x (iv vs ii) to 15x (iii vs ii) further from the true control, and curvature
tokens jitter far more. The targets become path-dependent dither rather than physical
controls, which may be harder to learn and interacts with expectation decoding. (ii)
alone also passes the rule (-74%) and keeps the tokens faithful. Your call; the rule
says (iv).

---

STOPPED here per instructions. Items 6-7 are not launched.

===========================================================================================
PLANNER DECISIONS ON e089d0f (D1 tokenizer (ii); D2 FROZEN v2 approved; D3 stage A only)
+ AMENDMENT. Work below.
===========================================================================================

## D2. FROZEN_RESULTS v2 -- done (fc22e97)

FROZEN_RESULTS.md is now the approved v2, with the edits applied: kinematic-rule
reference line (+0.184 m); F2 "survives in kind, 0.014 m, same order as zero-input
0.008"; F4 WITHDRAWN citing 6b. The v1 text is kept unchanged as
FROZEN_RESULTS_v1_LEAKY.md. (Interpretation: "keep v1 as ..._v1_LEAKY" was read as v2
taking the FROZEN_RESULTS.md name. Revert with git if that was not intended.)

## STEP 1 -- v0

### 1a. The old "constant turn rate 3.014"

v2_characterize.py computes yr = compute_ego_state(traj)[3,2]. With TODAY's
compute_ego_state that is wrap(future_yaws[0] - yaw0)/dt -- a future yaw -- and
re-running the script today gives a LEAKY 2.799. The published 3.014, however, was
produced before the W1 change, when compute_ego_state used backward differences:
a causal recomputation (yr = wrap(yaw0 - yaw_-1)/dt, 0 without a past pose, same loop,
custom test n=3614) reproduces it exactly, 3.014 / median 2.402. So 3.014 is causal.
Causal CTR - CV = -0.048 [-0.100,+0.004], p=0.074 (not significant).
Code: Alpamayo/code/w1/ctr_recheck.py.

### 1b. CAN bus

Not previously on the DGX, the NAS or the WS. Official source: the nuScenes public
bucket linked from the nuscenes.org download page,
https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/can_bus.zip -- 780,974,697 bytes (CAN
bus expansion, last-modified 2024-01-30). Downloaded and unzipped to
Alpamayo/nuscenes/can_bus/ (979 scenes x 8 message files).

Source used: the 'pose' message (50 Hz): vel[0] (m/s), accel[0] (m/s^2),
rotation_rate[2] (yaw rate, rad/s). Lookup = the LAST message with utime <= t0, with
t0 = sample['timestamp']. Never the nearest message: the nearest can be up to 18.9 ms
AFTER t0 on val. Unit test Alpamayo/code/w1/test_can_causal.py PASS (synthetic
boundaries; every built record has t0 - utime >= 0).
Offset t0 - utime on val: p50 9.96, p90 17.86, p99 20.59, max 38.26 ms.

Coverage (Alpamayo/w1_can_coverage.log):

    split     samples with CAN   scenes with NO CAN at all
    train     21097/22213        15/650: scene-0161..0168, 0170..0176 (devkit blacklist)
    holdout    1667/1717          0/50
    val        4979/5119          0/150

Every other missing sample is the FIRST sample of a scene: the scene's CAN log starts
just after it (609 train / 50 holdout / 140 val scenes, exactly 1 sample each). Those
samples also have no past pose, so the causal fallback (backward pose differences, per
the amendment) has nothing to use: v0 = a = yaw rate = 0 there.
(VAD's converter instead defaults to pose_list[0], a message AFTER t0; see
amendment 2.)

Official val, ALL 5,119 samples, LIDAR origin, same plan_metrics code as gate 3.
KIN = kinematic reference rule: the measured accel and yaw rate act over the first
0.5 s step, then speed and heading are held.

                       L2 NoAvg 1/2/3s   L2 TemAvg 1/2/3s   Col% NoAvg     Col% TemAvg   L2@3s med/p95
    CV  v0_fwd (side)  0.313 1.089 2.274 0.199 0.534 1.007  0.10 0.35 1.43 0.11 0.21 0.56 1.802/6.410
    KIN v0_fwd (side)  0.418 1.084 2.195 0.314 0.602 1.032  0.08 0.18 1.27 0.08 0.13 0.40 1.744/6.100
    CV  v0_back        0.668 1.706 3.127 0.485 0.953 1.552  0.14 0.63 2.83 0.14 0.29 0.95 2.308/8.205
    KIN v0_back        0.685 1.682 3.071 0.512 0.959 1.540  0.10 0.43 2.11 0.09 0.20 0.69 2.161/8.732
    CV  v0_back+.25a   0.677 1.710 3.128 0.494 0.960 1.558  0.14 0.55 2.60 0.14 0.27 0.88 2.291/8.602
    KIN v0_back+.25a   0.781 1.813 3.238 0.592 1.061 1.660  0.14 0.53 2.15 0.11 0.25 0.75 2.117/10.09
    CV  v0_can         0.539 1.461 2.779 0.390 0.797 1.340  0.12 0.55 2.27 0.12 0.27 0.81 1.947/7.489
    KIN v0_can         0.379 1.057 2.187 0.302 0.579 1.012  0.04 0.31 1.33 0.05 0.15 0.47 1.458/5.651

    (v0_back falls back to 0 on the 150 first-of-scene val samples; v0_can falls back to
    v0_back on the 140 samples without CAN at/before t0.)

### 1c. Default

v0 = v0_can (CAN pose vel at the last message <= t0), with backward-pose fallback. Ego
accel and yaw rate come from the same CAN message, with the same fallback. The
kinematic reference rule for all later tables is KIN v0_can: TemAvg 0.302/0.579/1.012,
NoAvg 0.379/1.057/2.187, collision TemAvg 0.05/0.15/0.47%. Evaluation covers all
5,119 samples; no causal-subset filtering.
Reference: VAD-Base TemAvg 0.41/0.70/1.05, col 0.07/0.17/0.41 (planner-verified). The
causal kinematic rule, with no learning and no perception, is at VAD-Base's L2.

## AMENDMENT 2 -- VAD converter ego status (tools/data_converter/vad_nuscenes_converter.py,
## hustvl/VAD main, 1005 lines, fetched 2026-09-30)

Verbatim, with line numbers:

    L39  def locate_message(utimes, utime):
    L40      i = np.searchsorted(utimes, utime)
    L41      if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
    L42          i -= 1
    L43      return i
      -> NEAREST message: can be AFTER the sample time.

    L170 def _get_can_bus_info(nusc, nusc_can_bus, sample):
    ...  last_pose = pose_list[0]
    L180     for i, pose in enumerate(pose_list):
    L181         if pose['utime'] > sample_timestamp:
    L182             break
    L183         last_pose = pose
    L184     _ = last_pose.pop('utime')  # useless
    L185     pos = last_pose.pop('pos')
    L186     rotation = last_pose.pop('orientation')
    L187     can_bus.extend(pos)
    L188     can_bus.extend(rotation)
    L189     for key in last_pose.keys():
    L190         can_bus.extend(pose[key])  # 16 elements
      -> pos / orientation come from the last message <= t0, but L190 indexes `pose`,
         the loop variable, which after the break is the FIRST MESSAGE AFTER t0. So
         can_bus[7:16] (accel, rotation_rate, vel) are ~0-20 ms in the FUTURE. And
         when the log starts after t0, last_pose = pose_list[0] is itself after t0.

    L232-242  pose_record_prev from sample['prev'], pose_record_next from sample['next']
    L473 if pose_record_prev is not None:
    L474     ego_w = (ego_yaw - ego_yaw_prev) / 0.5
    L475     ego_v = np.linalg.norm(ego_pos[:2] - ego_pos_prev[:2]) / 0.5
    L477 else:
    L478     ego_w = (ego_yaw_next - ego_yaw) / 0.5
    L479     ego_v = np.linalg.norm(ego_pos_next[:2] - ego_pos[:2]) / 0.5
      -> backward (causal) normally; FORWARD from the next keyframe (0.5 s future) on the
         first sample of every scene.

    L489 pose_index = locate_message(pose_uts, ref_utime)   (nearest, see L39)
    L491 steer_index = locate_message(steer_uts, ref_utime)
    L494 v0 = pose_data["vel"][0]
    L496 steering = steer_data["value"]  ... L501 Kappa = 2 * steering / 2.588
    L503 except:
    L504     delta_x = ego_his_trajs[-1, 0] + ego_fut_trajs[0, 0]
    L505     delta_y = ego_his_trajs[-1, 1] + ego_fut_trajs[0, 1]
    L506     v0 = np.sqrt(delta_x**2 + delta_y**2)
      -> v0 and Kappa from the NEAREST message (possibly future); the fallback reads
         ego_fut_trajs[0], the first FUTURE step.

    L508 ego_lcf_feat[:2] = np.array([ego_vx, ego_vy])
    L509 ego_lcf_feat[2:4] = can_bus[7:9]     <- ax, ay from the post-t0 message (L190)
    L510 ego_lcf_feat[4] = ego_w
    L512 ego_lcf_feat[7] = v0
    L513 ego_lcf_feat[8] = Kappa

ANSWER: yes. VAD's ego status reads future information in four places, all small in
time (<= ~20 ms for CAN; 0.5 s for the first-of-scene forward difference and the
except-path v0). Magnitude unmeasured; flagged only.

## STEP 2 -- targets + tokenizer

Code: Alpamayo/code/w1/targets.py -> Alpamayo/data/w1_targets.pkl.
Log: Alpamayo/w1_targets.log.

### 2b (done first, it defines the targets). Frame

The model's trajectory is now VAD's evaluation trajectory itself: LIDAR_TOP sensor
positions at future keyframes 1..n_fut (<= 12) in the t0 LIDAR frame, built exactly
as vad_nuscenes_converter.py builds gt_ego_fut_trajs. The first 6 steps equal the
gate-3 GT (asserted, < 1e-9). The rollout runs in that frame from (0,0) with heading
pi/2, so predictions need NO conversion, and there is no 0.06-0.08 m CAM/LIDAR offset
anymore.

### 2a. Targets

Chord construction from v0 = v0_can: segment speeds s_k and headings theta_k of the
lidar trajectory; acc[0] = (s_0 - v0)/dt (now nontrivial), acc[k] = (s_k - s_k-1)/dt,
cur[k] = wrap(theta_k - theta_k-1)/(s_k dt), theta_-1 = pi/2. The heading is held when
a segment is < 0.01 m. Continuous targets reproduce the trajectory to max 6.3 cm
(error only from that hold rule; 6341 of 29049 samples > 1e-6 m).

Tokenizer (ii) = 64 equal-mass bins per channel fit on the new official-train targets
(181,536 non-STOP steps), centre = MEDIAN of the bin. Mean centres (the item-5
definition) were tried first and failed on these targets: the chord curvature has a
heavy low-speed noise tail (p1/p99 -0.15/+0.15 rad/m, but the edge-bin means reach
-1.7/+2.9), so real turns decoded badly. Holdout floor with mean vs median centres:
0.785 vs 0.440 ADE@6s. Chosen on HOLDOUT (Alpamayo/w1_bins_variants.log); val was not
used. STOP rule unchanged (s_k < 0.1 -> STOP -> (0,0)).

FLOOR = rollout(detok(tok(GT targets))) from the true v0_can; L2 over n_fut >= 6,
ADE/FDE@6s over n_fut = 12:

    set / variant            n      L2 NoAvg 1/2/3s   L2 TemAvg 1/2/3s   ADE@6s mean/med/p95   FDE@6s
    holdout (ii) ALL       1717     0.088 0.198 0.328 0.064 0.117 0.176  0.440/0.079/0.908    0.918
    holdout (ii) CAN ok    1667     0.022 0.066 0.130 0.015 0.034 0.061  0.182/0.075/0.738    0.445
    holdout (ii) no CAN      50     2.285 4.589 6.906 1.714 2.863 4.017  7.494/5.893/22.13   13.860
    holdout (i) uniform    1717     0.158 0.436 0.835 0.112 0.235 0.399  1.125/0.511/3.782    2.692
    val     (ii) ALL       5119     0.122 0.274 0.453 0.089 0.162 0.243  0.600/0.103/1.245    1.249
    val     (ii) CAN ok    4979     0.028 0.087 0.175 0.019 0.045 0.080  0.240/0.099/0.846    0.599
    val     (ii) no CAN     140     3.464 6.920 10.34 2.597 4.328 6.046  11.09/10.41/27.19   20.20
    val     (i) uniform    5119     0.213 0.577 1.089 0.150 0.313 0.526  1.475/0.766/4.488    3.508

Roundtrip: tokens regenerate identically (deterministic) and detok is bit-stable:
PASS on both sets. (ii) cuts the uniform floor by 61% (holdout ADE@6s), so it passes
the >= 40% rule on the new targets too.

NOTE: the "no CAN" rows are the first sample of each scene. There, NOTHING causal
exists: no past keyframe, no CAN message <= t0, and no lidar sweep precedes the first
keyframe in any of the 850 scenes (checked). v0 falls back to 0, and their a0 target
lies beyond the bin range. They are 2.7% of val but 58% of val's floor ADE. They stay
in every evaluation, per instructions.

### 2c. CE weights

Would-be sqrt-inverse-frequency weights on the new train tokens: max/min = 3.67 over
the 65 classes, but 1.00 excluding STOP (the bins are equal-mass by construction).
Per slot, excluding STOP: 2.5 at slot 0 and 2.4 at slot 12, 1.1 everywhere else.
DECISION: plain CE. The literal rule (max/min < 2) is not met ONLY because of STOP, and
down-weighting STOP would miscalibrate the p(STOP) that the hybrid decode relies on.
FLAG: if the planner wants the rule read literally, weighting stays on.

### 2d. Calibration of the OLD causal_ego (custom split, val n=3318, free-running)

Code: w1/calib_2d.py. Distributions from leak/dump_ce.py --probs.
Log: Alpamayo/w1_calib_2d.log. Report only.

    slot  KL(pred||GT)  mean pred accel  mean GT accel  pred-GT   p(STOP) pred/GT
    a0      0.0013         -0.007           0.000        -0.007    0.159/0.160
    a1      0.3375         -0.017           0.036        -0.052    0.162/0.158
    a2      0.2837         -0.344           0.036        -0.379    0.172/0.157
    a3      0.3038          0.016           0.027        -0.010    0.174/0.156
    a4      0.1761         -0.186           0.025        -0.211    0.180/0.155
    a5      0.1723         -0.104           0.023        -0.127    0.193/0.155
    a6      0.1431         -0.212           0.028        -0.240    0.209/0.155
    a7      0.1476         -0.143           0.031        -0.174    0.225/0.156
    a8      0.1646         -0.139           0.034        -0.173    0.232/0.158
    a9      0.1387         -0.205           0.029        -0.234    0.236/0.160
    a10     0.1226         -0.202           0.031        -0.233    0.241/0.161
    a11     0.1327         -0.139           0.027        -0.167    0.247/0.161

## STEP 3 -- training wrapper (official split) + unit tests

Code: Alpamayo/code/w1/finetune_w1.py. It runs the UNCHANGED finetune.py loop (its own
__main__ exec'd verbatim) with these patches:
  - data: train = official train minus the 50 holdout scenes, n_fut=12 (18,313; the
    <2-past-pose exclusion is gone because features now fall back to CAN / zeros).
    Selection = AR val ADE@6s (argmax, corrected STOP) on finetune's fixed 400-sample
    subset (seed 1234) of the HOLDOUT n_fut=12 set -- 1,417 samples, not the 1,317 in
    the planner text, since the causal-subset filter no longer applies. Official val
    is never loaded.
  - ego features: w1/records.py ego_state_w1 = backward pose differences for the last
    4 poses; the current row = CAN (v0_can, yr_can, a_can) at the last message <= t0.
    Yaw is RELATIVE to t0 (+pi/2): previously it was the global yaw -- a normalisation
    change, noted for item 7c.
  - tokens: tokenizer (ii) (w1/w1tok.py).
  - rollout/GT: records carry future_speeds[0] = v0_can, the heading row pi/2, and the
    lidar-frame GT, so the unchanged ar_eval scores in VAD's frame.
  - loss: plain CE (2c).
  - extra tokens (w1/extra_tokens.py): --cmd adds 1 token. --meta adds 2 meta-action
    tokens, lat (= command, flippable) and lon, INSTEAD of cmd: an unflippable cmd
    token next to a flipped lat label would leak the true label. Rows 4+ of ego_state;
    each is its own embedding under ego_encoder.xtok.*, so it is saved and trained.
  - --no_vision: visual tokens REMOVED (no image IO, context = 4 ego tokens + extras).
  - --turn_weighted keeps the OLD weights (old CAM curvature > 0.05 -> ~40%).
  - --overfit N; checkpoints under models/checkpoints/_w1_<tag>/.
Unit tests, w1/test_w1.py -- 5/5 PASS:
  1. ego features read only past/current poses + the t0-causal CAN fields
     (whitelisted dict);
  2. scrambling future_* leaves them bit-identical;
  3. GT tokens through the UNCHANGED ar_eval rollout reproduce the step-2a floor, and
     GT = the lidar-frame trajectory (max diff < 1e-4 m, from the float32 pi/2 row);
  4. extra tokens: shapes with visual tokens present and removed, token identity,
     saved in state_dict, receive grad;
  5. flips: lon 10%, lat 10%, cmd 0%, rows carry the shown label.
Also: w1/test_can_causal.py (CAN never after t0), w1/test_causal_ego.py,
w1/test_command.py. leak/finetune_causal.py and leak/dump_ce.py gained a
--no_vision / 'remove' option for A0.
Speed with visual tokens removed: 0.36 s/step on 8 V100s (vs 4.6 s with 1,536 zeroed
visual tokens).

Meta-action classes (A2; lon from GT speed at 3 s vs v0_can, thresholds as specified):

    split     stop           accelerate     decelerate     maintain
    train     3506 (0.191)   4457 (0.243)   3331 (0.182)   7019 (0.383)
    holdout    324 (0.229)    367 (0.259)    214 (0.151)    512 (0.361)
    val        853 (0.167)   1370 (0.268)    891 (0.174)   2005 (0.392)

No class is below 2%. Note: "stop" is almost entirely already-stationary vehicles:
3503 of the 3506 train "stop" samples have command straight.

## STEP 3.5b -- ego-status MLP baseline (AD-MLP style)

Code: w1/ego_mlp.py. Inputs = the VLA's causal ego state (16) + one-hot command (3).
Output = 12 waypoints in the lidar frame (direct regression, L1 loss). MLP
19-512-512-24, 100 epochs, selection on holdout ADE@6s. 3 seeds, 1 GPU, ~25 s each.
Log: Alpamayo/w1_egomlp.log. Official val, ALL 5,119:

    method        L2 NoAvg 1/2/3s   L2 TemAvg 1/2/3s   Col% NoAvg      Col% TemAvg     ADE6 mean/med/p95   FDE6   L2@3s med/p95
    CV            0.527 1.448 2.762 0.376 0.784 1.326  0.10 0.47 2.07  0.14 0.27 0.75  3.742/2.763/9.889 8.735  1.937/7.427
    KIN           0.362 1.049 2.178 0.278 0.562 0.998  0.08 0.29 1.39  0.16 0.23 0.57  3.161/2.240/8.199 7.750  1.449/5.636
    ORACLE-KIN    0.477 1.145 1.955 0.348 0.658 1.022  0.12 0.70 1.58  0.14 0.35 0.70  2.671/1.708/8.901 6.342  1.181/6.466
    egoMLP s42    0.324 0.822 1.633 0.241 0.461 0.777  0.08 0.82 1.78  0.29 0.56 0.87  2.434/1.675/6.178 6.067  1.033/4.166
    egoMLP s123   0.328 0.823 1.636 0.244 0.463 0.779  0.59 0.23 1.76  0.53 0.37 0.75  2.437/1.691/6.096 6.073  1.047/4.171
    egoMLP s2024  0.326 0.829 1.641 0.243 0.464 0.782  0.59 0.80 1.86  0.53 0.65 0.98  2.440/1.696/6.154 6.075  1.054/4.166

  Scene-level paired bootstrap (s42; the other seeds are within 0.01):
    egoMLP - CV          L2@3s -1.129 [-1.279,-0.982]  L2T@3s -0.549 [-0.617,-0.481]  ADE6 -1.309 [-1.509,-1.110]
    egoMLP - KIN         L2@3s -0.545 [-0.634,-0.458]  L2T@3s -0.222 [-0.255,-0.188]  ADE6 -0.727 [-0.863,-0.592]
    egoMLP - ORACLE-KIN  L2@3s -0.322 [-0.508,-0.166]  L2T@3s -0.246 [-0.327,-0.177]  ADE6 -0.237 [-0.535,+0.000] p=0.050
    KIN - CV             L2@3s -0.584 [-0.652,-0.516]  L2T@3s -0.327 [-0.367,-0.288]  ADE6 -0.582 [-0.652,-0.509]
  Strata, L2@3s NoAvg: straight (3488) CV 2.43 / KIN 1.79 / O-KIN 1.79 / MLP 1.37;
  turning (638) 4.61 / 3.41 / 2.69 / 1.87; stationary (993) 2.75 / 2.74 / 2.08 / 2.42.

Oracle-kin table (train medians of the per-sample mean control over 3 s): lon accel
stop 0.000 / accelerate +0.680 / decelerate -0.596 / maintain +0.022 m/s^2; lat curvature
right -0.0292 / left +0.0300 / straight -0.0001 rad/m.

Notes (not interpreted):
  - A causal ego MLP + the (future-derived) command beats VAD-Base's published L2
    (TemAvg 0.41/0.70/1.05) at 0.24/0.46/0.78, but collides more (TemAvg 0.29/0.56/0.87%
    vs 0.07/0.17/0.41). The 1 s NoAvg collision swings 0.08-0.59% across seeds (4-30
    samples).
  - The CV here (0.527/1.448/2.762) differs slightly from step 1b's CV v0_can
    (0.539/1.461/2.779): the rollout now starts in the lidar frame along +y, whereas
    step 1b converted a CAM-heading rollout. The difference is the lidar mount yaw.
  - The "stationary" stratum (v0_can < 0.5) includes the 140 first-of-scene samples
    whose v0 falls back to 0 while actually moving; that is why CV scores 2.75 there.

===========================================================================================
PLANNER RESPONSE AFTER STEP 3 (approved: v2 rename, median centres, plain CE, A2 tokens;
holdout 1,417; the lidar-frame evaluator is authoritative).
===========================================================================================

## P4. Script hygiene -- why the old CTR script now reads a future yaw

The shared function dataset.compute_ego_state was rewritten on 2026-07-14 (875d3a2, "W1
FIX") from backward differences to forward differences that read future_positions[0],
future_yaws[0] and future_speeds[1]. v2_characterize.py (CTR 3.014, 947dee4, 2026-07-13)
imports it, so a rerun today silently uses the new, leaky version.
Guard: Alpamayo/code/LEGACY_SCRIPTS_README.md lists all 11 scripts that import it and
forbids rerunning them for new numbers; new work uses only the unit-tested causal
features (32d990c).

## P2. VAD-converter ego-status leak, quantified (CPU)

Features (b) = VAD's gt_ego_lcf_feat (9-dim), rebuilt EXACTLY as
vad_nuscenes_converter.py computes it (w1/vad_feats.py). Rebuilding it exposed one
more quirk, beyond the four future reads listed earlier: quart_to_rpy (L32-37) unpacks
(x,y,z,w), but nuScenes quaternions are (w,x,y,z). So for planar motion VAD's ego_yaw
is ~0, ego_w ~ 0 (train sd 0.011 rad/s, |max| 0.10), and (vx,vy) ~ (0, v) (vx sd 0.14).
Features (a) = ours (16 causal + CAN). Both get the one-hot command. The recipe is
identical (w1/ego_mlp.py train_one, 512-512 MLP, L1, holdout selection), on CPU, 3
seeds. KIN (VAD feats) = the kinematic rule with v0 = VAD v0, accel = VAD ax, yaw rate
= VAD ego_w. Code: w1/egomlp_ab.py; log: Alpamayo/w1_egomlp_ab.log. Holdout ADE@6s:
(a) 2.141/2.145/2.148, (b) 1.846/1.842/1.841.

ALL 5,119:
    method           L2 NoAvg 1/2/3s   L2 TemAvg 1/2/3s   Col% NoAvg      Col% TemAvg     ADE6 mean/med/p95   FDE6
    CV               0.527 1.448 2.762 0.376 0.784 1.326  0.10 0.47 2.07  0.14 0.27 0.75  3.742/2.763/9.889 8.735
    KIN (ours)       0.362 1.049 2.178 0.278 0.562 0.998  0.08 0.29 1.39  0.16 0.23 0.57  3.161/2.240/8.199 7.750
    KIN (VAD feats)  0.292 0.923 1.998 0.221 0.478 0.887  0.18 0.39 1.21  0.24 0.33 0.60  2.895/2.290/7.861 7.299
    MLP (a) s42      0.325 0.823 1.636 0.243 0.462 0.778  0.08 0.84 1.84  0.33 0.58 0.92  2.436/1.668/6.176 6.069
    MLP (a) s123     0.336 0.829 1.639 0.250 0.470 0.785  0.59 0.68 1.33  0.53 0.57 0.82  2.444/1.708/6.067 6.088
    MLP (a) s2024    0.329 0.833 1.646 0.245 0.469 0.786  0.59 0.74 1.78  0.54 0.65 0.95  2.442/1.694/6.159 6.071
    MLP (b) s42      0.222 0.627 1.345 0.166 0.335 0.604  0.08 0.21 0.94  0.09 0.15 0.34  2.065/1.696/5.437 5.440
    MLP (b) s123     0.245 0.624 1.346 0.177 0.340 0.608  0.10 0.18 0.98  0.16 0.26 0.44  2.068/1.695/5.411 5.452
    MLP (b) s2024    0.223 0.622 1.344 0.163 0.333 0.602  0.08 0.70 1.00  0.14 0.31 0.46  2.067/1.715/5.439 5.452
  (b)-(a), paired scene bootstrap, per seed (42 / 123 / 2024):
    L2@3s   -0.291 [-0.351,-0.232] / -0.293 [-0.352,-0.234] / -0.302 [-0.360,-0.245]
    L2T@3s  -0.174 [-0.208,-0.142] / -0.177 [-0.209,-0.146] / -0.184 [-0.216,-0.152]
    ADE6    -0.371 [-0.453,-0.288] / -0.376 [-0.454,-0.296] / -0.375 [-0.453,-0.296]
    Col TemAvg@3s  -0.576% [-1.485,-0.055] / -0.381% [-1.100,+0.052] / -0.498% [-1.443,+0.052]
    Col NoAvg@3s   -0.899% [-2.210,-0.117] / -0.352% [-0.879,+0.059] / -0.781% [-2.130,+0.039]
  KIN (VAD feats) - KIN (ours): L2@3s -0.180 [-0.245,-0.115], L2T@3s -0.112
  [-0.149,-0.074], ADE6 -0.265 [-0.346,-0.183]

EXCLUDING THE FIRST FRAME OF SCENE (n=4,979; drops the 140 samples with no causal speed):
    method           L2 NoAvg 1/2/3s   L2 TemAvg 1/2/3s   Col% NoAvg      Col% TemAvg     ADE6 mean/med/p95   FDE6
    CV               0.401 1.209 2.420 0.281 0.630 1.118  0.10 0.32 1.73  0.14 0.22 0.62  3.314/2.657/8.699 7.999
    KIN (ours)       0.231 0.799 1.820 0.180 0.403 0.782  0.08 0.14 1.02  0.16 0.18 0.43  2.712/2.187/7.113 6.980
    KIN (VAD feats)  0.291 0.923 1.997 0.220 0.477 0.886  0.18 0.40 1.25  0.25 0.34 0.61  2.893/2.290/7.851 7.291
    MLP (a) s42      0.207 0.596 1.309 0.154 0.317 0.581  0.08 0.76 1.67  0.34 0.55 0.85  2.024/1.626/5.331 5.356
    MLP (a) s123     0.218 0.603 1.312 0.162 0.326 0.588  0.60 0.60 1.14  0.54 0.55 0.75  2.032/1.654/5.313 5.375
    MLP (a) s2024    0.211 0.607 1.320 0.156 0.324 0.590  0.60 0.66 1.61  0.54 0.62 0.88  2.030/1.655/5.381 5.359
    MLP (b) s42      0.222 0.627 1.344 0.165 0.334 0.604  0.08 0.22 0.92  0.09 0.15 0.34  2.064/1.695/5.437 5.440
    MLP (b) s123     0.244 0.625 1.347 0.177 0.340 0.608  0.10 0.18 0.98  0.16 0.27 0.44  2.069/1.695/5.422 5.456
    MLP (b) s2024    0.223 0.622 1.343 0.163 0.333 0.602  0.08 0.68 1.00  0.14 0.31 0.46  2.066/1.712/5.441 5.453
  (b)-(a), per seed (42 / 123 / 2024):
    L2@3s   +0.035 [+0.007,+0.065] / +0.035 [+0.005,+0.065] / +0.023 [-0.005,+0.052]
    L2T@3s  +0.022 [+0.009,+0.036] / +0.020 [+0.007,+0.033] / +0.012 [-0.001,+0.026]
    ADE6    +0.040 [-0.003,+0.085] / +0.037 [-0.003,+0.080] / +0.036 [-0.006,+0.078]
    Col TemAvg@3s  -0.505% [-1.431,+0.020] / -0.305% [-1.038,+0.134] / -0.425% [-1.393,+0.137]
    Col NoAvg@3s   -0.743% [-2.070,+0.060] / -0.161% [-0.684,+0.241] / -0.603% [-1.971,+0.220]
  KIN (VAD feats) - KIN (ours): L2@3s +0.177 [+0.140,+0.216], L2T@3s +0.104
  [+0.083,+0.127], ADE6 +0.181 [+0.140,+0.227]

## P3. First-frame stratum

From here on every table reports ALL 5,119 and EXCLUDING the first frame of each scene
without causal speed (n=140, v0 = 0 kept). The 140 = the first-of-scene samples with no
CAN message <= t0 and no past pose. The other 10 first frames have CAN and stay in.

## STEP 3.5a -- OVERFIT GATE

Setup: ego + cmd, tokenizer (ii), visual tokens removed, plain CE, Y1 recipe otherwise
(lr 5e-5 peak, 100-step warmup, cosine to 0.1x, LoRA dropout 0.1). Trained AND selected
(AR argmax median ADE@6s) on the same 256 official-train samples; 150 epochs x 10
steps; 0.36 s/step; ~2.5 h wall on 8 GPUs (most of it per-epoch validation and saving).
Teacher-forced token accuracy on the 256 is logged every epoch.
Log: Alpamayo/w1_overfit256.log.

Rule (planner): PASS iff the final AR median ADE@6s <= 1.0 m (floor on these 256:
0.426 mean / 0.092 median).

  RUN 1 RESULT: FAIL (marginal). Epoch 150 median 1.058 (mean 2.336). The best
  (selected) checkpoint, epoch 128, has median 0.899 (mean 2.231). The last 5 epochs
  swing 0.93-1.43. TF token accuracy 97.1% at epoch 150 (0.2% -> 17.7% @7 -> 51.7% @25
  -> 90.1% @37 -> 97%).

DIAGNOSIS (w1/overfit_diag.py, free-running dumps of both checkpoints on the 256;
log Alpamayo/w1_overfit_diag.log):

                         best (ep128)       latest (ep150)
    ADE@6s floor         0.426 / 0.092      0.426 / 0.092     (mean / median)
    ADE@6s AR argmax     2.231 / 0.899      2.336 / 1.058
    ADE@6s AR expect     2.383 / 1.228      2.445 / 1.465
    exact 24-token seq   0.434              0.426
    first wrong slot     slot 0: 131, slot 1: 8, others 6 (of 145 failures)
    all-correct samples  n=111: ADE 0.413 = their floor 0.413
    first error in accel n=145: ADE 3.623 (median 2.585)
    wrong tokens         85% are >= 4 bins off (5% off by one)

  The failure is ONE slot: accel slot 0, the only token predicted from the context
  alone (no token prefix). It is right in ~49% of samples. When it is right, the other
  23 tokens are recalled exactly and the sample hits its tokenizer floor. When it is
  wrong, the model continues ANOTHER sample's memorised sequence (far-off bins, errors
  cascade; per-slot AR accuracy is flat at ~0.47 across all 24 slots).
  The planner's categories use aggregate TF accuracy, but the aggregate hides the
  failure: slot 0 has no prefix under teacher forcing either, so its TF accuracy is
  the same ~49%. The remaining 23 slots are ~100% given the true prefix, i.e. the
  sequence is keyed on its first token. So this is an OPTIMISATION failure of the
  context -> first-token mapping, and the downstream cascade is exposure.
  Expectation decode does not rescue it (median 1.23 vs argmax 0.90).
  -> The one allowed retry (LoRA dropout 0, constant LR, same epochs) targets exactly
  this and was launched: tag overfit256_retry, log Alpamayo/w1_overfit256_retry.log.
  The ego-MLP keeps its own dropout 0.1 (only the specified knobs changed).

===========================================================================================
PLANNER RESPONSE after overfit run 1 (slot-0 diagnosis accepted; gate 3.5a redefined:
PASS = final AR median ADE@6s <= 1.0 m AND slot-0 TF accuracy >= 90%, plus an
INPUT-SHUFFLE test) + AMENDMENT (items e, f).
===========================================================================================

## Retry status

The first retry launch died at epoch 27/150: the DGX was shut down at 09:01 UTC and
rebooted at 12:16 UTC on a new kernel (5.15.0-179 -> -194), and all tmux sessions were
lost. The log and checkpoints are kept as *_KILLED_*. Relaunched from scratch at 12:55
UTC (w1/run_retry.sh: train -> dumps of final / final+shuffle / best). Epoch 1
reproduces the killed run exactly (median 20.5344).

## Item 4 -- FROZEN_RESULTS.md history

Added "History -- origin of the leak": 875d3a2 (2026-07-14) rewrote compute_ego_state
backward -> forward differences. It is the origin of the leak and of the reported
"-36%" (6.610 -> 4.236 m), which was measured with leaked inputs (commit 67987a3).

## Amendment e -- are the context tokens distinguishable? (run 1 final checkpoint, CPU)

Code: w1/diag_distinct.py; log: Alpamayo/w1_diag_distinct_run1.log. The 256 overfit
samples; ego tokens = EgoEncoderMLP(ego rows) in eval mode; RMSNorm = LM layer-0
input_layernorm (eps 1e-6).

    token      L2 norm   cos raw   cos RMSNorm   mean-share raw   mean-share RMSNorm
    ego t-3    19.452    0.7198    0.7199        0.6203           0.7216
    ego t-2    19.365    0.7203    0.7204        0.6217           0.7221
    ego t-1    19.541    0.7156    0.7158        0.6195           0.7174
    ego t      19.253    0.7294    0.7294        0.6415           0.7311
    cmd         1.192    0.7711    0.7712        0.7734           0.7724
    (cos = mean pairwise cosine across the 256 samples; mean-share =
     ||mean_i x_i||^2 / mean_i ||x_i||^2. cmd has only 3 distinct vectors; the 256
     split 21 right / 11 left / 224 straight.)

Pairwise cosine after RMSNorm is 0.72-0.73, below the 0.95 "near-indistinguishable"
line. The across-sample mean vector carries 62-64% of the ego token's squared norm (72-73%
after RMSNorm). The Cosmos text-embedding mean norm is 0.870; the ego tokens are 22x
that (diag_inputs.py (a)). The fix run's final checkpoint gets the same table, if a
fix run happens.

## Amendment f -- padding

No padding anywhere: every train/holdout record has ego_state (5,4) and 12 target pairs
(checked on all 18,313 + 1,417), the visual token count is fixed per run (0 here), the
batches are stacked by the default collate (which errors on unequal shapes), and the
AR decode uses batch size 1. The "left-pad" in ego features repeats rows INSIDE the
fixed 4 rows, not the sequence.

## Fix run -- prepared, NOT launched (only if the retry fails)

w1/fixrun.py + --fix in finetune_w1.py / dump_w1.py:
  - features standardised with official-train stats (buffers ego_encoder.in_mean/in_sd);
  - no absolute heading/position exists to drop (yaw relative to t0; no x/y);
  - post-MLP LayerNorm + learnable scalar gain, init 0.870/sqrt(3584), so the token
    norm = 0.87 at init;
  - ego-MLP LR x10 via LrMultAdamW. finetune.py resets every group's lr each step, and
    Adam is gradient-scale invariant, so the multiplier is applied inside step(). The
    cmd embedding keeps x1.
Unit test w1/test_fixrun.py PASS (init norm 0.87; buffers/gain saved; cmd excluded
from the x10 group; first Adam step exactly 10x; group lr restored).

## QUEUE C1 -- mini information ladder on the Ego-MLP (a SMALL-MODEL demonstration of
## the ladder logic; not the VLA)

Code: w1/ladder.py; log: Alpamayo/w1_ladder.log; predictions: results/w1_ladder.pkl.
Ego-MLP recipe of 3.5b (512-512 MLP, L1 on 12 lidar-frame waypoints, holdout selection
with TRUE labels), 3 seeds, CPU. Official val. Privileged inputs are marked [P].
L5n = L5 trained with 10% meta-action flips (seed 42, as A2), tested at 0/10/20/40%
flips (seed 777).

Seed mean +- sd (3 seeds):

    rung                                   ALL 5,119              EXCL. FIRST FRAMES 4,979
                                           ADE@6s   L2@3s NoAvg   ADE@6s   L2@3s NoAvg
    L0  CV (no learning)                   3.742    2.762         3.314    2.420
    L0  kinematic rule (no learning)       3.161    2.178         2.712    1.820
    L1  Ego-MLP, ego only                  2.591    1.738         2.123    1.368
    L2  Ego-MLP, ego + cmd [P]             2.439    1.637         2.027    1.310
    L2b Ego-MLP, cmd only [P]              9.849    9.103         9.845    9.099
    L5  Ego-MLP, ego + oracle lon+lat [P]  1.750    1.193         1.537    1.013
    L5n flip 0 / 10 / 20 / 40% [P]         1.820 2.033 2.246 2.693   1.559 1.768 1.966 2.369
        (L2@3s)                            1.241 1.388 1.552 1.868   1.015 1.161 1.314 1.599
    oracle-kinematic, flip 0/10/20/40 [P]  2.671 3.626 4.447 6.082   2.337 3.314 4.154 5.808
    (all seed sds <= 0.011; full four-convention tables with mean/median/p95 and
     collision in the log)

Gap closed = (X - L2)/(L5 - L2), seed-mean ADE@6s | L2@3s, ALL 5,119:
    L1 -0.222 | -0.227;  L5n flip0 +0.899 | +0.891;  flip10 +0.589 | +0.560;
    flip20 +0.280 | +0.191;  flip40 -0.370 | -0.519;  oracle-kinematic -0.338 | -0.717.
    (L2b is -10.8 | -16.8: without ego status the MLP has no speed, 9.85 m ADE.)

Paired scene-level bootstrap, seed 42, ALL 5,119 (L2@3s | ADE6):
    L1 - CV   -1.012 [-1.149,-0.877] | -1.136 [-1.314,-0.956]
    L1 - KIN  -0.429 [-0.504,-0.355] | -0.554 [-0.670,-0.439]
    L1 - L2   +0.116 [+0.087,+0.145] | +0.173 [+0.130,+0.218]   (the cmd is worth 0.12-0.17 m)
    L5 - L2   -0.443 [-0.506,-0.379] | -0.685 [-0.772,-0.600]   (the oracle lon adds 0.44-0.69 m)
    OKIN - L2 +0.321 [+0.165,+0.506] | +0.237 [-0.001,+0.534]
    (excl. first frames: L1-L2 +0.072 | +0.117; L5-L2 -0.296 | -0.484; OKIN-L2 +0.359 |
     +0.316, all CIs excluding 0)
Figures: Alpamayo/viz/review/f6_mini_ladder.{png,pdf}, f7_flip_rate.{png,pdf}.

## QUEUE G2 (part 1) -- GATE 3.5a on the RETRY (LoRA dropout 0, constant LR)

Pre-registered PASS = final AR median ADE@6s <= 1.0 AND slot-0 TF acc >= 0.90 AND the
shuffled slot-0 acc drops to <= half of unshuffled. w1/gate35.py; log
Alpamayo/w1_gate35_retry.log. 256 overfit samples; floor 0.426 mean / 0.092 median.

    checkpoint                   epoch  AR median ADE@6s (mean)   slot-0 acc  exact 24-token seq
    retry final                   150   0.447 (1.711)             0.598       0.559
    retry final, INPUT-SHUFFLED   150   4.696 (5.882)             0.047       0.027
    retry best                    144   0.430 (1.590)             0.637       0.590
    run 1 final                   150   1.058 (2.336)             0.488       0.426
    run 1 best                    128   0.899 (2.231)             0.488       0.434
    TF token accuracy (all 24 slots), retry epoch 150: 97.8%

  median 0.447 <= 1.0: yes;  slot-0 0.598 >= 0.90: NO;  shuffle 0.598 -> 0.047 (<= half): yes
  RETRY: FAIL (on slot-0 accuracy only). Removing dropout and LR decay moves slot 0
  from 0.49 to 0.60 and the median from 1.06 to 0.45 m. The model reads its input
  (shuffling collapses slot 0 to 0.05 and the median ADE to 4.7 m), but the
  context -> first-token mapping still misses 40% of the 256 samples.
  The fix run (w1/fixrun.py) started 15:29 UTC; it is gated with the same rule.

## QUEUE G2 (part 2) -- GATE 3.5a on the FIX RUN, and G3 recipe decision

Fix = standardised ego features (train stats) + post-MLP LayerNorm + scalar gain (token
norm 0.87 at init) + ego-MLP LR x10; otherwise run 1's recipe (LoRA dropout 0.1, cosine
LR). Trained 15:29-18:00 UTC. Log Alpamayo/w1_overfit256_fix.log; gate log
Alpamayo/w1_gate35_fix.log.

    checkpoint                 epoch  AR median ADE@6s (mean)   slot-0 acc  exact 24-token seq
    fix final                   150   0.098 (0.498)             0.992       0.965
    fix final, INPUT-SHUFFLED   150   4.696 (5.971)             0.039       0.016
    fix best                     97   0.095 (0.507)             0.984       0.957
    floor on these 256                0.092 median / 0.426 mean
    TF token accuracy (all 24 slots), epoch 150: 99.84%

  median 0.098 <= 1.0: yes;  slot-0 0.992 >= 0.90: yes;  shuffle 0.992 -> 0.039: yes
  FIX RUN: PASS. The median sits on the tokenizer floor, and shuffling the inputs
  destroys it, so the model reads its ego + cmd inputs.

G3 (pre-registered): fix passes -> Stage A uses the FIX recipe. Launched as
`RECIPE=fix bash w1/run_stageA.sh` (tmux stageA). It starts when the diagnostic-c
probe (tmux fix) ends, and runs A1, A2, A3, A0, pausing for each report.

## QUEUE G1 -- diagnostic c: gradient norms (200-step probe, run-1 recipe)

w1/finetune_w1.py --probe_grad: per-group gradient L2 norm at every optimizer step,
after GradScaler unscale and before clip_grad_norm_(1.0); ego + cmd, no visual tokens,
the same 256 samples, run-1 recipe. Log Alpamayo/w1_probe_grad.log; summary
Alpamayo/w1_probe_grad_summary.txt. 10 of 200 steps were fp16 overflow steps (NaN,
skipped by GradScaler).

    steps      ego_mlp     ego_xtok(cmd)   lora        output_head   traj_embed   ego_mlp/lora
    1-10       1.859e+01   2.543e+00       1.915e+01   1.084e+02     4.920e-01    0.971
    11-50      1.220e+01   1.508e+00       1.203e+01   3.622e+01     3.741e-01    1.022
    51-100     1.399e+00   2.714e-01       2.325e+00   7.656e+00     5.216e-02    0.675
    101-200    1.999e+00   4.580e-01       5.226e+00   8.118e+00     6.482e-02    0.446
    (medians over the step range)

Numbers only: the ego-MLP gradient is the same order as LoRA's (ratio 0.45-1.02). The
global norm (dominated by output_head, 100+ early) is far above the clip threshold 1.0,
so every step is clipped. AdamW normalises per parameter, so gradient magnitude alone
does not set the update size of the ego MLP; its LR does.

## QUEUE C4/C5 -- mini rung 3 on the Ego-MLP: predicted vs oracle intent (small-model
## demonstration; CPU; G4 untouched)

Code: w1/rung3.py; log Alpamayo/w1_rung3.log; results/w1_rung3.pkl. 3 seeds. The L5 and
L5-noise models of C1 were retrained with the identical recipe/seed and reproduce C1's
predictions exactly (max |dev| 0.00e+00 m); classifier seed s feeds L5/L5n seed s.

C4a. Meta-action classifier: MLP on causal ego (16) + cmd [P] (3) -> lon (4 classes),
CE, holdout-accuracy selection. Val accuracy 0.677 / 0.676 / 0.672 (majority "maintain"
0.392); excl. first frames 0.686 / 0.686 / 0.681. Confusion, all 5,119, summed over 3 seeds
(rows true, cols predicted):

                stop   accel   decel   maint   recall
    stop        2303     112      66      78   0.900
    accel        634    2254      90    1132   0.548
    decel         72      65    1205    1331   0.451
    maint        258     671     481    4605   0.766
  (excl. first frames: identical except stop row 2225/112/66/78 and accel row
   370/2185/90/1132; recall 0.897 / 0.579 / 0.451 / 0.767)

C4b/C5. Seed means; gap closed = (X - L2)/(L5 - L2) against C1's L2 and L5:

    ALL 5,119                              ADE@6s   L2 TemAvg 1/2/3 s     gap ADE   gap TemAvg3s
    L5 oracle lon [P] (C1)                 1.750    0.240 0.420 0.637     +1.000    +1.000
    L5 PREDICTED lon                       2.498    0.249 0.471 0.796     -0.087    -0.097
    L5-noise oracle lon [P] (C1)           1.820    0.236 0.422 0.650     +0.899    +0.910
    L5-noise PREDICTED lon                 2.534    0.245 0.469 0.798     -0.139    -0.117
    L5-noise confusion-shaped 10%          2.005    0.243 0.440 0.695     +0.630    +0.596
    L5-noise confusion-shaped 20%          2.210    0.250 0.463 0.749     +0.332    +0.227
    L5-noise confusion-shaped 40%          2.553    0.263 0.501 0.835     -0.166    -0.372
    L5-noise uniform flip 10/20/40% (C1)   2.033 / 2.246 / 2.693          +0.589 / +0.280 / -0.370
    L2 ego + cmd [P] (C1)                  2.439    0.246 0.466 0.782      0.000     0.000

    EXCL. FIRST FRAMES 4,979
    L5 oracle lon [P] (C1)                 1.537    0.184 0.332 0.522     +1.000    +1.000
    L5 PREDICTED lon                       2.089    0.162 0.329 0.601     -0.126    -0.258
    L5-noise oracle lon [P] (C1)           1.559    0.162 0.307 0.503     +0.954    +1.318
    L5-noise PREDICTED lon                 2.120    0.155 0.324 0.600     -0.190    -0.247
    L5-noise confusion-shaped 10/20/40%    1.731 / 1.908 / 2.210          +0.602 / +0.242 / -0.374
    L5-noise uniform flip 10/20/40% (C1)   1.768 / 1.966 / 2.369          +0.528 / +0.124 / -0.699
    L2 ego + cmd [P] (C1)                  2.027    0.157 0.321 0.585      0.000     0.000
  Realised confusion-shaped error rates: 0.102 / 0.201 / 0.398.

C4c. The classifier's ~32% error rate is marked on f7 (orange star). It lands where the
L5-noise curve has already fallen back to the no-intent L2 line.
Notes (numbers only): a classifier that sees the same inputs as L2 adds no information,
and its predicted intent scores slightly WORSE than L2 (-0.09 to -0.14 gap). At the same
error rate, confusion-shaped noise costs less than uniform flips (e.g. 2.005 vs 2.033 at
10%, 2.553 vs 2.693 at 40%).

## STAGE A -- A1: VLA, ego + cmd [P], FIX recipe, official split (seed 42)

Run: w1/finetune_w1.py --tag A1 --cmd --fix --no_vision, Y1 epochs/patience,
batch 3 x 8 GPUs, natural sampling, plain CE, tokenizer (ii), v0_can. Selection = AR
median ADE@6s on the fixed 400-sample holdout subset: best at epoch 10 (2.052), i.e.
the LAST epoch -- selection was still improving (epochs 4-10: 2.20 2.28 2.30 2.36 2.10
2.19 2.05), so A1 may be under-trained at 10 epochs.
Cost: 0.43 s/step, 763 steps/epoch; train 18:25-19:33 UTC (9.1 GPU-h on 8 V100);
dumps (holdout 1,717 + val 5,119 + shuffled val 5,119) 42 min (5.6 GPU-h); ~15 GPU-h total.
Logs: Alpamayo/w1_stageA_A1.log, w1_stageA_A1_dump.log, w1_gateA_A1.txt (all tables).
Decode: STOP-aware hybrid, tau = 0.7 chosen on holdout.

Official val, ALL 5,119 (ADE/FDE@6s on the n_fut=12 part):
    method          L2 NoAvg 1/2/3s   L2 TemAvg 1/2/3s   Col% NoAvg      Col% TemAvg     ADE6 mean/med/p95   FDE6   L2@3s med/p95
    CV              0.527 1.448 2.762 0.376 0.784 1.326  0.10 0.47 2.07  0.14 0.27 0.75  3.742/2.763/9.889  8.735  1.937/7.427
    KIN             0.362 1.049 2.178 0.278 0.562 0.998  0.08 0.29 1.39  0.16 0.23 0.57  3.161/2.240/8.199  7.750  1.449/5.636
    Ego-MLP+cmd     0.324 0.822 1.633 0.241 0.461 0.777  0.08 0.82 1.78  0.29 0.56 0.87  2.434/1.675/6.178  6.067  1.033/4.166
    A1 argmax       0.433 1.188 2.355 0.315 0.642 1.106  0.16 0.68 1.47  0.21 0.45 0.80  3.450/2.350/10.06  8.563  1.519/6.898
    A1 expect       0.397 1.066 2.138 0.292 0.583 1.003  0.64 1.13 2.01  0.57 0.80 1.17  3.130/2.111/8.518  7.802  1.336/5.679
    A1 hybrid       0.373 1.024 2.080 0.274 0.555 0.967  0.04 0.43 1.33  0.10 0.25 0.57  3.082/2.124/8.652  7.738  1.342/5.698
    A1 blend a=.70  0.395 1.063 2.097 0.287 0.581 0.993  0.04 0.35 1.29  0.10 0.24 0.56  3.018/2.141/8.233  7.388  1.406/5.653

EXCL. FIRST FRAMES (4,979):
    CV              0.401 1.209 2.420 0.281 0.630 1.118  0.10 0.32 1.73  0.14 0.22 0.62  3.314/2.657/8.699  7.999  1.889/6.675
    KIN             0.231 0.799 1.820 0.180 0.403 0.782  0.08 0.14 1.02  0.16 0.18 0.43  2.712/2.187/7.113  6.980  1.413/5.022
    Ego-MLP+cmd     0.206 0.595 1.306 0.153 0.316 0.580  0.08 0.74 1.61  0.30 0.53 0.80  2.022/1.626/5.319  5.355  1.007/3.679
    A1 hybrid       0.248 0.785 1.740 0.180 0.403 0.760  0.04 0.34 1.04  0.10 0.21 0.46  2.658/2.068/7.336  7.021  1.312/5.035

Paired scene-level bootstrap (hybrid; negative = A1 better), L2@3s | L2T@3s | ADE6:
  ALL    A1 - CV       -0.682 [-0.796,-0.568] | -0.359 [-0.410,-0.309] | -0.661 [-0.826,-0.495]
         A1 - KIN      -0.098 [-0.163,-0.032] | -0.032 [-0.055,-0.008] | -0.079 [-0.192,+0.039] p=0.18
         A1 - Ego-MLP  +0.447 [+0.379,+0.516] | +0.190 [+0.161,+0.219] | +0.648 [+0.545,+0.755]
         expect-argmax -0.217 [-0.284,-0.147] | -0.103 [-0.138,-0.067] | -0.320 [-0.402,-0.235]
  EXCL.  A1 - CV       -0.681 [-0.795,-0.567] | -0.359 [-0.410,-0.308] | -0.656 [-0.823,-0.486]
         A1 - KIN      -0.080 [-0.147,-0.013] | -0.022 [-0.046,+0.002] | -0.054 [-0.169,+0.067] p=0.38
         A1 - Ego-MLP  +0.434 [+0.368,+0.500] | +0.180 [+0.153,+0.208] | +0.637 [+0.534,+0.743]
Strata, A1 - KIN (L2@3s): straight -0.041 [-0.127,+0.042] (n=3488); turning -0.326
  [-0.531,-0.123] (n=638); stationary -0.152 [-0.221,-0.095] (n=993).
Blend (alpha 0.70, fit on holdout): - CV -0.665 [-0.756,-0.575] L2@3s, -0.725 ADE6;
  - KIN -0.081 [-0.121,-0.041] L2@3s, -0.143 [-0.216,-0.069] ADE6. Nulls: all select
  alpha 0 (gain 0.0000); forced real alpha costs -0.005 / -0.727 / -0.591 m (mean /
  gauss / perm).
Slot-0 accuracy (val) 0.202; INPUT-SHUFFLE: slot-0 0.202 -> 0.050, ADE6 3.082 -> 5.657
  (x1.84) -> PASSES the shuffle test (not within 5%), so A2, A3 and A0 all run.
Per-slot argmax accuracy (val): accel 0.20 0.18 0.17 0.17 0.16 0.16 0.16 0.16 0.15 0.15
  0.15 0.14 | curv 0.21 0.19 0.18 0.18 0.16 0.16 0.16 0.16 0.15 0.15 0.15 0.14. Smooth
  decay, no single-slot spike (no leak fingerprint). Free-running CE a1-11 = 5.675.

## STAGE A -- A2: VLA, ego + ORACLE meta-action [P] (lat = cmd, lon), 10% training flips

Run: --meta --meta_flip 0.1 --fix --no_vision, otherwise as A1 (2 extra tokens lat + lon,
no separate cmd token). Selection (holdout 400, AR median ADE@6s) by epoch:
2.16 2.38 1.79 1.90 1.94 1.68 1.81 1.90 1.63 1.67 -> best epoch 9 (1.631).
Cost: 0.36 s/step; train 20:29-21:38 UTC (9.2 GPU-h); dumps 93 min (holdout + val at
4 flip rates + shuffled val; 12.4 GPU-h); ~22 GPU-h total. Decode: hybrid, tau 0.3
(holdout). Full tables: Alpamayo/w1_gateA_A2.txt.

    ALL 5,119       L2 NoAvg 1/2/3s   L2 TemAvg 1/2/3s   Col% NoAvg      Col% TemAvg     ADE6 mean/med/p95   FDE6   L2@3s med/p95
    ORACLE-KIN [P]  0.477 1.145 1.955 0.348 0.658 1.022  0.12 0.70 1.58  0.14 0.35 0.70  2.671/1.708/8.901  6.342  1.181/6.466
    A2 argmax       0.395 1.006 1.899 0.288 0.563 0.928  0.02 0.51 1.27  0.14 0.36 0.63  2.773/1.918/8.625  6.953  1.188/5.781
    A2 expect       0.355 0.878 1.665 0.262 0.498 0.816  0.55 0.90 1.82  0.50 0.68 1.03  2.496/1.610/7.499  6.334  0.996/4.925
    A2 hybrid       0.345 0.864 1.651 0.255 0.487 0.805  0.02 0.29 1.02  0.11 0.21 0.48  2.454/1.631/7.576  6.222  1.017/5.005
    A2 blend a=.80  0.352 0.878 1.658 0.260 0.496 0.814  0.02 0.21 1.00  0.10 0.18 0.43  2.435/1.657/7.356  6.094  1.042/4.926
    EXCL. FIRST FRAMES 4,979
    ORACLE-KIN [P]  0.360 0.933 1.666 0.259 0.518 0.839  0.12 0.58 1.43  0.14 0.32 0.64  2.337/1.677/6.917  5.857  1.158/5.265
    A2 hybrid       0.249 0.681 1.391 0.183 0.370 0.646  0.02 0.24 0.88  0.11 0.20 0.44  2.131/1.594/6.257  5.677  1.002/4.300
    (CV / KIN / Ego-MLP / A1 rows as in the A1 section)

Paired scene-level bootstrap (hybrid), L2@3s | L2T@3s | ADE6:
  ALL    A2 - CV          -1.111 [-1.234,-0.989] | -0.521 [-0.576,-0.467] | -1.288 [-1.462,-1.116]
         A2 - KIN         -0.527 [-0.599,-0.456] | -0.194 [-0.221,-0.167] | -0.706 [-0.826,-0.590]
         A2 - ORACLE-KIN  -0.304 [-0.464,-0.171] | -0.218 [-0.288,-0.158] | -0.217 [-0.482,-0.006]
         A2 - Ego-MLP     +0.018 [-0.056,+0.092] | +0.028 [-0.006,+0.061] | +0.020 [-0.078,+0.120]  (n.s.)
         A2 - A1          -0.429 [-0.489,-0.370] | -0.162 [-0.188,-0.136] | -0.628 [-0.715,-0.539]
  EXCL.  A2 - KIN         -0.429 [-0.504,-0.356] | -0.135 [-0.163,-0.109] | -0.581 [-0.705,-0.460]
         A2 - ORACLE-KIN  -0.275 [-0.440,-0.139] | -0.193 [-0.265,-0.132] | -0.206 [-0.481,+0.012] p=0.068
         A2 - Ego-MLP     +0.085 [+0.016,+0.154] | +0.066 [+0.037,+0.096] | +0.110 [+0.015,+0.205]
         A2 - A1          -0.349 [-0.410,-0.288] | -0.113 [-0.139,-0.089] | -0.527 [-0.616,-0.437]
Pre-registered check "A2 must beat the kinematic rule AND the oracle-kinematic rule, CI
excluding 0": PASS on ALL 5,119 (L2@3s and ADE6 vs both). Excluding first frames, the
ADE6 margin over oracle-kinematic is -0.206 [-0.481,+0.012] (not significant); L2@3s
-0.275 remains significant. No STOP; the queue continues to A3.
Strata, A2 - KIN (L2@3s): straight -0.375 [-0.466,-0.286]; turning -1.075
  [-1.254,-0.892]; stationary -0.709 [-0.949,-0.541].
Blend a=0.80: - CV -1.104 L2@3s; - KIN -0.520 [-0.580,-0.462] L2@3s, -0.725 ADE6. Nulls
  select alpha 0; forced alpha costs -0.020 / -1.093 / -0.903.
Slot-0 acc 0.215; INPUT-SHUFFLE: ADE6 2.454 -> 6.069 (x2.47), slot-0 0.215 -> 0.043: passes.
Per-slot argmax accuracy: accel 0.21 0.17 0.18 0.17 0.17 0.16 0.16 0.15 0.15 0.15 0.14 0.14 |
  curv 0.21 0.19 0.19 0.18 0.17 0.17 0.16 0.16 0.15 0.15 0.15 0.15 (no spike);
  free-running CE a1-11 5.110 (A1 5.675).
FLIP-RATE SWEEP (test flips on lat+lon, seed 777; A2 trained at 10%), ALL 5,119:
    test flip                 0%      10%     20%     40%
    A2 ADE6                   2.454   2.580   2.676   3.000
    A2 L2@3s                  1.651   1.744   1.839   2.067
    ORACLE-KIN ADE6 (same)    2.671   3.626   4.447   6.082
    (C1 Ego-MLP L5-noise ADE6, lon-only flips: 1.820 / 2.033 / 2.246 / 2.693)

## STAGE A -- A3: A1 + turn-weighted sampling (old weights: turning 16.7% -> ~40%)

Run: --cmd --turn_weighted --fix --no_vision, otherwise as A1. Selection by epoch:
3.20 2.31 2.23 2.34 2.06 1.94 2.14 2.04 2.23 2.09 -> best epoch 6 (1.940).
Cost: 0.42 s/step; train 23:39-00:47 UTC (9.1 GPU-h); dumps 45 min (6.0 GPU-h); ~15 GPU-h.
Decode hybrid, tau 0.7. Full tables: Alpamayo/w1_gateA_A3.txt.

    ALL 5,119       L2 NoAvg 1/2/3s   L2 TemAvg 1/2/3s   Col% NoAvg      Col% TemAvg     ADE6 mean/med/p95   FDE6   L2@3s med/p95
    A3 argmax       0.369 1.014 2.099 0.270 0.549 0.966  0.04 0.53 1.70  0.12 0.33 0.75  3.155/2.182/8.785  8.033  1.343/5.771
    A3 expect       0.362 0.952 1.938 0.268 0.526 0.905  0.64 0.98 2.31  0.57 0.76 1.21  2.932/1.998/7.786  7.466  1.205/5.018
    A3 hybrid       0.344 0.919 1.894 0.254 0.504 0.877  0.04 0.31 1.64  0.10 0.23 0.64  2.896/2.003/7.853  7.420  1.207/5.108
    A3 blend a=.70  0.366 0.965 1.922 0.268 0.532 0.908  0.02 0.25 1.43  0.09 0.19 0.54  2.846/2.045/7.441  7.101  1.282/5.009
    EXCL. FIRST FRAMES 4,979
    A3 hybrid       0.222 0.685 1.558 0.162 0.354 0.674  0.04 0.22 1.41  0.10 0.18 0.54  2.477/1.959/6.752  6.711  1.178/4.460
    (CV / KIN / ORACLE-KIN / Ego-MLP / A1 / A2 rows as in the A1 and A2 sections)

Paired scene-level bootstrap (hybrid), L2@3s | L2T@3s | ADE6:
  ALL    A3 - CV       -0.868 [-0.994,-0.744] | -0.449 [-0.506,-0.391] | -0.847 [-1.021,-0.674]
         A3 - KIN      -0.284 [-0.353,-0.215] | -0.121 [-0.147,-0.096] | -0.265 [-0.382,-0.150]
         A3 - Ego-MLP  +0.261 [+0.213,+0.309] | +0.100 [+0.080,+0.121] | +0.462 [+0.392,+0.533]
         A3 - A1       -0.186 [-0.230,-0.143] | -0.090 [-0.108,-0.072] | -0.186 [-0.260,-0.113]
  EXCL.  A3 - KIN      -0.263 [-0.331,-0.194] | -0.108 [-0.132,-0.084] | -0.235 [-0.352,-0.121]
         A3 - Ego-MLP  +0.251 [+0.203,+0.298] | +0.094 [+0.074,+0.113] | +0.455 [+0.385,+0.525]
         A3 - A1       -0.182 [-0.227,-0.138] | -0.086 [-0.104,-0.068] | -0.182 [-0.258,-0.106]
Strata, A3 - KIN (L2@3s): straight -0.203 [-0.287,-0.121]; turning -0.832 [-1.117,-0.557];
  stationary -0.216 [-0.307,-0.144]. (A1 - KIN: -0.041 n.s. / -0.326 / -0.152.)
Blend a=0.70: - KIN -0.256 [-0.303,-0.210] L2@3s, -0.315 [-0.400,-0.233] ADE6.
Slot-0 acc 0.211; INPUT-SHUFFLE: ADE6 2.896 -> 6.215 (x2.15), slot-0 0.211 -> 0.046: passes.
Per-slot argmax accuracy: accel 0.21 0.18 0.17 0.16 0.16 0.15 0.15 0.14 0.14 0.14 0.14 0.14 |
  curv 0.24 0.21 0.19 0.18 0.17 0.16 0.16 0.15 0.15 0.14 0.14 0.14 (no spike); CE a1-11 5.540.
