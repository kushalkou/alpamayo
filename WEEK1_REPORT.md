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
