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
