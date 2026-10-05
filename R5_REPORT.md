# R5_REPORT -- 2 Hz relabel + M1-v2a (track B); scene-type and baseline checks (track A)

HEADER (all new rules / thresholds fixed and committed BEFORE computing; val is read for
analysis only; nothing selects or tunes on val; CoC arm untouched)
B1 2 Hz relabel. Source = GT target controls (w1/targets: chord speed s_k over 0.5 s step
  k, acc_k = (s_k - s_k-1)/0.5, cur_k = heading change / (s_k * 0.5)). Slot t (t0+1 ..
  t0+6 s) uses step k = 2t - 1 (the step ending at t). Conversion of the R1.2 10 Hz rules:
  the 10 Hz rule used a TRAILING 0.5 s window (a = (v(t) - v(t-0.5)) / 0.5, yaw rate
  averaged over 0.5 s), which is exactly one 2 Hz step, so every threshold carries over
  numerically unchanged: stop |v| <= 0.2 m/s; reverse v < -0.2 (chord speed >= 0, so
  never); strong acc a >= 2.0, gentle 0.5 <= a < 2.0, maintain |a| < 0.5, gentle dec
  -2.0 < a <= -0.5, strong dec <= -2.0 m/s^2; curvature 0 if v < 1 m/s; straight |k| <
  0.01, steer 0.01-0.1, sharp >= 0.1 1/m. v = s_k (segment chord speed) instead of CAN
  vel; a, k from the chord instead of CAN accel / yaw rate. Implementation:
  ar1/r33_eval.derive (unchanged). Slots beyond n_fut repeat the last valid slot.
  GT-vs-GT consistency (GB1) = GT-trajectory-derived words vs the new labels, all 12
  slots, val n_fut = 12. It is 1.0 BY CONSTRUCTION (same derivation of the same
  controls). Side number (not a gate): the same check on the trajectory rolled out from
  the GT TOKENS (64-bin tokenizer), i.e. what a perfect token model can express.
B2 M1-v2a: ar1/finetune_meta.py --labels 2hz, seed 42; everything else as M1 (A3 recipe,
  B1 cameras, 10 epochs, holdout AR median ADE@6s selection on the fixed 400). Gates:
  GB2 |generated all-majority share - label all-majority share| <= 10 points (val n_fut =
      12; majority sequence = maintain x6, straight x6).
  GB3 F1 > 0 for every class (lon and lat separately) with >= 30 val label occurrences,
      pooled over the 6 slots, val n_fut = 12.
  GB4 L2@3s NoAvg M1-v2a - B1 s42, paired scene bootstrap, ALL 5,119 (indicative).
  GB5 consistency (trajectory-derived words == generated words, all 12 slots), all val
      5,119 (hybrid decode, tau from holdout) >= 0.70.
  Collapse share = generated all-majority share. Per-class P / R / F1 pooled over slots.
  False-go as R3 / R4. Strata as A3.
A2 WSS (whole-scene stationary): every consecutive pair of keyframes of the scene (all
  keyframes in sample.json, LIDAR_TOP ego poses) has displacement / dt < 0.5 m/s.
  "parking" scenes: description contains "parking" (case-insensitive).
  near = ego within 10 m of a stop-line, crosswalk or intersection polygon (as R4 1d);
  GT-go within 3 s = GT max displacement over steps 1-6 > 1.0 m (R3).
  GA2: train < 5 WSS scenes -> "train/val scene-type shift supported".
A1 R4 shuffle (B1 s42, B1_statshuf): recipient = sample; donor = the sample whose cached
  tokens it received (dump_w1 permutation: RandomState(99).permutation over the val
  stationary records in records order). False-go on GT-stopped recipients by donor type
  x recipient type. GA1 metric = false-go(donor WSS) - false-go(donor non-WSS) over all
  GT-stopped recipients; >= 20 points -> "image content drives false-go".
A3 Paired scene bootstrap (10,000) on L2@3s NoAvg and ADE@6s: no cam 3-seed mean vs B1
  3-seed mean; B1 3-seed vs M1 3-seed; B1 s42 vs B2 s42. Subsets (a) all, (b) excluding
  WSS scenes [POST-HOC], (c) WSS scenes only [POST-HOC]; each with and without first
  frames. GA3 on (b) with first frames: B1 - no cam L2@3s CI includes 0 -> "camera harm
  is scene-type specific".
A4 Li et al. CVPR 2024 (BEV-Planner) protocol: definitions taken from their paper / repo
  (quoted in the section). Rows CV, GoStraight, Ego-MLP; TemAvg and NoAvg; with the
  published converter features and with our corrected (causal) ones; deltas vs their
  Table 1 numbers as quoted.
A5 Experts (saved 6 samples; no training): minADE_6 (@3 s, @6 s); single-sample ADE =
  mean over the 6 draws of each draw's ADE; mean-of-6 trajectory; selected = the draw
  nearest (mean L2 over 12 steps) to the token-decoder trajectory of the same model;
  spread = mean pairwise L2 between draws at 3 s and at 6 s; stationary-hold rate = share
  of GT-stopped stationary samples whose prediction stays < 0.5 m within 3 s; collision
  = current metric (VAD port) and yaw-aware box collision (ego box 4.08 x 1.85 m rotated
  by the heading of each predicted segment, same occupancy grids).
A6 First speed word = lon word of slot t0+1 s; GT = CAN label (M1's training target);
  stationary val, seeds pooled.
ADDENDUM HEADER (fixed before computing C1 / C2)
C1 comfort (nuPlan bounds) on the predicted 2 Hz trajectory, val n_fut = 12 (4,219), 6 s:
  points p0 = (0, 0) at t0, p1..p12 at 0.5 s steps; dt = 0.5. Backward differences:
  v_k = (p_k - p_k-1) / dt (k = 1..12); speed s_k = |v_k|, s_0 = v0_can; heading h_k =
  atan2(v_k) if s_k > 0.1 m/s else h_k-1, h_0 = pi/2 (heading at t0 in this frame);
  lon accel a_k = (s_k - s_k-1) / dt (k = 1..12); yaw rate r_k = wrap(h_k - h_k-1) / dt
  (k = 1..12); lat accel = s_k * r_k; yaw accel = (r_k - r_k-1) / dt (k = 2..12); lon
  jerk = (a_k - a_k-1) / dt (k = 2..12); jerk = |A_k - A_k-1| / dt with the 2D
  acceleration A_k = (v_k - v_k-1) / dt (k = 2..12, v_0 = v0_can along h_0), jerk for
  k = 3..12. Bounds: lon accel in [-4.05, 2.40], |lat accel| <= 4.89, |yaw rate| <= 0.95,
  |yaw accel| <= 1.93, |lon jerk| <= 4.13, |jerk| <= 8.37. A trajectory is comfortable
  if every value at every step is within bounds; per-bound pass rates also reported.
  CAVEAT: 2 Hz finite differences make jerk (and yaw accel) crude: they are second / third
  differences over 0.5 s, far coarser than nuPlan's 10 Hz with filtering.
C2 B2 vs B1 false-go (seed 42) on WSS scenes only (GT-stopped stationary samples), with
  the paired scene bootstrap of the per-sample indicator difference.

GATE VERDICTS (pre-registered wording; raw outputs Alpamayo/r5_*.txt)
  GB1 PASS  GT-vs-GT consistency 1.000 (by construction) >= 0.90 -> B2 run. Side: GT-token
            rollout vs 2 Hz labels 0.808 (val, all 12 slots).
  GB2 FAIL  generated all-majority share 0.408 vs label share 0.023 (38.5 points > 10).
  GB3 FAIL  F1 = 0 for lon strong_acc (support 1,355) and strong_dec (1,188).
  GB4       M1-v2a - B1 s42 L2@3s -0.288 [-0.434,-0.160] (one seed, indicative, no claim).
  GB5 PASS  consistency 0.854 >= 0.70 (M1 with 10 Hz labels: 0.410).
  GA1       +9.5 points (< 20) -> "not image-content driven".
  GA2       train has 44 WSS scenes (not < 5): "train/val scene-type shift supported"
            does NOT apply.
  GA3       (b) excl. WSS: B1 - no cam L2@3s +0.036 [-0.013,+0.094], CI includes 0 ->
            "camera harm is scene-type specific" [POST-HOC subset].
  A8        not available (no AR1 PDF in the repo or on disk).
  Bug check (result looked too good): the M1-v2a gain replicates on HOLDOUT (L2@3s 1.627
  vs B1 1.777, no cam s42 1.711; ADE@6s 2.360 vs 2.592; r5_b2_holdout_check.txt); decoding
  uses GT only to log the GT token's log-prob, never to choose a token.

B1  2 Hz RELABEL (val n_fut = 12; train similar, r5_b1_relabel.txt)
  lon shares: gentle acc 0.201, strong acc 0.054, gentle dec 0.189, strong dec 0.047,
  maintain 0.361, stop 0.149 (10 Hz: 0.192 / 0.003 / 0.161 / 0.005 / 0.493 / 0.146).
  lat shares: straight 0.815, steer L / R 0.082 / 0.087, sharp L / R 0.007 / 0.010.
  all-majority share 0.023 (10 Hz 0.168). Agreement with 10 Hz labels: lon 0.688-0.695,
  lat 0.942-0.949 per slot. Logged in DEVIATIONS.md.

B2  M1-v2a seed 42 (M1 recipe, 2 Hz labels only). Best epoch 6 (holdout 1.732); 3.1 h
  train + 0.9 h dumps. Official val (L2 m, Col %):
                         L2@3s No/Tem  Col 3s No/Tem  ADE@6s mean/med  FG all/WSS/rest
  no cam 3-seed [P]      1.954 0.909   1.45 0.60      2.930 2.109      0.022 (3-seed)
  B1 s42 [P]             2.165 1.033   1.45 0.49      3.125 2.296      0.302 0.394 0.205
  M1 s42 (10 Hz) [P]     2.203 1.043   1.54 0.61      3.183 2.295      0.317 0.391 0.239
  M1-v2a s42 (2 Hz) [P]  1.877 0.884   1.58 0.57      2.808 1.883      0.061 0.012 0.112
  missed-go: B1 0.424, M1 0.473, M1-v2a 0.530. Excl. first frames: M1-v2a L2@3s 1.558.
  M1-v2a - B1 s42, L2@3s: (b) excl. WSS -0.180 [-0.270,-0.098]; (c) WSS -1.786
  [-3.179,-0.425] [both POST-HOC]; excl. first frames -0.314 [-0.461,-0.185].
  Words vs 2 Hz labels, pooled over slots: maintain P 0.473 R 0.766 F1 0.585; stop 0.645 /
  0.885 / 0.746; gentle acc 0.468 / 0.241 / 0.318; gentle dec 0.460 / 0.261 / 0.333;
  strong acc / dec 0.000; lat straight F1 0.910, steer L / R 0.379 / 0.426, sharp L / R
  0.129 / 0.268. Slot accuracy lon 0.603 -> 0.449, lat 0.895 -> 0.782 (t+1 -> t+6 s).
  Distinct generated sequences 345 (labels 2,893). Consistency stationary 0.904, moving
  0.842. Comfort (C1) 0.985.

A1  SHUFFLE BY SCENE TYPE (B1 s42, GT-stopped recipients; donor = images received)
                         WSS donor        non-WSS donor     unshuffled
  WSS recipient          0.379 (n 124)    0.290 (n 221)     0.394
  non-WSS recipient      0.394 (n 94)     0.291 (n 237)     0.205
  all recipients         0.385 (n 218)    0.290 (n 458)     0.302
A2  WSS scenes: train 44 (1,509 of 22,213 samples), holdout 6 (206), val 10 (345:
  scene-0100, 0344, 0553, 0554, 0770, 0771, 0775, 0777, 0781, 0784). "parking" in the
  description: train 156, holdout 10, val 33 scenes (lists in r5_trackA.txt).
  Stationary GT-go within 3 s: train near 0.209 (n 4,204), away 0.384 (n 513); val near
  0.330 (n 693), away 0.180 (n 300); WSS 0.000 in both.
A3  Paired scene bootstrap, L2@3s (ADE@6s in r5_trackA.txt); (b), (c) POST-HOC
  pair / subset          (a) all               (b) excl. WSS         (c) WSS only
  B1 - no cam (3 seeds)  +0.137 [+.036,+.255]  +0.036 [-.013,+.094]  +1.538 [+.385,+2.727]
    excl. first frames   +0.175 [+.076,+.291]  +0.077 [+.028,+.134]  +1.529 [+.383,+2.712]
  M1 - B1 (3 seeds)      +0.087 [+.005,+.179]  +0.020 [-.040,+.089]  +1.010 [+.289,+1.751]
  B2 - B1 (s42)          -0.035 [-.109,+.037]  -0.029 [-.110,+.049]  -0.125 [-.298,+.010]
  C2: B2 - B1 false-go on WSS (n 345) +0.009 [+0.000,+0.020]; non-WSS +0.027
  [-0.066,+0.145]. The 2 s camera history does not reduce false-go on WSS scenes.
A4  Li et al. protocol (0.1 m grid, yaw-aware box +0.986 m, max-over-horizon collision,
  L2 TemAvg; our frame), ALL 5,119; avg = mean of 1 / 2 / 3 s:
                          L2 1/2/3 s (avg)       Col % 1/2/3 s (avg)    d vs Li L2; Col
  GoStraight, v0 causal   0.38 0.78 1.33 (0.83)  0.12 1.04 3.36 (1.50)  -0.00; +0.42
  GoStraight, v0 VAD conv 0.28 0.63 1.11 (0.67)  0.12 0.88 2.97 (1.32)  -0.16; +0.24
  Ego-MLP causal, 3 s.    0.25 0.47 0.78 (0.50)  0.02 0.40 1.37 (0.60)  +0.15; +0.22
  Ego-MLP VAD conv, 3 s.  0.17 0.34 0.60 (0.37)  0.04 0.29 1.36 (0.56)  +0.02; +0.19
  Li Table 1 (as fetched): GoStraight 0.38 0.79 1.33 (0.83), Col 0.15 0.60 2.50; Ego-MLP
  0.15 0.32 0.59 (0.35), Col 0.00 0.27 0.85. VAD-port TemAvg / NoAvg rows in r5_a4.txt
  (Ego-MLP reproduces T4: L2@3s NoAvg 1.640 causal, 1.345 VAD converter).
A5  Experts (saved draws; numpy collision equals the stored VAD-port values exactly):
                           ADE 3s/6s    minADE6 3/6s  spread 3/6s  hold   Col No3 aa/yaw
  A3 token                   0.877 2.896  -              -             0.944  1.64 / 1.86
  A3 expert single draw      1.080 3.356  0.576 1.714    1.94 7.42     0.784  2.38 / 2.45
  A3 expert mean of 6        0.895 2.760  -              -             0.506  1.50 / 1.54
  A3 expert nearest token    0.911 2.803  -              -             0.951  1.74 / 1.76
  M1 token                   1.043 3.183  -              -             0.669  1.54 / 1.56
  M1 expert single draw      1.048 3.090  0.871 2.656    0.49 1.36     0.676  2.25 / 2.17
  M1 expert mean of 6        1.022 3.046  -              -             0.678  2.34 / 2.25
  M1 expert nearest token    1.020 3.031  -              -             0.673  2.13 / 2.09
  TemAvg 3 s collision and readout details in r5_a5.txt.
A6  M1 (10 Hz labels), stationary val, first speed word, seeds pooled (rows GT, cols gen):
  GT stop (2,238): stop 1,577, maintain 583, gentle acc 77, gentle dec 1; GT gentle acc
  (342): stop 159, gentle acc 136, maintain 46; GT maintain (306): stop 138, maintain 126,
  gentle acc 41. Accuracy 0.617. Strong / reverse words never generated.
A7  Turn signal = CAN vehicle_monitor left_signal / right_signal, last message <= t0
  (2 Hz; age median 0.25 s). Coverage train 0.947, val 0.970; on: left 0.074 / 0.075,
  right 0.092 / 0.072 (train / val).
C1  Comfort share (all six nuPlan bounds), val n_fut = 12: GT 0.415 (2 Hz jitter: the
  metric is crude); no cam 0.777 / 0.700 / 0.892; B1 0.690 / 0.723 / 0.714; M1 0.654 /
  0.772 / 0.685; B2 0.695; M1-v2a 0.985; A3 expert mean-6 0.935, single draw 0.519; M1
  expert mean-6 0.891, single draw 0.490 (per-bound rates in r5_c1.txt).
GPU: B2 only (M1-v2a train + dumps, ~31 GPU-h); track A used no GPU.
