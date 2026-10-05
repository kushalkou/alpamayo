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
