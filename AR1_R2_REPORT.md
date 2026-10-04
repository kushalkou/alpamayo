# AR1_R2_REPORT -- reproduction phase R2 (gate report)

Written 2026-10-03. Code: Alpamayo/code/ar1/; outputs: Alpamayo/ar1_*.txt. Pre-registered
rules applied without the planner. [P] = privileged input (VAD command, from GT future).

=========================================================================================
R2.0 PATCH-ORDER BUG -- confirmed and quarantined (c75c7af)
=========================================================================================
a. 5 val images from 5 scenes (CAM_FRONT / FRONT_LEFT / FRONT_RIGHT), frozen tower fp16
   (Alpamayo/ar1_r20_patch.txt):
     NEW R1 path vs HF Qwen2.5-VL processor + tower: token cosine 1.00000 on all 5
       (min 1.00000) -> PASS (>= 0.999)
     OLD week-1 path vs HF (448x448): mean token cosine 0.410 / 0.493 / 0.341 / 0.515 /
       0.327; min token cosine -0.018 .. 0.113
b. PATCH_ORDER_QUARANTINE.md lists every affected result: custom split (original cached
   training, RESULTS_OVERNIGHT full / vision-only rows, every y1_full row, FINDING 4,
   OVERNIGHT2 6b causal full vision, x1 attention, BRIEF) and official split (V8b: T10a,
   T10b, C7 V8b rows, f6 / f7 / f8 markers; G7: T11a-c, camera shuffle). Marked
   "INVALID (patch order)" in FROZEN_RESULTS.md and REVIEW_RESULTS.md; numbers kept.
   Zeroed / removed visual tokens never use the encoder: those results stand.

=========================================================================================
R2.1 BENCHMARK, live encoding, 3 cams x 4 frames (c9dd833; Alpamayo/ar1_r21_bench.txt)
=========================================================================================
  1,920 visual + 4 ego + 1 cmd = 1,925 context, 1,948 LM tokens per sample
  batch 2 x 8 V100: peak allocated 21.4 GB, reserved 32.8 GB (max over ranks)
  5.26 s/step mean (median 5.28; steps 11-60); 1,145 steps/epoch
  10 epochs = 16.7 h wall = 134 GPU-h (training only; + holdout AR selection)
  With the R2.2 cache (no live tower), B2 at batch 3 on 1 GPU: 6.2 s/step, peak 23.6 GB
  (smoke, 10 steps): 763 steps/epoch on 8 GPUs -> ~13 h wall for 10 epochs (+ selection).

=========================================================================================
R2.2 VISION CACHE (b8bc570; Alpamayo/ar1_r22_vcache_*.txt)
=========================================================================================
  One file per unique keyframe image of the 3 front cameras referenced by any history
  slot: 87,147 files (29,049 per camera), [160, 3584] fp16 = 1,147,136 B each,
  total 99,967,944,064 B (94 GiB) in Alpamayo/data/ar1_vcache/<CAM>/<image>.npy.
  Build: 8 GPUs, batch 16 per GPU, 480 s wall (~1.1 GPU-h incl. model load).
  Samples gather their files at load time (ar1/vcache.gather; t0 = 3 files, history =
  12 files).
  Unit test (ar1/vcache.py --test 50): PASS.
    (a) 3 build batches (48 images) recomputed: bit-exact equal to the cache; 0 missing.
    (b) single-image live encode vs cache: per-image mean cosine >= 0.99960 (median
        0.99997); per-token min cosine median 0.9990, worst 0.9495. This is fp16
        batch-shape noise: the same image encoded alone vs in a batch of 4 differs by the
        same order (ar1/batch_check.py: min 0.99886), and an fp32 tower differs from the
        fp16 one by more (per-token min 0.665, per-image mean >= 0.9959). The cache is
        therefore exactly "live encoding in batches of 16"; the first-try criterion
        (cos >= 0.9999 vs single-image live) was too strict for fp16 and was replaced by
        the bit-exact batch check.
  Side note: fp16 vision-tower outputs carry token-level error vs fp32 (some tokens
  cos 0.46-0.67). V100 has no bf16; every vision path here (and the HF reference in
=========================================================================================
R2.3 FLOW-MATCHING EXPERT on frozen A3 (our Table 12 analog; Alpamayo/ar1_r23_eval.txt)
=========================================================================================
Build (ar1/expert.py, kv_a3.py, train_expert.py). Option A as AR1: 28 expert layers, one
  per VLM layer. Each layer attends over [frozen A3 per-layer KV-cache (stop-grad) |
  its 12 action tokens]. 28 q / 4 kv heads, head dim 128, hidden 512, SwiGLU 1,536, RoPE
  at positions ctx_len + i (verified against the cached keys: layer-0 max |diff| 5.3e-3 on
  |k| <= 171). 184.2M params, fp32. Conditional flow matching on the Gaussian OT path,
  t ~ U(0,1). Inference: 10 Euler steps (dt = 0.1); unicycle rollout from v0_can (the
  token decoder's rollout).
  A3 context = 4 ego + 1 cmd token (no cameras), so the KV-cache was precomputed once
  (286 KB per sample; identical to stop-grad online).
  Targets: 12 x (accel, curvature) chord controls, clipped to +-6 m/s^2 and +-0.3 1/m
  (rollout floor 0.015 m mean ADE), then standardised. The clip was added after the first
  launch showed curvature sd 13.5 and before any evaluation.
  Selection (pre-registered): HOLDOUT median ADE@6s of one fixed-noise sample, every 5
  epochs, patience 4. Best epoch 20 of 40 (holdout 2.2685). Official val was used only
  for the final numbers.
Cost actuals: KV precompute 67 s on 1 GPU; training 0.41 GPU-h (1 V100); 6-sample
  inference for holdout + val 4.0 GPU-min. In total about 0.5 GPU-h, vs about 6-9 GPU-h
  for a no-camera token run (V8a 6 GPU-h, A0 8.7 GPU-h).
Official val, ALL 5,119 (EXCL. first frames n=4,979 in the txt). L2 in m, Col in %:
                        L2@3s NoAvg L2@3s TemAvg Col% 3s No/Tem ADE@6s mean med p95 FDE6
  KIN                       2.178      0.998     1.39 / 0.57   3.161 2.240 8.199  7.750
  Ego-MLP + cmd s42 [P]     1.633      0.777     1.78 / 0.87   2.434 1.675 6.178  6.067
  A3 token decoder [P]      1.894      0.877     1.64 / 0.64   2.896 2.003 7.853  7.420
  A3 + expert, 1 sample     2.257      1.067     2.40 / 1.09   3.336 2.522 8.102  8.367
  A3 + expert, mean of 6    1.875      0.895     1.50 / 0.82   2.760 2.002 6.571  6.868
                          ADE@3s   minADE_6@3s   minADE_6@6s  (deterministic rows: = ADE)
  KIN                       0.998     0.998         3.161
  Ego-MLP + cmd s42 [P]     0.777     0.777         2.434
  A3 token decoder [P]      0.877     0.877         2.896
  A3 + expert [P]           1.067     0.576         1.714
Paired scene bootstrap, ALL (95% CI):
  expert 1 sample - token   L2@3s +0.362 [+0.306,+0.418]  ADE6 +0.440 [+0.353,+0.526]
  expert mean-6 - token     L2@3s -0.019 [-0.075,+0.038]  ADE6 -0.136 [-0.221,-0.048]
  expert 1 sample - MLP     L2@3s +0.624 [+0.570,+0.675]  ADE6 +0.902 [+0.820,+0.983]
  expert 1 sample - KIN     L2@3s +0.078 [-0.005,+0.160]  ADE6 +0.176 [+0.046,+0.303]
  expert minADE_6@6s - token ADE@6s  -1.182 [-1.281,-1.080]
EXCL. first frames: same signs. Expert 1 sample - token L2@3s +0.386 [+0.328,+0.445];
  mean-6 - token ADE6 -0.116 [-0.203,-0.027].
Strata, L2@3s (token / expert 1 sample / MLP): straight 1.591 / 2.009 / 1.367; turning
  2.580 / 2.656 / 1.868; stationary 2.518 / 2.870 / 2.416 (excl. first frames 0.655 /
  1.146 / 0.637).
Reading: one expert sample is worse than the token decoder (+0.36 m L2@3s); the mean of
  6 samples matches it (L2@3s n.s., ADE6 -0.14 m). minADE_6 is far lower (1.71 vs
  2.90 m), as expected for 6 diverse samples; it is not comparable to a 1-sample ADE.
  The expert is weakest when stationary (no exact STOP: noise leaves small motions).
  Neither decoder reaches the Ego-MLP (1.633). Single seed, single run.

=========================================================================================
R2.4 NEW CAMERA BASE (B1, B2) -- Alpamayo/ar1_r24_eval.txt
=========================================================================================
Runs (A3 recipe: fix, cmd [P], turn-weighted, plain CE, batch 3 x 8, 10 epochs, seed 42,
AR median-ADE selection on 400 holdout samples; cached native-order vision):
  B1  3 front cams at t0 (480 visual tokens): 1.15 s/step, 08:06-10:55 UTC (~22.5 GPU-h
      train), best epoch 10 (holdout 1.899); dumps 44 min.
  B2  3 front cams x 4 keyframes (1,920 tokens): 6.28 s/step, peak 23.7 GB, 10-03 11:39 -
      10-04 01:41 UTC (~112 GPU-h train), best epoch 9 (holdout 1.951); dumps 68 min.
  Token feeding checked: the dataset returns the cached tokens of the sample's own
  frames (timestamps <= t0), equal to a direct gather.
Official val, ALL 5,119 (hybrid decode, tau fit on holdout: B1 0.3, B2 0.7):
                          L2@3s NoAvg TemAvg  Col% 3s No/Tem  ADE@6s mean med p95  FDE6
  VLA no cam 3-seed [P]     1.954    0.909    1.45 / 0.60   2.930 2.109 7.371  7.436
  B1 3 cams t0 [P]          2.165    1.033    1.45 / 0.49   3.125 2.296 7.716  7.766
  B2 3 cams x 4 kf [P]      2.130    1.018    1.47 / 0.59   3.093 2.219 8.102  7.650
  B1 cams shuffled          2.302    1.120    2.52 / 1.05   3.280 2.466 8.042  8.019
  B2 cams shuffled          2.250    1.084    2.11 / 0.84   3.230 2.417 8.046  7.963
  Ego-MLP + cmd 3-seed [P]  1.637    0.779    1.80 / 0.87   2.437 1.679 6.112  6.072
Paired scene bootstrap, L2@3s NoAvg (95% CI), ALL / EXCL. first frames:
  B1 - no cam       +0.211 [+0.081,+0.362]  /  +0.254 [+0.125,+0.401]
  B2 - no cam       +0.176 [+0.069,+0.305]  /  +0.214 [+0.108,+0.340]
  B2 - B1           -0.035 [-0.109,+0.037]  /  -0.040 [-0.114,+0.033]
  B1 shuffled - B1  +0.137 [-0.001,+0.277]  /  +0.132 [-0.004,+0.271]
  B2 shuffled - B2  +0.120 [+0.004,+0.233]  /  +0.112 [-0.005,+0.227]
Strata, L2@3s (no cam / B1 / B2), ALL: straight 1.678 / 1.810 / 1.758; turning
  2.570 / 2.545 / 2.706; stationary (993) 2.529 / 3.166 / 3.064. Stationary excl. first
  frames (853): 0.664 / 1.621 / 1.479; B1 - no cam +0.956 [+0.287,+1.675], B2 - no cam
  +0.815 [+0.272,+1.421].
  B2 - B1: straight -0.052 [-0.113,+0.009]; turning +0.161 [+0.015,+0.306]; stationary
  -0.102 [-0.400,+0.194].
Camera shuffle: ADE@6s x1.049 (B1), x1.044 (B2) on ALL; x1.053 / x1.046 excl. first
  frames. G7's rule (within 5% => unused) is borderline; the paired L2@3s differences
  are +0.12 to +0.14 m, with CIs touching 0.
Per-slot argmax accuracy (val, mean accel / curvature slot): no cam 0.156 / 0.168;
  B1 0.122 / 0.135; B2 0.127 / 0.137; shuffled B1 0.094 / 0.103, B2 0.113 / 0.120.
  Slot 0 accel: 0.205 / 0.176 / 0.183. The camera models are less accurate at every slot.
  Shuffling lowers accuracy further, so the cameras are read, but they do not help.
PRE-REGISTERED DECISION: B2 - B1 L2@3s CI includes 0 on ALL (-0.035 [-0.109,+0.037])
  and on stationary (-0.102 [-0.400,+0.194]) -> base input = B1.
Reading: with the correct patch order, cameras still make the token VLA worse than
  no cameras (+0.18-0.21 m L2@3s), most of all when stationary. The patch-order fix does
  not reverse the week-1 camera verdict for this recipe. Single seed per arm vs a 3-seed
  reference; the no-camera seed sd was 0.044 m.

=========================================================================================
R2.5 nuScenes MAP EXPANSION
=========================================================================================
Source: nuScenes download page (nuscenes.org/nuscenes#download, "Map expansion"), public
  bucket https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/nuScenes-map-expansion-v1.3.zip
  (same bucket as can_bus.zip). 398,535,531 bytes zipped, 472,910,112 unzipped;
  last-modified 2024-01-30. Licence: nuScenes Terms of Use (non-commercial; LICENSE file
  in the zip). Downloaded and unzipped to Alpamayo/nuscenes/maps/{expansion, basemap,
  prediction} (existing mask PNGs untouched).
Contents (devkit NuScenesMap, 4 maps): lanes, lane connectors (arclines + graph),
  road / lane dividers, stop lines with type (STOP_SIGN, TRAFFIC_LIGHT, PED_CROSSING,
  YIELD, TURN_STOP), crosswalks, walkways, traffic-light POSITIONS (no states), road
  segments with an intersection flag, car-park areas. boston-seaport for example:
  1,215 lanes, 775 stop lines, 340 crosswalks, 307 traffic lights.
Added to the CoC template plan (implemented in v2, R2.6):
  - ego lane + lane graph successors -> in-path test for objects (replaces the
    curvature corridor), lane count in the ego direction
  - lane change = final lane not reachable from the ego lane through the lane graph
  - stop line ahead with its type -> the cause of "stop for static constraint"
  - crosswalk ahead; pedestrian standing on it -> yield candidate
  - intersection flag -> turn vs curve, "at the intersection"
  Still missing: traffic-light STATE, sign classes beyond the stop-line type, routing
  (non-privileged).

=========================================================================================
R2.6 CoC TEMPLATES v2 (map-aware) -- AR1_COC_EXAMPLES.txt, same 20 samples as v1
=========================================================================================
Generator ar1/coc_template.py v2 (rules in its header). New map components: ego lane, lane
count, intersection, stop line ahead with type, crosswalk ahead. In-path now uses the
lane graph. Leads may be pedestrians or cyclists moving the same way. A pedestrian on the
crosswalk ahead can be a yield candidate. Lane change = the final lane is not reachable
in the lane graph. Turn = more than 30 deg, or more than 20 deg through an intersection.
Passing = lane change around a slower or stationary lead. Multi-phase: "..., then
proceed".
The four flagged examples:
  #07  FIXED. v1 "stop and hold, keeping the lane" -> v2 "Stop, then proceed, turning
       left at the stop-sign stop line 2 m ahead" (29 deg through an intersection is now
       a turn; the stop is explained by the mapped stop sign).
  #09  FIXED. The stationary pedestrian 3.9 m to the left is no longer in path (not on
       the ego's lanes). v2 = lead following behind the car 39 m ahead that is moving
       1.1 m/s.
  #15  FIXED. The pedestrian 13 m ahead, moving the same way, is now the lead: "Keep a
       safe gap ... because the pedestrian ... is moving the same way" (construction
       zone still listed).
  #20  CHANGED, NEEDS AN IMAGE CHECK. The map puts the stationary car 31 m ahead off the
       ego's lanes, and the -2.9 m end offset follows the lane graph through the
       intersection. So v2 says lane keeping at set speed, not "lane change to pass". If
       the image shows a real pass, the lane-graph test is wrong here.
The other 16 (one line each; "same" = same decision and cause as v1):
  #01  same decision; now notes "in an intersection", 4 lanes.
  #02  same (stay stopped behind the stationary car); now lists the turn stop line and
       crosswalk.
  #03  v1 lead = car 38 m at +2.6 m lateral -> v2 no lead (off the ego lanes); set
       speed. The traffic-light line 6 m ahead is listed.
  #04  same (lead following).
  #05  same decision; the stop is now explained: crosswalk stop line 8 m ahead.
  #06  CHANGED, DOUBTFUL: v1 "stay stopped, cause unknown" -> v2 "yield, then proceed"
       to an oncoming pedestrian 16 m ahead at -7.0 m lateral. That pedestrian is
       within 1.6 m of a successor lane (probably a turn connector), so it counts as in
       path.
  #07  see above.
  #08  v1 yield to a crossing car 21 m ahead -> v2 "stay stopped, then proceed at the
       traffic-light stop line 4 m ahead". The crossing car is no longer on the ego
       lanes; a red light is the likely cause, but its state is not annotated.
  #10  same (yield to the crossing pedestrian); adds "then proceed".
  #11  same; adds "at the intersection".
  #12  same; adds "at the intersection".
  #13  same.
  #14  CHANGED, DOUBTFUL: v1 lane keeping -> v2 "turning right" (-26 deg via an
       intersection, 40 m forward). This may be a bend, not a turn; the 20 deg
       intersection rule may be too loose.
  #16  same (turn right).
  #17  same.
  #18  same.
  #19  v1 lane change right -> v2 lane keeping: the -3.0 m offset stays in the lane graph
       through the intersection (as #20).
Count: 3 of 4 flagged fixed, #20 needs a human image check. Two new doubtful labels
(#06, #14) come from the successor-lane in-path test and the 20 deg intersection rule.
Decision distribution on all splits: NOT produced. The v2 map queries run at about 2 s
per sample single-threaded; the job was stopped after 19 h without output.

=========================================================================================
GATE -- STOPPED after B2. Nothing is running; GPUs idle. Commits: c75c7af (R2.0),
c9dd833 (R2.1), b8bc570 (R2.2), 596b09b + f8fd234 (R2.3), 5b24e4d (R2.5 / R2.6),
8b68caf (B1 interim), this commit (R2.4 final).
=========================================================================================
