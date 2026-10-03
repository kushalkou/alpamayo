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
  its 12 action tokens]. 28 q / 4 kv heads, head dim 128, hidden 512, SwiGLU 1,536, RoPE at
  positions ctx_len + i (verified against the cached keys: layer-0 max |diff| 5.3e-3 on
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
                        L2@3s NoAvg  L2@3s TemAvg  Col% 3s No/Tem  ADE@6s mean med p95  FDE6
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
R2.4 NEW CAMERA BASE (B1, B2)
=========================================================================================
PENDING

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
PENDING
