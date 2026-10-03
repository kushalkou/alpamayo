# AR1_R1_REPORT -- reproduction phase R1 (gate report)

Written 2026-10-03. Goal: rebuild the Alpamayo-R1 method at small scale on nuScenes.
No training was launched. Files: code in Alpamayo/code/ar1/, outputs in Alpamayo/ar1_*.txt.
R1.0 is DEVIATIONS.md.

=========================================================================================
R1.1 AR1-STYLE INPUTS (built and tested; benchmark NOT run)
=========================================================================================
Code: ar1/hist_index.py (index), ar1/test_hist.py (unit test), ar1/vision_ar1.py
(preprocess + native patchify + encode), ar1/test_vision.py, ar1/bench_inputs.py.
Output: Alpamayo/ar1_r11_tests.txt.

Images. 1600x900 -> 448x280 by bilinear resize (full field of view, aspect 1.78 -> 1.60).
  Patch grid [1, 20, 32] -> 2x2 merge -> 160 tokens per image (VERIFIED: equal to the
  transformers Qwen2VLImageProcessor output, max |diff| 2.4e-7; the frozen tower on 12
  images returns [1920, 3584], all finite).
Cameras. CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT.
History. 4 frames per camera. Camera sweeps are not on disk (samples/ only), so frame k is
  the keyframe image of the k-th previous sample. Measured age t0 - ts (median [p5, p95]):
    slot      CAM_FRONT            CAM_FRONT_LEFT       CAM_FRONT_RIGHT
    t0        0.036 [0.034,0.038]  0.043 [0.042,0.046]  0.028 [0.026,0.030]
    t0-0.5    0.536 [0.486,0.585]  0.543 [0.493,0.593]  0.528 [0.478,0.577]
    t0-1.0    1.036 [0.937,1.087]  1.043 [0.945,1.095]  1.028 [0.929,1.079]
    t0-1.5    1.536 [1.436,1.634]  1.543 [1.444,1.642]  1.528 [1.428,1.626]
  First frames of a scene repeat the oldest image: 3 samples per scene have 1, 2 or 3
  padded slots (train 650 each, holdout 50 each, val 150 each).
Unit test (ar1/test_hist.py): PASS. 29,049 samples, 348,588 image slots, no timestamp > t0
  (max ts - t0 = -0.0245 s); frames ordered oldest -> newest; paths exist.
Tokens per sample: 3 x 4 x 160 = 1,920 visual + 4 ego + 1 cmd = 1,925 context tokens,
  + 23 teacher-forced trajectory tokens = 1,948 LM tokens (week-1 V8b: 1,541).

Memory at batch 2 / s/step on 8 GPUs / GPU-h: NOT MEASURED. Launching the 8-GPU
  benchmark (ar1/bench_inputs.py, 60 steps, no checkpoint) was blocked by the permission
  classifier in this session. To measure:
    cd Alpamayo/code && python -m torch.distributed.run --nproc_per_node=8 \
        ar1/bench_inputs.py --steps 60
  ESTIMATE (not measured), scaled from the measured week-1 V8b step (6 cams x 256 tokens,
  batch 3, 5.66 s/step, 21.4 GB peak): LM tokens per step 3,896 vs 4,623 (x0.84), image
  patches per step 15,360 vs 18,432 (x0.83) -> about 4.5-5 s/step, peak about 18-22 GB
  at batch 2. Steps per epoch = 18,313 / 16 = 1,144; 10 epochs about 14-16 h wall =
  115-130 GPU-h, training only (plus holdout selection).

SIDE FINDING (affects the foundation section; nothing changed):
  The frozen week-1 vision recipe (vision_live.patchify, 448x448) does NOT produce the
  Qwen2-VL native patch layout. It holds the same pixel values in a different order:
  raster patch order instead of 2x2 merge blocks, and a (T, C, 14, 14) flatten instead of
  (C, T, 14, 14). The max |diff| vs the processor is 3.47. On the frozen tower, week-1
  layout vs native features for the same 448x448 image: token cosine mean 0.410, min 0.050.
  So every week-1 / queue-v2 camera result (V8b, G7, C7 "cameras add no decision
  information") used visual features the encoder was not trained to produce. Those
  results stand as measured for that pipeline, but "cameras add nothing" may partly
  reflect this. The new ar1 path uses the native layout. Re-running a camera arm with the
  native layout would test this; not done (no training in R1).

=========================================================================================
R1.2 META-ACTIONS (AR1 Table 5) -- built; output Alpamayo/ar1_r12_meta.txt
=========================================================================================
Code: ar1/meta_actions.py -> Alpamayo/data/ar1_meta.pkl (61 ticks per keyframe:
t0 .. t0 + 6.0 s at 10 Hz).
AR1 source. The class list is Table 5. "Automatically labeled at 10Hz" is in the text.
  Thresholds: NOT SPECIFIED in the paper. Table 5 lateral also has reverse left / right
  (included).
Signals. CAN 'pose' at about 50 Hz. Every tick uses data <= the tick only (trailing
  window): v = vel[0] of the latest message; a = (v(t) - v(t-0.5)) / 0.5; yaw rate = mean
  over (t-0.5, t]; curvature = yaw rate / v (0 when |v| < 1 m/s). Fallback when CAN does
  not cover a tick: LIDAR ego poses at about 20 Hz (sweep metadata).
Thresholds (OURS):
  lon  reverse v < -0.2 m/s; stop |v| <= 0.2; strong acc a >= 2.0 m/s^2; gentle acc
       [0.5, 2.0); maintain |a| < 0.5; gentle dec (-2.0, -0.5]; strong dec <= -2.0
  lat  straight |k| < 0.01 1/m (radius > 100 m); steer 0.01-0.1; sharp >= 0.1
       (radius < 10 m); stop -> straight; reversing -> reverse L / R or straight
Coverage of ticks: train CAN 0.927 / pose 0.022 / none 0.052; holdout and val CAN 0.948,
  none 0.052. "none" = ticks past the end of the scene.
Source cross-check (CAN vs pose, 187,640 ticks): lon agree 0.835, lat agree 0.990 (lon
  disagreements sit at the 0.5 m/s^2 boundary).
Sign check (val ticks by VAD command): cmd LEFT -> left 0.555 / right 0.025; cmd RIGHT ->
  left 0.024 / right 0.524; cmd STRAIGHT -> straight 0.885.
Class counts, all 10 Hz ticks in [t0, t0+6 s] (share of valid ticks):
  lon          train             holdout          val
  gentle_acc   206,898 (0.161)   16,149 (0.163)   55,221 (0.186)
  strong_acc     1,031 (0.001)      174 (0.002)      627 (0.002)
  gentle_dec   212,006 (0.165)   14,130 (0.142)   47,845 (0.162)
  strong_dec     4,747 (0.004)      377 (0.004)    1,309 (0.004)
  maintain     630,997 (0.491)   46,935 (0.473)  147,119 (0.497)
  stop         228,481 (0.178)   21,534 (0.217)   43,735 (0.148)
  reverse          785 (0.001)        0 (0.000)      264 (0.001)
  lat
  steer_L       98,788 (0.077)    8,600 (0.087)   21,480 (0.073)
  steer_R      107,707 (0.084)    6,419 (0.065)   25,803 (0.087)
  sharp_L        7,768 (0.006)      306 (0.003)    2,399 (0.008)
  sharp_R        7,754 (0.006)      552 (0.006)    1,957 (0.007)
  reverse_L/R        0                0                0
  straight   1,062,928 (0.827)   83,422 (0.840)  244,481 (0.826)
  Per keyframe at the t0 tick (train / holdout / val): see ar1_r12_meta.txt; shares are
  within 0.03 of the tick shares.
Notes: strong accel / decel and reverse are rare (<= 0.4%). Reverse L / R never occurs
  because reversing happens below 1 m/s, where curvature is set to 0. Train includes the
  CAN-blacklisted scenes (the pose fallback).

=========================================================================================
R1.3 PLAN ONLY -- flow-matching action expert
=========================================================================================
Conditioning. The spec says "VLA hidden states (stop-grad), as in AR1". The paper says
  the expert takes "the KV-cache from the sequence" with a stop-gradient on the VLM KV-cache.
  These differ:
  (A) AR1-faithful. The expert has 28 layers (one per VLM layer). Each expert layer attends
      over [VLM K/V of that layer (stop-grad) | its own action tokens]. Same heads (28 q,
      4 kv) and head dim (128) as the VLM; smaller hidden / MLP width.
  (B) As specified. The expert reads the last-layer VLM hidden states (stop-grad) through
      cross-attention.
  Recommendation: (A), because it is the paper's design and its extra cost is small (below).
  Planner to choose.
Architecture (proposal).
  (A) 28 layers, hidden 512, MLP 1,536 (SwiGLU). The q/o projections go to the 3,584-wide
      VLM attention space. About 6.6M params per layer, about 185M total. KV-cache of the
      1,925-token context: 28 x 2 x 4 x 128 x 2 B = 57 KB/token, about 110 MB per sample.
  (B) 8 layers, hidden 512, self-attn + cross-attn + SwiGLU 1,536. Memory projection
      3,584 -> 512. About 37M params.
  Action tokens. 12 tokens, one per control step: Linear((a, k) standardised) + step
  embedding + flow-time embedding (sinusoidal t -> MLP, added).
  Output head -> 12 x 2 velocity.
Targets. 12 x (accel, curvature) at 2 Hz (w1/targets.py), standardised with train stats.
  Variant: 60 x 2 at 10 Hz from CAN (closer to AR1's 64 x 10 Hz). Evaluation stays on the
  2 Hz nuScenes positions.
Loss. Conditional flow matching, Gaussian OT path: x_t = t a + (1 - t) eps,
  eps ~ N(0, I); target velocity u = a - eps; L = || v(x_t, t, ctx) - u ||^2 (AR1's loss,
  as in the paper). t ~ U(0, 1); AR1's t-sampling distribution is UNVERIFIED.
  Expert forward, loss and optimizer in fp32. VLM stays fp16 frozen (stop-grad).
Inference. Start from eps ~ N(0, I); 10 Euler steps with dt = 0.1 (AR1 default).
  Destandardise -> unicycle rollout from v0_can (dt 0.5 s, the existing rollout) -> (x, y,
  yaw). 6 noise seeds -> minADE_6. Seed 0 alone gives the single-sample number that is
  comparable with the week-1 tables.
Training schedule. Stage 1b: train the expert on the frozen stage-1 VLM. AR1 trains
  CE on tokens + FM jointly; joint training is the variant. LoRA weights come from the
  stage-1 token run with the R1.1 inputs.
Memory per GPU (estimate). VLM fp16 16.6 GB, no VLM backward in stage 1b. Context
  activations under no_grad about 1-2 GB at batch 4. Expert (A): fp32 weights + grads +
  Adam = 16 B x 185M = 3.0 GB, plus activations (12 query tokens over 1,937 keys x 28 layers)
  under 1 GB. Total about 22 GB at batch 4: fits. (B) about 19 GB.
GPU-h (estimate; pending the R1.1 benchmark). VLM forward-only is about 1/3 of a
  training step (no backward, no recompute), so about 1.5-2 s/step at batch 2. For
  10 epochs x 1,144 steps that is about 5-6 h wall = 40-50 GPU-h. Joint training (variant)
  costs about the stage-1 run + 5%: 120-140 GPU-h.
  Caching the VLM context instead of recomputing it: last-layer hidden states for (B) are
  1,925 x 3,584 x 2 B = 13.8 MB per sample = 253 GB for train. For (A) the KV-cache is
  110 MB per sample = 2 TB, which does not fit (1.1 TB free). Recompute is planned.

=========================================================================================
R1.4 PLAN ONLY -- template CoC generator (+ 20 examples for review)
=========================================================================================
Prototype: ar1/coc_template.py (rules in its header). It writes no training data.
Examples: AR1_COC_EXAMPLES.txt (20 traces, train split, stratified by decision, seed 0).
Structure, following AR1 Tables 1-3:
  (1) Driving decision (Table 1), labelled from the FUTURE 6 s: GT trajectory, heading
      change, and 10 Hz meta-actions (R1.2). This mirrors AR1, which labels the decision
      from the realised behaviour.
  (2) Critical components (Table 2), from the HISTORY WINDOW ONLY (t0-2 s .. t0):
      - ego speed now and 1 s ago (CAN), ego turn-signal state (CAN vehicle_monitor,
        <= t0)
      - nearest in-path agent: objects with visibility >= 40% and >= 1 lidar point, in a
        1.5 m corridor along the ego's current curvature, < 50 m. Velocity comes from
        the same instance 1.0 s earlier; class; relative motion (same way / oncoming /
        crossing / stationary). nuScenes "moving / parked" attributes are not used,
        because they are annotated with the whole clip in view.
      - construction zone (>= 3 cones / barriers within 30 m ahead)
      - routing: the VAD command (PRIVILEGED: computed from the GT future).
  (3) Trace: one composed sentence, action + lateral + cause. Training format (planned):
      "Decision: <lon>; <lat>. Components: ... Reasoning: ..." then the action tokens,
      about 60-100 text tokens.
Decisions emitted: lon = set-speed tracking, lead following, stop for static constraint,
  yield, speed adaptation (slowing for a turn); lat = lane keeping, turn L/R, lane change
  L/R, none. Never emitted (need lanes / signals): gap-searching, acceleration for
  passing, merge / split, in- / out-of-lane nudge, pull-over, maneuver abort.
Distribution (n_fut = 12 keyframes):
  lon    train (18,313)   holdout (1,417)   val (4,219)
  set speed      0.649    0.616    0.575
  stop static    0.153    0.143    0.109
  lead follow    0.133    0.159    0.269
  yield          0.044    0.070    0.031
  speed adapt.   0.022    0.011    0.016
  lat
  lane keeping   0.669    0.667    0.697
  none           0.145    0.189    0.117
  turn L / R     0.056 / 0.067    0.041 / 0.033    0.055 / 0.065
  lane chg L / R 0.025 / 0.038    0.041 / 0.029    0.032 / 0.034
What nuScenes cannot provide (Table 2 attributes):
  - traffic-light state and arrows: not annotated. So "stop for static constraint" has
    no observable cause in the labels (the camera may show it).
  - stop / yield signs and stop lines: not annotated. The map expansion (v1.3, a
    separate download, NOT on disk) has stop-line and crosswalk polygons and traffic-light
    positions, but no light states.
  - lane count, line types, lane membership, shoulder: need the map expansion; not on
    disk. As a result, lane change vs curve vs nudge is judged from the trajectory alone.
  - road grade, speed bumps, narrowing: no.
  - routing intent: only the privileged VAD command. CAN route.json (planned route
    polyline) is a candidate non-privileged source; not built.
  - ODD: scene.json descriptions hold free text (e.g. rain, night). It is scene-level,
    not used yet.
Issues visible in the 20 examples (my read; for the human review):
  #07  "stop and hold" while the future shows stop-then-go plus a 29 deg left turn. One
       decision per 6 s window misses sequences, and 29 deg is just under the 30 deg turn
       threshold.
  #09  a stationary pedestrian 3.0 m to the left is "in path" because the corridor bends
       with the current curvature; the yield label is doubtful.
  #15  slow following of an in-path pedestrian is labelled set-speed + construction; the
       lead rule excludes pedestrians.
  #20  a stationary car 31 m ahead and a lane change right: really "pass the stopped car",
       but labelled "path ahead is clear".
  The other 16 read as consistent with their [future] lines. Set-speed tracking is a
  57-65% catch-all.
Decision to make before scaling: whether to download the nuScenes map expansion (lanes,
  stop lines, crosswalks) to fill the lane / sign components.

=========================================================================================
GATE -- STOPPED. Nothing is training. GPUs were used only for the single-GPU encoder test
(R1.1, test 3).
=========================================================================================
