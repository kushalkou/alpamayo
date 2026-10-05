# PLANNER_HANDOFF -- Alpamayo-R1 reproduction on nuScenes, state after phase R3

Written 2026-10-05 for the planner. Plain summary of everything since the goal changed to
reproducing Alpamayo-R1 (AR1, arXiv 2511.00088). Detailed reports: AR1_GAP.md (R0),
AR1_R1_REPORT.md, AR1_R2_REPORT.md, AR1_R3_REPORT.md, DEVIATIONS.md,
PATCH_ORDER_QUARANTINE.md. All numbers are on official nuScenes val (5,119 samples)
unless stated. [P] = privileged input (the VAD command, derived from the GT future).

=========================================================================================
0. STATUS RIGHT NOW
=========================================================================================
- GPUs idle (8 x V100). Nothing queued. Every R3 GPU item and amendment A3 is finished.
- Kushal is auditing the CoC traces (AR1_COC_AUDIT.html, 100 val traces). The CoC
  training arm has NOT been launched and waits for that audit.
- Last commits: see section 8.

=========================================================================================
1. WHAT IS BUILT (AR1 component -> our status)
=========================================================================================
Backbone        Cosmos-Reason1-7B, frozen fp16, LoRA r16 (q, v, o). AR1: full 8.2B, bf16.
Vision          FIXED in R2: native Qwen2.5-VL patch order (verified vs HF processor, cos
                1.00000). 448x280 -> 160 tokens per image. Cache of all 87,147
                front-camera keyframe images (94 GiB, bit-exact with live encoding in batches).
Cameras         B1 = CAM_FRONT, FRONT_LEFT, FRONT_RIGHT at t0 (480 tokens), the base by
                pre-registered rule. B2 = same x 4 keyframes (1,920 tokens).
Ego / route     Causal ego state (CAN at t0); route = VAD command [P]. No non-privileged
                route yet (CAN route.json is a candidate).
Meta-actions    AR1 Table 5 classes at 10 Hz from CAN (R1.2). Thresholds are ours (the
                paper gives none). A 2 Hz GT-trajectory label set also exists (A1).
Action decoder  (a) discrete tokens, 12 x (accel, curvature) at 2 Hz, AR decode;
                (b) flow-matching expert, AR1 option A (28 layers reading the frozen VLM
                per-layer KV, stop-grad, 184M params, OT path, 10 Euler steps, 6 samples).
Reasoning text  M1 arm: meta-action words (plain LLM tokens, 1 Hz x 6 s) then trajectory.
CoC data        Template generator v3 with the nuScenes map expansion (lanes, stop lines
                with type, crosswalks, intersections). 29,049 traces, ~80 tokens each.
                Under audit.
Not built       RL / GRPO, closed-loop sim, LLM-written CoC, bf16, PhysicalAI-AV data.

=========================================================================================
2. HEADLINE RESULTS (official val, ALL 5,119; 3-seed means +- sd where marked)
=========================================================================================
                              L2@3s NoAvg      L2@3s TemAvg    ADE@6s          Col% Tem 3s
  KIN (kinematic rule)        2.178            0.998           3.161           0.57
  Ego-MLP + cmd [P], 3 seeds  1.637            0.779           2.437           0.87
  VLA no cam [P], 3 seeds     1.954 +- 0.054   0.909 +- 0.027  2.930 +- 0.036  0.61
  VLA B1 cams [P], 3 seeds    2.091 +- 0.064   0.993 +- 0.035  3.048 +- 0.067  0.49
  VLA B2 cams x4 kf [P], s42  2.130            1.018           3.093           0.59
  M1 words+traj [P], 3 seeds  2.178 +- 0.049   1.037 +- 0.013  3.134 +- 0.083  0.57
  M1 s42 + expert, mean of 6  2.111            1.022           3.046           1.22
  A3 (no cam) + expert, mean6 1.875            0.895           2.760           0.82
Key paired differences (scene bootstrap, 95% CI):
  B1 - no cam (3 vs 3 seeds)   L2@3s +0.137 [+0.036,+0.255]
  B2 - B1 (s42)                L2@3s -0.035 [-0.109,+0.037]  -> base = B1 (pre-registered)
  M1 - B1 (3 vs 3 seeds)       L2@3s +0.087 [+0.005,+0.179]
  M1 - no cam (3 vs 3)         L2@3s +0.224 [+0.059,+0.419]
  expert mean6 - token (A3)    L2@3s -0.019 [-0.075,+0.038]; ADE6 -0.136 [-0.221,-0.048]
  expert mean6 - token (M1)    L2@3s -0.092 [-0.173,+0.002]; ADE6 -0.137 [-0.227,-0.034]
Plain reading:
  1. Cameras, even with the patch order fixed, make the trajectory VLA WORSE than no
     cameras, by 0.14 m L2@3s over 3 seeds. The damage is concentrated in stationary
     scenes. Straight scenes are slightly worse; turning is slightly better (-0.08, n.s.).
  2. The camera failure mode is specific: FALSE-GO. With cameras the model drives off
     in 30-32% of the samples where the car stays stopped (no cam: 2%). This is in every
     seed of B1, B2 and M1. Cameras lower false-stop (0.63 -> 0.37-0.42).
  3. Not memorisation: B1 fits a 1,000-sample train subset no better than no-cam (median
     ADE 1.94 vs 1.97); its train-val gap is larger (+0.34 vs +0.19 mean).
  4. Meta-action words (AR1 Table 6 analog) do not help (+0.09 m vs B1). The words
     collapse toward "maintain / straight" (43-47% of samples get the all-majority
     sequence vs 17% in labels). Strong accel / decel and reverse are never generated.
     Word accuracy is lon 0.61, lat 0.84; macro-F1 lon 0.31, lat 0.41-0.44.
     Words vs own trajectory consistency is 0.41-0.57. GT self-consistency with 2 Hz
     labels is 1.00.
  5. Sampling the words (T = 0.7) reduces the collapse but hurts accuracy and trajectory
     (+0.07 m L2@3s). Greedy constrained re-decode is deterministic (identical).
  6. Flow-matching expert: the mean of 6 samples improves ADE@6s by about 0.14 m over the
     token decoder, on both the no-cam and the M1 model. It is not better at 3 s, it
     roughly doubles the collision rate, and it has a heavier p95 tail. One sample is
     worse than the token decoder. minADE_6 is much lower (by construction).
  7. No VLA variant beats the small Ego-MLP + cmd (1.637) or approaches it; the oracle
     meta-action results from week 1 remain the only large gains (privileged).

=========================================================================================
3. INTEGRITY ISSUES FOUND AND HANDLED (do not reuse the quarantined numbers)
=========================================================================================
- PATCH ORDER (R2.0): every camera result before R2 (custom split and week-1 V8b / G7 /
  C7 camera rows) used a non-native patch layout (feature cosine 0.33-0.52 to native).
  Marked INVALID (patch order) in FROZEN_RESULTS.md and REVIEW_RESULTS.md; list in
  PATCH_ORDER_QUARANTINE.md. Zeroed / removed-vision results are unaffected.
- fp16 vision tower: some tokens differ from fp32 (per-token cos down to 0.46-0.67);
  per-image mean cos >= 0.996. Shared by every path on V100 (no bf16).
- Bugs this phase, all fixed, no result lost:
  - CoC generator stall: devkit map queries rebuild polygons per call (19 h -> 51 s).
  - M1 val dump: NCCL 10-min timeout at the final all_gather; timeout raised to 3 h.
  - M1 expert: checkpoint reload put all ranks on cuda:0 (OOM); fixed with map_location.
  - Expert targets: chord curvature tail (sd 13.5) -> clipped +-0.3 1/m before any eval.
- Consistency metric: the first ceiling (0.18) was a CAN-vs-2 Hz label mismatch, not a
  model property; fixed in A1 (2 Hz labels, self-consistency 1.00).
- CoC rules: the first lane-polygon "in path" test made yield 16.5% (crossing lanes in
  intersections). Replaced before any audit by proximity to the realized path.
- Labels: the VAD command used as route is privileged everywhere ([P]).

=========================================================================================
4. DATA AND ARTIFACTS
=========================================================================================
  data/ar1_hist.pkl       3 front cams x 4 keyframes per sample, all <= t0 (unit-tested)
  data/ar1_vcache/        native-order vision features, 94 GiB
  data/ar1_meta.pkl       10 Hz meta-actions (Table 5) from CAN, 61 ticks per sample
  data/ar1_coc.pkl        CoC v3: decision, trace, full CoC text per sample (n_fut >= 6)
  data/ar1_kv_A3_*.pt     frozen A3 KV cache (no-cam context) for the A3 expert
  nuscenes/maps/expansion map expansion v1.3 (downloaded R2.5)
  results/w1_dump_{B1,B1_s123,B1_s2024,B2,M1,M1_s123,M1_s2024,M1_G,M1_T07}_*  AR dumps
  results/ar1_expert_{A3,M1}.pkl   expert 6-sample predictions (holdout, val)
  checkpoints: _w1_B1*, _w1_B2, _w1_M1*, _ar1_expert_{A3,M1}
  Code: Alpamayo/code/ar1/ (finetune_meta.py = M1 arm; train_expert*.py = experts;
  coc_template.py = CoC; r2x/r3x eval scripts reproduce every table).

=========================================================================================
5. COMPUTE (8 x V100 32 GB)
=========================================================================================
  B1 run (480 cam tokens)      1.18 s/step, ~27 GPU-h end to end (train 20 GPU-h)
  B2 run (1,920 tokens)        6.28 s/step, ~120 GPU-h
  M1 run (B1 + 18 words)       1.25 s/step, ~31 GPU-h with dumps (dump ~45 min wall)
  Expert on short context (A3) 0.5 GPU-h; expert on M1 (503-token KV online) 54 GPU-h
  Vision cache build           1.1 GPU-h
  R2 + R3 total roughly 650 GPU-h.

=========================================================================================
6. OPEN DECISIONS FOR THE PLANNER
=========================================================================================
D1. Cameras hurt (false-go when stationary). Options, cheapest first (each ~27 GPU-h
    per B1-sized run unless noted):
    a. CPU analysis first (no GPU): which stationary scenes false-go (traffic light, lead
       stopped, queue)? Do they share visual context? Uses existing dumps.
    b. Camera dropout during training (drop the 480 tokens with p = 0.3-0.5) so the model
       cannot replace ego / kinematic cues with image cues.
    c. Stationary-aware loss weight or STOP-token reweighting on the B1 recipe.
    d. Fewer visual tokens (pool 160 -> 40 per image) to test dilution.
    e. Accept no-cam as the trajectory base and use cameras only for the reasoning
       text (decision head), as in week-1 C7, now with correct patch order.
D2. Meta-action words collapse. Options: train on the 2 Hz labels (consistent with the
    trajectory), class-balanced CE on word tokens, coarser classes (no strong / reverse),
    or 2 Hz words instead of 1 Hz.
D3. Flow-matching expert: keep as the AR1 decoder? It gains ADE@6s (mean of 6) but
    doubles collisions. Options: more Euler steps, mean-of-K vs median, train on B1
    rather than M1 (the M1 expert costs 54 GPU-h because of the online KV).
D4. CoC training arm (Table 6 "CoC + traj"): after Kushal's audit. Cost about one M1 run
    plus longer text (80 tokens): estimate 35-45 GPU-h per seed with dumps.
D5. Route: replace the privileged VAD command by a causal route from CAN route.json
    before any claim that is meant to be deployable.
D6. Out of reach on this hardware (unchanged): RL / GRPO with a reasoning critic,
    closed-loop AlpaSim, full-backbone fine-tuning.

=========================================================================================
7. CAVEATS
=========================================================================================
- Single seed: B2, both experts, M1_T07. Three seeds: no cam, B1, M1.
- All VLA arms use the privileged command [P]; compare only with [P] baselines.
- Meta-action thresholds and CoC rules are ours (AR1 gives none); CoC is template text,
  not LLM-written; traffic-light state is not observable in nuScenes.
- Horizon 6 s at 2 Hz (AR1: 6.4 s at 10 Hz); minADE_6 is not comparable to 1-sample ADE.

=========================================================================================
8. COMMITS (R2 onward)
=========================================================================================
R2: c75c7af c9dd833 b8bc570 596b09b f8fd234 5b24e4d 8b68caf 69a2bf2
R3: 13c7f2f 4b4ebc7 b360904 2104dfc c0a9eca 83a24c1 85508b5 a70665d 6ab2cff ecabe0e
    da0fcdb 9d79895 7b9540e d7ef324 a685505 (audit sheet); this handoff: see git log.
