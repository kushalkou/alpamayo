# FINAL_RESULTS -- Alpamayo-R1 (arXiv 2511.00088) rebuilt at small scale on nuScenes

Frozen 2026-10-13, tag v1.0-review. Every number is copied from the report or dump named
next to it; nothing here is recomputed except the mean / sd of three per-seed values that
a report lists (marked "derived"). Official nuScenes val, all 5,119 samples, unless
stated. [P] = privileged input (VAD command, built from the GT future). (1) = one seed.
Figures: figures/fig1-7 (PNG + script each). Reproduction: REPRO.md.

=========================================================================================
1. SETUP
=========================================================================================
Data      nuScenes v1.0-trainval, official split. Train 650 scenes minus a 50-scene
          holdout = 18,313 training records (n_fut = 12); holdout 1,717 (selection and
          tau only); val 150 scenes, 5,119 samples (analysis only, never selection).
          2 Hz keyframes, 6 s horizon (12 steps). Map expansion v1.3. [WEEK1_REPORT.md,
          PLANNER_HANDOFF.md]
Hardware  8 x V100 32 GB (no bf16, no FlashAttention): fp16 backbone, fp32 adapters.
Recipe    Cosmos-Reason1-7B (Qwen2.5-VL) frozen fp16 + LoRA r16 on q/v/o (10.2M train.);
          ego MLP with standardised inputs + post-LayerNorm + gain ("fix" recipe), ego-MLP
          lr x10; lr 5e-5, warmup 100, cosine to 0.1x; batch 3 x 8 GPUs; turn-weighted
          sampler; 10 epochs, patience 5; selection = holdout AR median ADE@6s on 400
          fixed samples; STOP-aware hybrid decode with tau chosen on holdout. Trajectory =
          24 tokens (12 accel + 12 curvature, 2 Hz), unicycle rollout from v0_can.
Cameras   B1 = CAM_FRONT / FRONT_LEFT / FRONT_RIGHT at t0, native patch order, 160 tokens
          each (480); B2 = same x 4 keyframes. Vision cache 87,147 images, 94 GiB.
Arms      M1 = 18 meta-action word tokens (AR1 Table 5 classes; 10 Hz CAN labels), then
          trajectory; M1-v2a = same with 2 Hz GT-trajectory labels; CoC = CoC text
          (decision, critical components, reasoning; template rules v3, audited) before
          the words; flow-matching expert (AR1 option A: per-layer KV of the frozen VLM,
          stop-grad, 184M params, OT path, 10 Euler steps, 6 samples).
NVIDIA released [AR1_GAP.md R1, checked 2026-10-03]: Alpamayo-R1-10B weights (gated,
          8.2B backbone + 2.3B expert, BF16, not RL-trained), inference code
          (NVlabs/alpamayo), recipes incl. SFT, open-loop RL and the CoC auto-labelling
          pipeline (NVlabs/alpamayo-recipes), PhysicalAI-AV dataset (gated; 1,728 + 349
          CoC annotations public; the 700K CoC traces NOT released).
We rebuilt on nuScenes: the training pipeline on the public Cosmos-Reason1-7B, the
          tokenizer and rollout, meta-action labels (thresholds ours; AR1 gives none), the
          CoC traces (rule-based templates, not LLM-written), the expert, all evaluation.
          Not built: RL / GRPO, closed-loop AlpaSim, full fine-tuning, bf16, AR1 data.

=========================================================================================
2. BUGS CAUGHT (symptom -> fix -> effect)
=========================================================================================
1 Ego-state future leak (custom split). compute_ego_state used forward differences:
  ego accel = GT accel slot 1, yaw rate = v0 x GT curvature slot 12 (exact). Fix: causal
  ego. Blend gain vs CV +0.164 [.14,.19] -> +0.014 [.003,.024] m; headline withdrawn.
  [FROZEN_RESULTS.md, OVERNIGHT2_REPORT.md]
2 v0 leak. A CV seeded with the forward speed |p1-p0|/dt scores L2 TemAvg 0.199 / 0.534
  / 1.007 vs 0.390 / 0.797 / 1.340 with causal v0_can. Fix: v0 = CAN at the last message
  <= t0 (backward-pose fallback). [WEEK1_REPORT.md 1b]
3 Input scale. Ego tokens had L2 norm 19.4 vs 0.87 for text embeddings (22x),
  unstandardised; the VLA could not overfit 256 samples (median ADE@6s 1.058, slot-0
  accuracy 0.49). Fix: standardise + LayerNorm + gain, ego lr x10 -> 0.098 (= tokenizer
  floor 0.092), slot-0 0.992. [REVIEW_RESULTS.md T5]
4 Vision patch order. Raster patches with a (T,C,14,14) flatten instead of Qwen2.5-VL's
  2x2 merge blocks: token cosine 0.33-0.52 to the native features. Fix: native path (cos
  1.00000 vs the HF processor). Every earlier camera result marked INVALID.
  [PATCH_ORDER_QUARANTINE.md]
5 VAD converter ego status. Four future reads (CAN message after t0, nearest-message v0,
  forward difference on first frames, except-path v0) + quaternion order (yaw ~ 0). Ego-
  MLP + cmd: VAD features 1.345 vs ours 1.640 L2@3s (-0.291 [-0.351,-0.232]); excluding
  the 140 first frames +0.035 [+0.007,+0.065]. We use causal features. [REVIEW_RESULTS T4]
6 10 Hz meta-action labels (CAN) inconsistent with the 2 Hz trajectory: word-trajectory
  consistency 0.497 (M1) vs 0.882 with 2 Hz labels (M1-v2a); false-go 0.315 vs 0.119;
  L2@3s 2.178 vs 1.942 (3 seeds). [R9 A3, R6 F3, R6 Part 2]
7-9 CoC trace rules (audit 1 by Kushal, 100 traces; verified on all 29,049) [R11]:
  7 route command in the causes: it is the GT LiDAR lateral offset at t0+3 s (>= 2 m),
    "straight" on 33-42% of turn decisions. Fix: removed from causes.
  8 rear-axle distances / out-of-path yields: 1,265 traces flagged (x < 3.5 m or |y| >
    half-width + 1.5 m); 620 of 1,286 yield decisions rested on one. Fix: front-bumper
    distances (3.528 m), path corridor, crosswalk branch removed.
  9 turns only inside intersection polygons: 606 traces with |dpsi| > 45 deg labelled
    lane keeping. Fix: also turn if |dpsi| > 45 deg.
  v3 changes 7.2% of decisions; audit 2 (32 cards): 0 wrong decisions, 0 wrong causes,
  8 unsure -> PASS; signed off (AUDIT_SIGNOFF.md).

=========================================================================================
3. BASELINES (L2@3s NoAvg unless stated) [REVIEW_RESULTS T4/T6/T7, R7A, PLANNER_HANDOFF]
=========================================================================================
  CV (no learning)                  2.762      kinematic rule (v0_can)        2.178
  Ego-MLP + cmd [P], 3 seeds        1.637      Ego-MLP no cmd, 3 seeds        1.740
  Ego-MLP + cmd, VAD-converter feats 1.345 (leaky inputs, bug 5)
  Ego-MLP + oracle meta-action [P]  1.193      VAD-Base reported L2 TemAvg 0.41 0.70 1.05
Li et al. CVPR 2024 protocol (0.1 m grid, yaw-aware box, max-over-horizon collision, L2
TemAvg; ALL 5,119) [R5 A4, R6 3a]:   L2 1/2/3 s (avg)       Col % 1/2/3 s (avg)
  GoStraight, v0 causal             0.38 0.78 1.33 (0.83)  0.12 1.04 3.36 (1.50)
  GoStraight, v0 VAD converter      0.28 0.63 1.11 (0.67)  0.12 0.88 2.97 (1.32)
  Ego-MLP causal (3)                0.25 0.47 0.78 (0.50)  0.02 0.40 1.37 (0.60)
  Ego-MLP VAD converter (3)         0.17 0.34 0.60 (0.37)  0.04 0.29 1.36 (0.56)
  no-cam VLA (3)                    0.26 0.52 0.91 (0.56)  0.02 0.82 2.44 (1.09)
  B1 (3)                            0.30 0.58 0.99 (0.63)  0.04 0.63 1.95 (0.87)
  M1 10 Hz (3) / M1-v2a 2 Hz (3)    (0.65) / 0.27 0.54 0.92 (0.58)   (1.04) / (1.05)
  Li Table 1: GoStraight 0.38 0.79 1.33 (0.83), Col 0.15 0.60 2.50; Ego-MLP 0.15 0.32
  0.59 (0.35), Col 0.00 0.27 0.85. Our GoStraight matches; no VLA beats the Ego-MLP.

=========================================================================================
4. TABLE 6 ANALOG [R9 A1, R13 B1 (CoC; mean (sd) derived from per-seed values), fig2]
=========================================================================================
  3-seed mean (sd)   L2@3s        ADE@6s       Col aa       FG (R3)      MG (R3)
  with route [P]
   traj (B1)         2.091(.064)  3.048(.067)  1.231(.215)  0.307(.012)  0.409(.027)
   +meta (M1-v2a)    1.942(.056)  2.891(.074)  1.569(.196)  0.119(.071)  0.512(.016)
   +CoC v3           2.070(.026)  3.024(.054)  1.680(.151)  0.154(.019)  0.541(.020)
  without route
   traj (B1-NR)      2.144(.028)  3.140(.010)  1.537(.200)  0.317(.005)  0.337(.019)
   +meta (M1v2a-NR)  2.082(.012)  3.072(.027)  1.608(.226)  0.250(.015)  0.499(.044)
   +CoC              not run
  Paired deltas, L2@3s (scene bootstrap 10,000, 3-seed means):
   +meta - traj: with route -0.150 [-0.232,-0.073]; without route -0.063 [-0.114,-0.008]
   +CoC - +meta (with route) +0.129 [+0.059,+0.215] (GC1-3 "CoC hurts")
   no route - route: traj +0.053 [+0.012,+0.091]; +meta +0.140 [+0.043,+0.253]
  AR1 Table 6 (p.23, quoted; minADE_6 @3 s / @6.4 s, CoC test set) -- DIFFERENT METRIC,
  COMPARE DIRECTION ONLY:
   0.5B no route: Base 0.284/0.996; +Traj 0.282/0.971; +Meta 0.291/0.988; +CoC 0.279/0.955
   3B   no route: Base 0.291/0.977; +Traj 0.293/0.976; +Meta 0.280/0.927; +CoC 0.275/0.908
   0.5B route:    Base 0.264/0.848; +Traj 0.262/0.834; +Meta 0.264/0.821; +CoC 0.254/0.794
  Direction: AR1 meta helps except 0.5B no route; CoC best everywhere. Ours: meta helps
  in both columns; CoC (as a generated prefix) HURTS vs meta.

=========================================================================================
5. TABLE 7 / 9 / 12 ANALOGS
=========================================================================================
Table 7 (hard cases) [R9 A2]. AR1 gives no selection rule ("challenging long-tail cases
in D_hard"). Ours: HARD = C2 START or |dpsi| > 30 deg or (in-path agent within 20 m AND
intersection / stop line < 20 m): 1,178 samples (23%), REST 3,941.
  L2@3s HARD / REST, 3 seeds: B1 3.266 / 1.740; M1-v2a 3.258 / 1.548; no-cam VLA 3.433 /
  1.512; Ego-MLP + cmd 2.975 / 1.236 (best on HARD). CoC on HARD: not computed.
  M1-v2a - B1: HARD -0.008 [-0.078,+0.062]; REST -0.192 [-0.296,-0.099].
  AR1 Table 7 (route, 0.5B): Traj 0.315/0.994; Meta 0.301/0.928; CoC 0.290/0.868.
  Direction: AR1's gain grows on hard cases; ours vanishes there (words hurt on START).
Table 9 (reasoning-action consistency) [R9 A3, R13 B1]. Ours = trajectory-derived words
  equal the generated words (all 12 slots): M1 10 Hz 0.497; M1-v2a 0.882; M1-v2a-NR
  0.858; no-cam M1-v2a (1) 0.830; CoC v3 (words AND decision) 0.845 / 0.846 / 0.825.
  AR1 Table 9 (LLM-judged score): SFT 0.62; +RL r_reason 0.53; +r_consistency 0.85;
  +r_safety 0.83. Direction: consistent (2 Hz) labels reach AR1's post-RL level by SFT.
Table 12 (decoders) [R10 B2, 3 seeds; fig6]   token     expert mean of 6
  ADE@6s                                       2.891     2.732  (-0.159 [-0.217,-0.099])
  nearest-to-token ADE@6s                      -         2.725  (-0.166 [-0.218,-0.113])
  hold rate (GT-stopped)                       0.863     0.871
  C2 STOPPED / START / MOVING L2@3s            0.606 / 9.273 / 1.755   0.906 / 8.682 /
                                               1.611 (STOPPED +0.300 [+0.141,+0.481])
  Col aa / yaw                                 1.57 / 1.54             1.60 / 1.52
  spread 3 s (per seed) 0.13-0.24 m; no-cam (A3) expert spread 1.94 m, hold 0.506 (R9).
  AR1 Table 12: AR 0.6811 vs flow 0.6440 minADE6@6.4s. Direction: same (flow better).

=========================================================================================
6. MECHANISM RESULTS
=========================================================================================
Mediation GM1 [R7A C1]: M1-v2a false-gos have a go-word first in 0.830; P(false-go |
  slot-1 "stop") 0.022 -> start decision mediated by the generated words. Forcing slot 1
  to gentle_acc gives false-go 1.000 (R6 GV3).
Blank image GBL [R7A C3, batched]: M1-v2a blank - real +0.244 [+0.170,+0.316] -> cameras
  used; blank B1 never starts (FG 0.000, MG 0.996).
C2 strata trade (ADOPTED AFTER R4) [R7A C2, R8, R13; fig3, fig4], L2@3s:
  M1-v2a - B1: STOPPED -0.528 [-0.984,-0.072]; START +0.358 [+0.221,+0.515]; MOVING
  -0.117 [-0.170,-0.065]. M1-v2a - no-cam: +0.490 / -0.590 / -0.062 (CIs exclude 0).
  Removing the route (M1-v2a-NR - M1-v2a, 3 seeds): STOPPED +0.806, MOVING +0.039.
Route leak (3 s) [R8 A3, R11 A1]: the command is the GT lateral offset at t0+3 s; a
  stationary sample with a turn command is ALWAYS a START (val 28 / 28, train 92 / 92);
  M1-v2a misses 0 of those vs 0.754 of straight-command STARTs. GL1 rule not met (only
  12% of STARTs carry the cue).
CoC decisions [R13 A2; fig5], seed 42: decision correct 0.584; lon recall Set speed
  0.954, Stop 0.707, Lead 0.103, Yield 0.000, Speed adaptation 0.000; 80% of the GC1 gap
  comes from wrong decisions; on correct decisions CoC - M1-v2a +0.059 [-0.010,+0.120].
Causes test GC6b [R13]: shuffled causes - GT causes (both with the true decision)
  +0.107 (seed 42), +0.098 [+0.074,+0.123] (3 seeds) -> "causes matter".
Oracle rows: GT CoC forced - M1-v2a, 3 seeds -0.123 [-0.189,-0.064] [R13]; GT 2 Hz words
  forced (M1-v2a s42, batched) 1.401 vs own 1.871 [R6 GV1]; Ego-MLP + oracle meta-action
  1.193 vs 1.637 [REVIEW_RESULTS T6]. CoC-AUX (reasoning off, (1)) - M1-v2a s42 +0.060
  [+0.013,+0.107] -> "hurts" [R13 GX1].

=========================================================================================
7. CLAIMS WITHDRAWN (R5-R13)
=========================================================================================
- R6 / R7: "tau 0.9 goes with a high WSS false-go" -- tau changes almost nothing (R8 A2).
- R8: "command from the 6 s GT endpoint" -- it is the LiDAR offset at t0+3 s (R11 A1).
- R7 / R8 route-leak explanation stated with the 6 s horizon -- counts stand, horizon 3 s.
- R9: "the M1-v2a expert lowers collisions (1.33 vs 1.58)" -- seed 42 only; 3 seeds 1.60
  vs 1.57 (R10).
- R9: "worse on STOPPED (+0.118)" -- understated; 3 seeds +0.300 [+0.141,+0.481] (R10).
- R12 GC2 and GC6 -- "uninformative by design" (GC2 tests a mode never trained; GC6
  keeps the true decision) (R13 A3).
- Earlier (pre-R5, kept for completeness): ego-leak headline (2.894 vs 3.062, -5.5%);
  "-36%" speed gain; every patch-order camera result (V8b, G7, custom-split vision).

=========================================================================================
8. LIMITATIONS
=========================================================================================
- Privileged route [P] in every headline VLA row (GT future at 3 s); no causal route.
- No traffic-light state in nuScenes: red-light stops are "stop line (light state not
  annotated)" in the CoC and invisible to the decision rules.
- WSS ambiguity: 10 val wait-at-standstill scenes (345 samples) dominate false-go and the
  seed spread; parked whole-scene clips (0771, 0777) and scene-0778 have ambiguous intent.
- 2 Hz keyframes (AR1 10 Hz): comfort and jerk metrics are crude (GT first-order 0.675).
- Single t0 frames for the headline cameras (B2 with 4 keyframes did not help, (1)).
- LoRA r16 on a frozen fp16 backbone vs AR1 full fine-tuning in bf16.
- Experts: M1-v2a s42 and s123 stopped at the 100-epoch cap (not converged).
- One-seed rows: B2, no-cam M1-v2a, CoC-AUX, the batched verification (R6 V1/V2), R7A
  NR seed 42 diagnostics, and every R6 Part 1 number; all others 3 seeds as marked.
- CoC is template text from annotations the model cannot see (GT boxes, map); generated
  CoC is judged by rules, not an LLM critic. No RL, no closed loop.
