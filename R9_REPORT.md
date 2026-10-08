# R9_REPORT -- Table 6 / 7 / 9 / 12 analogs; flow expert on M1-v2a

HEADER (all rules and definitions fixed and committed BEFORE computing; holdout selection
as before; val read for analysis only; CoC arm untouched unless AUDIT_SIGNOFF.md appears).
Definitions as R8 (R3 false-go / missed-go; WSS; C2 STOPPED / START / MOVING, ADOPTED
AFTER R4). Collision = @3 s NoAvg vehicles, VAD port (aa) and yaw-aware (r5_a5c1).
A1 Table 6 analog. Rows traj-only (B1 / B1-NR), +meta 2 Hz words (M1-v2a / M1-v2a-NR),
   +CoC ("pending"); columns with route [P] / without route. 3-seed mean +- sd (sd over
   per-seed values) of L2@3s NoAvg, ADE@6s (n_fut = 12), collision aa / yaw, R3 FG, R3 MG.
   Paired deltas (scene bootstrap 10,000, 3-seed means, L2@3s and ADE@6s): +meta - traj
   (with route; without route) and without - with route (traj; +meta). AR1 Table 6 quoted
   from docs/ar1_2511.00088.pdf p.23 (pypdf text; only the check / cross route marks are
   transcribed to ASCII), all 0.5B and 3B rows, labelled "different metric (minADE_6, CoC
   test set); compare direction only".
A2 Table 7 analog. AR1's definition of its hard set is quoted first (exact text). Our
   rule-based nuScenes HARD set (evaluation-only; uses GT future and annotations, which
   no model sees), features from ar1/coc_template.py functions, val all 5,119:
     H_START  C2 START (v0_can <= 0.2 m/s, GT max displacement 3 s > 1.0 m).
     H_TURN   |dpsi| > 30 deg, dpsi = GT yaw at the last available future keyframe
              (<= 6 s) minus yaw at t0 (coc_template.future 'dpsi').
     H_AGENT  any annotated agent at t0 (vehicle, pedestrian, cyclist; not cone/barrier;
              visibility >= 40%, >= 1 lidar point) that is in path (coc_template.objects:
              within 1.5 m + half its width of the realized GT path, or in the ego lanes
              with |y| < 2.5 m when the path is < 5 m) and 0 < x < 20 m ahead.
     H_INT    ego at t0 on an intersection road segment, or a stop-line centroid ahead
              (0 < x < 20 m, |y| < 6 m).
     HARD = H_START or H_TURN or (H_AGENT and H_INT); REST = the others.
   Report n per component and HARD / REST, L2@3s and ADE@6s for CV, KIN, Ego-MLP + cmd
   (3), Ego-MLP-NR (3), no-cam VLA (3), B1 (3), B2 (s42), M1 (3), M1-v2a (3), no-cam
   M1-v2a (s42), B1-NR (3), M1-v2a-NR (3); paired CIs (3-seed means) M1-v2a - B1 and
   M1-v2a-NR - B1-NR on HARD and on REST. AR1 Table 7 rows quoted beside it.
A3 Table 9 analog. Consistency = trajectory-derived words (r33_eval.derive on the
   hybrid-decoded controls, holdout tau) vs the generated words in the sequence: share of
   samples with all 12 slots equal; mean per-slot agreement. Runs: M1 10 Hz (3), M1-v2a
   (3), M1-v2a-NR (3), no-cam M1-v2a (s42); overall and per C2 stratum. derive uses the
   2 Hz rules, so for M1 (10 Hz words) it measures agreement with the 2 Hz reading of its
   own trajectory. AR1 Table 9 rows quoted for direction (their score is an LLM-judged
   consistency on CoC, not this rule).
B1 Flow expert on M1-v2a s42 (with route): ar1/train_expert_meta.py unchanged recipe
   (option A per-layer KV online from the frozen model, stop-grad, 184M params, OT path,
   selection holdout median ADE@6s of one fixed-noise draw, patience 4 x 5 epochs), with
   --labels 2hz (training words = the 2 Hz labels M1-v2a was trained on; inference words
   = M1-v2a's own generated words from its dumps). Est. ~54 GPU-h (M1 expert: 6.8 h x 8).
   Queue ar1/run_r9b.sh (tmux r9b); checks AUDIT_SIGNOFF.md before starting.
B2 Table 12 analog, R5 A5 metric set (ar1/r5_a5c1.py definitions), val: token decoder vs
   expert single draw (per-draw metrics averaged over the 6 draws), mean of 6, nearest to
   token (draw with the lowest mean L2 to the token trajectory), minADE6 @3 s / @6 s,
   spread (mean pairwise L2 between draws at 3 s / 6 s), stationary-hold rate (GT-stopped
   stationary samples whose prediction stays < 0.5 m within 3 s), collision aa / yaw,
   C2 strata L2@3s. Rows: M1-v2a s42 token + expert; A3 (no-cam) token + expert, M1 token
   + expert for comparison. AR1 Table 12 quoted.
   GE1 M1-v2a expert mean-of-6 - token, ADE@6s (paired scene bootstrap): upper < 0 ->
       "expert improves the long horizon".
   GE2 expert mean-of-6 hold rate >= token hold rate - 0.05 -> "expert holds stops";
       else "expert drifts at standstill".
   GE3 nearest-to-token - token, ADE@6s: upper < 0 -> "expert adds value even when
       anchored to the token decision".

=========================================================================================
GATE VERDICTS
  GE1, GE2, GE3: pending (Part B, flow expert on M1-v2a s42, tmux r9b, started 10:35 UTC).
  Part A has no gates. Headline: +meta beats traj-only with and without route (A1), but
  NOT on the HARD set (A2: M1-v2a - B1 HARD -0.008 [-0.078,+0.062]; REST -0.192).

A1 Table 6 analog (val; 3-seed mean (sd); per-sample dumps, holdout tau)
                 L2@3s       ADE@6s      Col aa      Col yaw     FG (R3)     MG (R3)
  with route [P]
   traj-only B1  2.091(.064) 3.048(.067) 1.231(.215) 1.322(.197) 0.307(.012) 0.409(.027)
   +meta M1-v2a  1.942(.056) 2.891(.074) 1.569(.196) 1.537(.186) 0.119(.071) 0.512(.016)
   +CoC          pending (AUDIT_SIGNOFF.md absent)
  without route
   traj B1-NR    2.144(.028) 3.140(.010) 1.537(.200) 1.582(.238) 0.317(.005) 0.337(.019)
   +meta M1v2aNR 2.082(.012) 3.072(.027) 1.608(.226) 1.647(.243) 0.250(.015) 0.499(.044)
   +CoC          pending
  Paired deltas, 3-seed means: L2@3s | ADE@6s
   +meta - traj, with route      -0.150 [-0.232,-0.073] | -0.157 [-0.256,-0.061]
   +meta - traj, without route   -0.063 [-0.114,-0.008] | -0.068 [-0.147,+0.016]
   no route - route, traj        +0.053 [+0.012,+0.091] | +0.092 [+0.028,+0.155]
   no route - route, +meta       +0.140 [+0.043,+0.253] | +0.181 [+0.072,+0.304]
  AR1 Table 6 (p.23, quoted; [x] = no route, [v] = route; minADE_6 @3s, @6.4s). DIFFERENT
  METRIC (minADE_6, CoC test set); COMPARE DIRECTION ONLY:
   ID Model Name Route Parameters minADE6@3s minADE6@6.4s
   1 Base model (action modality) [x] 0.5B 0.284 0.996
   2 + Ft. w/ Traj. [x] 0.5B 0.282 0.971
   3 + Ft. w/ Meta-action & Traj. [x] 0.5B 0.291 0.988
   4 + Ft. w/ CoC & Traj. (AR1) [x] 0.5B 0.279 0.955
   5 Base model (action modality) [x] 3B 0.291 0.977
   6 + Ft. w/ Traj. [x] 3B 0.293 0.976
   7 + Ft. w/ Meta-action & Traj. [x] 3B 0.280 0.927
   8 + Ft. w/ CoC & Traj. (AR1) [x] 3B 0.275 0.908
   9 Base model (action modality) [v] 0.5B 0.264 0.848
   10 + Ft. w/ Traj. [v] 0.5B 0.262 0.834
   11 + Ft. w/ Meta-action & Traj. [v] 0.5B 0.264 0.821
   12 + Ft. w/ CoC & Traj. (AR1) [v] 0.5B 0.254 0.794
  Verified: 3B without route, traj -> +meta = 0.976 -> 0.927 at 6.4 s (0.293 -> 0.280 at
  3 s). Table 6 has NO 3B with-route rows. Direction: AR1 0.5B no route, meta hurts;
  AR1 3B no route and 0.5B with route, meta helps (at 6.4 s). Ours: meta helps in both
  columns at 3 s (with route also at 6 s; without route the 6 s CI includes 0).

A2 Table 7 analog. AR1's definition (quoted exactly): Table 7 caption "Open-loop
  evaluation of models on the challenging dataset. All models are finetuned on the CoC
  dataset and evaluated on the challenging dataset."; Sec. 6.1: "challenging long-tail
  cases in D_hard to thoroughly test the model's ability to handle rare, safety-critical
  events." No selection rule is given (D_hard: symbol transcribed). AR1 Table 7 (route
  [v], 0.5B): 1 Ft. w/ Traj. 0.315 0.994; 2 Ft. w/ Meta-action & Traj. 0.301 0.928;
  3 Ft. w/ CoC & Traj. (AR1) 0.290 0.868.
  Our HARD set: START 231, TURN 579, AGENT 883, INT 2,962, AGENT and INT 506;
  HARD 1,178 (23%, 139 of 150 scenes), REST 3,941.
  L2@3s ADE@6s     HARD        REST       |                 HARD        REST
  CV               4.773 6.449 2.161 2.883 | B2 (1)          3.339 4.769 1.768 2.561
  KIN              4.161 5.839 1.586 2.310 | M1 10 Hz (3)    3.188 4.564 1.876 2.680
  Ego-MLP+cmd (3)  2.975 4.279 1.236 1.852 | M1-v2a (3)      3.258 4.699 1.548 2.317
  Ego-MLP-NR (3)   3.330 4.810 1.265 1.890 | no-cam M1v2a(1) 3.258 4.785 1.442 2.231
  no-cam VLA (3)   3.433 4.963 1.512 2.284 | B1-NR (3)       3.443 4.951 1.756 2.565
  B1 (3)           3.266 4.721 1.740 2.517 | M1-v2a-NR (3)   3.378 4.893 1.694 2.494
  M1-v2a - B1:       HARD -0.008 [-0.078,+0.062] (ADE6 -0.022 [-0.136,+0.091]);
                     REST -0.192 [-0.296,-0.099] (ADE6 -0.200 [-0.325,-0.077])
  M1-v2a-NR - B1-NR: HARD -0.065 [-0.138,+0.008] (ADE6 -0.058 [-0.176,+0.065]);
                     REST -0.062 [-0.126,+0.005] (ADE6 -0.071 [-0.167,+0.026])
  By component, M1-v2a - B1 | NR: START +0.358 [+0.221,+0.515] | +0.161 [+0.004,+0.322];
  TURN -0.069 [-0.189,+0.049] | -0.144 [-0.267,-0.016]; AGENT and INT -0.087 [-0.182,
  +0.002] | +0.003 [-0.089,+0.095].
  Reading: the opposite of AR1's Table 7 direction. The meta gain is on REST; on HARD it
  is ~0, because words HURT on START (missed starts) and only help a little on turns.
  The Ego-MLP + cmd is the best model on HARD (2.975), i.e. privileged command + ego
  state beat every VLA there.

A3 Table 9 analog (consistency, 2 Hz rules): all-12-slot match | mean per-slot agreement
                   all         STOPPED     START       MOVING      seeds (all-12)
  M1 10 Hz (3)     0.497 0.918 0.796 0.970 0.709 0.956 0.437 0.907 .410 .562 .517
  M1-v2a (3)       0.882 0.975 0.971 0.995 0.905 0.987 0.867 0.971 .854 .884 .908
  M1-v2a-NR (3)    0.858 0.973 0.968 0.996 0.921 0.991 0.836 0.969 .840 .868 .864
  no-cam M1-v2a(1) 0.830 0.964 0.982 0.998 0.965 0.997 0.798 0.957
  AR1 Table 9 (quoted; "Reasoning-Action Consistency Score", LLM-judged; ADE, score):
  SFT 2.12m 0.62; SFT + RL (r_reason) 2.19m 0.53; SFT + RL (r_reason + r_consistency)
  1.92m 0.85; SFT + RL (r_reason + r_consistency + r_safety) 1.94m 0.83.
  Direction: consistent labels (2 Hz) give 0.88 all-12 consistency without any RL, in the
  range AR1 reaches only after consistency-reward RL; 10 Hz labels give 0.50.
