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
