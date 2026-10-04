# AR1_R3_REPORT -- reproduction phase R3 (gate report)

Written 2026-10-04. Code: Alpamayo/code/ar1/; outputs: Alpamayo/ar1_r3*.txt.
Pre-registered rules applied without the planner.
[P] = privileged input (VAD command, from the GT future).

=========================================================================================
R3.0 CAMERA-HARM DIAGNOSIS (CPU, existing dumps; ar1/r30_diag.py)
=========================================================================================
a. Stationary val samples (v0_can < 0.5 m/s). Thresholds fixed before looking, first 3 s:
   GT stopped = max displacement < 0.5 m, GT moves = > 1.0 m; false-go = P(pred > 1.0 m |
   GT stopped); false-stop = P(pred < 0.5 m | GT moves). Alpamayo/ar1_r30a.txt.
                       ALL (993: 676 stop, 283 move)  EXCL. first frames (853: 653 / 168)
   model                 false-go   false-stop             false-go   false-stop
   no cam s42 (A3)        0.016      0.625                  0.017      0.506
   no cam s123            0.012      0.629                  0.012      0.512
   no cam s2024           0.037      0.629                  0.038      0.512
   no cam 3-seed mean     0.022      0.628                  0.022      0.510
   B1 3 cams t0           0.302      0.424                  0.300      0.458
   B2 3 cams x 4 kf       0.320      0.371                  0.319      0.417
   The camera models start moving in about 30% of the samples where the car stays
   stopped, against about 2% without cameras.
b. AR ADE@6s, fixed 1,000-sample train subset (n_fut = 12, seed 0) vs val (n_fut = 12),
   hybrid decode with the holdout tau. Alpamayo/ar1_r30b.txt.
   model                 train mean / med   val mean / med   gap val - train (mean / med)
   no cam s42 (A3)        2.708 / 1.970      2.896 / 2.003    +0.188 / +0.033
   B1 3 cams t0           2.783 / 1.935      3.125 / 2.296    +0.342 / +0.360
   B1 fits the train subset no better than the no-camera model but has a larger
   train-val gap.

=========================================================================================
R3.1 COST WITH THE CACHE (B1, 480 visual tokens; Alpamayo/ar1_r24_B1.log)
=========================================================================================
  1.18 s/step mean (152 logged intervals), batch 3 x 8 V100, peak 17.8 GB,
  763 steps/epoch.
  10 epochs training only: 2.50 h wall = 20.0 GPU-h. With per-epoch AR selection (400
  holdout samples): 2.82 h wall (08:06-10:55 UTC) = 22.6 GPU-h. Holdout + val dumps
  about 0.5 h wall (4 GPU-h). One B1 run end to end: about 27 GPU-h.
  For scale: B2 (1,920 tokens, cached) 6.28 s/step, 14.0 h wall = 112 GPU-h training.

=========================================================================================
R3.2 B1 SEEDS (123, 2024; same recipe) -- Alpamayo/ar1_r32_eval.txt
=========================================================================================
Runs: B1_s123 05:21-08:37 UTC, B1_s2024 08:37-11:33 UTC (train + holdout / val dumps).
Official val, ALL 5,119; per seed (42 / 123 / 2024), mean +- sd (ddof 1; T9b quoted the
ddof-0 sd 0.044 for the same no-camera numbers):
                     no camera (A3, G6C)                    B1 (3 cams t0)
  L2@3s NoAvg   1.894 2.001 1.967 = 1.954 +- 0.054   2.165 2.048 2.061 = 2.091 +- 0.064
  L2@3s TemAvg  0.877 0.926 0.923 = 0.909 +- 0.027   1.033 0.974 0.971 = 0.993 +- 0.035
  ADE@6s        2.896 2.968 2.926 = 2.930 +- 0.036   3.125 3.009 3.010 = 3.048 +- 0.067
  ADE@6s med    2.003 2.077 1.941 = 2.007 +- 0.068   2.296 2.215 2.150 = 2.220 +- 0.073
  L2@3s p95     5.108 5.293 5.654 = 5.352 +- 0.278   5.570 5.308 5.328 = 5.402 +- 0.146
  Col% Tem 3s   0.641 0.514 0.658 = 0.605 +- 0.078   0.495 0.466 0.498 = 0.486 +- 0.018
Seed-mean difference B1 - no cam (paired scene bootstrap):
                L2@3s                  L2@3s TemAvg           ADE@6s
  ALL           +0.137 [+0.036,+0.255]  +0.084 [+0.023,+0.157]  +0.118 [+0.005,+0.244]
  EXCL. first   +0.175 [+0.076,+0.291]  +0.106 [+0.046,+0.178]  +0.167 [+0.053,+0.291]
  strata L2@3s  straight +0.081 [+0.039,+0.123]; turning -0.082 [-0.173,+0.012];
                stationary +0.476 [-0.024,+0.978] (excl. first frames +0.751
                [+0.234,+1.310])
  stationary false-go: no cam 0.016 / 0.012 / 0.037; B1 0.302 / 0.321 / 0.299
Over three seeds B1 stays worse than no cameras on L2 and ADE, with lower collision rates;
the false-go failure is present in every B1 seed.

=========================================================================================
R3.3 META-ACTION + TRAJ ARM (M1), seed 42 -- Alpamayo/ar1_r33_eval.txt
=========================================================================================
Build (ar1/finetune_meta.py): B1 base + 18 text tokens from the frozen LLM vocabulary (6
  lon words at t0+1..6 s, 1 token each; 6 lat words, 2 tokens each), then 24 trajectory
  tokens. Teacher forcing, plain CE on all 42 targets. Text logits come from the frozen
  lm_head. A3 recipe otherwise; 10 epochs. Inference: greedy decoding constrained to the
  valid words, then the trajectory.
Run: 11:35-14:41 UTC (1.25 s/step, peak 17.8 GB, about 25 GPU-h); best epoch 8 (holdout
  AR median ADE@6s 1.704; B1 s42 1.899). Text CE 0.25 at the end of training.
  Val dump: the first attempt timed out at the end-of-split all_gather (NCCL 10 min;
  per-rank decode times uneven); rerun with a 3 h timeout.
Expert (ar1/train_expert_meta.py, option A on the frozen M1 sequence, KV online, ctx 503;
  GT words in training, M1's own generated words at selection / inference): 80 epochs,
  best epoch 60 (holdout median 1.617). 4.25 s/step on 8 GPUs, 6.8 h wall = 54 GPU-h
  training; 0.43 GPU-h inference. The first final step OOMed (checkpoint loaded onto
  cuda:0 by all ranks); fixed with map_location, predictions rerun from the saved best.
Official val (L2 m, Col %):
                        L2@3s No / Tem  Col% 3s No / Tem  ADE@6s mean med p95  minADE6 3s/6s
  ALL 5,119
  no cam 3-seed [P]     1.954  0.909     1.45 / 0.60      2.930 2.109 7.371   = ADE
  B1 3-seed mean [P]    2.091  0.993     1.23 / 0.49      3.048 2.371 6.999   = ADE
  B1 seed 42 [P]        2.165  1.033     1.45 / 0.49      3.125 2.296 7.716   = ADE
  M1 token decoder [P]  2.203  1.043     1.54 / 0.61      3.183 2.295 7.758   = ADE
  M1 + expert, 1 sample 2.148  1.047     2.32 / 1.15      3.084 2.123 8.931   0.871 2.656
  M1 + expert, mean 6   2.111  1.022     2.34 / 1.22      3.046 2.079 8.849   0.871 2.656
  EXCL. first frames (4,979)
  no cam 3-seed [P]     1.618  0.706     1.21 / 0.50      2.512 2.064 6.318
  B1 seed 42 [P]        1.872  0.854     1.31 / 0.44      2.763 2.248 6.741
  M1 token decoder [P]  1.907  0.861     1.41 / 0.56      2.820 2.248 7.177
  M1 + expert, mean 6   1.832  0.852     2.31 / 1.18      2.702 2.026 8.117   0.702 2.312
Paired scene bootstrap, ALL (L2@3s; ADE@6s):
  M1 token - B1 s42           +0.038 [-0.039,+0.124]; +0.058 [-0.055,+0.179]
  M1 token - B1 3-seed mean   +0.111 [+0.023,+0.210]; +0.135 [+0.016,+0.269]
  M1 token - no cam 3-seed    +0.249 [+0.087,+0.440]; +0.253 [+0.062,+0.471]
  expert mean-6 - M1 token    -0.092 [-0.173,+0.002]; -0.137 [-0.227,-0.034]
  expert 1 sample - M1 token  -0.055 [-0.136,+0.040]; -0.099 [-0.190,+0.004]
  expert minADE_6@6s - token ADE@6s  -0.527 [-0.607,-0.440]
Strata L2@3s, ALL (straight / turning / stationary): no cam 1.678 / 2.570 / 2.529;
  B1 s42 1.810 / 2.545 / 3.166; M1 token 1.790 / 2.426 / 3.508.
Meta-action words, val n_fut = 12, vs the CAN labels used in training (slot t+1 .. t+6 s):
  lon acc 0.716 0.644 0.604 0.576 0.564 0.559 (mean 0.610); macro-F1 mean 0.309
  lat acc 0.915 0.862 0.833 0.807 0.803 0.794 (mean 0.836); macro-F1 mean 0.437
  Unconstrained argmax equals the constrained word in 0.999 of slots.
  Against the 2 Hz GT labels: see Amendment A1.
Consistency (Table 9 analog; predicted-trajectory-derived meta == generated words):
  token decoder: all 12 slots 0.412, lon 6/6 0.462, lat 6/6 0.886
  expert mean-of-6: 0.404 (lon 0.437, lat 0.932); expert sample 0: 0.264
  GT self-consistency with 2 Hz labels: 1.000 (A1)
Reading: adding meta-action words does not change the token decoder against its B1
  twin. Still worse than no cameras and the B1 3-seed mean. Worst when stationary
  (3.51). The expert's mean of 6 is better than the token decoder on ADE@6s (-0.14 m)
  but has 2x the collision rate and a heavier p95 tail. The words collapse toward
  maintain / straight (A2). Single seed; seeds 123 / 2024 running.

=========================================================================================

=========================================================================================
R3.4 CoC DATASET (ar1/coc_template.py v3; 4b4ebc7; Alpamayo/ar1_r34_coc_all.txt)
=========================================================================================
a. Stall cause (cProfile, 40 val samples: 216 s): 172 s of it were 545 calls to the
   devkit's NuScenesMap.get_records_in_radius (0.32 s each), which scan EVERY record of
   the layer and
   rebuild each shapely polygon from its nodes on every call (1.18M polygon tests, 2.5M
   Polygon constructions); record_on_point has the same linear scan (16 s). About 5 s per
   sample = about 33 h for the 24k samples, single-threaded. The job was not hung, only
   slow, with no progress output.
   Fix: build each layer's polygons ONCE per map and query an STRtree (shapely 2.0.7);
   scenes run in a 40-process pool (map built lazily per process). 40 samples: 2.3 s.
   All 29,049 samples: 36-39 s build, 51 s end to end including tokenisation.
b. Rule fixes (pre-registered in the generator header):
   - turn: heading change accumulated INSIDE map intersection polygons > 30 deg (a bend
     within a lane = lane keeping).
   - yield: only agents on the ego path or pedestrians on a crosswalk the path crosses.
     First implementation = lane polygons of the ego lane + successors (depth 3). It
     produced yield in 16.5% of train, with candidates 5-10 m to the side (successor
     branches cover crossing and turn lanes in intersections). Replaced, before any
     audit, by: in path = centre within 1.5 m + half width of the ego's REALIZED 6 s path
     (GT future positions; used for labelling only, like AR1's labeller that sees the
     future). If the ego covers < 5 m: inside the ego lane or its direct successors and
     |y| < 2.5 m. The same in-path test is used for leads.
c. Generated for every keyframe with n_fut >= 6 (data/ar1_coc.pkl: decision, trace, full
   CoC text = "Driving decision: ... Critical components: ... Reasoning: ...").
   Token lengths use the Cosmos (Qwen2.5) tokenizer.
   lon decision     train (22,213)   holdout (1,717)   val (5,119)
   set speed          0.684            0.645             0.611
   stop (static)      0.171            0.127             0.125
   lead following     0.092            0.116             0.235
   yield              0.042            0.101             0.021
   speed adaptation   0.011            0.011             0.008
   passing            0.000 (7)        0 (0)             0 (0)
   lat decision
   lane keeping       0.761            0.726             0.796
   none (stopped)     0.149            0.195             0.119
   turn L / R         0.028 / 0.040    0.024 / 0.035     0.027 / 0.034
   lane change L / R  0.009 / 0.012    0.009 / 0.010     0.010 / 0.013
   tokens, trace only     mean 16.5 / 17.5 / 17.6, median 15, p95 26-27, max 34
   tokens, full CoC text  mean 78.4 / 81.8 / 82.5, median 74-76, p95 114-121, max 156
   Train and val differ (lead following 9% vs 24%, yield 4% vs 2%); val has no passing.
d. AR1_COC_AUDIT.txt: 100 random VAL traces, stratified by lon decision (20 each of the 5
   classes present in val), seed 0. Each has sample id, scene, decision, causes
   (components), trace, CAM_FRONT path, and a [future] line for the auditor.
   The CoC training arm is NOT launched (waits for Kushal's audit).

=========================================================================================
AMENDMENT A1-A3 (R3.3 M1 seed 42; ar1/r33_amend.py; Alpamayo/ar1_r33_amend.txt)
=========================================================================================
Val n_fut = 12 (n = 4,219). NEW labels = R1.2 rules on the GT 2 Hz trajectory controls.
CAN labels = R1.2 on 10 Hz CAN (what M1 was trained on); side number.
A1. GT self-consistency (GT-trajectory-derived vs NEW labels): 1.000 (all 12 slots);
    identical by construction. NEW vs CAN labels agree on all 12 slots in 0.180 of
    samples (slot mean lon 0.693, lat 0.945): the old ceiling was a label mismatch.
                                   lon acc / mF1 (slot mean)   lat acc / mF1 (slot mean)
    M1 words vs NEW labels          0.478 / 0.313               0.825 / 0.418
    M1 words vs CAN labels (side)   0.610 / 0.309               0.836 / 0.437
    M1 CONSISTENCY (predicted-trajectory-derived == generated words; label-free):
      all 12 slots 0.412; lon 6/6 0.462; lat 6/6 0.886; slot mean lon 0.828, lat 0.976
    Predicted-trajectory-derived vs NEW labels: all 12 0.080 (slot mean lon 0.434, lat
      0.816); vs CAN 0.128.
    M1's words match its own trajectory far better (0.41) than they match the GT labels.
A2. Collapse. Generated vs GT class shares per slot (full table in the txt):
    lon maintain: generated 0.544 (t+1 s) -> 0.670 (t+6 s); CAN 0.487 -> 0.506; NEW 0.36.
    lon gentle acc / gentle dec: generated 0.158 -> 0.088 / 0.157 -> 0.062 (CAN ~0.19 /
      ~0.16). strong acc, strong dec, reverse: never generated (CAN 0.1-0.5%; NEW ~5%).
    lat: generated straight 0.843 -> 0.881 (CAN 0.816 -> 0.834); sharp L / R rare.
    Distinct 12-word sequences: generated 389, CAN labels 1,548, NEW 2,893.
    Share equal to the majority sequence (maintain x6, straight x6), which is also the
      most frequent train sequence (2,808 / 18,313): generated 0.433, CAN 0.168,
      NEW 0.023.
    The generated words collapse toward maintain / straight, more so at longer horizons.
A3. Re-decode with valid-word-constrained decoding: QUEUED after the M1 seeds (tmux a3).
    Note: the original M1 dump already used constrained greedy decoding (unconstrained
    argmax agreed on 0.999 of word slots), so M1_G is a determinism check. M1_T07 =
    sampling at T = 0.7 over the valid words. Results pending.
