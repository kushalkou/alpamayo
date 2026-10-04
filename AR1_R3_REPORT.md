# AR1_R3_REPORT -- reproduction phase R3 (gate report)

Written 2026-10-04. Code: Alpamayo/code/ar1/; outputs: Alpamayo/ar1_r3*.txt. Pre-registered
rules applied without the planner. [P] = privileged input (VAD command, from GT future).

=========================================================================================
R3.0 CAMERA-HARM DIAGNOSIS (CPU, existing dumps; ar1/r30_diag.py)
=========================================================================================
a. Stationary val samples (v0_can < 0.5 m/s). Thresholds fixed before looking, first 3 s:
   GT stopped = max displacement < 0.5 m, GT moves = > 1.0 m; false-go = P(pred > 1.0 m |
   GT stopped); false-stop = P(pred < 0.5 m | GT moves). Alpamayo/ar1_r30a.txt.
                         ALL (993: 676 stopped, 283 move)  EXCL. first frames (853: 653 / 168)
   model                 false-go   false-stop             false-go   false-stop
   no cam s42 (A3)        0.016      0.625                  0.017      0.506
   no cam s123            0.012      0.629                  0.012      0.512
   no cam s2024           0.037      0.629                  0.038      0.512
   no cam 3-seed mean     0.022      0.628                  0.022      0.510
   B1 3 cams t0           0.302      0.424                  0.300      0.458
   B2 3 cams x 4 kf       0.320      0.371                  0.319      0.417
   The camera models start moving in about 30% of the samples where the car stays
   stopped, against about 2% without cameras.
b. AR ADE@6s on a fixed 1,000-sample train subset (n_fut = 12, seed 0) vs val (n_fut = 12),
   hybrid decode with the holdout tau. Alpamayo/ar1_r30b.txt.
   model                 train mean / med   val mean / med   gap val - train (mean / med)
   no cam s42 (A3)        2.708 / 1.970      2.896 / 2.003    +0.188 / +0.033
   B1 3 cams t0           2.783 / 1.935      3.125 / 2.296    +0.342 / +0.360
   B1 fits the train subset no better than the no-camera model but has a larger
   train-val gap.

=========================================================================================
R3.1 COST WITH THE CACHE (B1, 480 visual tokens; Alpamayo/ar1_r24_B1.log)
=========================================================================================
  1.18 s/step mean (152 logged intervals), batch 3 x 8 V100, peak 17.8 GB, 763 steps/epoch.
  10 epochs training only: 2.50 h wall = 20.0 GPU-h. With per-epoch AR selection (400
  holdout samples): 2.82 h wall (08:06-10:55 UTC) = 22.6 GPU-h. Holdout + val dumps
  about 0.5 h wall (4 GPU-h). One B1 run end to end: about 27 GPU-h.
  For scale: B2 (1,920 tokens, cached) 6.28 s/step, 14.0 h wall = 112 GPU-h training.

=========================================================================================
R3.3 META-ACTION + TRAJ ARM (M1) -- PENDING (GPU queue)
=========================================================================================

=========================================================================================
R3.4 CoC DATASET (ar1/coc_template.py v3; 4b4ebc7; Alpamayo/ar1_r34_coc_all.txt)
=========================================================================================
a. Stall cause (cProfile, 40 val samples: 216 s): 172 s of it were 545 calls to the devkit's
   NuScenesMap.get_records_in_radius (0.32 s each), which scan EVERY record of the layer and
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
