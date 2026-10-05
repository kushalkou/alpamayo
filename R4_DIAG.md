# R4_DIAG -- camera false-start, label and word diagnosis (diagnosis only)

HEADER (fixed and committed BEFORE any R4 number was computed; val is read for analysis
only, nothing here selects or tunes anything)
R3 definitions (code: ar1/r30_diag.py, w1/evalw1.strata):
  stationary = v0_can < 0.5 m/s at t0.  Over the first 3 s (6 steps), d = max distance
  from the origin: GT stopped d < 0.5 m; GT goes d > 1.0 m (between: neither).
  false-go = P(pred d > 1.0 m | GT stopped); missed-go = P(pred d < 0.5 m | GT goes)
  (= R3 "false-stop"); GT-go rate = share of stationary samples with GT d > 1.0 m.
  Predictions: saved AR dumps, STOP-aware hybrid decode, tau from holdout (q2_traj).
  Models: no cam = A3 / G6C_s123 / G6C_s2024; B1 = B1 / B1_s123 / B1_s2024;
  M1 = M1 / M1_s123 / M1_s2024.
New thresholds and choices (fixed here):
  1b TRAIN subset = 50 train scenes drawn with np.random.RandomState(0).choice over the
     sorted list of the 650 train scenes (holdout excluded), stationary samples only
     (n_fut >= 6): 314 samples. New dumps (GPU, counted in the cap) for all 3 seeds of
     no cam and B1. The R3.0b 1,000-sample random train dumps are a side number.
  1c "scene" = val scene with >= 1 GT-stopped stationary sample. Scenes ranked by pooled
     B1 false-go count (3 seeds; ties by scene name); top 10% = ceil(0.1 x n scenes).
  1d pooled B1 false-go per GT-stopped sample = mean over the 3 seeds; 95% CI = scene
     bootstrap (10,000 draws, seed 0).
     lead: an in-path agent (any class except cone / barrier) with 0 < x < 20 m, using
       the CoC v3 in-path test for a stopped ego (centre inside the ego lane or its
       direct successors, |y| < 2.5 m; ar1/coc_template.py).
     map: ego position within 10 m of a stop-line, crosswalk or intersection polygon
       (map expansion v1.3), reported combined and separately.
     nav command: VAD command (left / right / straight).
     first frame: q2_traj ff (no CAN and no past pose).
     time stopped before t0: t0 minus the last time the speed exceeded 0.5 m/s before t0,
       from CAN pose (50 Hz), else LIDAR ego poses; bins 0-2, 2-5, > 5 s; "unknown" if
       no source; if never above 0.5 m/s since the source starts, the elapsed time counts.
  1e Jaccard of the B1 false-go sample sets, per seed pair, plus 3-way.
  2a B1 seed 42, the 993 val stationary samples only, each gets the cached vision tokens
     of another stationary val sample (dump_w1 --shuffle_cams, permutation seed 99 over
     the stationary list). Unshuffled reference = the existing B1 val dump on the same
     samples. The existing all-val B1 camera shuffle is a side number.
  3a 2 Hz labels = the R1.2 rules applied to GT 2 Hz controls (ar1/r33_eval.derive:
     speed after step k = 2t-1, accel = speed change over that step, curvature = cur[k]
     if speed >= 1 m/s). Val n_fut = 12 (n = 4,219); train n_fut = 12 as a side line.
  3b "accel word" = the lon word of slot t0+1 s; GT = CAN label of that slot (training
     target); 2 Hz label as a side split. Stationary val samples, M1 seeds pooled.
  3c consistency = all 12 slots of words derived from the predicted trajectory equal the
     generated words; stationary vs moving (non-stationary) val; per seed and pooled.
Gate metrics (fixed here):
  G1 B1 train / val false-go = pooled over the 3 B1 seeds (train = the 314-sample subset).
  G2 share of pooled B1 val false-gos in the top 10% scenes (1c).
  G3 |shuffled - unshuffled| B1 s42 false-go on the 993 val stationary samples, in
     percentage points.
  G4 "per-slot agreement" = agreement of each of the 12 slots separately (val n_fut =
     12); the gate condition (< 80%) holds if ANY slot is below 80%. Note: these
     agreements were already computed in R3 amendment A1 (lon 0.69, lat 0.95 per slot).
  G5 consistency (3c) on all val samples, M1 seeds pooled; condition < 60%. Note: the
     per-seed values were already reported in R3 (0.41 / 0.57 / 0.52).
Ledger: inference only, 1.52 GPU-h of the 3 GPU-h cap (Alpamayo/r4_gpu_ledger.txt; one
  A3 train dump segfaulted at start when 4 jobs loaded at once, rerun alone). Raw output:
  Alpamayo/r4_diag.txt (ar1/r4_diag.py).

GATE VERDICTS (pre-registered wording, not reinterpreted)
  G1 B1 train false-go 0.010 (<= 10%) AND val 0.307 (>= 25%)  -> "overfitting"
  G2 top 10% scenes (5 of 46) hold 0.759 of false-gos (>= 50%) -> "scene-specific"
  G3 shuffled 0.321 vs real 0.302: 1.9 points (within 5)      -> "cameras act as a generic
     go bias"
  G4 min per-slot 2 Hz vs 10 Hz agreement 0.688 (< 80%)        -> "relabel at 2 Hz before
     any M1/CoC"
  G5 M1 follows its own words 0.497 (< 60%)                    -> "words are decorative"
  Facts that bear on several gates (no reinterpretation): 4 of the top 5 false-go scenes
  are whole-scene stationary parking-lot clips (scene-0770, 0771, 0775, 0777, nuScenes
  descriptions "Stationary ego vehicle, parking lot ..."; 5th: scene-0635 "Stationary in
  rain"); they hold 138 of 676 GT-stopped samples and 407 of 623 pooled false-gos.

PART 1  FALSE-START ANATOMY (stationary samples)
1a val, 993 stationary (676 GT stopped, 283 GT go; GT-go rate 0.285 for all rows)
   model   seed  false-go  missed-go  false-go excl. first frames (853)
   no cam  42    0.016     0.625      0.017
   no cam  123   0.012     0.629      0.012
   no cam  2024  0.037     0.629      0.038
   B1      42    0.302     0.424      0.300
   B1      123   0.321     0.378      0.320
   B1      2024  0.299     0.424      0.299
   M1      42    0.317     0.473      0.319
   M1      123   0.324     0.396      0.325
   M1      2024  0.303     0.364      0.303
   pooled: no cam 0.022 / 0.628; B1 0.307 / 0.409; M1 0.315 / 0.411
1b train 50-scene subset, 314 stationary (223 GT stopped, 80 GT go, GT-go rate 0.255)
   model   seed  false-go  missed-go
   no cam  42 / 123 / 2024   0.004 / 0.009 / 0.004   0.713 / 0.713 / 0.700
   B1      42 / 123 / 2024   0.013 / 0.013 / 0.004   0.487 / 0.450 / 0.487
   pooled: no cam 0.006, B1 0.010; side (R3.0b train dumps, n 197): 0.028 / 0.021.
   On train stationary samples B1 rarely false-goes (1%), against 31% on val.
1c 46 val scenes with GT-stopped samples; top 10% = 5 scenes hold 0.759 of the 623 pooled
   B1 false-gos (and 0.253 of GT-stopped samples); 25 scenes have >= 1 false-go.
   False-gos are concentrated in a few scenes.
1d B1 pooled false-go, 676 GT-stopped (mean over seeds; scene bootstrap 95% CI)
   all                n 676  0.307 [0.146,0.478]
   factor             yes: n  false-go [CI]         no: n  false-go [CI]
   lead < 20 m        329  0.179 [0.053,0.361]      347  0.428 [0.186,0.655]
   <= 10 m stop line  289  0.060 [0.008,0.157]      387  0.492 [0.250,0.704]
   <= 10 m crosswalk  225  0.007 [0.000,0.032]      451  0.457 [0.244,0.651]
   <= 10 m intersect. 437  0.096 [0.039,0.179]      239  0.693 [0.370,0.900]
   <= 10 m any        439  0.096 [0.039,0.176]      237  0.699 [0.377,0.906]
   first frame         23  0.333 [0.159,0.522]      653  0.306 [0.144,0.478]
   stopped before t0  0-2 s n 159 0.258 [0.158,0.369]; 2-5 s n 146 0.260 [0.124,0.413];
                      > 5 s n 348 0.348 [0.135,0.571]; unknown (no CAN) n 23 0.333
   nav command [P]: all 676 are "straight" (VAD command of a car that stays stopped).
   False-go is high away from stop lines, crosswalks and intersections, lower with a lead.
1e B1 false-go sets: Jaccard s42-s123 0.846, s42-s2024 0.699, s123-s2024 0.696;
   3-way intersection 163, union 254 (Jaccard 0.642).

PART 2  CAMERA SHUFFLE AT STANDSTILL (B1 s42, 993 val stationary, perm seed 99)
                                    false-go  missed-go  L2@3s mean
   unshuffled                       0.302     0.424      3.166
   shuffled among stationary        0.321     0.466      3.333
   side: all-val shuffle (R2.4)     0.501     0.311      3.788
   Swapping in another standstill scene's images leaves false-go almost unchanged.

PART 3  META-ACTION LABELS
3a 10 Hz CAN vs 2 Hz GT-trajectory labels, n_fut = 12 (val 4,219; train 18,313)
                         val                         train
   per-slot agreement    lon 0.688-0.695             lon 0.711-0.717
                         lat 0.942-0.949             lat 0.949-0.950
                         min 0.688, mean 0.819       min 0.711, mean 0.832
   lon shares CAN        acc 0.192 sacc 0.003 dec 0.161 sdec 0.005 maint 0.493 stop 0.146
   lon shares 2 Hz       acc 0.201 sacc 0.054 dec 0.189 sdec 0.047 maint 0.361 stop 0.149
   lat shares CAN / 2 Hz straight 0.825 / 0.815, steer L+R 0.160 / 0.169 (val)
   all-majority share    CAN 0.168, 2 Hz 0.023 (train 0.153 / 0.017)
   GT-vs-GT consistency  CAN 0.180, 2 Hz 1.000 (train 0.203 / 1.000)
   The label sets disagree on ~30% of lon slots; 2 Hz has far fewer all-maintain rows.
3b M1 seeds pooled, 993 val stationary (2,979 sample-seeds)
   generated lon, slot t+1: stop 0.639, maintain 0.266, gentle acc 0.093, gentle dec 0.002
   generated lon, all slots: stop 0.625, maintain 0.275, gentle acc 0.100 | CAN labels:
     stop 0.639, gentle acc 0.199, maintain 0.128, gentle dec 0.027
   generated lat, all slots: straight 0.933 (CAN 0.886)
   slot-1 accel word == CAN label: 0.617; L2@3s match 1.373 (n 1,839) vs mismatch 6.846
   (n 1,140). Side, 2 Hz label: 0.610; 1.280 vs 7.550.
   When the first speed word is wrong at standstill, the trajectory error is 5x larger.
3c consistency (trajectory-derived words == generated words, all 12 slots)
   seed     all val   stationary   moving
   42       0.410     0.684        0.344
   123      0.562     0.723        0.524
   2024     0.517     0.771        0.456
   pooled   0.497     0.726        0.441
   M1's trajectory follows its words more often at standstill than when moving.

PART 4  COST LEDGER (per seed, GPU-h, estimates from measured B1 / M1 runs; no runs)
   camera dropout (B1 recipe)              ~22-27 (B1 is 27; dropped samples cheaper)
   stationary-weighted loss (B1 recipe)    ~27
   80 tokens per camera (pooled cache)     ~19 (est. 0.8 s/step vs 1.18; no cache rebuild)
   M1 retrained on 2 Hz labels             ~31 (M1 measured: 25 train + 6 dumps)
   class-balanced M1                       ~31
