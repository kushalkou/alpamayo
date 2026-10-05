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
