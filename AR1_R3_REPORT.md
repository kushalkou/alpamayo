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
