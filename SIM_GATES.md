# SIM WEEK 1 -- gate log

Synthetic benchmark, code in Alpamayo/code/sim/{generate,model,gates,g1_diag}.py.
Model is a NON-AUTOREGRESSIVE MLP (24 independent heads x 65 classes) -- a
simplification relative to the nuScenes AR decoder. Metric ADE@6s (m), test n=3614.
Blend alpha fit on val (grid .05), applied to test. Bootstrap = shrink_lib.paired.

## Generator settings (seed 0)

    v0            resampled from nuScenes dump_test.pkl v0 (all 3614, with replacement)
    lon           brake/hold/accel, uniform 1/3
    lat           P(turn) = p, left/right 50/50
    A (accel mag) U(0.2, 1.0) m/s^2   HAND-SET (chosen so CV ADE ~ nuScenes at p=.2:
                  4.25 / 2.66 med vs nuScenes 3.06 / 2.41)
    K (curv mag)  nuScenes turning subset (max|curv|>0.05, n=638), drawn within the
                  sample's v0-decile band of that subset
    turn shape    plateau 6 segments (3 s) + 1-segment half ramps, onset U{-4..8}
                  (onset < 0 = turn in progress, visible in history yaw-rate)
    noise sigma   accel 0.3 m/s^2 per step, curvature 0.005 rad/m per step
    obs vector    9 dims: onehot lon | onehot lat | A | K | onset. With prob 1-rho it
                  encodes an independent z' drawn from the same prior at the same v0.
    history       past poses only. nuScenes compute_ego_state's current row uses
                  future_speeds[1] and future_yaws[0]; the sim does NOT copy that.
    conventions   as nuScenes: v0 = first-segment speed, accel slot 0 == 0,
                  accel[j] = ds/dt, STOP when speed < 0.1. Token floor 0.94 m mean.
    MLP           25 -> 512 -> 512 -> 24x65, GELU, dropout .1, AdamW 1e-3, 80 ep,
                  best val CE, unweighted CE. Train 11.6 s on one V100.

## G1 -- rho=0, p=0: PASS (no future leak)

    ADE@6s mean (median)
      CV                 2.545 (2.223)
      CV + meanctl       2.552 (2.242)     train-set mean control per slot
      mlp expect         2.792 (2.453)
      mlp argmax         3.081 (2.876)
      blend (a*=0.10)    2.540 (2.204)

    paired, TEST                         delta     95% CI            p
      mlp(expect) - CV                  +0.248  [+0.222,+0.274]  0.0000
      mlp(expect) - CV+meanctl          +0.240  [+0.216,+0.264]  0.0000
      blend - CV+meanctl                -0.012  [-0.020,-0.004]  0.0022

The standalone model has no advantage over CV. The blend beats CV+meanctl by 0.012 m.
Diagnosis (g1_diag.py):

    blend - (v0-only binned mean control, blended with CV, a*=0.20)
                                        +0.002  [-0.000,+0.005]  0.092
    mlp with obs permuted across test - mlp
                                        +0.005  [-0.001,+0.010]  0.137
    blend - CV+meanctl, moving only     -0.016  [-0.025,-0.007]
    blend - CV+meanctl, stationary      +0.011  [-0.001,+0.022]

A predictor that sees ONLY v0 matches the blend. The observation carries nothing.
The 0.012 m comes from legitimate v0 dependence (braking saturates at v=0, a stopped
car cannot brake), not from future information.

Calibration hook at G1: accel slots 1-11, CE_model 1.969, CE_unigram 2.155,
info gain 0.087. Mechanism: the unigram does not condition on v0 (STOP tokens at
v0~0). The rho=0 floor of this statistic is therefore NOT zero. (Not interpreted.)
