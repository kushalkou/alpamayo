# OVERNIGHT REPORT — decode-rule / shrinkage queue

Run 2026-09-06 19:05 UTC -> 21:46 UTC. All 10 items COMPLETE. No training run started.

## MORNING SUMMARY

1.  All 10 items completed. Nothing failed. Verification passed EXACTLY (0.000e+00).
2.  SOMETHING FINALLY BEATS CV. A model+CV shrinkage ensemble, alpha fit on val and
    applied to test: ego-only 2.894 m vs CV 3.062 (-0.168, 95% CI [-0.194,-0.142],
    p<1e-4); full-vision 2.910 (-0.152, [-0.181,-0.123], p<1e-4).
3.  alpha* = 0.30 [0.25,0.35] full-vision, 0.25 [0.25,0.30] ego-only,
    0.10 [0.05,0.15] zero-input. ALL THREE CIs EXCLUDE 0.
4.  One-sentence answer: alpha* differs significantly from 0, so the model carries
    information CV does not have, and the blend is a working system.
5.  LEAD WEDNESDAY WITH: 2.894 m vs 3.062 m CV, -5.5%, p<1e-4 -- the first
    configuration in this project to beat constant velocity. Caption it honestly as a
    model+CV ENSEMBLE, not the model beating CV on its own.
6.  Biggest effect on turning: blend 4.868 vs CV 5.218 (-0.349, [-0.435,-0.263]),
    win rate 0.666 [0.629,0.702].
7.  Stationary hybrid works: tau=0.5 makes V1s strictly dominate argmax in EVERY
    stratum for both trained models (was +0.082 worse on stationary, now -0.019 better).
8.  Per-stratum alpha does NOT help deployably (detector version +0.015, p=0.059,
    i.e. slightly WORSE). Global alpha is the honest headline.
9.  Adaptive shrinkage (3.3) is negligible: -0.005 and -0.002 m. Not worth the
    complexity; recommending neither, as instructed.
10. Shrinkage does NOT statistically subsume the decode fix (+0.036, p<1e-4) but
    absorbs 89% of it: the 0.333 m decode gain shrinks to 0.036 m once both are blended.
11. NEEDS A HUMAN: your zeroboth alpha*~1.0 cross-check came out 0.10. The premise was
    inverted, not the sweep -- details in item 2/4 below. I am confident but you should
    check my reasoning.
12. NEEDS A HUMAN: item 2 shows the Gate 2.3(c) oracle-switch headroom is largely an
    artifact of the min() operator, not complementary information. That is a correction
    to a Gate 2 claim I made. Details below.

---

## 1. VERIFICATION — PASS

Re-decoded 100 test samples the slow way (one independent GPU pass per variant through
decode_trajectory_ex) and compared with the offline reconstruction from dump_test.pkl.

    worst max|ADE_slow - ADE_offline| over all 3 models x 2 variants = 0.000e+00
    worst max|value diff| per slot                                    = 0.000e+00
    max|ADE(alpha=0) - ADE(CV)| over 200 test samples                 = 0.00e+00

Not merely under 1e-6 -- bitwise identical, which is what should happen when the two
decodes are deterministic functions of the same forward pass. Every offline sweep below
is therefore valid. Raw: `Alpamayo/overnight/01_verify.txt`.

## 2. ZEROBOTH vs CV — prediction CONFIRMED, plus a correction to Gate 2.3(c)

Trajectory distance |zeroboth - CV| on TURNING (n=638): 0.045 m @1s, 0.140 @2s,
0.324 @3s, 1.477 @6s.

Decomposition (mean L2 from CV @6s, TURNING):

    zeroboth accel + CV curvature      1.4509     <- essentially all of it
    CV accel + zeroboth curvature      0.2273
    zeroboth (both)                    1.4765

YOUR PREDICTION HELD: the divergence is longitudinal only. Swapping in zeroboth's
curvature barely moves the trajectory; swapping in its acceleration reproduces almost the
entire divergence. The 2.4 "predicts zero turns" conclusion stands and needs no restating.

It is in fact stronger than 2.4 implied. Across all 3,614 test samples zeroboth emits ONE
CONSTANT control sequence -- max|predicted curvature| has mean = p50 = p95 = max = 0.00638
under V1s (0.00430 under argmax), and max|predicted accel| is likewise constant at 0.3694.
It is not "a model that rarely predicts turns"; it is a model that ignores its input
entirely and emits a fixed constant-acceleration, near-straight plan.

CORRECTION TO GATE 2.3(c) -- I need to flag this against my own earlier claim. Per-sample
ADE correlation between zeroboth and CV on TURNING is pearson r = 0.955, and the mean
difference is only 0.072 m, yet the oracle min(zeroboth, CV) still "gains" 0.424 m over CV.
A model that emits a single constant plan cannot possibly hold complementary scene
information, so that 0.424 is the min() operator harvesting sample-level noise between two
near-identical predictors. By the same mechanism the Gate 2.3(c) headroom I reported for
the trained models (0.712 overall, 1.162 on turning) is INFLATED and overstates what any
real selector could reach. zeroboth is a useful calibration of that inflation: about 0.42 m
of "headroom" is available between two predictors that agree at r=0.955. I should not have
presented those oracle numbers without this null.

Raw: `Alpamayo/overnight/02_zeroboth.txt`.

## 3. GATE 3.1 — STATIONARY HYBRID — works, tau=0.5

Rule: per slot, p(STOP) > tau -> argmax value, else expectation. tau fit on VAL.
VAL-selected tau: 0.5 (y1_full), 0.7 (y1_ego), 0.3 (zeroboth, curve is flat).

TEST ADE@6s mean (median), n=3614. CV = 3.062 (2.409) / stationary 0.736 (0.001):

    model / variant              ALL              STATIONARY        NON-STAT
    CV baseline               3.062 (2.409)      0.736 (0.001)    3.546 (2.936)
    y1_full / argmax          3.921 (3.241)      0.786 (0.001)    4.572 (3.675)
    y1_full / V1s             3.588 (2.974)      0.868 (0.183)    4.154 (3.363)
    y1_full / hybrid tau=0.5  3.570 (2.979)      0.767 (0.001)    4.153 (3.363)
    y1_ego  / hybrid tau=0.7  3.620 (2.969)      0.690 (0.001)    4.229 (3.414)

ANSWER: yes. For both trained models the hybrid beats argmax in EVERY stratum with all
CIs excluding zero (y1_full: ALL -0.350 p<1e-4, STATIONARY -0.019 p=0.018, NON-STAT
-0.419 p<1e-4). The stationary regression that V1s introduced (+0.082 m) is gone, and the
stationary median returns to 0.001 from 0.183. The hybrid costs nothing off-stationary
(-0.000, p=0.26), so it is a strict improvement.

One exception worth noting: for zeroboth the hybrid is slightly WORSE than argmax on
stationary (+0.015, p=0.025). Its p(STOP) never crosses the low tau it selected, so the
rule degenerates. Irrelevant in practice -- zeroboth is not a deployable model.

Raw: `Alpamayo/overnight/03_05_shrink.txt`.

## 4. GATE 3.2 — GLOBAL SHRINKAGE — THE HEADLINE

This is a model+CV ENSEMBLE, not the model beating CV.

VAL alpha curve, ADE@6s (alpha=0 is CV exactly, verified):

    y1_full/V1s  0.00:3.093  0.20:2.948  0.30:2.930  0.40:2.944  0.60:3.063  1.00:3.628
    y1_ego/V1s   0.00:3.093  0.20:2.962  0.30:2.958  0.40:2.988  0.60:3.145  1.00:3.790
    zeroboth/V1s 0.00:3.093  0.10:3.090  0.20:3.097  0.30:3.111  0.60:3.197  1.00:3.400

    alpha*  y1_full  0.30  95% CI [0.25, 0.35]
            y1_ego   0.25  95% CI [0.25, 0.30]
            zeroboth 0.10  95% CI [0.05, 0.15]

THE ANSWER, UNHEDGED: alpha* differs significantly from zero for every checkpoint -- all
three bootstrap CIs exclude 0 -- so the model carries information constant velocity does
not have, and the blend is a working system.

TEST at alpha*, y1_full/V1s (mean / median / p95):

    stratum        n      CV     alpha=1   blend   medCV  med_bl   p95 a=1  p95 blend
    ALL         3614    3.062     3.588    2.910   2.409   2.280     9.595     8.017
    STRAIGHT    2976    2.600     3.245    2.490   2.006   1.952     8.456     6.865
    TURNING      638    5.218     5.189    4.868   4.953   4.330    12.877    10.641
    STATIONARY   622    0.736     0.868    0.729   0.001   0.066     4.592     4.358

Paired bootstrap on TEST, blend vs CV:

    y1_full   ALL         -0.152  [-0.181,-0.123]  p<1e-4   win 0.513 [0.497,0.529]
    y1_full   STRAIGHT    -0.110  [-0.139,-0.081]  p<1e-4   win 0.480
    y1_full   TURNING     -0.349  [-0.435,-0.263]  p<1e-4   win 0.666 [0.629,0.702]
    y1_full   STATIONARY  -0.007  [-0.037,+0.022]  p=0.66   (n.s.)
    y1_ego    ALL         -0.168  [-0.194,-0.142]  p<1e-4   win 0.535 [0.518,0.551]
    y1_ego    TURNING     -0.332  [-0.406,-0.257]  p<1e-4   win 0.719 [0.683,0.753]
    zeroboth  ALL         -0.014  [-0.018,-0.010]  p<1e-4

The tail is where the blend earns most of it: p95 on TURNING falls 12.877 -> 10.641, and
overall 9.595 -> 8.017. That is exactly the failure mode Gate 2.3 identified.

CROSS-CHECK YOU ASKED FOR -- zeroboth alpha* came out 0.10, NOT near 1.0. The sweep is
fine; the premise was inverted. If a model equals CV plus noise, then
blend = CV + alpha*noise, whose error INCREASES with alpha, so the optimum is near 0, not
near 1. alpha* -> 1 would be expected only for a model BETTER than CV. The real signature
of "zeroboth is CV-like" is that its curve is nearly FLAT (3.090 to 3.111 across
alpha 0.1-0.3, against 2.930 to 3.628 for y1_full), which is what we observe. Its alpha*
is small but strictly positive (CI [0.05,0.15] excludes 0), consistent with its one
constant non-zero acceleration carrying a sliver of real signal. I am confident in this
reading but flagging it since it contradicts your stated expectation.

## 5. GATE 3.2b — PER-STRATUM ALPHA — does NOT help deployably

VAL alpha* per stratum (y1_full/V1s): straight 0.25 [0.20,0.25], turning 0.60 [0.50,0.65],
stationary 0.65 [0.55,0.75]. The turning optimum really is much higher than the straight
one -- the model is worth more where the road bends.

TEST ADE@6s (mean / median / p95), y1_full/V1s:

    CV baseline                    3.062   2.409   8.382
    (a) global alpha*              2.910   2.280   8.017
    (b) per-stratum ORACLE label   2.902   2.267   8.110   [UPPER BOUND, not a system]
    (c) per-stratum DETECTOR (i)   2.925   2.299   7.991   [deployable]

    (b) - global   -0.0075  [-0.0236,+0.0086]  p=0.36   n.s.
    (c) - global   +0.0149  [-0.0005,+0.0298]  p=0.059  WORSE, marginal
    (c) - CV       -0.1374  [-0.1723,-0.1018]  p<1e-4

CONCLUSION: global alpha is the honest headline. Even with the ORACLE stratum label the
gain is not significant (p=0.36), so per-stratum tuning is overfitting the val split; with
the deployable past-curvature detector it is slightly worse than global. The detector is
too weak (precision 0.365) to route on, consistent with Gate 2.4.

## 6. GATE 3.3 — ADAPTIVE SHRINKAGE — negligible; recommending neither

    y1_full/V1s   fits: (A) alpha0=0.35 lambda=4 | (B) alpha0=0.35 gamma=0.125
    CV                3.062   2.409   8.382
    global alpha*     2.910   2.280   8.017
    (A) curv-adaptive 2.905   2.273   7.965    (A)-global -0.0053 [-0.0104,-0.0001] p=0.044
    (B) per-horizon   2.908   2.281   8.012    (B)-global -0.0022 [-0.0030,-0.0013] p<1e-4

Both are statistically significant and practically irrelevant: 5 mm and 2 mm against a
0.152 m effect from the global blend. (B)'s fitted profile is nearly flat (0.35 decaying
to 0.26 over 6 s), i.e. the data does not want much per-horizon decay. For y1_ego the
curvature-adaptive version is not even significant (-0.0012, p=0.78). Reporting both,
recommending neither, as instructed.

## 7. GATE 3.4 — DOES SHRINKAGE SUBSUME THE DECODE FIX? — statistically no, practically
   mostly

2x2 on TEST, alpha* fit on VAL separately per decode (y1_full):

    cell                    mean    median     p95
    argmax / alpha=1       3.921     3.241   10.530
    argmax / alpha*        2.946     2.318    8.046
    V1s    / alpha=1       3.588     2.974    9.595
    V1s    / alpha*        2.910     2.280    8.017

    blended-argmax - blended-V1s   +0.0357  [+0.0256,+0.0460]  p<1e-4  -> NOT subsumed
    unblended  V1s - argmax        -0.3327  [-0.3703,-0.2950]  p<1e-4

The honest reading is the ratio, not the p-value: the decode fix is worth 0.333 m on its
own and 0.036 m once both are blended, so shrinkage absorbs 89% of it. Both are variance
reduction acting on the same over-commitment; they are 89% the same finding. With
n=3,614 the residual 11% is detectable, so they are not literally identical, and V1s
remains the better decode to blend. For Wednesday I would present them as one mechanism
with two implementations rather than two independent wins.

## 8. VISUALS — `Alpamayo/viz/`

    a_alpha_curve.png       VAL alpha curve, 3 checkpoints, alpha* marked with CI band
    b_turning_cdf.png       per-sample ADE@6s CDF on TURNING: CV vs model vs blend
    c_decode_vs_vision.png  decode fix -0.333 p<1e-4 vs vision effect -0.045 p=0.35
    d_bev_turning.png       two BEV turning scenes: a median win and a p99 blow-up

Panel (d) is the most useful slide: left, the model tracks a real turn at 0.87 m while CV
flies off at 11.20 m; right, the model turns the wrong way at 23.32 m where CV manages
9.40 m. Same model, same decode -- that is the over-commitment story in one image.

## NOTE ON STRATUM DEFINITIONS (do not misread the tables)

Items 3-5 use the Gate 2 stratum definition, where STRAIGHT/TURNING are cut on
max|GT curvature| and STATIONARY OVERLAPS both (2976 / 638 / 622, summing past 3614).
Items 6-7 use disjoint strata, carving stationary out first (2441 / 551 / 622 = 3614).
Same underlying samples; the CV column differs between the two (3.062 overall either way,
but 2.600 vs 3.101 on "straight") because of that overlap. Compare like with like.


---

## GATE 4.1 — IS alpha* VARIANCE REDUCTION OR REAL SIGNAL? — HEADLINE SURVIVES

Raw: `Alpamayo/overnight/08_gate41.txt`. All in the EGO frame (rotate by -yaw0; ADE is
rotation invariant, but it makes "mean deviation" meaningful).

Setup. With M=model, C=CV, G=ground truth, D=M-C and R=C-G:

    blend(a) - G = a*D + R      =>  squared-error optimum  a* = <D, G-C> / ||D||^2

Under "CV + zero-mean noise independent of the residual" that inner product has
expectation zero, so a* = 0. A strictly positive a* means D points along the correction
CV actually needs.

### (a) bias / variance vs ground truth (TEST, ego frame)

    subset     predictor      MSE    bias^2      var   |bias| m
    ALL        CV           27.845    0.483   27.362      0.507
    ALL        model        38.783    1.828   36.956      1.038
    TURNING    CV           61.332    0.259   61.073      0.356
    TURNING    model        73.763    0.783   72.980      0.681

The model is a WORSE standalone predictor than CV on every component -- more bias and
more variance -- which is why it loses head-to-head. That is not incompatible with it
being a useful *component*; see (b).

### (b) closed-form optimum, fit on VAL

    a*_MSE = <D, G-C> / ||D||^2                        y1_full +0.3301   y1_ego +0.2824
    corr(D, G-C)                                       y1_full +0.3392   y1_ego +0.3131
    empirical a* minimising ADE (the headline)         y1_full  0.30     y1_ego  0.25
    a* predicted by "CV + zero-mean noise, NO signal"           0.0000

The closed form (0.3301) and the empirical ADE optimum (0.30) agree to within one grid
step, and both are far from the no-signal prediction of 0.

### (c) null controls — identical val-fit / test-eval pipeline

    predictor      alpha*   test ADE   gain vs CV      sd (20 seeds)
    CV                  -      3.062       0.0000       -
    real            0.300      2.910       0.1523       -
    null_mean       0.000      3.062       0.0000       -
    null_gauss      0.000          -       0.0000       0.0000
    null_perm       0.000          -       0.0000       0.0000

Every null selects alpha*=0 and therefore gains exactly nothing. `null_mean` is the
constant population-level correction (mean D learned on val); `null_perm` takes D from a
random OTHER sample, preserving its magnitude and temporal structure while destroying
scene alignment; `null_gauss` is the specified zero-mean Gaussian matched to D's
per-timestep variance.

### (d) sharper tests — because a null pinned at the alpha=0 boundary is one-sided

Zero gain for the nulls is *necessary* (alpha=0 recovers CV), so three tests that are not
boundary-limited:

    d1. force the real alpha*=0.30 onto each predictor (TEST):
        real          2.910  -> +0.1523 m vs CV
        null_gauss    3.424  -> -0.3618 m
        null_perm     3.378  -> -0.3161 m
        null_mean     3.120  -> -0.0581 m

    d2. permutation test on corr(D, G-C): observed +0.3392,
        0 of 2000 permutations reach it  ->  p = 0.0005

    d3. VAL curve shape, ADE@6s at alpha = 0.0 / 0.1 / 0.3 / 0.5
        real        3.0926  3.0001  2.9304  2.9890   dips then rises
        null_gauss  3.0926  3.1681  3.4538  3.8619   monotone increasing

### VERDICT — the item-4 conclusion is NOT retracted

The gap is the whole effect: real gain 0.1523 m, every null 0.0000 m, i.e. **100% of the
blend improvement requires signal and none of it is variance reduction**. Blending a
signal-free predictor of identical magnitude at the same strength does not merely fail to
help, it *costs* 0.32-0.36 m. The correlation between the model's deviation and the
correction CV needs is +0.34 with p=0.0005.

Two further points worth stating:

1. **The gain is entirely scene-specific, not a constant prior correction.** `null_mean`
   -- the average deviation applied to every sample -- gains nothing (alpha*=0) and costs
   0.058 m when forced. So the model is not just supplying "CV decelerates too little on
   average"; it is supplying a per-scene correction.
2. **A model can be worse than CV standalone and still carry usable information.** From
   (a) the model has 39% higher MSE and double the bias of CV. Those facts coexist because
   ADE ranks predictors while corr(D, G-C) measures whether the model's *disagreement*
   with CV is informative. This reconciles the whole project: every head-to-head test said
   the models were useless, and they were -- as replacements. As correction terms they are
   not.
