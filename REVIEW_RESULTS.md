# REVIEW RESULTS -- "Where does the signal come from?"

One file for the review deck. Plain ASCII. Every number carries the commit that produced
it. [P] = privileged input (derived from the ground-truth future: the navigation command
or the oracle meta-action). "measured" = computed here; "reported" = copied from a paper.
Figures: Alpamayo/viz/review/ (f1-f7, png + pdf).

---------------------------------------------------------------------------------------
## T1. The ego-state input contained two ground-truth targets (commit 495cb86)

compute_ego_state's current row equals GT accel slot 1 (max err 3.4e-7) and GT curvature
slot 12 (yaw_rate/v0, max err 1.5e-8) on all 3,614 custom-split test samples.
The trained models read it (argmax accuracy per slot, free-running; figure f1):

    slot                    a0     a1     a2     a3     k0
    leaky y1_ego           0.998  0.774  0.305  0.272  0.693
    causal_ego (fixed)     0.988  0.283  0.254  0.244  0.700     (commit 7f0e24e)
    zero-input model       0.844  0.138  0.138  0.139  0.611

Origin: commit 875d3a2 (2026-07-14) rewrote compute_ego_state from backward to forward
differences; the same commit produced the old "-36%" gain (FROZEN_RESULTS.md history,
commit 67987a3).

## T2. Removing the leak takes the blend gain over CV from 0.164 m to 0.014 m

Custom split, test, causal subset n=3,358, ADE@6s, CV 3.059; blend alpha fit on val;
paired bootstrap 95% CI (figure f2).

    predictor, blended with CV                 gain vs CV (m)            alpha*  commit
    leaked features only, no learning          +0.652 [+0.612,+0.692]    1.00    495cb86
    kinematic rule (causal), no learning       +0.184 [+0.138,+0.228]    1.00    495cb86
    leaky ego-only (frozen headline)           +0.164 [+0.137,+0.191]    0.25    495cb86
    causal ego-only (retrained)                +0.014 [+0.003,+0.024]    0.10    7f0e24e
    causal full vision (retrained)             +0.014 [+0.002,+0.025]    0.10    2e45c04
    zero-input model                           +0.008 [+0.006,+0.010]    0.05    495cb86
    causal full - causal ego, TURNING subset   -0.014 [-0.050,+0.021] (vision adds nothing)
Status per claim: FROZEN_RESULTS.md v2 (commit fc22e97).

## T3. Our causal, perception-free baselines sit at published planner L2 (figure f3)

Official nuScenes val, all 5,119 samples, VAD-style evaluation frame (lidar), v0 from
CAN at the last message <= t0. Measured: commits 72d60a1 (CV, KIN) and 950101a
(Ego-MLP). Reported: planner-verified (GaussianAD table; UniAD paper).

                                    L2 TemAvg 1/2/3 s     L2 NoAvg 1/2/3 s      Col% TemAvg 1/2/3
    CV (measured)                   0.376 0.784 1.326     0.527 1.448 2.762     0.14 0.27 0.75
    kinematic rule (measured)       0.278 0.562 0.998     0.362 1.049 2.178     0.16 0.23 0.57
    Ego-MLP + cmd [P] (measured)    0.241 0.461 0.777     0.324 0.822 1.633     0.29 0.56 0.87
    VAD-Base (reported)             0.41  0.70  1.05      0.54  1.15  1.98      0.07 0.17 0.41
    UniAD (reported)                --                    0.48  0.96  1.65      --
Metric code: VAD metric_stp3.py functions ported verbatim; gate: GT L2 = 0 exactly,
unmasked GT collision 0.37-0.53% per step (commit e089d0f).
FLAG: a CV seeded with the NON-causal forward speed |p1-p0|/dt scores TemAvg
0.20/0.53/1.01 (commit e089d0f) -- lower than every causal row above.

## T4. VAD's converter reads the future; its ego features help only where they do

vad_nuscenes_converter.py (quoted with line numbers, commit 72d60a1): (1) can_bus accel/
vel from the first CAN message AFTER t0 (L180-190 loop-variable bug); (2) nearest-message
v0 and steering (L39-43, can be after t0); (3) forward difference from the next keyframe
on the first sample of every scene (L477-480); (4) except-path v0 from the first future
offset (L503-506). Also (commit 762c5af): quart_to_rpy (L32-37) reads (w,x,y,z)
quaternions as (x,y,z,w), so ego yaw ~ 0 and yaw rate ~ 0.
Ego-MLP + cmd [P], 3 seeds, identical recipe; (a) our causal features vs (b) VAD's;
L2@3s NoAvg seed means (figure f4; commit 762c5af):

                                  (a) ours    (b) VAD     (b) - (a), paired scene bootstrap, seed 42
    all 5,119                      1.640       1.345      -0.291 [-0.351,-0.232]
    excluding 140 first frames     1.314       1.345      +0.035 [+0.007,+0.065]
    kinematic rule, all 5,119      2.178       1.998      -0.180 [-0.245,-0.115]
    kinematic rule, excl. first    1.820       1.997      +0.177 [+0.140,+0.216]

## T5. The VLA cannot even memorise 256 samples: slot 0 is the bottleneck (figure f5)

Overfit test, run 1 (Y1 recipe, ego + cmd [P], no visual tokens), 150 epochs on 256
official-train samples, commit c41cdfc. Floor on these samples 0.426 mean / 0.092 median.

    final AR median ADE@6s            1.058  (best checkpoint 0.899)
    teacher-forced token accuracy     97.1% overall, but slot 0 = 49%
    samples with slot 0 right         n=125, median ADE 0.12 m (all 24 tokens right, = floor)
    samples with slot 0 wrong         n=131, median ADE 2.80 m (85% of wrong tokens >= 4 bins off)
Input diagnostics on run 1 (commits c4fc454, 982d1ec): ego tokens have L2 norm 19.4 vs
0.87 for Cosmos text embeddings (22x); ego features unstandardised; mean pairwise cosine
of ego tokens across samples 0.72 after the layer-0 RMSNorm; mean-share 0.62-0.73;
no padding; slot 0 reads the last context token (cmd) under a full causal mask.
Same test, two follow-up runs (commits e0c6680, ed26214):

    run                                         median ADE@6s   slot-0 acc   shuffled slot-0   gate
    run 1 (Y1 recipe)                           1.058           0.488        --                FAIL
    retry (LoRA dropout 0, constant LR)         0.447           0.598        0.047             FAIL (slot 0)
    fix (standardised features, LayerNorm +     0.098           0.992        0.039             PASS
         gain to text norm 0.87, ego-MLP LR x10)
The fix reaches the tokenizer floor (0.092 median). The input pathway, not the
decoder, was the bottleneck. Stage A uses the fix recipe.

## T6. Mini information ladder on a small Ego-MLP (a demonstration, not the VLA)

Official val, 3 seeds, seed mean (sd <= 0.011); commit e0f7516; figures f6, f7.

    rung                                   ADE@6s all / excl. first    L2@3s all / excl. first
    CV (no learning)                       3.742 / 3.314               2.762 / 2.420
    kinematic rule (no learning)           3.161 / 2.712               2.178 / 1.820
    L1 ego only                            2.591 / 2.123               1.738 / 1.368
    L2 ego + cmd [P]                       2.439 / 2.027               1.637 / 1.310
    L2b cmd only, no ego [P]               9.849 / 9.845               9.103 / 9.099
    L5 ego + oracle meta-action [P]        1.750 / 1.537               1.193 / 1.013
    oracle-kinematic rule [P]              2.671 / 2.337               1.955 / 1.666
    L5 trained with 10% flips, test flips  0%: 1.820   10%: 2.033   20%: 2.246   40%: 2.693  (ADE, all)
    gap closed (X-L2)/(L5-L2), ADE, all    L1 -0.22; L5n 0.90 / 0.59 / 0.28 / -0.37 at 0/10/20/40%

## T7. Stage A: the VLA on the official split

T7-A1. With ego status and the command, the VLA beats CV by 0.68 m at 3 s and the
kinematic rule by 0.10 m, but stays 0.45 m behind a small MLP on the same inputs.

    official val, all 5,119; A1 = ego + cmd [P], fix recipe, no visual tokens, hybrid decode
                    L2 TemAvg 1/2/3 s    L2 NoAvg 3 s   Col% TemAvg 3 s   ADE@6s
    CV              0.376 0.784 1.326    2.762          0.75              3.742
    kinematic rule  0.278 0.562 0.998    2.178          0.57              3.161
    Ego-MLP + cmd   0.241 0.461 0.777    1.633          0.87              2.434
    A1              0.274 0.555 0.967    2.080          0.57              3.082
    A1 - KIN, L2@3s: -0.098 [-0.163,-0.032]; ADE6 -0.079 [-0.192,+0.039] (n.s.)
    A1 - Ego-MLP, L2@3s: +0.447 [+0.379,+0.516]
    input-shuffle: ADE6 3.08 -> 5.66, slot-0 accuracy 0.20 -> 0.05 (reads its inputs)
    (commit fe97d9c; later runs A2, A3, A0 appended below)

T7-A2. Given the oracle meta-action, the VLA beats CV by 1.1 m and the oracle-kinematic
rule by 0.30 m at 3 s, but only ties the small MLP; it still does better than the
rule as the labels get noisier.

    all 5,119; A2 = ego + oracle meta-action [P] (lat + lon tokens), trained with 10% flips
                        L2 TemAvg 1/2/3 s    L2 NoAvg 3 s   Col% TemAvg 3 s   ADE@6s
    oracle-kinematic    0.348 0.658 1.022    1.955          0.70              2.671
    A2                  0.255 0.487 0.805    1.651          0.48              2.454
    A2 - oracle-kinematic, L2@3s: -0.304 [-0.464,-0.171]; ADE6 -0.217 [-0.482,-0.006]
    A2 - Ego-MLP + cmd,  L2@3s: +0.018 [-0.056,+0.092] (n.s.)
    A2 - A1,             L2@3s: -0.429 [-0.489,-0.370]
    test flips 0/10/20/40%: A2 ADE6 2.454 / 2.580 / 2.676 / 3.000;
                            oracle-kinematic 2.671 / 3.626 / 4.447 / 6.082 (figure f7)
    (commit 3409c31)  NOTE: single seed. Over 3 seeds (T9c) the ADE6 advantage over the
    oracle-kinematic rule is not significant; the L2@3s advantage holds.

T7-A3. Oversampling turning scenes improves the VLA by 0.19 m at 3 s over A1, mostly on
turns, but it is still 0.26 m behind the small MLP.

    all 5,119; A3 = A1 + turn-weighted sampling (turning 17% -> 40% of draws)
                        L2 TemAvg 1/2/3 s    L2 NoAvg 3 s   Col% TemAvg 3 s   ADE@6s
    A1                  0.274 0.555 0.967    2.080          0.57              3.082
    A3                  0.254 0.504 0.877    1.894          0.64              2.896
    A3 - A1, L2@3s: -0.186 [-0.230,-0.143];  A3 - KIN: -0.284 [-0.353,-0.215]
    A3 - Ego-MLP + cmd, L2@3s: +0.261 [+0.213,+0.309]
    turning subset (n=638), L2@3s: KIN 3.412, A1 3.086, A3 2.580, Ego-MLP 1.868
    (commit 2137420)

T7-A0. Removing the 1,536 zeroed visual tokens instead of feeding zeros makes training
11x faster and leaves the blend with CV unchanged, but it worsens the standalone mean
(the median improves).

    custom-split test, causal subset n=3,358, ADE@6s; causal ego-only, old recipe
                         standalone mean (median)   blend with CV   blend - CV
    zeroed visual tokens 4.170 (3.469)              3.045           -0.014 [-0.025,-0.003]
    removed (A0)         4.319 (3.195)              3.039           -0.020 [-0.026,-0.014]
    removed - zeroed: standalone +0.149 [+0.020,+0.282]; blend -0.006 [-0.016,+0.004]
    (commit 31faa75)
    second seed (G9, removed, seed 123): standalone 3.868 (3.136), blend 3.040;
    removed s123 - zeroed: standalone -0.303 [-0.406,-0.197], blend -0.006 [-0.016,+0.005];
    removed s123 - removed s42: standalone -0.452 [-0.614,-0.298] -> the +0.149 was seed
    noise (commit 525a91d)

## T8. Mini ladder: predicted vs oracle intent (small Ego-MLP, a demonstration; commit b1917e1)

T8a. A classifier that predicts the longitudinal intent from ego status + command gets 68%
of val samples right (the majority class alone gets 39%).

    3 seeds, official val all 5,119: accuracy 0.677 / 0.676 / 0.672
    recall: stop 0.900, accelerate 0.548, decelerate 0.451, maintain 0.766
    (decelerate is mostly mistaken for maintain; accelerate for maintain or stop)

T8b. Feeding the predicted intent into the oracle-trained model is no better than not
having intent at all, while the true (oracle) intent is worth about 0.7 m.

    seed-mean ADE@6s (L2 TemAvg 3 s), all 5,119         gap closed (L2 = 0, L5 = 1)
    L2 ego + cmd [P]                      2.439 (0.782)   0.000
    L5 ego + ORACLE intent [P]            1.750 (0.637)   1.000
    L5 ego + PREDICTED intent             2.498 (0.796)  -0.087
    L5-noise + PREDICTED intent           2.534 (0.798)  -0.139
    (excl. first frames: 2.027 / 1.537 / 2.089 / 2.120)

T8c. Realistic (confusion-shaped) intent errors hurt a little less than random flips at
the same rate, and a 68%-accurate classifier sits past the point where intent stops
helping (figure f7).

    L5-noise, ADE@6s at 10 / 20 / 40% error    confusion-shaped 2.005 / 2.210 / 2.553
                                               uniform flips    2.033 / 2.246 / 2.693
    classifier-predicted intent (about 32% error)               2.534

## T9. Seeds: three seeds per VLA arm (commits d7e6afc, G5 54383f1)

T9a. Training 3x longer does not help: a 30-epoch schedule again peaks at epoch 6 and
ties the 10-epoch A3 on official val.

    all 5,119; A3 recipe, seed 42      L2 NoAvg 3 s   L2 TemAvg 3 s   ADE@6s mean/med/p95
    A3, 10 epochs                       1.894          0.877           2.896 / 2.003 / 7.853
    G5, 30 epochs (stopped at 11)       1.906          0.880           2.890 / 1.893 / 7.855
    G5 - A3, L2@3s: +0.012 [-0.022,+0.045]; ADE6 -0.006 [-0.063,+0.050]
    pre-registered rule (>= 3% better on holdout): -0.54% -> keep 10 epochs

T9b. Over three seeds the VLA with ego + command is 0.06 m worse than the single
seed reported in T7 and stays 0.32 m behind the small MLP at 3 s.

    all 5,119; 3-seed mean +- sd     L2 NoAvg 3 s    L2 TemAvg 3 s   ADE@6s          ADE@6s median
    (i) VLA ego + cmd [P] (A3 rec.)  1.954 +- 0.044  0.909 +- 0.022  2.930 +- 0.029  2.007 +- 0.055
    Ego-MLP + cmd [P], 3 seeds       1.637           0.779           2.437
    kinematic rule                   2.178           0.998           3.161
    per seed L2@3s: 1.894 (A3) / 2.001 / 1.967
    (i) - KIN, L2@3s: -0.224 [-0.294,-0.154];  (i) - Ego-MLP: +0.317 [+0.269,+0.368]
    (bootstrap on per-sample errors averaged over the 3 seeds)

T9c. With the oracle meta-action the VLA is stable across seeds. It still beats the
oracle-kinematic rule at 3 s, but its 6 s advantage is no longer significant (this
corrects the single-seed A2 claim in T7-A2).

    all 5,119; (ii) = A2 design + turn weighting, 3 seeds
                                       L2 NoAvg 3 s    L2 TemAvg 3 s   ADE@6s
    (ii) VLA ego + oracle meta [P]     1.690 +- 0.005  0.828 +- 0.007  2.544 +- 0.030
    oracle-kinematic rule [P]          1.955           1.022           2.671
    (ii) - oracle-kinematic: L2@3s -0.266 [-0.422,-0.137]; ADE6 -0.127 [-0.385,+0.080]
    (ii) - (i):              L2@3s -0.264 [-0.312,-0.218]; ADE6 -0.386 [-0.457,-0.319]
    (ii) - Ego-MLP + cmd:    L2@3s +0.053 [-0.018,+0.123]; ADE6 +0.107 [+0.010,+0.204]
    test flips 0/10/20/40%, ADE6: (ii) 2.544 / 2.702 / 2.842 / 3.171
                                  oracle-kinematic 2.671 / 3.626 / 4.447 / 6.082 (figure f7)
    excl. first frames: (i) 1.618 +- 0.047, (ii) 1.414 +- 0.029 L2@3s
    all 5 new runs pass the input-shuffle test (ADE x2.10 to x2.48)

## T10. Decision bottleneck: VLA predicts the meta-action, a fixed small head drives

T10a. A VLA that predicts the longitudinal meta-action does no better than a small MLP
without cameras, worse with cameras, and no predicted meta-action helps the fixed head.

    official val, all 5,119; lon meta-action (4 classes); head = C1 L5 architecture,
    3 head seeds, h1 hard (pre-registered primary)
                            acc     macro-F1   downstream ADE@6s   downstream L2@3s
    none (ego + cmd) [P]    --      --         2.439               1.637
    MLP classifier (3 s.)   0.677   0.669      2.463               1.654
    VLA, no cameras (V8a)   0.671   0.670      2.458               1.649
    VLA + 6 cameras (V8b)   0.614   0.610      2.501               1.694
    oracle, 40% flips [P]   --      --         2.468               1.677
    oracle [P]              --      --         1.750               1.193
    V8b - V8a: macro-F1 -0.060 [-0.097,-0.024]; L2@3s +0.044 [-0.016,+0.113];
               ADE6 +0.043 [-0.057,+0.159]  -> pre-registered: cameras add no decision
               information (excl. first frames V8b is worse: L2@3s +0.093 [+0.037,+0.158])
    V8a - MLP: macro-F1 +0.001 [-0.007,+0.009]; L2@3s -0.004 [-0.015,+0.006]
    holdout macro-F1 (selection): MLP 0.672, V8a 0.672, V8b 0.675 (figure f8)
    (commit c06c06f)

T10b. Near-threshold labels and per-class behaviour (amendment C8).

    val, all 5,119; near-threshold = within 0.3 m/s of a lon boundary (20.7% of val,
    20.2% of train: accel 10.0%, decel 8.1%, stop 3.0% of val)
                  acc / macro-F1 all     acc / macro-F1 excl. near (n=4,057)
    MLP (3 s.)    0.677 / 0.669          0.718 / 0.711
    V8a           0.671 / 0.670          0.714 / 0.713
    V8b           0.614 / 0.610          0.652 / 0.648
  Removing near-threshold samples raises every predictor by about 0.04 in both metrics.

                  recall V8a   recall V8b   V8b - V8a [95% CI]
    stop  (853)   0.925        0.580        -0.345 [-0.487,-0.208]
    accel (1370)  0.612        0.666        +0.054 [+0.020,+0.089]
    decel (891)   0.554        0.593        +0.038 [+0.003,+0.074]
    maint (2005)  0.654        0.601        -0.053 [-0.088,-0.018]
  V8b beats V8a on accelerate and decelerate recall and loses on stop and maintain.

    confusion (rows true; cols stop accel decel maint)   V8a            | V8b
    stop                                                 789  37  17  10 | 495 300  19  39
    accel                                                220 838  37 275 |  89 912  62 307
    decel                                                 41  30 494 326 |  44  43 528 276
    maint                                                 89 318 286 1312|  62 303 434 1206
  V8b predicts "accelerate" for 300 of 853 true stops.

    agreement V8a/V8b 0.736; on the 1,351 disagreements: V8a correct 0.557, V8b
    correct 0.341, neither 0.101
  When the two disagree, V8a is right more often than V8b.

## T11. Vision end to end: cameras make the trajectory VLA slightly worse (commit d75359f)

T11a. Adding the six cameras to the ego + command VLA raises L2 at 3 s by 0.15 m against
its no-camera twin (0.09 m against the 3-seed no-camera mean).

    official val, all 5,119; A3 recipe, 10 epochs, seed 42
                                   L2 NoAvg 3 s   L2 TemAvg 3 s   Col% TemAvg 3 s   ADE@6s
    Ego-MLP + cmd [P]              1.633          0.777           0.87              2.434
    VLA ego + cmd [P], no cameras  1.894          0.877           0.64              2.896
      (3-seed mean, T9b)           1.954          0.909           --                2.930
    VLA + 6 cameras (G7) [P]       2.043          0.960           0.61              3.001
    VLA + cameras, shuffled        2.115          1.009           0.75              3.047
    G7 - no-camera twin: L2@3s +0.149 [+0.074,+0.231]; ADE6 +0.105 [+0.004,+0.212]
    G7 - 3-seed no-camera mean: L2@3s +0.089 [+0.016,+0.170]; ADE6 +0.071 [-0.026,+0.176]

T11b. Shuffling the camera images between samples barely changes the vision VLA (ADE x1.015),
except in stationary scenes.

    shuffled - G7, L2@3s: straight -0.027 [-0.087,+0.027]; turning +0.015 [-0.120,+0.153];
    stationary +0.454 [+0.072,+0.811]

T11c. End to end vs decision bottleneck (ADE@6s, all 5,119).

    VLA end to end, ego + cmd (3 seeds)              2.930
    VLA end to end, + cameras (G7)                   3.001
    bottleneck: fixed head + V8a meta-action         2.458
    bottleneck: fixed head + V8b meta-action         2.501
    fixed head, no meta-action (ego + cmd)           2.439
    fixed head + oracle meta-action [P]              1.750
  The fixed-head routes beat both end-to-end VLAs by about 0.5 m, and none uses the cameras
  to its benefit.
