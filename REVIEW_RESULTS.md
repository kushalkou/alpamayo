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

## T7. Stage A (VLA, official split) -- pending

Filled in as runs finish: A1 (ego + cmd [P]), A2 (+ oracle meta-action [P]), A3 (A1 +
turn-weighted sampling), A0 (visual tokens removed vs zeroed, custom split).
