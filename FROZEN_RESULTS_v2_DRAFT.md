# FROZEN RESULTS v2 -- DRAFT FOR HUMAN REVIEW (not frozen)

Written 2026-09-29 by the OVERNIGHT2 run. FROZEN_RESULTS.md is untouched. This file
proposes how each frozen claim should change in light of the ego-state leak found in
OVERNIGHT2 item 1. Evidence: OVERNIGHT2_REPORT.md sections 1 and 6.

## The defect

dataset.compute_ego_state builds its current (t) row from future poses:
  - ego[t, accel]    = (future_speeds[1] - future_speeds[0]) / dt
                     == future_accelerations[1], the GT for accel slot 1
                        (max error 3.4e-7 over all 3,614 test samples)
  - ego[t, yaw_rate] = wrap(future_yaws[0] - yaw_0) / dt
                     == v0 * future_curvatures[0], the GT for curvature slot 12
                        (max error 1.5e-8)
Every trained model with ego input received two ground-truth targets as input. CV did
not. The trained models use this: slot-1 argmax accuracy is 0.774 (y1_ego) vs 0.305 at
slot 2, and it falls to 0.283 once the leak is removed.

## Evidence summary (test, causal subset n=3358 = samples with >= 2 past poses)

    predictor, blended with CV (alpha fit on val)        gain vs CV (m)     alpha*
    leaked features only, no learning                    +0.652 [.61,.69]   1.00
    causal features only, no learning                    +0.184 [.14,.23]   1.00
    y1_ego   (leaky input)   -- the frozen headline      +0.164 [.14,.19]   0.25
    causal_ego (retrained, leak-free, Y1 recipe)         +0.014 [.003,.024] 0.10
    zeroboth (no inputs)                                 +0.008 [.006,.010] 0.05

## Proposed status per claim

HEADLINE (2.894 vs 3.062, -5.5%): WITHDRAWN.
  Leak-free, the same recipe gives 3.045 vs 3.059 (-0.014 m, CI [-0.024,-0.003],
  p=0.011) on the causal subset. The leak is 91% of the headline gain. Proposed
  replacement: "A leak-free ego-only model blended with CV beats CV by 0.014 m
  (0.5%), about what a zero-input model buys. A hand-written causal rule (CV + last
  observed accel and yaw rate) beats CV by 0.18 m and beats every trained model."

FINDING 1 -- decode rule (expectation over argmax): SURVIVES.
  Leak-free: expectation 4.170 vs argmax 4.460, -0.290 m [-0.335,-0.243]. The sim
  replicates it at every G3 point (0.11-0.37 m). It is an estimator property,
  independent of the leak.

FINDING 2 -- shrinkage toward CV, nulls select alpha=0: SURVIVES IN KIND, MAGNITUDE
WITHDRAWN.
  Leak-free alpha* = 0.10 [0.05,0.10] (was 0.25-0.30). All three nulls still select
  alpha=0 and lose 0.02-0.07 m at the forced alpha. The mechanism holds, but the
  0.15-0.17 m gains and the alpha values in the frozen text were produced mostly by
  the leak. "89% the same mechanism as Finding 1" needs re-measuring on causal models.

FINDING 3 -- correction vs replacement, r=+0.34: WITHDRAWN AS STATED.
  Leak-free corr(D, G-C) = +0.16 (perm p=0.0005), down from +0.31 for y1_ego on the
  same subset. A positive correlation survives, but the frozen r=+0.34 and the MSE/bias
  numbers were measured on models that were handed a target slot.

FINDING 4 -- vision helps on turning (+0.122 m): AT RISK, PENDING.
  Every checkpoint in that analysis (full and ego, all seeds) had the leak. Leak-free
  ego-only still gains on TURNING (-0.062 m [-0.093,-0.031]) and nothing on STRAIGHT.
  The causal full-vision retrain is running (6b) and settles this.

CORRECTIONS C1-C6: unaffected by the leak.
  C1, C2, C3 and C5 concern decode, STOP and floor details. C4's zero-input model
  cannot see ego features. C6 is already a withdrawal.

METHODOLOGICAL NOTES: M1-M5 unaffected. PROPOSED M6: audit every input feature for the
pose indices it reads, and check whether any input equals a target. A per-slot
accuracy profile (a spike at one slot) is a cheap detector.

LIMITS, additions:
  - The causal retrain is a single seed (Y1 seed-to-seed sd was 0.025 m; the causal
    vs leaky gap is 0.150 m).
  - The causal subset drops the 7% of samples with < 2 past poses. The frozen
    full-set numbers are not directly comparable.
  - The decode seed v0 = |p1 - p0|/dt reads pose +1. It is shared with CV, so it is
    not a differential leak, but no deployed system can measure it (a CV with backward
    speed scores 3.538 vs 3.061).

## UPDATE 2026-09-30 -- causal full-vision retrain (6b) landed

FINDING 4 -- vision on turning: proposed WITHDRAWN. Leak-free, the full-vision blend
equals the ego-only blend (ALL -0.000 [-0.012,+0.012]; TURNING -0.014 [-0.050,+0.021],
p=0.44). The causal full-vision blend beats CV by 0.014 m [0.002,0.025], exactly like
causal ego-only. Single seed.
