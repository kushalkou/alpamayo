# FROZEN RESULTS — decode rule, shrinkage, and the vision question

Alpamayo VLA / nuScenes. Results frozen 2026-09-07. Nothing below may be changed by
further analysis; anything new goes in a separate document.

Scope: a Cosmos-Reason1-7B model fine-tuned to emit 6-second driving futures as 24
discrete tokens (12 acceleration + 12 curvature), decoded through a unicycle model.
Scene-based split, 16,763 train / 3,572 val / 3,614 test. All numbers are autoregressive
(no teacher forcing). Metric is ADE@6s in metres unless stated: mean L2 distance between
predicted and true positions over the 12 future steps.

Baseline throughout: **CV**, a constant-velocity rollout from the true current speed and
heading. Full test set: **3.062 mean / 2.409 median**.

---

## HEADLINE

**An ego-only model blended with constant velocity reaches 2.894 m vs CV's 3.062 m
on the 3,614-sample test set: -5.5%, p < 1e-4 (paired bootstrap, 95% CI
[-0.194, -0.142]).**

The blend weight alpha was fit on the validation split and applied unchanged to test.
It is seed-robust: +0.154 +/- 0.025 m across three training seeds (alpha* = 0.25/0.25/0.30).
The full-vision model gives 2.910 m (+0.152, CI [-0.181,-0.123], seeds +0.144 +/- 0.009).

**This is a model+CV ENSEMBLE, not the model beating CV.** The formula is
`trajectory = alpha * model + (1 - alpha) * CV` applied per timestep in position space,
with alpha ~= 0.25-0.30. No trained model beats CV on its own; the best standalone learned
model scores 3.266 m and has no inputs at all (see Correction 4).

---

## THE FOUR FINDINGS

### 1. Decode rule: take the distribution's mean, not its mode

Replacing argmax decoding with the probability-weighted mean over the 64 bins of each
token's distribution improves ADE@6s by **0.333 m** (CI [-0.371, -0.295], p < 1e-4,
n = 3,614) on the full-vision model; 0.346 m ego-only; 0.382 m zero-input.

The same fix applied to a model **trained with both vision and ego inputs zeroed** —
a model that cannot see anything — captures **86%** of the gain (+0.252 vs +0.294).
It is therefore an **estimator fix, not a perception improvement**, and should generalise
to any VLA emitting discretised control tokens.

Two implementation details mattered. The STOP token must be kept in the distribution's
support with its tokenizer value of 0.0; dropping it renormalises over a residual noise
floor and drives stationary vehicles away (ADE@6s 3.589 vs 1.253 on a stationary-heavy
subset). And a hybrid rule -- argmax when p(STOP) > 0.5, mean otherwise -- makes the fix
strictly dominant over argmax in every stratum (3.570 vs 3.921 overall; on stationary
0.767 vs 0.786, with the median back to 0.001).

### 2. Shrinkage toward the baseline

alpha* = 0.30 (full-vision, 95% CI [0.25, 0.35]) and 0.25 (ego-only, CI [0.25, 0.30]);
every CI excludes 0. Three matched null predictors -- Gaussian noise scaled to the model's
deviation, the model's deviations permuted across samples, and its constant mean deviation
-- all select alpha* = 0 and gain exactly **0.0000**. Forcing the real alpha* = 0.30 onto a
signal-free predictor of the same magnitude **costs 0.32-0.36 m**.

Findings 1 and 2 are **89% the same mechanism**: the decode gain is 0.333 m alone but
0.036 m once both are blended. Both are variance reduction against an over-committing
model. Present them as one finding with two implementations.

### 3. Correction versus replacement -- why every earlier result was negative

The model is a worse standalone predictor than CV on every component: **39% higher MSE**
(38.783 vs 27.845) and **double the bias** (1.038 vs 0.507 m). Yet its *disagreement* with
CV correlates with the correction CV actually needs at **r = +0.34** (permutation test,
0 of 2000 permutations reach it, p = 0.0005).

Those facts are compatible because ADE ranks predictors, while that correlation asks
whether the model's deviation is informative. This reconciles the whole project: **every
head-to-head test said the models were useless, and they were -- as replacements. As
correction terms they are not.**

### 4. Vision: nothing overall, a real effect on turning

Adding one model at a time to a fixed baseline span, repeated across all three choices of
base seed (marginal test ADE gain, mean by kind):

    base ego     ALL: full   ALL: ego   diff  | TURN: full  TURN: ego   diff
    ego_s42       +0.0612    +0.0651  -0.0039 |  +0.1689    +0.0272   +0.1416
    ego_s123      +0.1155    +0.1311  -0.0156 |  +0.2997    +0.1675   +0.1323
    ego_s2024     +0.0520    +0.0594  -0.0074 |  +0.2116    +0.1183   +0.0934

**Overall: no vision effect.** Adding an ego-only checkpoint helps as much as adding a
full-vision one (difference negative in 3/3 configurations). This is ensemble diversity.

**On turning (n = 638, max|GT curvature| > 0.05 rad/m): +0.122 m, sd 0.026, positive in
3/3 configurations**, with full-vision and ego-only ranges separating completely within
each configuration. Vision helps **only as a correction term, never as a replacement**,
and only where the road bends.

---

## CORRECTIONS TO PRIOR CLAIMS

All six are self-corrections found by this work.

**C1. "Turn-weighted training makes vision net-positive overall" (Y1) does not replicate.**
Claimed: full-vision 3.924 beats ego-only 3.978 overall. True: with the corrected decode on
the full test set the gap is **-0.045 m, CI [-0.137, +0.048], p = 0.35** — not significant.
Why it was wrong: the original claim rested on a 0.054 m difference with no CI and no seed
replication; Z1 had already flagged it as within seed noise.

**C2. The published 3.924 used a legacy STOP mapping.** Argmax decoded the STOP token to
bin 32 -- which is *not* zero (-0.336 m/s2, -0.064 rad/m) -- rather than the tokenizer's own
value of (0.0, 0.0). Corrected, the same checkpoint scores **3.921**. Worth +0.0071 m
(CI [-0.0002, +0.0154], p = 0.057): real but marginal. Why it was wrong: a comment said
"clamp to 0..63" and the equality branch was never checked against the tokenizer.

**C3. The 0.885 m "immovable floor" is a 100-sample figure.** It is the mean over the first
100 trajectories of the whole dataset. On the full test set the hard-token floor is
**1.348 m mean / 0.889 m median**; the oracle sub-bin floor is 0.266 m, so interpolation
headroom is 1.082 m. The medians agree; the means differ because of a test-set tail the
100-sample check never saw. Why it was wrong: a development-time sanity check was quoted as
a dataset constant.

**C4. The best standalone learned model has no inputs.** A checkpoint trained with vision
AND ego zeroed scores **3.266 m**, beating full-vision (3.588) and ego-only (3.633),
p < 1e-4 against both. It wins by being closest to CV: across all 3,614 test samples it
emits a single constant control sequence (max|predicted curvature| = 0.00638 and
max|predicted acceleration| = 0.3694 for *every* sample). It does not rarely turn; it
ignores its input entirely.

**C5. Gate 2.3(c)'s oracle-switch headroom was inflated.** Claimed: per-sample
min(model, CV) reaches 2.350 vs CV 3.062, read as "they fail on different samples, so a
selector is worth building". True: the min() operator manufactures most of that. The
zero-input model correlates with CV at **r = 0.955** with a mean difference of 0.072 m, yet
min(zeroboth, CV) still "gains" **0.424 m**. A model emitting one constant plan cannot hold
complementary information. Why it was wrong: no null model was run alongside.

**C6. "Vision contributes as a correction term" (Gate 4.5) is WITHDRAWN for the overall
number.** Claimed: in a nested blend, the term (full - ego) took coefficient +0.2546 with a
CI excluding 0, marginal gain +0.068 m. True: that gain is ensemble diversity —
adding a second *ego-only* checkpoint buys as much (+0.065). Why it was wrong: the two
checkpoints differ by training run as well as by modality, and the contrast design was
additionally confounded by basis-span structure (see M3). The turning-only effect in
Finding 4 survives this correction; the overall claim does not.

---

## METHODOLOGICAL NOTES (carry these beyond this project)

**M1. Teacher-forced validation loss is invalid for token trajectory models.** A model
given zero inputs matches it. Selection and early stopping must use leak-free
autoregressive rollout error.

**M2. Oracle-switch headroom is uninterpretable without a null model.** `min(A, B)` sits
below `mean(B)` whenever A and B are imperfectly correlated, with no complementary
information required. Always report a null predictor's oracle gain beside the real one.
Here a near-copy of the baseline (r = 0.955) showed 0.424 m of illusory headroom.

**M3. Basis-span confounding in nested-blend decomposition.** When decomposing a blend into
contrast terms `(P - Q)`, a contrast whose `Q` already lies in the fitted span adds exactly
one new direction `P`, while a contrast with neither endpoint in the span adds only the
difference direction. The first kind gains and the second does not, *regardless of what the
contrast is labelled*. Vary one model at a time against a fixed span instead, and repeat
across base choices.

**M4. A null test pinned at a parameter boundary is one-sided.** If alpha = 0 recovers the
baseline exactly, a signal-free predictor can never show a positive gain, so "the null
gained nothing" is necessary rather than informative. Add a forced-parameter test (apply the
real alpha to the null and measure the cost), a permutation test on the underlying
correlation, and a curve-shape check for an interior versus boundary minimum.

**M5. Report medians and tail quantiles, not only means.** The mean alone hid the actual
mechanism here for the whole project. The blend's straight-road win rate is 0.480 -- below
half -- while its mean improves: it removes bad outcomes rather than improving typical ones.

---

## LIMITS

- **One dataset** (nuScenes), one architecture, one tokenizer. No claim of generality
  beyond "should apply to VLAs emitting discretised control", which is untested.
- **The turning subset is 638 samples** and is the same subset used throughout, so the
  turning results are not independent of each other.
- **The modality contrast is necessarily cross-run**: no two checkpoints differ in vision
  alone, so "vision" is always confounded with training run to some degree. Varying the
  base seed mitigates this but does not eliminate it.
- **The headline is an ensemble**, not a model. Deployed, it requires running both the
  network and a constant-velocity predictor and blending them.
- **No closed-loop evaluation.** Every number is open-loop displacement error against a
  logged future. No collision, off-road, or comfort metric.
- **Turn detection is not deployable**: both inference-time detectors tested fail, so the
  turning-specific gains cannot currently be routed to at runtime.
