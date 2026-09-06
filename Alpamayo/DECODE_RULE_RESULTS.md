# Decode-rule fix — results log

Token-based trajectory VLA. 24 discrete slots, flattened `[a0..a11, k0..k11]`.
Slots 0-11 = acceleration, 12-23 = curvature. Inference is autoregressive,
KV-cached, then unicycle rollout (dt=0.5s, 12 steps), seeded with the true
current speed.

**Token id layout (non-obvious).** Vocab is 129 wide but only 65 ids are used.
Accel AND curvature tokens both occupy ids **0-63**; the SLOT POSITION
disambiguates them, not the id. Ids **64-127 are dead classes**, never a target.
STOP = 128.

---

## GATE 0 — reproduce and reconcile

- **Tokenizer roundtrip: 0.885m** mean / 0.729 median / 2.748 max (100 traj).
  Matches the 0.8845m reference. No tokenizer bug.
- **Argmax AR, full 3,614-sample test set**, `_y1_full_turnw/alpamayo_best.pt`:

| subset | model | ADE@1s | ADE@2s | ADE@3s | ADE@6s | med@6s | FDE@6s |
|---|---|---|---|---|---|---|---|
| ALL (3614) | CV | 0.150 | 0.476 | 0.943 | **3.062** | 2.409 | 7.666 |
| | argmax | 0.219 | 0.601 | 1.172 | 3.924 | 3.247 | 10.044 |
| STRAIGHT (2976) | CV | 0.134 | 0.418 | 0.816 | **2.600** | 2.006 | 6.463 |
| | argmax | 0.211 | 0.566 | 1.086 | 3.566 | 3.030 | 9.082 |
| TURNING (638) | CV | 0.229 | 0.751 | 1.538 | **5.218** | 4.953 | 13.274 |
| | argmax | 0.258 | 0.766 | 1.578 | 5.594 | 4.514 | 14.533 |

Reproduces the published Y1 table to three decimals.

- **4.236 vs 3.924 reconciled.** Different checkpoints, only one turn-weighted:
  4.236 = `_w2_full_fixed` (W2, fixed-ego, NOT turn-weighted);
  **3.924 = `_y1_full_turnw`** (Y1, fixed-ego + turn-weighted). The handoff doc
  mislabelled 4.236 as turn-weighted.

---

## GATE 1 — expectation decoding (400-sample fixed val subset) — PASS

**Bug found in the original decode spec.** "Drop id 128, softmax over the
remaining 64" discards the STOP mass, which is exactly where a stationary
vehicle's probability sits; the residual over 0-63 is noise and the decoded mean
drives a parked car away. Fix: renormalise over **{0..63} U {128}** with STOP
contributing the tokenizer's own `detokenize_step(STOP) = 0.0`.

On the 68 stationary samples (|v0|<0.5 m/s) the literal family scores ~3.3m vs
~1.0m for the STOP-aware family — the bug in isolation. On the 332
non-stationary samples the two families are identical (4.101 vs 4.098), so the
fix is exactly a stationary-case correction and not a confound.

**V1s (pure STOP-aware expectation) beats argmax**, paired bootstrap 20k:

| stratum | V1s - argmax | 95% CI | p |
|---|---|---|---|
| ALL (400) | **-0.301** | [-0.402, -0.202] | <1e-4 |
| STRAIGHT (325) | -0.286 | [-0.407, -0.169] | <1e-4 |
| TURNING (75) | -0.369 | [-0.539, -0.210] | <1e-4 |

Sharpening monotonically HURTS: V1s 3.591 < V2s(top-k5) 3.639 < V3s(T=0.5)
3.753 < argmax 3.893. Mixed variants both lose to expectation-on-both-halves
(V4 3.767, V5 3.720).

**No configuration beats CV.** V1s - CV: ALL +0.509 [+0.242,+0.788] p<1e-4.
V1s appears to beat CV on turning (4.769 vs 4.860) but this is NOT significant:
-0.091 [-0.778,+0.581] p=0.78 at n=75.

**Diagnostics (n=400).** Dead-class mass ids 64-127: max 1.1e-5 — no leak.
Curvature bimodality (top-2 bins non-adjacent): 9.2% mean. Argmax selects STOP
on 17.4% of slots, matching the 17% stationary rate. Entropy / top-1 mass:
accel slots 1-11 **1.355 nats / 0.536**, curvature slots 13-23 **0.520 / 0.824**.

---

## GATE 2.0 — STOP/bin-32 fairness fix

Argmax mapped STOP -> bin 32, which is NOT zero (accel -0.336 m/s2, curv
-0.064 rad/m), while the tokenizer's `detokenize_step(STOP)` is (0.0, 0.0).
Corrected in `decode_trajectory_ex` and in the legacy
`inference.decode_trajectory` / `ar_eval.ar_decode` paths.

Within-run delta: legacy 3.893 -> corrected **3.886**, i.e. **+0.0071m**
[-0.0002,+0.0154] p=0.057. About 2% of the gap; V1s vs CORRECTED argmax is
**-0.294m** [+0.195,+0.395] p<1e-4. The Gate 1 result is not inflated.

Results predating this fix (incl. Y1's 3.924 and the Gate 0 repro) used bin 32.
`ar_eval` is used by `finetune.py` for AR-val-ADE selection, so future training
runs select under the corrected arithmetic.

---

## GATE 2.1 — mechanism tests

### (a) Residual |E[value] - argmax_center|, in bin widths (n=4,800/half)

| half | mean | median | p90 | p99 | frac>0.5 | frac>1.0 |
|---|---|---|---|---|---|---|
| accel | 0.849 | 0.400 | 2.167 | 5.925 | **0.440** | **0.269** |
| curv | 0.106 | 0.010 | 0.309 | 1.339 | 0.044 | 0.017 |

**The hypothesis SPLITS by half.** Curvature: 95.6% inside half a bin, median
0.010 — genuine sub-bin **interpolation**. Acceleration: 26.9% move more than a
FULL bin, p99 at 5.9 bin widths — **distributional mass shift**, not
interpolation. Consistent with the entropy table (accel diffuse 1.355 nats /
top-1 0.536; curvature confident 0.520 / 0.824), and it explains why the mixed
V4/V5 variants both lose: each half contributes a DIFFERENT real gain.

### (b) Coarsening dose-response — prediction did NOT hold

| bins | argmax | expect | gap | vs 64-bin gap |
|---|---|---|---|---|
| 64 | 3.886 | 3.591 | 0.295 | 1.00x |
| 32 | 4.177 | 3.937 | 0.240 | 0.81x |
| 16 | 9.530 | 8.937 | 0.593 | 2.01x |

Predicted ~2x / ~4x. The gap shrinks at 2x width before growing at 4x. Expected
given (a): only the curvature half does quantization refinement. The 16-bin
point should be discounted anyway — absolute ADE 9.53m vs 3.89m is outside the
regime where a quantization argument applies.

### (c) Zero-input transfer — prediction HELD

V1s vs corrected argmax, ADE@6s, paired bootstrap:

| model | argmax | V1s | gain | 95% CI | p |
|---|---|---|---|---|---|
| y1_full | 3.886 | 3.591 | **+0.294** | [+0.195,+0.395] | <1e-4 |
| y1_ego | 4.035 | 3.741 | **+0.294** | [+0.189,+0.403] | <1e-4 |
| zeroboth_jul12 | 3.695 | 3.443 | **+0.252** | [+0.128,+0.375] | 1e-4 |
| y1_full_zeroboth (OOD) | 11.579 | 11.383 | +0.197 | [+0.178,+0.216] | <1e-4 |

A model trained with BOTH modalities zeroed — which knows nothing about the
scene — still gets **86%** of the full model's gain. **This is a decode /
estimator fix, NOT a perception improvement, and must be described that way.**

Incidental: `zeroboth_jul12` argmax (3.695) beats `y1_full` argmax (3.886),
consistent with the earlier finding that the zero-input model collapses to a
near-constant-velocity predictor. CV (3.062) still beats all of them.

---

## GATE 2.3 — oracle soft floor (full test set, n=3,614)

| decode | ADE@1s | ADE@2s | ADE@3s | ADE@6s | med@6s |
|---|---|---|---|---|---|
| bin centers (hard floor) | 0.138 | 0.294 | 0.496 | **1.348** | 0.889 |
| oracle sub-bin (soft floor) | 0.063 | 0.105 | 0.146 | **0.266** | 0.188 |

**Total interpolation headroom @6s = 1.082m** (1.348 -> 0.266).

Note on provenance: the familiar **0.885m** floor is the mean over the first 100
trajectories of the whole dataset; on the full TEST set the hard floor is 1.348m
mean / 0.889m median. The medians agree; the means differ because the test set
has a long tail the 100-sample check never saw. Use 1.348 -> 0.266 for a
like-for-like comparison.

The headroom is computed with GROUND-TRUTH token assignments, so it is a ceiling
the model cannot actually reach, not a target.


---

## GATE 2.2 / 2.3 / 2.4 — full test set (n=3,614)

All decodes use CORRECTED argmax (STOP -> 0.0). Y1's published 3.924 and the
Gate 0 reproduction used the legacy STOP -> bin-32 mapping (delta -0.003).

### 2.2 ADE@6s (mean), all strata

| model / decode | ALL | STRAIGHT | TURNING | STATIONARY |
|---|---|---|---|---|
| **CV baseline** | **3.062** | **2.600** | 5.218 | 0.736 |
| y1_full / argmax | 3.921 | 3.565 | 5.579 | 0.786 |
| y1_full / V1s | 3.588 | 3.245 | 5.189 | 0.868 |
| y1_ego / argmax | 3.979 | 3.612 | 5.691 | 0.712 |
| y1_ego / V1s | 3.633 | 3.299 | 5.190 | 0.765 |
| **zeroboth / argmax** | 3.648 | 3.304 | 5.252 | 0.737 |
| **zeroboth / V1s** | **3.266** | **2.864** | **5.146** | 0.751 |

n: ALL 3614, STRAIGHT 2976, TURNING 638, STATIONARY 622.

### Paired bootstrap, ADE@6s (negative = first term better)

| stratum | comparison | delta | 95% CI | p |
|---|---|---|---|---|
| ALL | y1_full: V1s - argmax | **-0.333** | [-0.371,-0.295] | <1e-4 |
| ALL | y1_ego: V1s - argmax | **-0.346** | [-0.382,-0.309] | <1e-4 |
| ALL | zeroboth: V1s - argmax | **-0.382** | [-0.422,-0.342] | <1e-4 |
| ALL | y1_full: V1s - CV | +0.526 | [+0.434,+0.617] | <1e-4 |
| ALL | **zeroboth - y1_full [V1s]** | **-0.322** | [-0.406,-0.240] | <1e-4 |
| ALL | **y1_full - y1_ego [V1s]** | -0.045 | [-0.137,+0.048] | **0.35 (n.s.)** |
| TURNING | y1_full: V1s - CV | -0.028 | [-0.269,+0.217] | 0.83 (n.s.) |
| TURNING | y1_ego: V1s - CV | -0.027 | [-0.285,+0.240] | 0.85 (n.s.) |
| TURNING | zeroboth: V1s - CV | -0.072 | [-0.146,+0.001] | 0.054 (n.s.) |
| STATIONARY | y1_full: V1s - argmax | **+0.082** | [+0.057,+0.107] | <1e-4 |

**Two headline answers.**
1. **V1s does NOT beat CV anywhere.** The closest is zeroboth on turning
   (-0.072, p=0.054) — still not significant at n=638.
2. **The decode fix is worth more than the cameras.** V1s-vs-argmax is
   -0.33 to -0.38m (p<1e-4) in every model; full-vision-vs-ego-only is
   -0.045m (p=0.35, NOT significant). y1_ego/V1s (3.633) beats
   y1_full/argmax (3.921) — a better decode on the camera-free model beats the
   camera model on the old decode.

**The zero-input model wins.** `zeroboth_jul12` beats BOTH trained models at
every stratum (ALL 3.266 vs 3.588/3.633; p<1e-4 vs y1_full). It is the closest
thing to CV among the learned models, which is why it is best.

**Y1's "vision helps overall" does not replicate.** With the corrected decode on
the full test set the full-vs-ego gap is -0.045m [-0.137,+0.048] p=0.35 under
V1s and -0.058 p=0.32 under argmax. Consistent with the Z1 seed-robustness
finding that the effect is within seed noise.

**V1s HURTS on stationary samples** (+0.082m, p<1e-4; median 0.001 -> 0.183).
Expectation smears the STOP mass that argmax commits to cleanly. A hybrid
(argmax when p(STOP) is high, expectation otherwise) is the obvious fix and is
NOT yet implemented.

### 2.3 Distribution

**(a) Win rate vs CV** — on TURNING the models win the MAJORITY of samples while
tying on the mean:

| stratum | model/decode | win rate | Wilson 95% | mean gap |
|---|---|---|---|---|
| TURNING | y1_full/V1s | **0.552** | [0.513,0.590] | -0.028 |
| TURNING | y1_ego/V1s | **0.577** | [0.538,0.615] | -0.027 |
| TURNING | y1_full/argmax | 0.459 | [0.421,0.498] | +0.362 |
| ALL | y1_full/V1s | 0.374 | [0.359,0.390] | +0.526 |
| ALL | zeroboth/V1s | 0.484 | [0.467,0.500] | +0.204 |
| STATIONARY | zeroboth/argmax | 0.724 | [0.687,0.757] | +0.001 |

On turning, V1s beats CV on 55-58% of samples (CI excludes 50%) yet only ties on
the mean. **It loses through the tail, not through uniform inferiority.**

**(b) Tail** — TURNING ADE@6s quantiles:

| series | p50 | p75 | p90 | p95 | p99 | mean |
|---|---|---|---|---|---|---|
| CV | 4.953 | 7.157 | 9.541 | 10.579 | 13.323 | 5.218 |
| y1_full/V1s | **4.349** | 7.128 | 10.390 | 12.877 | **19.374** | 5.189 |
| y1_ego/V1s | **4.070** | 6.872 | 10.358 | 14.069 | **20.030** | 5.190 |

Better at the median, far worse at p95/p99. The mean gap is entirely a tail
phenomenon: the models occasionally commit to a badly wrong turn.

**(c) Oracle switch** — ceiling on any CV/model selector:

| stratum | model | CV | model | oracle min | gain vs CV |
|---|---|---|---|---|---|
| ALL | y1_full/V1s | 3.062 | 3.588 | **2.350** | **0.712** |
| TURNING | y1_full/V1s | 5.218 | 5.189 | **4.055** | **1.162** |
| ALL | zeroboth/V1s | 3.062 | 3.266 | 2.652 | 0.410 |

A perfect selector would reach 2.350m vs CV's 3.062 (-23%), and 4.055 vs 5.218
on turning (-22%). **CV and the model fail on DIFFERENT samples**, so a switch is
worth building in principle — the problem is building the selector (see 2.4).

### 2.4 Deployable turn detection — NO

The TURNING stratum uses max|GT FUTURE curvature|, an oracle label. Two
inference-time detectors were built and evaluated.

| detector | precision | recall | F1 | n_pos | CV on subset |
|---|---|---|---|---|---|
| (i) PAST curvature >0.05 | 0.365 | 0.511 | 0.426 | 892 | 2.839 |
| (ii) PREDICTED curv >0.05 [y1_full] | 0.464 | 0.365 | 0.409 | 502 | 4.601 |
| (ii) PREDICTED curv >0.05 [y1_ego] | 0.439 | 0.403 | 0.420 | 585 | 4.593 |
| (ii) PREDICTED curv >0.05 [zeroboth] | — | — | — | **0** | — |

V1s vs CV **on the detected subset**:

| detector | model | model ADE | CV | delta | 95% CI | p | win rate |
|---|---|---|---|---|---|---|---|
| (i) past | y1_full/V1s | 3.087 | 2.839 | +0.248 | [+0.076,+0.421] | 0.005 | 0.298 |
| (ii) pred | y1_full/V1s | **6.255** | 4.601 | **+1.654** | [+1.250,+2.070] | <1e-4 | 0.377 |
| (ii) pred | y1_ego/V1s | **6.059** | 4.593 | **+1.466** | [+1.102,+1.824] | <1e-4 | 0.368 |

**NOT DEPLOYABLE. The turn advantage is an analysis artifact.**
- Detector (i) selects EASIER-than-average samples (subset CV 2.839 < overall
  3.062), and the model loses to CV on them (+0.248, p=0.005).
- Detector (ii) selects genuinely harder samples (subset CV 4.601 > 3.062), and
  the model loses CATASTROPHICALLY there (+1.654m, p<1e-4, win rate 0.377).
  **Each model is at its WORST precisely on the samples where it predicts a
  turn.** The self-detector is anti-selective.
- Cross-model is milder (y1_ego's detector on y1_full: +0.156, p=0.27 n.s.),
  confirming the effect is each model'''s own over-commitment, not scene difficulty.
- **zeroboth predicts ZERO turns** on all 3,614 samples, confirming directly
  that it has collapsed to a constant-velocity-like predictor.

So the oracle-switch headroom in 2.3(c) is real but currently unreachable: the
only turn signals available at inference time select exactly the samples where
the model should NOT be trusted.
