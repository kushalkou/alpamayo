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
