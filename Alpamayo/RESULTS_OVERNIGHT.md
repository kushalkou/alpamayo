# Alpamayo VLA — Overnight Results (2026-07-12 → 07-13)

> Append-only log. Each queue item (P1→P4) is written as it lands and committed/pushed
> immediately, so a partial file is still readable if a later item fails.

---

## EXECUTIVE SUMMARY (updated as items land)

**As of P1 (leak-free AR eval of all 5 checkpoints, full test set, n=3614):**

1. **Teacher-forced val loss is confirmed dead as a metric.** The zero-input null
   ("zero-both") does not merely *tie* the input-fed models on the only valid metric —
   it **beats every one of them** on autoregressive ADE at every horizon.
2. **Does vision help? No — it hurts.** Full (vision+ego) AR ADE@6s = **6.978m** vs
   ego-only = **4.919m**: adding vision makes the model **2.06m worse** @6s (mean;
   −0.70m median). Vision-only (5.268m) is also worse than ego-only.
3. **Does anything beat zero-both? No.** The true null (no vision, no ego) is the best
   model at every horizon: ADE@6s **4.545m mean / 3.642m median**. It beats ego-only by
   0.37m, vision-only by 0.72m, and full by 2.43m @6s (mean).
4. **Interpretation:** every model is essentially regressing to the trajectory prior;
   the input-fed models overfit input-correlations that do not generalize, so inputs act
   as *net noise*. Perception is not being used. This is the disease P2/P3 target.
5. All models sit above the (recomputed) roundtrip discretization floor of 2.504m @6s,
   so no result is "too good"/buggy. The old baseline is far worse (12.5m @6s) — a
   genuinely weaker pre-migration checkpoint.

_Recommendation forming: fix the metric (P2), then attack the trivially-solvable slots
that starve perception gradients (P3 position-weighted loss, P4 scheduled sampling)._

_(P2/P3/P4 summaries appended below as they complete.)_

---

## P1 — Full-test autoregressive eval, all 5 checkpoints

- **Metric:** autoregressive (KV-cache greedy) decode → unicycle rollout → ADE/FDE vs GT
  in the global-axes ego-origin frame. **No GT-token leak** (unlike teacher-forced val).
- **Set:** test split, 128 scenes / **3614 samples**, augment OFF, seed 42.
- **Code:** `code/inference.py` (KV cache verified token-identical to full-recompute:
  0/15 mismatches, 6.7× faster). Per-checkpoint zeroing flags verified honored:
  ego-only→`--zero_vision`, vision-only→`--zero_ego`, zero-both→both.

### Table — ADE / FDE (meters), autoregressive, test n=3614

| Checkpoint | sel. (TF val) | ADE@1s | ADE@2s | ADE@3s | ADE@6s | FDE@6s | tok-acc | seq-acc |
|---|---|---|---|---|---|---|---|---|
| (a) old baseline        | 2.852* | 1.202 | 2.714 | 4.701 | 12.467 | 28.143 | 25.99% | 6.45% |
| (b) full live-vision    | 2.0806 | 1.051 | 1.923 | 2.976 | 6.978 | 15.129 | 43.80% | 10.24% |
| (c) ego-only (zero_vis) | 2.0035 | 0.517 | 1.064 | 1.789 | 4.919 | 11.665 | 49.93% | 11.37% |
| (d) vision-only (zero_ego)| 2.0942 | 0.557 | 1.163 | 1.957 | 5.268 | 12.258 | 37.49% | 0.00% |
| (e) **ZERO-BOTH (null)**| 2.0392 | **0.503** | **1.029** | **1.710** | **4.545** | **10.544** | 40.57% | 0.00% |
| — roundtrip floor —     |   —    | 0.466 | 0.814 | 1.193 | 2.504 | 5.108 | (100%) | (100%) |

_All ADE/FDE are **means**. Medians below. *(a)'s stored TF val_loss is 2.852; the
dir name "val1.9098" reflects an older, differently-computed val metric._

### Table — ADE / FDE medians (meters)

| Checkpoint | ADE@1s | ADE@2s | ADE@3s | ADE@6s | FDE@6s |
|---|---|---|---|---|---|
| (a) old baseline        | 0.685 | 1.796 | 3.355 | 10.095 | 24.391 |
| (b) full live-vision    | 0.405 | 0.891 | 1.550 | 4.504 | 10.538 |
| (c) ego-only            | 0.329 | 0.755 | 1.308 | 3.807 | 9.361 |
| (d) vision-only         | 0.323 | 0.755 | 1.363 | 4.205 | 10.373 |
| (e) **ZERO-BOTH**       | **0.319** | **0.731** | **1.275** | **3.642** | **8.768** |
| — roundtrip floor —     | 0.268 | 0.509 | 0.776 | 1.730 | — |

### The scientific question — answered in prose

**Does vision improve AR ADE over ego-only?**
**No — it makes it worse.** Ego-only (vision zeroed) gets ADE@6s **4.919m** mean /
3.807m median. The full model, which additionally receives the 1536 vision tokens, gets
**6.978m** mean / 4.504m median — i.e. **+2.059m worse @6s mean** (+0.697m median).
The degradation is monotone across horizons (Δmean = +0.53m@1s, +0.86m@2s, +1.19m@3s,
+2.06m@6s). Vision-only (ego zeroed, 5.268m @6s) also trails ego-only. On this dataset
and recipe, **vision tokens carry no usable trajectory signal and act as a distractor.**

**Does anything beat zero-both?**
**No.** The true null — no vision *and* no ego — is the single best checkpoint at every
horizon on both mean and median ADE/FDE. ADE@6s **4.545m mean / 3.642m median**.
Margins @6s mean: beats **ego-only by 0.374m**, **vision-only by 0.723m**, **full by
2.433m**, and the old baseline by 7.922m. The implication is stark: **feeding the model
any input, under the current recipe, is worse than feeding it nothing.** Every model is
effectively fitting the marginal trajectory prior; the input-conditioned models learn
train-set input correlations that fail to generalize, so inputs reduce to net noise.

**Caveat on token metrics.** Token-acc does *not* track ADE: ego-only has the highest
tok-acc (49.93%) yet zero-both wins ADE; vision-only/zero-both have 0% seq-acc yet best
ADE. ADE is the decision metric — tok/seq-acc are recorded for the log only.

**Not a bug.** Every model is above the recomputed roundtrip floor (2.504m @6s mean),
so nothing is suspiciously good; "worse than null" is a legitimate, if damning, finding.

**Artifacts:** `results/res_{a..e}.json`, logs `Alpamayo/.../scratchpad/eval_*.log`.
Preserved zero-both ckpt: `models/checkpoints/_zeroboth_run_jul12/alpamayo_best_e1_val2.0392.pt`.

---

## P2 — AR val-ADE model selection (infra foundation)

**What changed.** Selection + early-stopping now run on autoregressive median ADE@6s
over a seed-fixed 400-sample val subset (KV-cache decode, sharded across DDP ranks),
computed each epoch. Teacher-forced val loss is still logged but **cannot** drive
selection. New: `code/ar_eval.py` (device-parametrized decode identical to inference.py
+ `compute_val_ade` + `fixed_val_indices`); `finetune.py` epoch-end rewritten;
checkpoints now store both `val_loss` (record) and `val_ade6` (selection).

**Sanity — full-live checkpoint (val set):**

| set | ADE@1s | ADE@2s | ADE@3s | ADE@6s (mean/med) |
|---|---|---|---|---|
| 400-subset (seed 1234) | 1.035 | 1.867 | 2.901 | 6.946 / **4.525** |
| full val (n=3572)      | 1.052 | 1.911 | 2.947 | 6.961 / **4.403** |

**Δ median ADE@6s = 0.122m (2.8%)** — the 400-subset is a faithful proxy. It also
matches the P1 *test* number for full-live (median 4.504m), confirming val/test/code
consistency. Selection infra validated. Committed before P3.
