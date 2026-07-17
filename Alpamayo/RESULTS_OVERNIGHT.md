# Alpamayo VLA — Overnight Results (2026-07-12 → 07-13)

> Append-only log. Each queue item (P1→P4) is written as it lands and committed/pushed
> immediately, so a partial file is still readable if a later item fails.

---

## EXECUTIVE SUMMARY — for the advisor (5-minute read)

> **LATEST (post-verification V1–V3 + fix-and-retrain W1–W4). Read this block; the P1–P4
> sections below are the earlier investigation and contain one now-fixed decode bug.**

**The two things that matter.**
1. **We found and fixed the real bug: the model was never given the car's true speed.**
   `compute_ego_state` computed speed as a backward difference over `past_poses` — zero when
   history was padded, under-estimated otherwise (W1). Fixing it (speed = `future_speeds[0]`)
   improved full-live test ADE@6s from **6.61 → 4.24 m** (−36%) and ego-only to **3.82 m**.
   Biggest lever in the whole study.
2. **Even so, vision does not help.** In the clean retrain (W2: correct ego, valid AR-val-ADE
   selection, 10 epochs), **ego-only (no camera) beats full live-vision at every horizon**
   (ADE@6s 3.82 vs 4.24 m mean; 3.05 vs 3.28 m median). The camera is net-negative for
   trajectory prediction on this dataset.

**The honest scoreboard (test ADE@6s, mean / median):** tokenizer floor 1.35 / 0.89 · **naive
constant-velocity baseline 3.06 / 2.41** · W2 ego-only **3.82 / 3.05** (best model) · W2 full
4.24 / 3.28. **No learned model beats constant velocity yet** — ego-only is closest and was
still improving at epoch 10 (val-ADE 2.90 and falling), so a longer ego-only run is the one
promising open thread.

**But it's subtler than "vision is useless" (X1–X2).** X1 (attention rollout): the trajectory
tokens **do attend heavily to vision** (~65% of attention mass) — so vision *reaches* the plan;
it isn't a routing/capacity failure. X2 (turn/straight split on the fixed-ego W2 models): **on
the turning subset (18%), full-vision beats ego-only at every horizon** (ADE@6s 5.58 vs 5.87m) —
**vision carries real, turn-specific signal once ego is correct.** It's just swamped by the 82%
straight-line majority (where the camera adds noise), so vision loses on the overall average.

**Deliverable shipped:** demo visualizer with BEV (GT/prediction/CV) + 6-camera grid, 3 animated
scenes, regenerated with the W2 model (W3, `viz/`).

**Recommend next:** (a) **run ego-only longer (15–20 ep)** — still improving at e10 (2.90 and
falling); most likely path to beat CV. (b) **Amplify the turn signal** rather than adding adapter
width (X1 shows vision is already read): balance/curvature-weight training so the 18% turning
cases aren't drowned by straights, and consider **unfreezing the vision encoder** (frozen features,
not attention access, are the likely ceiling). (c) X3 (LoRA-capacity probe) is **not indicated** —
its trigger was "vision mass ≈ 0", but it's 0.65. Selection must always use AR val-ADE.

---

### Earlier investigation (P1–P4) — records; superseded where noted

**What ran.** P1: leak-free autoregressive (AR) ADE eval of all 5 checkpoints on the full test
set. P2: replaced the (disqualified) teacher-forced val loss with AR val-ADE as the selection
metric in `finetune.py`, validated. P3: position-weighted-loss experiment. P4: scheduled-sampling.
_(Absolute ADE magnitudes in P1/P3/P4 were computed with the pre-W1 decode; see V1.)_

**Does vision help? No.** On the only trustworthy metric (AR ADE, no GT-token leak), the
full vision+ego model (6.98m @6s mean) is *worse* than ego-only (4.92m) and worse than a
model given **no inputs at all** ("zero-both" null, 4.55m mean / 3.64m median — the best of
all checkpoints). Teacher-forced val loss hid this completely: a zero-input model matches it.

**Did position-weighting change anything? Yes — it made things worse, informatively.** The
perception-dependent accel_1..11 slots did not improve (CE ~2.39, unchanged), proving they
are **information-starved, not gradient-starved**; meanwhile down-weighting curvature wrecked
geometry (ADE 12.2m). You cannot reweight your way to signal that isn't being extracted.

**Did anything help? Yes — scheduled sampling, partially.** It beat the teacher-forced full
model by 1.36m mean / 0.55m median @6s (best vision+ego model) by curing exposure bias — but
its 1-step CE is identical to full-live's, so it improved *rollout robustness*, not
perception, and still lost to the zero-input null. Adopt it (low p≈0.08) as a free win.

**Bottom line for the advisor:** across three independent training interventions, the model
never uses the camera. On 6-second nuScenes trajectories the ego's own kinematic state (and
the trajectory prior) already determine the future; the vision tokens are not reaching /
informing the trajectory tokens, so they act as net noise. **The problem is architectural
(does perception connect to the plan?), not a loss or a metric problem — and the metric is
now fixed.**

**Recommend next (in priority order):**
1. **Prove the vision→trajectory pathway is broken before training more.** Attention-rollout
   from trajectory tokens back to the 1536 vision tokens on the current model; if they attend
   ~uniformly/negligibly, the adapter/LoRA capacity or token routing is the culprit.
2. **Increase the perception pathway's capacity/priority:** LoRA on the cross-attention to
   vision (not just q/v/o), and/or a small trainable vision→LM adapter; consider unfreezing
   the last 1–2 vision blocks.
3. **Add an auxiliary loss that provably needs the camera** (e.g. predict a lead-agent's
   relative position / lane occupancy) so perception gets a gradient it *can* satisfy.
4. **Stress-test on perception-critical subsets** (intersections, turns, dense agents) — the
   flat result may partly reflect that straight-ahead nuScenes 6s futures are near-deterministic
   from ego state. If vision doesn't help even there, it's the data, not the model.
5. Keep **AR val-ADE (P2)** as the only selection metric; fold in **scheduled sampling p≈0.08**
   by default. Never rank by teacher-forced val loss again.

---
### Appendix — per-item detail below (P1 table, P2 sanity, P3, P4).

---

## P1 — Full-test autoregressive eval, all 5 checkpoints

- **Metric:** autoregressive (KV-cache greedy) decode → unicycle rollout → ADE/FDE vs GT
  in the global-axes ego-origin frame. **No GT-token leak** (unlike teacher-forced val).
- **Set:** test split, 128 scenes / **3614 samples**, augment OFF, seed 42.
- **Code:** `code/inference.py` (KV cache verified token-identical to full-recompute:
  0/15 mismatches, 6.7× faster). Per-checkpoint zeroing flags verified honored:
  ego-only→`--zero_vision`, vision-only→`--zero_ego`, zero-both→both.

> ⚠️ **SUPERSEDED — this P1 table was computed with a BUGGY rollout (V1).** The initial
> speed was seeded from `ego_state[3,0]` (a backward difference that is 0 when past_poses
> are missing), so every rollout undershot. Absolute ADE magnitudes here are inflated ~2×
> and the ranking is not trustworthy. See **V1** below for the fix and **V3** for the
> corrected table. Kept for the record only.

### Table — ADE / FDE (meters), autoregressive, test n=3614  [BUGGY — see V1/V3]

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

---

## P3 — Position-weighted loss (accel_1..11 ×1.0, curv+accel_0 ×0.2)

**Hypothesis:** the 13 trivially-solvable slots (curv-copy + accel_0 ego-persistence)
dominate the loss and starve the 11 perception-dependent accel_1..11 slots of gradient.
Reweight to concentrate gradient there → force perception use.

**Recipe:** 8-GPU, aug ON, eff batch 24, 3 epochs, lr 5e-5, selected on AR val-ADE (P2).
Pos-weights ×sqrt-inv-freq class weights (multiplied, not replaced). Best = epoch 3.

**Result — decisively negative. Position-weighting makes ADE ~2.7× worse.**

| model | ADE@6s mean | ADE@6s median | tok-acc |
|---|---|---|---|
| zero-both (null)        | 4.545 | 3.642 | 40.6% |
| full live-vision        | 6.978 | 4.504 | 43.8% |
| **P3 pos-weighted**     | **12.226** | **9.334** | 26.2% |

Val-ADE@6s median per epoch (selection metric): 8.63 → 8.52 → **8.47** (best). It never
approached the 4.4m baseline. AR ADE full test @1/2/3/6s mean = 1.50 / 3.05 / 4.99 /
12.23m; median = 0.83 / 1.86 / 3.30 / 9.33m. (All above the 2.504m floor — not a bug,
genuinely worse.)

**Per-position CE tells us WHY (val, teacher-forced):**

| slot group | P3 plain CE | full-live | zero-both |
|---|---|---|---|
| accel_0            | 0.12 | 0.11 | 0.46 |
| accel_1..11 (mean) | ~2.39 | ~2.24 | ~2.17 |
| curv_0             | **2.92** | 1.44 | 1.11 |
| curv_1..11 (mean)  | ~0.48 | ~0.41 | ~0.37 |

Two findings:
1. **The accel_1..11 slots did NOT improve** (~2.39 CE, ~31% acc — same as before, even
   slightly worse). Pouring gradient onto them changed nothing. So those slots are not
   *gradient*-starved — they are **information-starved**: the signal to predict accel_1..11
   is not being extracted from vision/ego (or is not present). This is the same wall P1
   hit, now confirmed from the loss side.
2. **Curvature prediction collapsed** (curv_0 CE 1.11→2.92). Down-weighting the curv
   slots removed the model's one reliable behavior (smooth curvature copy), and since
   curvature controls heading/geometry, ADE nearly tripled.

**Does it narrow the full-vs-zero-both gap? No — the opposite.** Perception is still not
used (accel_1..11 unchanged); ego-only would not get worse than full because the model
never started using vision. The intervention traded away the load-bearing curv-copy for
no accel gain. **Conclusion: the bottleneck is information/architecture, not loss
weighting.** Preserved: `models/checkpoints/_posweighted_run_jul12/`,
`results/res_p3_posweighted.json`.

---

## P4 — Scheduled sampling (feed own prediction w.p. p, ramped 0→0.25)

**Hypothesis:** the free curv-copy / accel-persistence shortcut survives because teacher
forcing always hands the model the true previous token. Replace it, with prob p, by the
model's own prediction → remove the crutch, force reliance on inputs. Efficient 2-pass
impl (pass 1 no-grad gets 1-step preds; pass 2 grad on mixed inputs). 3 epochs, aug ON,
eff batch 24, selected on AR val-ADE (P2). Best = **epoch 1** (p peaked ~0.083).

**Result — the first intervention that HELPS (vs teacher forcing), but does not beat the null.**

Val-ADE@6s median per epoch: **4.079** (p~0.08) → 4.309 (p~0.17) → 4.259 (p~0.25).
Gentle scheduling wins; heavier p regresses (own-error exposure destabilizes). Best 4.079
beats the teacher-forced full-live baseline on the identical 400-subset (4.525).

Full-test AR (n=3614):

| model | ADE@6s mean | ADE@6s median | tok-acc |
|---|---|---|---|
| zero-both (null)        | 4.545 | 3.642 | 40.6% |
| ego-only                | 4.919 | 3.807 | 49.9% |
| **P4 scheduled sampling** | **5.618** | **3.959** | 46.0% |
| vision-only             | 5.268 | 4.205 | 37.5% |
| full live-vision (TF)   | 6.978 | 4.504 | 43.8% |
| P3 pos-weighted         | 12.226 | 9.334 | 26.2% |

P4 vs full teacher-forced (same inputs, only training differs): **−1.360m mean / −0.545m
median @6s** — a real, clean win from cutting exposure bias, and the best-scoring of all
vision+ego models. **But P4 vs zero-both null: +1.073m mean / +0.317m median @6s worse.**
Scheduled sampling narrows the gap to the null but does not cross it.

**Per-position CE (val, TF) — why the gain is robustness, not perception:**

| slot group | P4 plain CE | full-live | P3 pos-w |
|---|---|---|---|
| accel_0            | 0.11 | 0.11 | 0.12 |
| accel_1..11 (mean) | ~2.23 | ~2.24 | ~2.39 |
| curv_0             | 1.40 | 1.44 | 2.92 |
| curv_1..11 (mean)  | ~0.41 | ~0.41 | ~0.40 |

P4's teacher-forced CE is **indistinguishable from full-live's** — accel_1..11 still stuck
at ~2.23 (~33% acc), curv preserved (unlike P3). So scheduled sampling did **not** teach
the model to extract perception; it made the *autoregressive rollout* robust to its own
mistakes (accumulated drift), which is exactly what improves AR ADE while leaving 1-step CE
flat. The accel_1..11 wall — the perception-dependent slots — is untouched by all three
interventions. Preserved: `models/checkpoints/_schedsamp_run_jul13/`,
`results/res_p4_schedsamp.json`, `results/perpos_ce_p4.log`.

---

## V1 — Roundtrip floor reconciliation (BLOCKING) — **BUG FOUND & FIXED**

The validated tokenizer floor is **ADE 0.885m / FDE 2.137m** (`test_roundtrip.py`). P1's
recomputed floor was 2.504m / 5.108m — 2.8× too high. A floor cannot change, so a decode
path was wrong. It was.

**(a) `test_roundtrip.py` as-is:** ADE mean **0.885** / median 0.729 / FDE mean **2.137**. ✓

**(b) Same 100 trajectories through inference.py's rollout path (GT tokens injected):**
ADE mean **1.807** / median 1.314 / FDE mean 3.687. ✗ — 2× worse. (Tokens identical:
dataset tokens == `tokenizer.tokenize`, verified.)

**(c) Step-by-step diff, trajectory 0 — the divergence:**

| | test_roundtrip | inference.py (buggy) |
|---|---|---|
| v0 (initial speed) | **4.1895** (`future_speeds[0]`) | **0.0000** (`ego_state[3,0]`) |
| yaw0 | 1.4361 | 1.4361 (same ✓) |
| frame | absolute global | origin + translated GT (equivalent ✓) |

The rollout started at **rest** (v0=0) and never caught up: by t=11 the buggy path is at
y=6.3m while GT is at y=22.6m → **24m error** on a trajectory the floor rolls out to 2.1m.

**Root cause.** `ego_state[3,0]` is a **backward difference** over `past_poses`
(`compute_ego_state`): it is exactly 0 when `past_poses` are missing (4/100 here) and
systematically **under-estimates** the true initial speed otherwise. `yaw0` and the frame
were already correct — the sole bug is the speed seed.

**Fix.** Seed `v0 = future_speeds[0]` (the reference `test_roundtrip.py` uses) in
`inference.py`, `ar_eval.py`, `eval_test_ar.py`. **Verification:** inference.py's own
`unicycle_rollout` on GT tokens now reproduces **ADE 0.8845m / FDE 2.1371m — MATCH.**
Gate cleared. (Commit `1d116d3`.)

**Why this matters for P1.** The bug's error scales with how far the trajectory travels
(speed undershoot), and a model predicting larger accelerations partially self-compensates
by ramping speed back up — so the bug does **not** hit all models equally. Every P1/P3/P4
**test-set ADE magnitude is invalid** and the P1 ranking must be recomputed with the fixed
decode (**V3**). The per-position CE analyses (teacher-forced, no rollout) are unaffected.

**Bonus finding (feeds V3).** Fixing v0 halved the floor even though only 4% of trajectories
had speed==0 — so `ego_state[3,0]` is wrong for *most* trajectories. **The model's own ego
speed input is corrupted**, which may itself explain why ego/vision "don't help". Worth a
follow-up: fix `compute_ego_state` to a forward/centered speed estimate and retrain.

---

## V2 — Characterizing "zero-both" honestly

**(a) It is a constant output.** Greedy-decoding the zero-both checkpoint on 400 test
samples yields **exactly 1 unique 24-token sequence** (as it must — with vision and ego
zeroed, the LM context is identical for every sample):

```
tokens = [33]*12 + [35]*12
accels = [-0.008]*12  m/s^2   (≈ 0)
curvs  = [ 0.0043]*12 rad/m   (≈ 0)
```

i.e. **hold speed, go essentially straight**. It is a constant-velocity controller.

**(b) The rollout is seeded from GROUND-TRUTH ego state even when ego is zeroed.**
`inference.py` (post-V1-fix):
```
196:  v0   = float(traj['future_speeds'][0])   # true current speed  (GT)
197:  yaw0 = float(ego_state[3, 1])            # current global yaw   (GT)
209:  pred_positions, _ = unicycle_rollout(pred_accels, pred_curvs, v0, yaw0)
```
`--zero_vision` (line 202) and `--zero_ego` (model.py:151) zero only the model **input
embeddings**; `traj['future_speeds']` and `ego_state` used for the rollout are untouched.

**(c) Relabel.** "zero-both" is **not a null model** — it is a **CONSTANT-VELOCITY
BASELINE SEEDED WITH GROUND-TRUTH EGO SPEED + HEADING**. Its low P1 score reflects a strong
prior (most 6-second futures ≈ hold speed and heading), not model skill.

**True naive baselines (no model at all), full test n=3614, V1-fixed frame:**

| baseline | ADE@1s | ADE@2s | ADE@3s | ADE@6s | (median@6s) |
|---|---|---|---|---|---|
| Constant-velocity (hold speed, straight) | 0.150 | 0.477 | 0.943 | **3.062** | 2.409 |
| Constant-turn-rate (hold speed + yaw-rate) | 0.143 | 0.434 | 0.869 | **3.014** | 2.402 |

**This is the honest reference line.** Any model that claims to use perception must beat
**~3.0m ADE@6s** — the free score from GT ego kinematics + "keep doing what you're doing."
(The tokenizer floor 0.885m is the *upper* bound of achievable skill; the CV baseline
~3.0m is the *no-skill* line. A useful model lives between them.) `results/res_v2_cv_baselines.json`.

---

## V3 — Corrected evals (V1-fixed decode) + clean AR-selected training

### V3a — P1 five-checkpoint table, RE-RUN with the fixed decode (replaces P1)

Test n=3614, autoregressive, V1-fixed `v0`. **This supersedes the P1 table.**

**Mean ADE / FDE (m):**

| Checkpoint | ADE@1s | ADE@2s | ADE@3s | ADE@6s | FDE@6s | tok-acc |
|---|---|---|---|---|---|---|
| (a) old baseline        | 1.017 | 2.479 | 4.437 | 12.233 | 28.05 | 26.0% |
| (b) full live-vision    | 0.825 | 1.627 | 2.636 | 6.610 | 14.79 | 43.8% |
| (c) ego-only            | 0.268 | 0.737 | 1.411 | 4.488 | 11.21 | 49.9% |
| (d) vision-only         | 0.272 | 0.752 | 1.435 | 4.486 | 11.07 | 37.5% |
| (e) zero-both = **CV, learned** | 0.205 | 0.588 | 1.138 | 3.648 | 9.12 | 40.6% |
| **naive CV baseline (no model)** | **0.150** | **0.477** | **0.943** | **3.062** | 7.67 | — |
| naive const-turn-rate | 0.143 | 0.434 | 0.869 | 3.014 | 7.84 | — |
| **tokenizer floor** | 0.139 | 0.294 | 0.496 | **1.348** | 3.18 | (100%) |

**Median ADE@6s:** full 4.087 · ego-only 3.592 · vision-only 3.853 · zero-both 3.205 ·
CV 2.409 · CTR 2.402 · floor 0.889.

**Two conclusions, now on solid ground:**

1. **The fix changed magnitudes, not the ranking.** Buggy→fixed ADE@6s mean: full
   6.978→6.610, ego 4.919→4.488, vision 5.268→4.486, zero-both 4.545→3.648. The order
   (zero-both < ego ≈ vision < full) is **unchanged** — so "more inputs → worse AR ADE,
   full is worst" was **not** a decode artifact. It survives the corrected decode.

2. **But the honest reference (V2) rewrites the interpretation: every learned checkpoint
   is WORSE than a no-model constant-velocity baseline (3.06m mean / 2.41m median @6s).**
   Even zero-both (the closest to CV, 3.65m/3.21m) slightly trails true CV because its
   constant tokens aren't exactly straight (curv≈0.004). The full vision+ego model is
   ~2× the CV baseline. **The model has negative skill vs CV; adding inputs makes it
   worse.** This is a training/architecture failure deeper than "vision doesn't help":
   the trajectory head never learned to beat "keep going straight."

The vision-vs-ego delta @6s: full − ego-only = **+2.12m mean / +0.49m median** (vision
still hurts). But both are moot until a model beats CV.

### V3b — clean AR-val-ADE-selected training (full-live vs ego-only), 10 ep, patience 5

_Running (launched after V3a; ~many hours). Both select on AR median ADE@6s (P2), the
only valid metric, trained past epoch 1. This is the first apples-to-apples vision-vs-ego
test free of the disqualified TF-val-loss selection. Results appended when complete._

---

## W1 — Ego-state speed channel was corrupted (root cause)

`compute_ego_state` used a BACKWARD difference over `past_poses`: speed=0 when history
padded (many samples; some have 0 past poses), under-estimated otherwise; accel derived
from it, also corrupted. The model never saw the car's true speed — the deepest cause of
"no model beats CV" and "more inputs → worse".

FIX (`dataset.py`): forward differences over `[past…, current, future[0]]`; current-row
speed = `dist(current→future[0])/dt` = `future_speeds[0]` exactly (matches rollout seed);
no padded zeros. VALIDATION (200 samples, |ego[3,0]−future_speeds[0]|): OLD mean 0.576 /
max 9.98 / 8 zeros → NEW 0.0000 / 0 zeros. yaw unchanged. No pkl regen (live). Commit 875d3a2.

## W3 — Demo visualizer (`code/visualize.py`)

BEV (ego origin, forward up, 6 s): GT green / prediction red / const-vel grey-dashed, over
the 6-camera grid; animates a scene → GIF. Auto-picks straight/turning/braking scenes;
`--checkpoint` CLI arg. Shipped `viz/demo_{straight,turning,braking_accel}.gif`. Commit aeaf4da.

## W4 — Scenario stratification (straight vs turning)

Split by max|GT curv| over 6 s > 0.05 rad/m: **straight n=2976 (82%) / turning n=638 (18%)**.
ADE mean (m), AR, V1-fixed rollout. ⚠️ PRE-W2 checkpoints (old ego, evaluated with old ego
to stay matched); definitive run pending on W2 models.

| subset | CV | zero-both | ego-only | full |
|---|---|---|---|---|
| STRAIGHT @6s | **2.600** | 3.304 | 4.229 | 6.627 |
| TURNING @6s | **5.218** | 5.252 | 5.695 | 6.531 |

**Does vision help on turns? No — it still hurts.** On turning, full (6.531) is worse than
ego-only (5.695) by 0.836m and both trail CV (5.218). Full's ADE is ~scenario-independent
(~6.6m both), i.e. it barely conditions on the situation. No model beats CV on either subset.
Must be reconfirmed on W2 fixed-ego models. `results/res_w4_stratified.json`.

---

## W2 — The real experiment: full-live vs ego-only, FIXED ego, AR-val-ADE selection

10 epochs, patience 5, aug ON, eff batch 24, 8-GPU, selected on AR median ADE@6s (P2).
First clean vision-vs-ego test: correct ego input, valid metric, trained past epoch 1.

**Val-ADE@6s median per epoch** (selection metric):
- full-live: 3.735 · 3.341 · 3.416 · **3.135**(e4 best) · 3.196 · 3.183 · 3.372 · 3.305 · 3.359 → early-stop e9
- ego-only:  3.231 · 3.360 · 3.359 · 3.138 · 3.311 · 3.165 · 3.207 · 3.191 · 3.113 · **2.902**(e10 best, still improving)

**Full test-set AR ADE (n=3614), fixed ego:**

| model | ADE@1s | ADE@2s | ADE@3s | ADE@6s | (median@6s) | tok-acc |
|---|---|---|---|---|---|---|
| **tokenizer floor** | 0.139 | 0.294 | 0.496 | **1.348** | 0.889 | — |
| **naive CV baseline** | 0.150 | 0.477 | 0.943 | **3.062** | 2.409 | — |
| **W2 ego-only (fixed ego)** | 0.185 | 0.545 | 1.095 | **3.820** | **3.052** | 53.1% |
| W2 full live-vision (fixed ego) | 0.275 | 0.695 | 1.305 | **4.236** | 3.277 | 51.6% |
| — (pre-fix full, V3a) | 0.825 | 1.627 | 2.636 | 6.610 | 4.087 | 43.8% |

**Two clean conclusions:**

1. **The ego bug (W1) was doing real damage.** Fixing it improved full-live test ADE@6s from
   **6.610 → 4.236m (−36% mean, −0.81m median)** and ego-only from 4.488 → 3.820m. tok-acc rose
   43.8% → 51–53%. The model genuinely could not work without its true speed. This is the single
   biggest lever found across the whole investigation.

2. **Vision still does not help — it slightly hurts.** With the fixed ego, valid metric, and full
   training, **ego-only (no camera) beats full live-vision at every horizon**: ADE@6s **3.820 vs
   4.236m mean (−0.42m), 3.052 vs 3.277m median (−0.23m)**; vision is worse at 1s/2s/3s too. The
   camera tokens remain net-negative for trajectory prediction on this dataset/recipe.

**And the bar still stands: no learned model beats the naive CV baseline.** W2 ego-only (3.820m
mean / 3.052m median @6s) is the best model produced, but it is still **+0.76m mean / +0.64m
median above CV (3.062 / 2.409)** — though ego-only was *still improving at epoch 10* (val-ADE
2.90 and falling), so more epochs may finally cross CV. That is the one open thread.

Checkpoints: `models/checkpoints/_w2_full_fixed/`, `_w2_egoonly_fixed/`;
`results/res_w2_full.json`, `res_w2_egoonly.json`.

---

## X1 — Attention rollout: do the trajectory tokens attend to vision? (W2 full-live)

For the 24 trajectory-prediction query positions, average (causal, row-normalized) attention
mass on each key group, 28-layer Qwen2.5-VL LM, 16 val samples (eager attention):

**Pooled over layers: vision = 0.649, ego = 0.125, prior-traj = 0.226.**
Per-layer vision mass climbs from ~0.2 (layers 0–3) to **0.76–0.93 in the middle/late layers
(8–22)**. (Layer 27 not captured — minor artifact; excluded.)

**Vision is NOT ignored — it receives ~65% of the trajectory tokens' attention mass.** So the
failure of vision to help is **not** a routing/attention-access problem: the plan tokens read
the camera heavily. Per-key, ego is ~74× denser (0.125 mass over 4 tokens vs 0.649 over 1536),
so ego is the *concentrated* signal — but vision is far from zero.

**Implication:** vision reaches the plan and is attended to; it simply is not *useful* —
the (frozen) vision features don't add trajectory-relevant information beyond ego on this
dataset. **This means X3 (more adapter capacity) is NOT indicated** — the planner's trigger
was "vision mass ≈ 0", and it is 0.65, not 0. Adding capacity to a pathway that is already
heavily used but uninformative would not help. `code/x1_attention.py`.

---

## X2 — Straight/turning stratification on the W2 FIXED-EGO models (the real table)

Split by max|GT curv| over 6 s > 0.05 rad/m: straight n=2976 (82%) / turning n=638 (18%).
W2 checkpoints, correct ego, V1-fixed rollout. ADE mean (m).

| subset | model | ADE@1s | ADE@2s | ADE@3s | ADE@6s |
|---|---|---|---|---|---|
| **ALL** | CV | 0.150 | 0.476 | 0.943 | **3.062** |
| | ego-only | 0.185 | 0.545 | 1.095 | 3.820 |
| | full | 0.275 | 0.695 | 1.305 | 4.236 |
| **STRAIGHT** | CV | 0.134 | 0.418 | 0.816 | **2.600** |
| (n=2976) | **ego-only** | 0.172 | 0.498 | 0.985 | **3.379** |
| | full | 0.282 | 0.689 | 1.261 | 3.949 |
| **TURNING** | CV | 0.229 | 0.751 | 1.538 | **5.218** |
| (n=638) | **full** | 0.239 | 0.724 | 1.510 | **5.579** |
| | ego-only | 0.244 | 0.764 | 1.610 | 5.873 |

**The nuanced answer to "does vision help on turns?" — YES, once ego is correct.** On the
turning subset, **full-vision beats ego-only at every horizon** (ADE@6s 5.579 vs 5.873, −0.29m;
also better at 1/2/3s). This is the first place vision provides a genuine benefit — exactly
where perception should matter. But:
- On STRAIGHT (82% of samples) vision **hurts** (full 3.949 vs ego-only 3.379), which dominates
  the overall average → full worse overall. The straight-line future is near-deterministic from
  ego; the camera only adds noise there.
- **No model beats CV on either subset** (turning: full 5.579 > CV 5.218; straight: ego 3.379 >
  CV 2.600). Vision helps *relative to ego-only on turns*, but not enough to beat constant velocity.

This overturns the pre-fix W4 read (where vision hurt even on turns): with the corrected ego,
**vision carries real, turn-specific signal** — it is small and currently swamped by the
straight-line majority. `results/res_x2_w2_stratified.json`.

### X3 decision

**Not triggered.** The gate was "only if X1 shows near-zero vision attention." X1 shows vision
attention mass = **0.65** (heavily attended), so the bottleneck is not routing/adapter-capacity —
adding LoRA capacity to an already-heavily-used pathway is not the indicated fix. **However**,
X2's turn-specific benefit means vision is not inert; if the goal is to *amplify* the turn signal,
the higher-value levers are (a) a **curvature/turn-weighted or turning-subset-balanced** training
signal so the 18% turning cases aren't drowned by straights, and (b) **unfreezing the vision
encoder** (the frozen features, not adapter width, are the likely ceiling — attention already
reaches them fine). Deferring X3 as written; recommend the above instead. Awaiting direction.

---

## Y1 — Turn-weighted training: THE PAYOFF (vision finally helps overall)

Oversampled turning cases (max|GT curv|>0.05) from **16.2% → ~40%** of drawn samples
(binary weight w_turn=3.44, `DistributedWeightedSampler`). Fixed ego, AR-val-ADE selection,
10ep/patience5, else W2 recipe. Retrained full-live AND ego-only.

**Headline — full-test ADE@6s (mean, m), turn-weighted vs the W2 (non-weighted) baseline:**

| subset | CV | W2 full | W2 ego | **Y1 full** | **Y1 ego** |
|---|---|---|---|---|---|
| STRAIGHT (n=2976) | 2.600 | 3.949 | 3.379 | **3.566** | 3.610 |
| TURNING (n=638) | 5.218 | 5.579 | 5.873 | **5.594** | 5.699 |
| **OVERALL (n=3614)** | 3.062 | 4.236 | 3.820 | **3.924** | 3.978 |

Per-horizon (Y1, overall mean): full 0.219/0.601/1.172/**3.924** @1/2/3/6s;
ego-only 0.210/0.601/1.184/**3.978**. (Y1 ego best == latest == epoch 10; Y1 full best =
epoch 8; full_latest epoch 10 is worse at 4.103, so selection was fine.)

**Turn-weighting flips the vision-vs-ego result — vision now helps.**
- In W2, full-vision LOST to ego-only overall (4.236 vs 3.820).
- In Y1, **full-vision BEATS ego-only overall (3.924 vs 3.978) and on BOTH subsets**
  (straight 3.566 vs 3.610; turning 5.594 vs 5.699).

**Mechanism (clean and interpretable):** turn-weighting **improved the vision model**
(full overall 4.236→3.924) while **degrading the ego-only model** (3.820→3.978). Oversampling
turns gives the full model camera signal it can exploit; for ego-only there is no camera to
leverage, so the extra hard turning cases only pull its overall fit down. This is exactly the
hypothesis: once turns are properly represented, **vision pays off**.

**Caveats (honest):** margins are small (~0.05m overall, ~0.1m on turns) and **no learned
model beats the CV baseline yet** (overall 3.062; best learned 3.92). The turn *gap* did not
widen vs W2 (W2 turn gap 0.29m → Y1 0.11m) — the win came mostly from full improving on
STRAIGHT and ego degrading overall, not from a bigger turn margin. So the result is
"**vision is now net-positive**", not "vision is now clearly good". Checkpoints:
`models/checkpoints/_y1_full_turnw/`, `_y1_egoonly_turnw/`; `results/res_y1_turnw_stratified.json`.
