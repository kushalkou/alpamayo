# Alpamayo VLA — one-page brief

**Kushal Koujalagi | 2026-09-07 | full detail in FROZEN_RESULTS.md**

## What the system does

A 7B vision-language model (Cosmos-Reason1) is fine-tuned on nuScenes to predict where the
car will be over the next 6 seconds. It writes the future as 24 discrete tokens — 12 for
acceleration, 12 for steering curvature — which are converted back into a path by a simple
bicycle-style motion model. Trained on 16,763 clips, tested on 3,614 held-out clips from
scenes the model never saw.

**The number that matters:** ADE, the average distance in metres between the predicted path
and what the car actually did, at 6 seconds.

**The baseline that matters:** CV, "constant velocity" — assume the car keeps its current
speed and heading in a straight line. On the test set CV scores **3.062 m**.

## What was tried, and what happened

For most of this project **no trained model beat that straight-line baseline.** The best
standalone model scores 3.588 m against CV's 3.062 m. That was the central problem.

Two things fixed it, and they turned out to be the same thing.

**1. The model was asked the wrong question at output time.** The model produces a
probability distribution over 64 possible accelerations, then we took the single most likely
one. Taking the *average* of the distribution instead — its centre of mass — improves ADE by
**0.333 m** (p < 1e-4). The most likely bin is a committed guess; the average is a hedge.

**2. The model over-commits, so hedge again at the path level.** Blending the model's path
with the constant-velocity path, roughly 25–30% model and 70–75% CV, gives:

> ### **2.894 m vs CV's 3.062 m — a 5.5% improvement, p < 1e-4**
> First configuration in this project to beat constant velocity. Blend weight was chosen on
> validation data and applied unchanged to test. Holds across three training seeds.

**This is an ensemble of the model and the baseline, not the model winning on its own.**
Stated plainly because the distinction matters for what we can claim.

## Why everything looked negative before

The model is a *worse* predictor than CV: 39% higher squared error, double the systematic
bias. But the direction in which it disagrees with CV correlates at **+0.34** (p = 0.0005)
with the correction CV actually needs.

**The models were useless as replacements for the baseline. They are not useless as
corrections to it.** Every earlier experiment asked the first question and got a negative
answer. That single reframing explains the whole project history.

We verified this is real, not a statistical artifact: three different "fake models" with the
same error magnitude but no real information gain **exactly zero**, and forcing the same
blend weight onto them *costs* 0.32–0.36 m.

## What the cameras contribute

**Overall: nothing measurable.** Full-vision versus ego-only (speed/heading only) is
**−0.045 m, p = 0.35** — not significant. Adding a second camera-free model helps as much as
adding a camera model, so what looked like a vision benefit is just averaging two different
networks. An earlier claim that turn-weighted training made vision net-positive **does not
replicate** and has been withdrawn.

**On turns, vision does help: +0.122 m**, consistent across all three seed configurations,
and only as a correction term. That is the first camera benefit here that survives seed
replication. It is real but modest, and only where the road bends.

## The four findings

1. **Decode the distribution's mean, not its mode.** Worth 0.333 m. Works just as well on a
   model given *no inputs at all* (86% of the gain), so it is a fix to the estimator, not to
   perception — it should transfer to any similar system that emits discrete control tokens.
2. **Shrink the prediction toward the baseline.** Worth 0.152–0.168 m and seed-robust. This
   is 89% the same underlying mechanism as finding 1.
3. **Correction, not replacement.** A model can be worse than a baseline and still improve
   it. This is the conceptual result and the most reusable one.
4. **Vision helps only on turns, only as a correction.** Nothing overall.

## Honest limits

One dataset; no closed-loop or collision testing; the turning result rests on 638 samples;
"vision versus no vision" always also means "different training run"; and the headline
requires running the baseline alongside the network. We also cannot yet *detect* turns at
runtime well enough to route to the turning-specific behaviour.

## Three open decisions — your input needed

1. **What is the deliverable?** A paper on the decode/shrinkage method (generalises beyond
   this dataset, and finding 3 is the interesting claim), or a system for the buggies (in
   which case closed-loop metrics matter more than ADE)?
2. **Is ADE the right benchmark?** It hid the actual mechanism for months — the blend wins
   on 48% of straight-road samples while improving the mean, because it removes rare bad
   predictions rather than improving typical ones. Should we move to tail metrics or
   collision-based ones?
3. **The Amogh comparison** (Cosmos versus Qwen2.5-VL on the same split) is still not done
   and is the stated headline contribution. Do we run it now against these frozen numbers,
   or redirect that effort?
