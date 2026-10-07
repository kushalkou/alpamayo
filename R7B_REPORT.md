# R7B_REPORT -- CoC arm (AR1 "CoC + traj")

STATUS: not launched. Part B launches ONLY if AUDIT_SIGNOFF.md exists in the repo root
when R7 Part A finishes. Implementation details (exact sequence layout, parser, decode
order) are added to this header and committed before any Part B compute.
HEADER (rules fixed before computing; val read for analysis only):
NOTE CoC causes include GT-box agents (nuScenes annotations) and HD-map elements that are
   NOT model inputs: the CoC text is annotation-derived supervision.
B1 Smoke test on data/ar1_coc_2hz.pkl (<= 1 GPU-h): loss decreases, generated CoC parses,
   trajectory tokens decode. No metrics reported.
B2 CoC arm, seeds 42 / 123 / 2024: M1-v2a recipe + CoC text BEFORE the meta-action words
   (AR1 order: reasoning, then words, then trajectory). Selection identical to M1-v2a
   (holdout, incl. tau). Eval per seed, inference only:
   (i) reasoning generated; (ii) reasoning disabled at inference; (iii) causes shuffled
   across samples WITHIN the same GT decision class, teacher-forced, model generates words
   + trajectory; (iv) GT full CoC teacher-forced [ORACLE].
GATES (3 seeds, paired scene bootstrap 10,000, L2@3s NoAvg; WSS stratum pre-registered):
   GC1 CoC (i) - M1-v2a: upper < 0 -> "CoC improves on meta-action"; lower > 0 -> "CoC
       hurts"; else "no measurable CoC effect".
   GC2 CoC (i) - CoC (ii): CI includes 0 -> "gain does not need generated reasoning".
   GC3 consistency (trajectory follows generated decision and words) >= 0.70.
   GC4 false-go <= M1-v2a 3-seed value (0.119).
   GC5 decision-field collapse: majority decision share within 10 points of label share.
   GC6 (iii) - (i) CI includes 0 -> "generated causes not used by the action decoder".
   Also per-class F1, collisions, strata (incl. R7A C2), comfort (first-order).
Conditional arm (NOT launched; cost only): decision-only CoC padded to matched length, run
   only if GC1 = "CoC improves".
