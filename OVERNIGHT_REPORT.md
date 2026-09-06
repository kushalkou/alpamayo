# OVERNIGHT REPORT — decode-rule / shrinkage queue

STATUS: IN PROGRESS. Started 2026-09-06 19:05 UTC.
This file is written incrementally so it survives a dropped session.

## MORNING SUMMARY

(pending — written last, max 15 lines)

---

## Queue status

| item | what | status |
|---|---|---|
| 0 | decode dumps (val n=3572, test n=3614) | RUNNING |
| 1 | VERIFICATION (blocking) | pending |
| 2 | zeroboth / CV divergence | pending |
| 3 | Gate 3.1 stationary hybrid | pending |
| 4 | Gate 3.2 global shrinkage | pending |
| 5 | Gate 3.2b per-stratum alpha | pending |
| 6 | Gate 3.3 adaptive shrinkage | pending |
| 7 | Gate 3.4 does shrinkage subsume the decode fix | pending |
| 8 | visuals for Wednesday | pending |
| 9 | writeup / corrections | pending |
| 10 | morning summary | pending |

Raw outputs land in `Alpamayo/overnight/`. Reference line for every table:
CV = 3.062 mean / 2.409 median ADE@6s on the full 3,614-sample test set.

## Standing reference numbers (established Gates 0-2, already committed)

- CV baseline, full test: ADE@6s 3.062 mean / 2.409 median.
- y1_full  argmax(corrected) 3.921 | V1s 3.588
- y1_ego   argmax(corrected) 3.979 | V1s 3.633
- zeroboth argmax(corrected) 3.648 | V1s 3.266   <- best learned model
- Decode fix (V1s - argmax): -0.333 [-0.371,-0.295] p<1e-4 (y1_full)
- Vision effect (y1_full - y1_ego, V1s): -0.045 [-0.137,+0.048] p=0.35 (n.s.)
- Tokenizer hard floor (full test): 1.348 mean / 0.889 median
- Oracle sub-bin soft floor: 0.266 mean -> interpolation headroom 1.082 m
