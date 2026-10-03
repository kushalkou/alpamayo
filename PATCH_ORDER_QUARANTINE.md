# PATCH_ORDER_QUARANTINE -- results computed with the non-native vision patch layout

Phase R2.0, 2026-10-03. Numbers are kept everywhere; nothing is deleted.

The defect. vision_live.patchify (the live path since the DGX migration) and
precompute_visual_tokens.py (the old 246 GB cache it was validated against, cos 0.9955)
build the vision-tower input in raster patch order with a (T, C, 14, 14) flatten. The
Qwen2.5-VL tower expects 2x2 merge blocks and (C, T, 14, 14) (its window attention, 2D
rotary positions and 2x2 merger all assume that order). Measured on 5 val images from 5
scenes (Alpamayo/ar1_r20_patch.txt, ar1/r20_patch_check.py):
  NEW R1 path (ar1/vision_ar1.py) vs HF processor + tower: token cosine 1.00000 (min
    1.00000) on all 5 -> PASS (>= 0.999)
  OLD path vs HF processor + tower (448x448): token cosine mean 0.33-0.52, min -0.02-0.11
Unaffected: runs with visual tokens ZEROED after the encoder (--zero_vision, zeroboth)
or REMOVED (--no_vision): they never use the encoder output.

AFFECTED (INVALID (patch order)):
Custom split (old, pre-official):
  1. Original Gate 1-4 training on cached vision tokens (val loss 1.9098 @ epoch 1;
     token acc 41.49%), and the live-vision migration gates (CLAUDE.md section 3).
     Also leaky (FROZEN_RESULTS_v1_LEAKY.md).
  2. RESULTS_OVERNIGHT.md: full live-vision rows (e.g. ADE@6s 6.978 / 4.236), the
     vision-only (ego zeroed) row, and "full beats ego-only on turning" (5.58 vs 5.87).
  3. OVERNIGHT_REPORT.md / DECODE_RULE_RESULTS.md: every y1_full row (decode rule,
     alpha*, blend gains, y1_full - y1_ego, zeroboth - y1_full). The decode-rule finding
     itself (expectation over argmax) is replicated on ego-only and zeroboth models and
     stands; the y1_full instances of it are invalid.
  4. FROZEN_RESULTS.md FINDING 4 (vision helps on turning; already WITHDRAWN for the
     leak) and the OVERNIGHT2 6b causal full-vision retrain (blend +0.014; full - ego on
     TURNING -0.014). Marked in FROZEN_RESULTS.md and REVIEW_RESULTS.md T2.
  5. x1_attention.py attention analysis and visualize*.py outputs on vision models.
  6. BRIEF.md "Overall: nothing measurable" (full vision vs ego-only).
Official split (week 1 / queue v2):
  7. V8b (6 cameras + ego + cmd intent predictor): WEEK1_REPORT G8; REVIEW_RESULTS T10a
     V8b row and V8b - V8a; T10b (C8) V8b rows, recall / confusion / agreement; C7 V8b
     rows (w1/c7_bottleneck.py, results/w1_c7.pkl); figure f8 V8b bar, f6 / f7 V8b
     markers. The pre-registered verdict "cameras add no decision information" is VOID.
  8. G7 (6 cameras end to end): WEEK1_REPORT G7; REVIEW_RESULTS T11a-c G7 rows, the
     camera-shuffle test (T11b) and G7 differences; figure f8 G7 reference line.
     "Cameras make the trajectory VLA slightly worse" is VOID.
  9. V8b / G7 cost numbers (5.66 s/step, ~92 GPU-h) stand: cost does not depend on the
     patch order.
NOT affected (stand): every no-camera result (A0-A3, G5, G6 / T9, V8a, MLP, C7 non-V8b
rows, G9, the leak findings, ladders T6 / T8, Ego-MLP, kinematic rule, CV).
