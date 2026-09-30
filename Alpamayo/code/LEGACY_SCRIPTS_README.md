# LEGACY SCRIPTS -- DO NOT RERUN FOR NEW NUMBERS

dataset.compute_ego_state changed semantics on 2026-07-14 (commit 875d3a2, "W1 FIX"):
it went from BACKWARD differences over past poses (causal) to FORWARD differences that
read future_positions[0], future_yaws[0] and future_speeds[1] (the ego-state leak,
OVERNIGHT2 item 1). Any script that imports it computes something different today
from what it computed when its numbers were published. Example: v2_characterize.py
produced the causal CTR 3.014 on 2026-07-13 (947dee4); rerun today it silently reads
a future yaw and gives 2.799.

Scripts that import compute_ego_state (frozen or historical; rerun only to reproduce
a result dated AFTER 2026-07-14, and never for new numbers):
  dataset.py (definition)  decode_eval.py  dump_decode.py  eval_stratified.py
  eval_stratified_w2.py  eval_stratified_y1.py  eval_z1_seeds.py  gate3_overfit.py
  v2_characterize.py  visualize.py  visualize_y2.py

New work uses only causal features:
  leak/causal_ego.py (custom split) and w1/records.py ego_state_w1 (official split,
  CAN at the last message <= t0). Both are guarded by unit tests
  (w1/test_causal_ego.py, w1/test_w1.py, w1/test_can_causal.py).
