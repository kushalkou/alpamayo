# R11_REPORT -- CoC audit bugs verified and fixed (v3); re-audit page

Replaces the earlier R11 (text in git at c378628). The earlier R11's CoC seed-42 run
(launched 11:47 UTC) was STOPPED at 13:37 UTC (epoch 5 of 10) on this prompt and is
discarded; no CoC numbers exist. Nothing is launched by this R11.
HEADER (rules fixed and committed BEFORE computing). Input: Kushal's audit CSV (100 rows,
AR1_COC_AUDIT.html v2; ids mapped to A001-A100 by token via AR1_COC_AUDIT.txt). v2 traces
= data/ar1_coc_2hz.pkl, rows recomputed by coc_template (speed_src 2hz), all 29,049.
A1 Command [P]: quote w1/build_data.py; P(cmd = straight | v2 lateral decision = Turn
   left / Turn right), train and val (pickle split field).
A2 v2 agent frame = nuScenes ego_pose (rear-axle midpoint). Count v2 traces whose lead or
   yield candidate has x < 3.5 m, or |y| > agent half-width (size[0] / 2) + 1.5 m; list
   the audit ids among them.
A3 Count v2 traces with |dpsi| (GT yaw at the last future keyframe <= 6 s minus yaw at t0)
   > 45 deg and lateral decision not a turn; list audit ids.
A4 Quote AR1 Table 5 exactly; also check AR1 Table 1 (the decision taxonomy our CoC
   decision field follows) for nudge classes.
B  v3 = ar1/coc_template_v3.py (v2 code unchanged; imported and overridden) ->
   data/ar1_coc_2hz_v3.pkl (v2 pickle kept). Fixes:
   B1 "route: <cmd>" removed from the critical components (cmd stays a model input).
   B2 Front bumper x_f = x - 3.528 m (ego box of the yaw-aware metric: length 4.084 m,
      centre 0.5 m ahead of the LiDAR, LiDAR 0.986 m ahead of the rear axle). Lead / yield
      candidates: centre ahead of the bumper (x_f > 0) and inside the corridor: distance
      from the agent centre to the realized GT path (polyline from the front bumper) <=
      0.925 (ego half-width) + 0.5 + agent half-width; if the GT path is < 5 m long, a
      straight corridor |y| <= that bound. Distances in the text are x_f. Lead x_f < 40 m,
      yield x_f < 25 m (as v2). Stationary for the whole horizon (v0_can < 0.2 m/s and GT
      max displacement over the available future <= 0.5 m): action wording "stay stopped"
      (never "slow down").
   B3 Lateral turn if |heading change inside intersections| > 30 deg (v2) OR |dpsi| > 45
      deg (side from the sign of the triggering quantity).
   B4 AR1 Table 1 has "In-lane nudge" and "Out-of-lane nudge"; added as "In-lane nudge
      left/right" (my rule): not a turn and not a lane change; GT path lateral offset y
      (ego frame at t0, x_f in (0, 40] m) reaches max |y| >= 0.5 m and its end offset is
      <= 0.5 x that max (returns); a stationary vehicle / object outside the corridor but
      within 3 m of it on the opposite side, x_f in (0, 30] m, is the cause.
   B5 Report share changed per fix (decision field, causes field; each fix alone vs v2),
      class distributions v2 vs v3, and which of the 100 audit ids change.
C1 audit/coc_audit_v3.html (v2 layout: filmstrip, speed chart, top-down, rule evidence,
   decision, causes, trace, radios OK / wrong / unsure, note, CSV with the v2 columns
   sample_id, decision, causes, peeking, note). (a) every audit id whose v3 trace differs
   from v2, with v2 text beside v3; (b) 30 fresh val traces, one per scene, scenes not in
   the v2 audit, stratified over v3 lon decision classes: RandomState(11); classes in
   descending size order; quota = 30 split as evenly as possible (each class <= its size,
   remainder to the largest classes); within a class, scenes drawn without replacement and
   one random sample per scene.
C2 ar1/audit_summary.py --v3 on the new CSV: PASS if wrong decisions <= 2 AND no single
   cause error type >= 3 among (a)+(b); cause error = Causes "wrong"; type = the trace's
   cause type (R11-earlier categories). "unsure" counted separately. Prints only.
