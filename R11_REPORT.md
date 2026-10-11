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
   from v2 (decision field, or causes / trace text after deleting "; route: <cmd>" from
   the v2 causes, since B1 alone changes every trace), with v2 text beside v3; (b) 30
   fresh val traces, one per scene, scenes not in
   the v2 audit, stratified over v3 lon decision classes: RandomState(11); classes in
   descending size order; quota = 30 split as evenly as possible (each class <= its size,
   remainder to the largest classes); within a class, scenes drawn without replacement and
   one random sample per scene.
C2 ar1/audit_summary.py --v3 on the new CSV: PASS if wrong decisions <= 2 AND no single
   cause error type >= 3 among (a)+(b); cause error = Causes "wrong"; type = the trace's
   cause type (R11-earlier categories). "unsure" counted separately. Prints only.
=========================================================================================
BUG CONFIRMATIONS (all 29,049 v2 traces; r11_bugs.txt)
A1 CONFIRMED, and R8 was WRONG about the horizon. w1/build_data.py: "x6 = gl[-1][0];
   command = 0 if x6 >= 2 else (1 if x6 <= -2 else 2)" with gl = LiDAR positions t0..t0+
   N_EVAL keyframes, N_EVAL = 6 at 2 Hz: the command is the lateral offset of the GT
   LiDAR position at t0+3 s (>= 2 m right / <= -2 m left / else straight; VAD rule).
   P(cmd = straight | v2 turn): train left 0.382, right 0.331; val left 0.421, right
   0.353 (holdout 0.333 / 0.500). Slow turns do not reach 2 m in 3 s; curves do.
   13 audit ids have a turn decision with a contradicting command; Kushal's route
   notes: A012 A043-A046 A051-A054 A056 A057 A059 A080.
A2 CONFIRMED. v2 x is from the nuScenes ego_pose origin = REAR-AXLE midpoint; in-path =
   centre within 1.5 m + half its width of the GT path polyline starting at the rear axle
   (path >= 5 m), else in the ego lanes with |y| < 2.5 m; plus a crosswalk branch: any
   pedestrian inside the crosswalk ahead, with NO lateral test. Flagged lead / yield
   (x < 3.5 m or |y| > half-width + 1.5 m): 1,265 traces (4.4%); yield candidates 785
   |y|, 97 x; leads 357 |y|, 64 x. 620 of 1,286 v2 yield decisions rest on a flagged
   candidate. Audit ids: A082 A087 A092 A094 A096 A099 A100 (all expected) plus A006
   A022 A085 A086 A097 (|y| only; A022 audited wrong, A085-A097 audited OK).
A3 CONFIRMED. |dpsi| > 45 deg but not a turn: 606 traces (2.1%; 606 of 2,261 with
   |dpsi| > 45), all "Lane keeping": the turn rule only counted heading change inside
   map intersection polygons. Audit ids: A025, A065, plus A034.
A4 AR1 Table 5 (p.15, quoted): "Longitudinal: Gentle accelerate, Strong accelerate,
   Gentle decelerate, Strong decelerate, Maintain speed, Stop, Reverse | Lateral: Steer
   left, Steer right, Sharp steer left, Sharp steer right, Reverse left, Reverse right,
   Go straight". NO nudge (atomic meta actions). AR1 Table 1 (driving decisions, which
   our CoC decision field follows) HAS "Out-of-lane nudge (straddle avoidance)" and
   "In-lane nudge ... Temporary offset within the lane (no line crossing) to increase
   clearance around a blockage/hazard; specify left/right" -> B4 applied.

B v3 = data/ar1_coc_2hz_v3.pkl (v2 kept; ar1/coc_template_v3.py; r11_v3.txt). Changed vs
   v2, each fix alone: decision / causes / trace
     B1 route   0.000 / 1.000 / 0.000 (check: identical to v2 minus the route clause)
     B2 bumper  0.050 / 0.672 / 0.204 (bumper distances change most causes texts)
     B3 turn    0.021 / 0.000 / 0.021;  B4 nudge 0.002 / 0.002 / 0.002
     all (v3)   0.072 / 1.000 (0.673 beyond the route clause) / 0.225
   After v3: 0 leads / yield candidates behind the bumper or outside the corridor (4,744
   leads, 1,420 yield candidates). lon v2 -> v3: Set speed 0.658 -> 0.641, Stop 0.161 ->
   0.156, Lead 0.120 -> 0.140, Yield 0.044 -> 0.039, Speed adaptation 0.016 -> 0.024,
   Acceleration 0.0003 =. lat: Lane keeping 0.766 -> 0.743, None, Lane change unchanged,
   Turn right 0.039 -> 0.048, Turn left 0.028 -> 0.039, In-lane nudge L 0.0009 R 0.0008.
   Audit ids changed: 81 of 100 (decision 13: A025, A034, A065 -> turn; A069 Stop ->
   Lead; A082 Yield -> Set speed; A085 A086 A087 A092 A094 A096 A099 A100 Yield -> Stop
   (no pedestrian ahead of the bumper in the corridor); the other 68 change causes /
   trace text only, mostly bumper distances). All 606 new turns move >= 2 m.
C1 audit/coc_audit_v3.html (9.9 MB, 111 cards: (a) 81 + (b) 30; v2 layout, CSV as v2;
   no network refs; JS parses, not run in a browser). (b) quota: Set speed, Lead, Stop,
   Yield, Speed adaptation 6 each (Acceleration: no eligible scene); drawn rarest first.
C2 ar1/audit_summary.py --v3 <csv> (tested on a synthetic CSV). For reference, Kushal's
   v2 CSV has 3 wrong decisions (A022, A031, A082), 11 wrong and 17 unsure causes.
