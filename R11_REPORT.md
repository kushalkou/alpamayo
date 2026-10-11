# R11_REPORT -- CoC audit page (40 traces); CoC arm seed 42 (PROVISIONAL)

HEADER (all rules fixed and committed BEFORE computing; val read for analysis only).
A0 Existing AR1_COC_AUDIT.html (ar1/build_audit_html_v2.py) is inspected and reported; it
   is kept unchanged. A new page is built if it does not match A1-A2.
A1 Sample (val split, data/ar1_coc_2hz.pkl). First the 8 traces whose decision changed in
   R6 0b (A031, A041, A042, A043, A044, A048, A052, A053; tokens from AR1_COC_AUDIT.txt).
   Then 4 per pool, pools in this order, one np.random.RandomState(7) for all draws,
   rng.choice(4, replace=False) over the pool's sample tokens sorted ascending, minus
   tokens already chosen:
     WSS        scene in the R5 A2 WSS list (data/r5_wss.pkl)
     START      C2 START (v0_can <= 0.2 m/s, GT max displacement 3 s > 1.0 m)
     TURN       |dpsi| > 30 deg (R9 A2 H_TURN, data/r9_hard_val.pkl)
     AGENT      in-path agent 0 < x < 20 m (R9 A2 H_AGENT)
     INT        R9 A2 H_INT (on an intersection segment or stop line ahead < 20 m)
     RAIN/NIGHT nuScenes scene description contains "rain" or "night" (case-insensitive)
     MOVING STRAIGHT  v0_can > 0.2 m/s and |dpsi| <= 10 deg (my rule)
     RANDOM     all val samples in the pickle
   40 traces, no duplicates; ids R01-R40 in draw order (the 8 also show their A-id).
A2 audit/coc_audit.html: one self-contained file (images as base64 JPEG, SVG inline, no
   network), opens on a Mac browser. Per trace: CAM_FRONT_LEFT / CAM_FRONT /
   CAM_FRONT_RIGHT at t0; CAM_FRONT at t0+3 s (6 keyframes later) labelled "future, for
   checking only"; BEV in the ego frame at t0 (x forward = up): ego past 2 s (keyframe
   poses), GT future 6 s, ALL annotated agents at t0 within 50 m with their class
   (nuScenes category), stop lines, crosswalks, intersection road segments; the trace text
   from the 2 Hz pickle (decision, causes, trace); radios Decision (ok / wrong) and Causes
   (ok / wrong / missing); a Notes box. Progress counter (traces with both radios
   answered) at the top; one "Download CSV" button writing coc_audit_results.csv with
   columns id, decision, causes, notes. Answers kept in the browser's localStorage when
   available. A sidecar audit/coc_audit_meta.json holds id, token, pool, decision and
   cause type.
   Cause type (my rule, for the sign-off count): Yield -> "yield agent"; Lead -> "lead
   agent"; lane change past a lead -> "passing lead"; Stop with stop line -> "stop line";
   Stop with crosswalk -> "crosswalk"; Stop without either -> "unexplained stop"; Speed
   adaptation -> "turn"; Set speed -> "clear lane".
A3 ar1/audit_summary.py reads coc_audit_results.csv (+ the sidecar) and prints counts and
   the pre-registered rule: PASS if wrong decisions <= 2 AND no single cause error type
   appears >= 3 times, where a cause error = Causes "wrong" or "missing", and its type =
   the trace's cause type above. It prints only; it never writes AUDIT_SIGNOFF.md.

PART A RESULTS
A0 Existing AR1_COC_AUDIT.html (8.8 MB, ar1/build_audit_html_v2.py): 100 val traces
   (stratified by lon decision); reads no CoC pickle: it recomputes rows with
   coc_template.build_rows using the 10 Hz CAN meta-actions (the OLD decisions, as
   data/ar1_coc.pkl), not data/ar1_coc_2hz.pkl. Against A2 it has: CAM_FRONT at t0 (not
   FRONT_LEFT / FRONT_RIGHT); CAM_FRONT t0+3 s inside a 7-frame filmstrip; top-down with
   past 2 s, GT future, lanes, stop lines, crosswalks, but only the NAMED agents (lead /
   yield) and no intersection polygons; trace text; radios OK / wrong / unsure for
   decision, causes, no-peeking (no "missing"); note; counts; CSV download. -> does not
   match; new page built, old file kept unchanged.
A1 40 traces drawn (pool sizes after removals: WSS 345, START 231, TURN 570, AGENT 881,
   INT 2,943, RAIN/NIGHT 1,331, MOVING STRAIGHT 3,026, RANDOM 5,083). Decisions of all 40
   recomputed rows equal the pickle (asserted). Sample decisions: Set speed 18, Lead 11,
   Stop 9, Yield 1, Speed adaptation 1 (lon); ids and pools in audit/coc_audit_meta.json.
A2 audit/coc_audit.html (3.3 MB, one file, 160 base64 JPEGs, 40 inline SVG BEVs, no
   network references; JS syntax-checked with esprima, not run in a browser here: no
   browser on the DGX). Map fact found while checking a BEV: many nuScenes crossings
   have no ped_crossing polygon, only PED_CROSSING-type stop lines, so the BEV draws those
   stop lines in blue dashes (other stop-line types red dashes; hover shows the type).
A3 Alpamayo/code/ar1/audit_summary.py [csv]: counts per answer, wrong decisions, cause
   errors by cause type, per pool, and the PASS / FAIL rule (tested on a synthetic CSV).
   Prints only.
