# R12_REPORT -- v3 mechanical check, short re-audit page; CoC seed 42 on v3 (PROVISIONAL)

HEADER (rules fixed and committed BEFORE computing; val read for analysis only).
A1 The 68 audit ids whose v3 trace differs from v2 in text only (R11 B5). For each id a
   PREDICTED v3 text is built from the v2 row using only (i) deleting "; route: <cmd>",
   (ii) the same element (stop line / crosswalk: same map token; lead / yield candidate:
   same annotation) with its printed distance max(0, d - 3.528) .0f, (iii) dropping a v2
   lead / yield candidate that fails the v3 rule (x_f <= 0, outside the v3 corridor, or
   the v2 crosswalk-pedestrian branch), and composing with the v3 text functions WITHOUT
   any other change (in particular without the B2 "stay stopped" wording). PASS iff the
   predicted causes AND trace equal the actual v3 ones; else one line with the clause-
   level diff and the reason class (e.g. new candidate, stationary wording).
A2 audit/coc_audit_v3_short.html, v3 layout, CSV as v3 (sample_id, decision, causes,
   peeking, note); card count in the page title and header; coc_audit_v3.html untouched.
   (a) the 13 decision-changed ids, v2 text in grey beside v3; (b) every A1 id not PASS
   (v2 text beside v3); (c) 4 val traces whose v3 turn comes only from the > 45 deg rule
   (|dpsi| > 45, |dpsi in intersections| <= 30) and 3 val traces with v3 "In-lane
   nudge"; (d) 5 val traces, one per v3 lon decision class (the 5 largest classes that
   have an eligible scene). (c)+(d): one per scene, scenes not seen before (not in the
   v2 audit, not on coc_audit_v3.html, not already used), one np.random.RandomState(13)
   for all draws in the order (c) turns, (c) nudges, (d) classes by descending size;
   scene drawn uniformly from the sorted eligible scenes, then one sample uniformly.
A3 ar1/audit_summary.py --v3short <csv>: PASS if wrong decisions <= 2 AND no single
   cause error type >= 3 over (a)-(d) (cause error = Causes "wrong"; types as v3,
   + "nudge obstacle"); "unsure" reported separately. Prints only.
B  CoC seed 42 on data/ar1_coc_2hz_v3.pkl: ar1/finetune_coc.py unchanged except a --coc
   path flag (default v2 file) and the tag COC3; all R11-earlier Part B rules otherwise
   (sequence, loss, recipe, selection, modes i-iv, smoke criteria <= 1 GPU-h, GC1-GC6
   definitions, (iii) donors from the v3 pickle). Every number PROVISIONAL; GC1-GC6 one
   seed, indicative; discarded if the short re-audit reports a new systematic bug; seeds
   123 / 2024 only after AUDIT_SIGNOFF.md. Queue ar1/run_r12b.sh (tmux r12b).
