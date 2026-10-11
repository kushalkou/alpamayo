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

=========================================================================================
PART A RESULTS
A1 Mechanical check of the 68 text-only changes (r12_a1.txt): PASS 60 / 68. Explained by
   (ii) same lead for A001-A022 (22), same yield candidate for A081 A084 A093 A095, and
   route / map distances only for the other 34. NOT PASS (8), all intended v3 rules but
   outside (i)-(iii):
     A088 A089 A090 A091 A098  "Slow down and yield" -> "Stay stopped and yield" (B2
                               stationary wording; same agent, distance -3.528)
     A083 A097                 a different, closer pedestrian is now the yield candidate
                               (the v2 one fails the narrower corridor; v3 picks the
                               nearest in-corridor pedestrian)
     A045                      "construction zone" dropped: the >= 3 cones within 30 m
                               window is now 0 < x_f < 30 m (3.5-33.5 m from the rear
                               axle), so the cones near the ego no longer count
   No unexplained diff remains: no case changed agents or map items for another reason.
A2 audit/coc_audit_v3_short.html, 32 cards (2.6 MB; title and header show 32), v3
   layout and CSV; JS parses, no network refs; coc_audit_v3.html unchanged.
   (a) 13 decision changes; (b) the 8 A1 non-PASS ids; (c) 4 turns only from the > 45 deg
   rule (val pool 133: S01-S04) and 3 in-lane nudges: val has 6 nudges in 5 scenes, only
   2 unseen (S05 scene-0271, S06 scene-0914), so S07 is a TRAIN nudge (scene-0527);
   (d) 4 fresh, not 5: "Speed adaptation" had 0 unseen val scenes after (c) (pools Set
   speed 1,253, Lead 422, Stop 43, Yield 5).
A3 ar1/audit_summary.py --v3short <csv> (tested on a synthetic CSV). Prints only.
