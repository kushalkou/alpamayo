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
PART B GATES -- PROVISIONAL (traces not signed off), ONE SEED, INDICATIVE (r12b_eval.txt)
  GC1 CoC (i) - M1-v2a s42 L2@3s +0.171 [+0.101,+0.255] -> "CoC hurts" (vs 3-seed M1-v2a
      +0.106 [+0.028,+0.199]; ADE@6s +0.171 [+0.074,+0.281]).
  GC2 (i) - (ii) -5.378 [-5.870,-4.927] -> UNINFORMATIVE BY DESIGN (R13 A3): the model
      never saw a sequence without reasoning; with it removed it fails completely (L2@3s
      7.43, false-go 0.999).
  GC3 consistency 0.845 (>= 0.70) -> PASS (words 0.867, decision agrees 0.929).
  GC4 false-go (i) 0.132 (> 0.119) -> FAIL (WSS 0.055, non-WSS 0.211).
  GC5 "Set speed tracking" generated 0.780 vs label 0.572 + 0.10 -> FAIL (collapse).
  GC6 (iii) - (i) -0.114 [-0.213,-0.039] -> UNINFORMATIVE BY DESIGN (R13 A3): (iii)
      also forces the GT decision field; see R13 GC6b for the causes test.
GPU: smoke 0.42 + train 34.8 + dumps 30.6 = 65.8 GPU-h. AUDIT_SIGNOFF.md absent.

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

PART B DETAIL (PROVISIONAL; seed 42; v3 traces; holdout tau 0.5; best epoch 6 of 10,
  holdout median ADE@6s 1.858 vs M1-v2a s42 1.732)
  L2@3s     all    excl.WSS WSS    STOPPED START  MOVING ADE@6s FG    MG    Col aa/yaw
  (i) gen   2.048  2.179    0.229  0.576   9.407  1.881  2.980  0.132 0.523 1.54/1.56
  (ii) none 7.425  7.512    6.228  5.948   9.005  7.577  10.205 0.999 0.000 7.79/8.42
  (iii) shuf 1.934 2.074    0.002  0.190   8.183  1.872  2.763  0.052 0.187 1.89/1.88
  (iv) GT   1.827  1.959    0.002  0.131   7.854  1.770  2.591  0.036 0.194 1.33/1.33
  M1-v2a s42 1.877 2.008    0.063  0.292   9.348  1.723  2.808  0.061 0.530 1.58/1.54
  (i) - B1 3-seed -0.044 [-0.160,+0.060]; MOVING (i) - M1-v2a s42 +0.158 [+0.112,+0.207].
  Generated CoC: parse 1.000, 69 tokens mean; decision accuracy lon 0.668, lat 0.859.
  By GT class: Set speed 0.954, Stop 0.707, Lead 0.103, Yield 0.000, Speed adaptation
  0.000: leads and yields (annotation-derived) are almost never generated.
  First-order comfort 0.979 (i); word collapse 0.422; stop F1 0.72 (i) vs 0.91 (iv).
  Reading: with the GT CoC forced (iv) the arm reaches 1.827 (M1-v2a s42 1.877; (iii)
  with the GT decision but shuffled causes 1.934), but its own decision is wrong 33% of
  the time and collapses to Set speed: the limit is predicting the annotation-derived
  decision, not using it.
