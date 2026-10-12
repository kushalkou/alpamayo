# R13_REPORT -- CoC causes test, decision errors, CoC seeds 123 / 2024, CoC-AUX

HEADER (rules fixed and committed BEFORE computing; val read for analysis only).
AUDIT_SIGNOFF.md exists (Kushal, 2026-10-12; traces data/ar1_coc_2hz_v3.pkl). The R12 CoC
seed-42 run on v3 traces (tag COC3) is the official seed 42: no PROVISIONAL label in new
tables. All CoC runs: ar1/finetune_coc.py, v3 traces, recipe and holdout selection as
R12 (tau from each run's own holdout dump).
A1 Causes test, seed 42: (iii) shuffled causes within the GT (lon, lat) class [true
   decision] - (iv) GT CoC [true decision], L2@3s NoAvg, paired scene bootstrap 10,000,
   all val and strata. GC6b: CI includes 0 -> "given the true decision, the causes add
   nothing"; upper < 0 -> "shuffled causes better (suspect)"; lower > 0 -> "causes matter".
A2 Seed 42 mode (i): confusion matrix GT lon x generated lon (and GT lat x generated lat);
   "decision correct" = generated lon AND lat equal the v3 GT. L2@3s for correct vs wrong;
   share of the GC1 gap explained by wrong decisions = sum over wrong samples of
   (CoC - M1-v2a s42) / sum over all samples of the same (per-sample L2@3s).
A3 R12_REPORT.md: GC2 and GC6 marked "uninformative by design" (GC2: the model never saw
   a sequence without reasoning; GC6: (iii) forces the GT decision).
B1 CoC seeds 123, 2024 (tags COC3_s123, COC3_s2024): identical recipe; dumps holdout +
   val (i) gen, val (iii) shuf, val (iv) gt. Gates on 3-seed means (per-sample metrics
   averaged over seeds; scene bootstrap 10,000; L2@3s NoAvg):
   GC1-3 CoC (i) - M1-v2a (3 seeds): upper < 0 "CoC improves"; lower > 0 "CoC hurts";
         else "no measurable CoC effect".
   GC4-3 pooled R3 false-go of (i) over the 3 seeds <= 0.119.
   GC5-3 pooled generated lon decisions (3 seeds): majority share <= its val label share
         + 0.10.
   GC6b-3 (iii) - (iv) on 3-seed means, wording as A1.
B2 CoC-AUX seed 42 (tag COC3AUX), after B1 regardless of outcome: same recipe, but each
   training sample carries its CoC text with p = 0.5, fixed per sample for all epochs
   (np.random.RandomState(42).rand(n_train) < 0.5 over the train records in order);
   without CoC the words follow the context directly (no terminator). Holdout selection
   with reasoning OFF (mode none), since that is the evaluated mode. Dumps: holdout + val
   mode none (reasoning OFF; tau from its holdout dump) and holdout + val mode gen
   (reasoning ON). GX1 CoC-AUX (OFF) - M1-v2a s42, L2@3s: upper < 0 "CoC helps as
   auxiliary supervision"; lower > 0 "hurts"; else "no effect". One seed, indicative.
Every run: collisions aa / yaw, R3 FG / MG, C2 + WSS strata, per-class word F1,
   consistency (as R12). Queue ar1/run_r13b.sh (tmux r13b): s123, s2024, then AUX.

=========================================================================================
GATE VERDICTS
  GC6b (seed 42) (iii) shuffled causes - (iv) GT CoC, both with the true decision:
       +0.107 [+0.071,+0.145] (ADE@6s +0.172) -> "causes matter".
  GC1-3, GC4-3, GC5-3, GC6b-3, GX1: pending (queue r13b: CoC s123, s2024, then CoC-AUX).

A1 Causes test by stratum (r13_partA.txt): excl. WSS +0.114 [+0.075,+0.156]; MOVING
   +0.102 [+0.063,+0.142]; START +0.329 [+0.126,+0.540]; STOPPED +0.059 [-0.008,+0.152];
   WSS +0.000 (both hold perfectly). With the right decision forced, the true causes
   still lower L2@3s by 0.11 m vs another sample's causes of the same decision class.
A2 Seed 42, mode (i): decision correct (lon AND lat) 0.584 (2,991 of 5,119).
   lon recall: Set speed 0.954, Stop 0.707, Lead 0.103, Yield 0.000, Speed adaptation
   0.000 (Lead -> Set speed 893, -> Stop 277; Yield -> Stop 70; Speed adapt. -> Set
   speed 92). The model never generates Yield or Speed adaptation.
   lat recall: Lane keeping 0.926, None 0.832, Turn left 0.522, Turn right 0.561, Lane
   change 0.000 (left 51, right 68 -> mostly Lane keeping), In-lane nudge 0.000 (6).
   L2@3s: correct 1.649 (M1-v2a s42 1.590), wrong 2.608 (M1-v2a 2.280).
   CoC - M1-v2a s42: correct decisions +0.059 [-0.010,+0.120]; wrong +0.328
   [+0.202,+0.496]. Share of the GC1 gap from wrong decisions 0.798 (697 of 874 m summed
   L2@3s); by GT class: Lead +0.301, Set speed +0.244, Stop +0.197, Speed adaptation
   +0.032, Yield +0.024. When the decision is right, CoC is not measurably worse.
A3 R12_REPORT.md: GC2 and GC6 now read "UNINFORMATIVE BY DESIGN" with the reasons.
