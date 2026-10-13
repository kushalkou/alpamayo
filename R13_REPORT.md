# R13_REPORT -- CoC causes test, decision errors, CoC seeds 123 / 2024, CoC-AUX

HEADER (rules fixed and committed BEFORE computing; val read for analysis only).
AUDIT_SIGNOFF.md exists (Kushal, 2026-10-12; traces data/ar1_coc_2hz_v3.pkl). The R12 CoC
seed-42 run on v3 traces (tag COC3) is the official seed 42: no PROVISIONAL label in new
tables. All CoC runs: ar1/finetune_coc.py, v3 traces, recipe and holdout selection as
R12 (tau from each run's own holdout dump).
A1 Causes test, seed 42: (iii) shuffled causes within the GT (lon, lat) class [true
   decision] - (iv) GT CoC [true decision], L2@3s NoAvg, paired scene bootstrap 10,000,
   all val and strata. GC6b: CI includes 0 -> "given the true decision, the causes add
   nothing"; upper < 0 -> "shuffled causes better (suspect)"; lower > 0 "causes matter".
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
  GC1-3  CoC (i) - M1-v2a, 3 seeds: +0.129 [+0.059,+0.215] (ADE@6s +0.133) -> "CoC hurts".
  GC4-3  pooled false-go 0.153 (> 0.119) -> FAIL (seeds 0.132 / 0.166 / 0.163).
  GC5-3  generated "Set speed tracking" 0.775 vs label 0.572 + 0.10 -> FAIL (collapse).
  GC6b-3 (iii) - (iv), 3 seeds: +0.098 [+0.074,+0.123] (ADE@6s +0.154) -> "causes matter".
  GX1    CoC-AUX (reasoning OFF) - M1-v2a s42: +0.060 [+0.013,+0.107] -> "hurts" (one seed,
         indicative), although its holdout selection score is better (1.685 vs 1.732).
GPU: Part B 173.6 GPU-h (s123 53.5, s2024 62.3, AUX 57.8, AUX code check 0.06).

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

B1 CoC seeds 123 / 2024 (v3, recipe = seed 42; r13b_coc3seeds.txt). Holdout best median
   ADE@6s 1.858 / 1.873 / 1.779 (M1-v2a 1.732 / 1.695 / 1.712); tau 0.5 / 0.9 / 0.5.
   mode (i)   L2@3s  excl.WSS WSS   STOP  START MOVE  ADE6  FG    MG    Col aa/yaw cons.
   s42        2.048  2.179   0.229 0.576 9.407 1.881 2.980 0.132 0.523 1.54/1.56  0.845
   s123       2.064  2.195   0.255 0.709 9.246 1.889 3.008 0.166 0.537 1.84/1.80  0.846
   s2024      2.099  2.211   0.549 0.641 9.555 1.925 3.084 0.163 0.562 1.66/1.76  0.825
   M1-v2a 3s  1.942  2.035   0.657 0.606 9.273 1.755 2.891 0.119 0.512 1.57/1.54  0.882*
   (iv) GT CoC [ORACLE] L2@3s 1.827 / 1.797 / 1.833 (FG 0.036 / 0.055 / 0.036);
   (iii) shuffled causes 1.934 / 1.870 / 1.947. (* words-only consistency, R9 A3.)
   GC1-3 strata: excl. WSS +0.161 [+0.093,+0.251], MOVING +0.144 [+0.106,+0.182], START
   +0.129 [-0.051,+0.310], STOPPED +0.036 [-0.410,+0.626], WSS -0.312 [-0.620,-0.044].
   Oracle (iv) - M1-v2a, 3 seeds: -0.123 [-0.189,-0.064] (ADE@6s -0.322); MOVING +0.008.
   GC6b-3 strata: START +0.521 [+0.374,+0.687], MOVING +0.086, STOPPED +0.027 (n.s.).
   Generated decision accuracy lon 0.668 / 0.686 / 0.687; parse 1.000 every seed; pooled
   generated lon: Set speed 0.775, Stop 0.182, Lead 0.040, Yield 0.003, Speed adapt. 0.
   Word F1 (i): straight 0.89-0.91, stop 0.66-0.72, maintain 0.57-0.58, steer 0.35-0.44,
   sharp 0.13-0.21; collapse 0.397-0.450 (labels 0.023).
   Reading: replicates seed 42. CoC hurts as a reasoning prefix because the model rarely
   generates the right lead / yield decision; with the GT CoC it beats M1-v2a mostly
   through the decision (STOPPED, START), and the true causes add 0.10 m on top.

B2 CoC-AUX seed 42 (CoC kept on 9,196 of 18,313 train samples; selection with reasoning
   OFF; best epoch 7, holdout median ADE@6s 1.685; r13b_eval.txt)
               L2@3s  excl.WSS WSS   STOP  START MOVE  ADE6  FG    MG    Col aa/yaw cons.
   OFF         1.937  2.067   0.127 0.267 9.494 1.791 2.912 0.087 0.601 1.72/1.62  0.873
   ON          2.305  2.231   3.318 2.599 8.459 1.919 3.283 0.571 0.276 2.32/2.38  0.646
   M1-v2a s42  1.877  2.008   0.063 0.292 9.348 1.723 2.808 0.061 0.530 1.58/1.54  0.854
   GX1 strata (OFF - M1-v2a s42): MOVING +0.069 [+0.014,+0.122], STOPPED -0.025 (n.s.),
   START +0.147 (n.s.), WSS +0.064 [+0.010,+0.127]. ON - M1-v2a s42 +0.428 [+0.264,+0.614];
   ON parses only 0.763 of its reasoning (decision accuracy lon 0.489) and starts at
   standstill (FG 0.571). Word F1 (OFF): stop 0.72, straight 0.91, maintain 0.58.
   Reading: half-time CoC supervision does not improve the word + trajectory decoder;
   the holdout median (selection) and val mean L2@3s disagree here.
