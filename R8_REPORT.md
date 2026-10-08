# R8_REPORT -- tau sensitivity, command leak check, no-route seeds

HEADER (all rules fixed and committed BEFORE computing; val read for analysis only; CoC
arm untouched unless AUDIT_SIGNOFF.md appears). Headline numbers stay HOLDOUT-SELECTED
(tau from holdout). Everything in Part A is labelled SENSITIVITY or DIAGNOSTIC and is
never used to select or report a headline.
Definitions (unchanged): R3 false-go (stationary v0_can < 0.5, GT max disp. 3 s < 0.5 m,
pred > 1.0 m), R3 missed-go (stationary, GT > 1.0 m, pred < 0.5 m); WSS = R5 A2 rule;
C2 strata (ADOPTED AFTER R4) STOPPED / START / MOVING with C2 FG (pred > 1.0 m in STOPPED)
and C2 MG (pred <= 1.0 m in START), as in R7A.
A1 Hybrid decode (w1/gate_a.preds), described from the code. Tau is applied AFTER
   decoding: the 24 trajectory tokens are decoded greedily (argmax, fed back) and the
   dump stores per slot argmax value, expectation and p(STOP); tau only selects, per slot,
   argmax (p(STOP) > tau) vs expectation. So changing tau needs NO re-decoding: Part A is
   CPU only (0 GPU-h). Table: selected tau of every run with dumps (q2_traj.run: holdout,
   all n_fut = 12 holdout samples, mean ADE@6s over {0.3, 0.5, 0.7, 0.9}).
A2 SENSITIVITY. Fixed grid tau in {0.3, 0.5, 0.7, 0.9} for M1-v2a s42 / s123 / s2024,
   M1-v2a-NR s42, no-cam M1-v2a s42: L2@3s NoAvg (all 5,119), R3 false-go (all, WSS),
   R3 missed-go, C2 strata L2@3s and C2 FG / MG.
   GT1: at tau = 0.3, 3-seed M1-v2a WSS false-go (R3) max - min <= 0.05 -> "seed
   fragility is a threshold-selection artifact"; else "not explained by tau".
A3 DIAGNOSTIC. Command [P] (VAD rule from the GT future: final 6 s waypoint lateral
   offset >= 2 m right, <= -2 m left, else straight) share per C2 stratum, train (the
   finetune train split, 18,313) and val (5,119); C2 strata computed identically on train.
   GL1: val P(cmd != straight | START) >= 0.20 AND P(cmd != straight | STOPPED) <= 0.02
   -> "privileged command leaks start information"; else "no start leak via command".
   Also M1-v2a s42 (holdout tau) C2 FG on STOPPED and C2 MG on START, split by command.
PART B (GPU, tmux r8b, ar1/run_r8b.sh, markers results/R8_<tag>_{TRAINED,DUMPED}):
   M1v2a_NR_s123, B1_NR_s123, M1v2a_NR_s2024, B1_NR_s2024; recipe and holdout selection
   identical to the s42 NR runs (R7A header). Before each run the queue checks for
   AUDIT_SIGNOFF.md; if present it stops (status R8B_PAUSED_FOR_COC), R7 Part B runs as
   pre-registered in R7B_REPORT.md, then this queue resumes.
   Gates (3 seeds, paired scene bootstrap 10,000, L2@3s NoAvg, 3-seed means):
   GN1-3 M1-v2a-NR - M1-v2a and B1-NR - B1: CIs reported (all four R7A strata + C2).
   GN2-3 M1-v2a-NR - B1-NR: upper < 0 -> "meta helps without route (3 seeds)"; lower > 0
         -> "meta hurts without route"; else "flat".
   Per run: C2 strata, R3 FG (all / WSS / non-WSS), MG, collision @3 s NoAvg / TemAvg
   vehicles, VAD port (aa) and yaw-aware.

=========================================================================================
GATE VERDICTS
  GT1 tau 0.3, M1-v2a WSS false-go s42 / s123 / s2024 = 0.012 / 0.290 / 0.070; max - min
      0.278 (> 0.05) -> "not explained by tau". [SENSITIVITY]
  GL1 val P(cmd != straight | START) 0.121 (< 0.20); P(cmd != straight | STOPPED) 0.000
      -> "no start leak via command" by the pre-registered rule. [DIAGNOSTIC; but see A3:
      a one-sided leak exists for the 28 turn-command samples]
  GN1-3 M1-v2a-NR - M1-v2a (3 vs 3 seeds) +0.140 [+0.043,+0.253]: the route helps M1-v2a.
        B1-NR - B1 +0.053 [+0.012,+0.091]: the route helps B1 a little (s42 alone: flat).
  GN2-3 M1-v2a-NR - B1-NR -0.063 [-0.114,-0.008] -> "meta helps without route (3 seeds)".
        Not the AR1 direction (AR1 0.5B no route: meta 0.291 / 0.988, traj 0.282 / 0.971).
GPU: Part A 0 GPU-h (tau needs no re-decoding); Part B 4 runs 14.2 h x 8 = 113.8 GPU-h.
CoC: AUDIT_SIGNOFF.md never appeared (checked before each run and at 09:25 UTC); the queue
never paused; R7 Part B not launched.

A1 HYBRID DECODE. The model decodes its 24 trajectory tokens (12 accel, 12 curvature
  slots) greedily: at each slot the argmax token is fed back. For every slot the dump also
  keeps the expectation of the slot value under the softmax (the STOP token counts as 0)
  and p(STOP). The hybrid decode then takes, per slot, the argmax value when p(STOP) > tau
  (a confident stop is kept exactly as 0) and the expectation otherwise. The 12 accel /
  curvature values are rolled out from v0_can into waypoints. Tau never changes the
  tokens, only how they are read out. It is chosen per run on the HOLDOUT split (all
  n_fut = 12 samples) as the grid value with the lowest mean ADE@6s.
  Selected tau (holdout ADE@6s at 0.3 / 0.5 / 0.7 / 0.9 in r8_partA.txt):
  no-cam VLA A3 0.7, s123 0.7, s2024 0.5 | B1 0.3, s123 0.5, s2024 0.7 | B2 0.7 |
  M1 0.3, s123 0.7, s2024 0.3 | M1-v2a 0.3, s123 0.9, s2024 0.5 | no-cam M1-v2a 0.3 |
  B1-NR 0.7 | M1-v2a-NR 0.9.
  For every M1 / M1-v2a run the holdout ADE@6s spread over the grid is <= 0.004 m: the
  selection is close to a tie, and the selected value carries no information.

A2 SENSITIVITY (fixed grid, NOT headline). Per run, tau 0.3 -> 0.9 changes L2@3s by
  <= 0.001, R3 false-go by <= 0.002, WSS false-go by 0.000, missed-go by <= 0.007.
                 L2@3s  FG    FG WSS  MG     C2 STOPPED START MOVING  C2 FG  C2 MG
  M1-v2a s42     1.877  0.061 0.012   0.530  0.292      9.348 1.723   0.052  0.662
  M1-v2a s123    1.967  0.200 0.290   0.509  0.998      9.090 1.733   0.190  0.636
  M1-v2a s2024   1.979  0.098 0.070   0.498  0.518      9.366 1.809   0.094  0.602
  M1-v2a-NR s42  2.074  0.243 0.362   0.519  1.384      9.405 1.783   0.234  0.645
  no-cam M1-v2a  1.860  0.038 0.000   0.633  0.163      9.633 1.707   0.027  0.784
  (rows at tau 0.3; full grid in r8_partA.txt.) The R6/R7 remark that "tau 0.9 goes with
  a high WSS false-go" is withdrawn: tau has almost no effect. The s123 / NR standstill
  behaviour is in the generated tokens themselves, i.e. a training-seed effect.

A3 DIAGNOSTIC command [P] per C2 stratum (right / left / straight):
  train STOPPED 2,843: 0.000 / 0.000 / 1.000; START 844: 0.066 / 0.043 / 0.891;
        MOVING 14,626: 0.101 / 0.073 / 0.826
  val   STOPPED 679:   0.000 / 0.000 / 1.000; START 231: 0.078 / 0.043 / 0.879;
        MOVING 4,209:  0.090 / 0.066 / 0.844
  The command is built from the GT 6 s endpoint, so a stationary sample with a turn cmd
  is ALWAYS a START (train 92 / 92, val 28 / 28). Only 12% of val STARTs carry that cue,
  so GL1 is not met, but the cue is perfect where present. M1-v2a s42 (holdout tau) by
  command: right START n 18 MG 0.000; left START n 10 MG 0.000; straight START n 203 MG
  0.754, STOPPED n 679 FG 0.052. M1-v2a starts on every turn-command START and misses
  three in four straight STARTs: it uses the leak where present.

PART B (r8b_eval.txt; per-sample dumps, holdout tau; 3-seed means, scene bootstrap 10,000)
  Holdout tau: M1-v2a-NR 0.9 / 0.3 / 0.3; B1-NR 0.7 / 0.7 / 0.7 (A2: tau is immaterial).
  L2@3s NoAvg per seed (s42 s123 s2024) | mean +- sd:
    M1-v2a     1.877 1.968 1.980 | 1.942 +- 0.056
    B1         2.165 2.048 2.061 | 2.091 +- 0.064
    M1-v2a-NR  2.074 2.096 2.075 | 2.082 +- 0.012
    B1-NR      2.175 2.119 2.139 | 2.144 +- 0.028
  L2@3s [CI]     M1-v2a-NR - M1-v2a     B1-NR - B1             M1-v2a-NR - B1-NR
    all          +0.140 [+0.043,+0.253] +0.053 [+0.012,+0.091] -0.063 [-0.114,-0.008]
    excl. ff     +0.141 [+0.043,+0.256] +0.042 [+0.002,+0.080] -0.069 [-0.121,-0.013]
    excl. WSS    +0.046 [-0.001,+0.100] +0.050 [+0.008,+0.090] -0.101 [-0.146,-0.056]
    WSS only     +1.445 [+0.366,+2.606] +0.097 [-0.014,+0.230] +0.462 [+0.113,+0.855]
    C2 STOPPED   +0.806 [+0.143,+1.507] +0.113 [+0.016,+0.226] +0.166 [-0.063,+0.410]
    C2 START     +0.022 [-0.224,+0.285] +0.219 [+0.022,+0.438] +0.161 [+0.004,+0.322]
    C2 MOVING    +0.039 [+0.011,+0.068] +0.034 [-0.008,+0.076] -0.112 [-0.160,-0.063]
  ADE@6s: M1-v2a-NR - M1-v2a +0.181 [+0.072,+0.304]; B1-NR - B1 +0.092 [+0.028,+0.155];
  M1-v2a-NR - B1-NR -0.068 [-0.147,+0.016] (CI includes 0 at 6 s).
  Per run: C2 L2@3s STOPPED START MOVING | C2 FG MG | R3 FG all/WSS/non-WSS, MG |
  collision @3 s NoAvg/TemAvg, aa ; yaw
    M1v2a_NR      1.383 9.409 1.782|.234 .645|.243/.362/.118 .519|1.35/0.56;1.37/0.59
    M1v2a_NR_s123 1.561 9.344 1.784|.258 .658|.268/.371/.160 .530|1.72/0.65;1.80/0.68
    M1v2a_NR_s2024 1.295 9.134 1.814|.230 .554|.240/.368/.106 .449|1.76/0.66;1.78/0.68
    B1_NR         1.246 8.955 1.952|.321 .411|.318/.388/.245 .314|1.31/0.54;1.31/0.55
    B1_NR_s123    1.299 9.135 1.866|.311 .450|.311/.400/.218 .346|1.68/0.70;1.70/0.77
    B1_NR_s2024   1.196 9.312 1.898|.311 .472|.321/.400/.239 .350|1.62/0.58;1.74/0.61
    (with route: M1-v2a FG 0.061/0.198/0.098, WSS 0.012/0.290/0.070; B1 FG 0.30-0.32)
  Reading. Without the route, all three M1-v2a seeds start wrongly in WSS (0.362-0.371),
  and the seed spread vanishes (L2@3s sd 0.012 vs 0.056). The route mainly buys M1-v2a the
  standstill decision (STOPPED +0.806, MOVING only +0.039). Words still help without the
  route, but only while moving (MOVING -0.112); at standstill and on START, M1-v2a-NR is
  no better than B1-NR. Collisions: no consistent route effect (1.35-1.76 vs 1.37-1.76).
