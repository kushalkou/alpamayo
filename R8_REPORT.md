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
