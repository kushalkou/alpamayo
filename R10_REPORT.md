# R10_REPORT -- flow expert on M1-v2a, 3 seeds

HEADER (all rules fixed and committed BEFORE computing; val read for analysis only; CoC
arm untouched unless AUDIT_SIGNOFF.md appears).
B1 Experts on frozen M1-v2a s123 and s2024: ar1/train_expert_meta.py --tag <tag>
   --labels 2hz, recipe identical to the R9 s42 expert: same code, max 100 epochs, eval
   every 5, patience 4, expert init / data-order / noise seeds as in the recipe (fixed;
   the seed axis is the frozen VLM's training seed), holdout selection (median ADE@6s of
   one fixed-noise draw), inference words = each VLM's own generated words. Queue
   ar1/run_r10b.sh (tmux r10b), s123 then s2024; before each run it checks for
   AUDIT_SIGNOFF.md and stops (R10B_PAUSED_FOR_COC) so R7 Part B runs first.
   Est. ~68 GPU-h per seed (R9 s42: 67.5).
B2 Table 12 analog, same metric set as R9 B2 (ar1/r9_b2.py definitions) per seed and as
   3-seed means (per-sample metrics averaged over seeds, then paired scene bootstrap
   10,000; ADE@6s on n_fut = 12):
   GE1-3 expert mean-of-6 - token ADE@6s: upper < 0 -> "expert improves the long horizon".
   GE2-3 3-seed mean hold rate, mean-of-6 >= token - 0.05 -> "expert holds stops"; else
         "expert drifts at standstill".
   GE3-3 nearest-to-token - token ADE@6s: upper < 0 -> "expert adds value even when
         anchored to the token decision".
   Also STOPPED (C2, adopted after R4) L2@3s mean-of-6 - token with CI (3 seeds).
B3 Stop reason of the A3 and M1 experts from their training logs (patience vs cap).
Audit page path sent to Kushal at the start (this prompt has no Part A).

=========================================================================================
GATE VERDICTS (3 seeds; paired scene bootstrap 10,000; per-sample metrics averaged over
seeds; ADE@6s on n_fut = 12)
  GE1-3 mean-of-6 - token ADE@6s -0.159 [-0.217,-0.099] -> "expert improves the long
        horizon".
  GE2-3 hold rate mean-of-6 0.871 vs token 0.863 (s42 0.929/0.919, s123 0.793/0.787,
        s2024 0.892/0.885) -> "expert holds stops".
  GE3-3 nearest-to-token - token ADE@6s -0.166 [-0.218,-0.113] -> "expert adds value even
        when anchored to the token decision".
  STOPPED (C2) L2@3s mean-of-6 - token +0.300 [+0.141,+0.481]; per seed +0.118 [+0.039,
        +0.230], +0.548 [+0.190,+0.963], +0.232 [+0.110,+0.395]: the expert is WORSE on
        STOPPED in every seed, although its hold rate is not lower.
GPU: s123 67.0 + 0.4 GPU-h; s2024 60.0 + 0.4 GPU-h (total 127.8). AUDIT_SIGNOFF.md
absent throughout; the queue never paused; R7 Part B (CoC) not launched.

B1 Expert training (holdout median ADE@6s of one fixed-noise draw):
  s42   best epoch 100 of 100 (1.425) -> stopped by the CAP, still improving (R9).
  s123  best epoch 95 of 100 (1.479)  -> stopped by the CAP (20 evaluations).
  s2024 best epoch 70, stopped at 90 by PATIENCE (1.562).

B2 Table 12 analog (val; ADE@3s all 5,119; hold = GT-stopped stationary, n 676; Col @3 s
  NoAvg vehicles aa/yaw; C2 L2@3s STOPPED START MOVING; r10_b2.txt)
  run / readout       ADE3  ADE6  minADE6 3/6 spread3/6 hold  Col aa/yaw C2STOP START MOVE
  s42 token           0.884 2.808 -           -         0.919 1.58/1.54  0.292 9.348 1.723
  s42 exp 1 draw      0.824 2.655 0.772/2.524 0.15/0.41 0.929 1.33/1.23  0.410 8.764 1.590
  s42 exp mean6       0.822 2.651 -           -         0.929 1.33/1.23  0.410 8.762 1.585
  s42 exp near-tok    0.820 2.642 -           -         0.928 1.41/1.29  0.394 8.790 1.579
  s123 token          0.935 2.915 -           -         0.787 1.37/1.35  1.005 9.095 1.732
  s123 exp 1 draw     0.914 2.813 0.865/2.693 0.13/0.36 0.793 1.22/1.15  1.554 8.597 1.599
  s123 exp mean6      0.912 2.809 -           -         0.793 1.23/1.13  1.554 8.596 1.595
  s123 exp near-tok   0.908 2.802 -           -         0.793 1.25/1.17  1.497 8.645 1.592
  s2024 token         0.938 2.950 -           -         0.885 1.76/1.72  0.522 9.378 1.809
  s2024 exp 1 draw    0.883 2.747 0.798/2.525 0.24/0.72 0.892 2.17/2.11  0.754 8.695 1.663
  s2024 exp mean6     0.878 2.735 -           -         0.892 2.23/2.19  0.754 8.689 1.651
  s2024 exp near-tok  0.876 2.732 -           -         0.888 2.03/2.01  0.719 8.729 1.647
  3-seed token        0.919 2.891 -           -         0.863 1.57/1.54  0.606 9.273 1.755
  3-seed exp 1 draw   0.874 2.738 -           -         0.871 1.57/1.50  0.906 8.685 1.617
  3-seed exp mean6    0.871 2.732 -           -         0.871 1.60/1.52  0.906 8.682 1.611
  3-seed exp neartok 0.868 2.725 -           -         0.869 1.56/1.49  0.870 8.721 1.606
  Comparison rows (R9, single seed): A3 no-cam token 2.896 ADE6, hold 0.944; A3 expert
  mean6 2.760, hold 0.506, spread 1.94/7.42; M1 (10 Hz) token 3.183, expert mean6 3.046.
  AR1 Table 12 (quoted in R9): AR 0.6811 vs flow 0.6440 minADE6@6.4s.

READING
  - The long-horizon gain replicates: -0.159 ADE@6s at 3 seeds (s42 alone -0.158), better
    on START and MOVING in every seed.
  - The expert stays near-deterministic given the words in every seed (spread 0.13-0.24 m
    at 3 s), so mean of 6, single draw and nearest-to-token are nearly the same.
  - STOPPED is worse in every seed although hold (< 0.5 m in 3 s) is unchanged: on the
    C2 STOPPED samples that the expert does not hold, its drift is larger.
  - Collisions: no consistent effect (3-seed 1.60 vs 1.57 aa); s2024 is worse (2.23 vs
    1.76).

B3 A3 expert: best epoch 20, stopped at 40 by PATIENCE. M1 expert: best epoch 60, stopped
  at 80 by PATIENCE. Only the M1-v2a experts reach the cap (s42, s123).

DEVIATIONS FROM THE PROMPT
  - The prompt has no Part A; the audit page path was sent at the start of R10 instead.
RULES I CHOSE MYSELF
  - Expert init / data-order / noise seeds kept at the recipe's fixed values for all three
    experts; the seed axis is the frozen VLM's seed only.
  - 3-seed gates use per-sample metrics averaged over seeds, then one scene bootstrap.
CLAIMS WITHDRAWN
  - R9: "the M1-v2a expert lowers collisions (1.33 vs 1.58 aa)". This was s42 only; at 3
    seeds it is 1.60 vs 1.57 (s2024 2.23 vs 1.76).
  - R9: "It is worse on STOPPED (+0.118)" was understated: 3-seed +0.300 [+0.141,+0.481].
