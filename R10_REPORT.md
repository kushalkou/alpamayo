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
