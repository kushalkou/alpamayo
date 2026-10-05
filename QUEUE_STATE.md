# QUEUE_STATE -- R5 (2026-10-05)

Running: tmux r5 = Alpamayo/code/ar1/run_r5b2.sh (M1-v2a seed 42 train -> holdout / val
dumps). Resume: `cd Alpamayo/code && tmux new -d -s r5 "bash ar1/run_r5b2.sh"`.
Status: Alpamayo/r5_status.log. Launch nothing beyond B2 seed 42.
Left to do (CPU, plus at most 6 GPU-h of inference): B2 eval (GB2-GB5, strata, comfort),
A4 Li et al. baselines, A5 expert re-eval (yaw-aware collision), C1 comfort, C2.
Rules for all of these are already fixed in the R5_REPORT.md header.
