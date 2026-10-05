# QUEUE_STATE -- R5 (started 2026-10-05)

Running: tmux r5 = Alpamayo/code/ar1/run_r5b2.sh (M1-v2a seed 42: train, then holdout /
val dumps). Resume after reboot: `cd Alpamayo/code && tmux new -d -s r5 "bash
ar1/run_r5b2.sh"` (markers results/R5_M1v2a_{TRAINED,DUMPED}; a training interrupted
mid-run restarts from scratch). Status: Alpamayo/r5_status.log. Launch nothing beyond
B2 seed 42. Track A is CPU, plus at most 6 GPU-h of inference after B2.
