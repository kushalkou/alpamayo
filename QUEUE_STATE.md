# QUEUE_STATE -- R7 (2026-10-07)

Running: tmux r7a = Alpamayo/code/ar1/run_r7a.sh: B1_NR s42 (train + holdout/val dumps)
-> M1v2a_NR s42 (train + dumps) -> Ego-MLP-NR (CPU). Ends R7A_DONE.
Waiting: tmux r7c3 = ar1/run_r7c3.sh, starts on R7A_DONE: C3 blank-image control (B1 check
200, B1 real/cams/blank, M1v2a blank; batched; cap 2 GPU-h; ledger Alpamayo/r7_gpu_ledger.txt).
Ends R7C3_DONE.
Resume after reboot: cd Alpamayo/code && tmux new -d -s r7a "bash ar1/run_r7a.sh" &&
tmux new -d -s r7c3 "bash ar1/run_r7c3.sh" (markers results/R7_*; done steps skip).
Status: Alpamayo/r7_status.log. Then: ar1/r7a_eval.py, ar1/r7_c12.py (rerun with R7 runs),
ar1/r7_c3_eval.py -> R7A_REPORT.md. Part B (CoC) only if AUDIT_SIGNOFF.md exists after C3.
