# QUEUE_STATE -- R7 Part A (2026-10-07)

Running: tmux r7a = Alpamayo/code/ar1/run_r7a.sh: B1_NR s42 (train + holdout/val dumps)
-> M1v2a_NR s42 (train + dumps) -> Ego-MLP-NR (CPU). ~26-32 GPU-h per arm.
Resumable via results/R7_<tag>_{TRAINED,DUMPED}, results/R7_egomlp_NR_DONE; resume after
reboot: cd Alpamayo/code && tmux new -d -s r7a "bash ar1/run_r7a.sh".
Status: Alpamayo/r7_status.log (ends R7A_DONE). Then: python ar1/r7a_eval.py -> R7A_REPORT.
Part B (CoC arm) only if AUDIT_SIGNOFF.md exists in the repo root when Part A finishes.
