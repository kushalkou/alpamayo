# QUEUE_STATE -- R9 (2026-10-08)

Running: tmux r9b = Alpamayo/code/ar1/run_r9b.sh: flow expert on frozen M1-v2a s42
(--labels 2hz), started 10:35 UTC, est. ~7 h / ~54 GPU-h. Log Alpamayo/ar1_r9_M1v2a_expert.log;
status Alpamayo/r9_status.log (ends R9B_DONE). Resume after reboot: cd Alpamayo/code &&
tmux new -d -s r9b "bash ar1/run_r9b.sh" (restarts training from scratch; if training had
finished, set EXTRA="--predict_only --train_log <log>" in the script).
Then: python ar1/r9_b2.py -> R9_REPORT.md Part B (GE1-GE3). R9 Part A done (d15c3eb).
If AUDIT_SIGNOFF.md appears: finish this run, then R7 Part B (CoC) before anything else.
