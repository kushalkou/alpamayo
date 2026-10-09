# QUEUE_STATE -- R10 (2026-10-09)

Running: tmux r10b = Alpamayo/code/ar1/run_r10b.sh: flow experts on frozen M1-v2a s123
then s2024 (recipe = R9 s42 expert, --labels 2hz), ~8.5 h / ~68 GPU-h each, started 05:29
UTC. Markers results/R10_EXPERT_<tag>_DONE; status Alpamayo/r10_status.log (ends
R10B_DONE). Resume after reboot: cd Alpamayo/code && tmux new -d -s r10b "bash
ar1/run_r10b.sh" (an interrupted run restarts from scratch). Pauses before a run if
AUDIT_SIGNOFF.md exists (then R7 Part B first). Then: python ar1/r10_b2.py -> R10_REPORT.md.
