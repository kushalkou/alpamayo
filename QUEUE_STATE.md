# QUEUE_STATE -- R13 (2026-10-12)

Running: tmux r13b = Alpamayo/code/ar1/run_r13b.sh (started 07:36 UTC): CoC v3 seed 123
(train ~4.3 h + dumps gen / shuf / gt ~2.3 h), seed 2024 (same), then CoC-AUX seed 42
(train ~4 h + dumps none / gen ~3 h); ~22 h total. Markers results/R13_<tag>_<step>_DONE;
status Alpamayo/r13_status.log (ends R13B_DONE). Resume after reboot: cd Alpamayo/code &&
tmux new -d -s r13b "bash ar1/run_r13b.sh". Then: python ar1/r13b_eval.py -> R13_REPORT.
