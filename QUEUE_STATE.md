# QUEUE_STATE -- R8 (2026-10-07)

Running: tmux r8b = Alpamayo/code/ar1/run_r8b.sh: M1v2a_NR_s123 DONE 23:08 -> B1_NR_s123 DONE 02:05 ->
M1v2a_NR_s2024 -> B1_NR_s2024 (~3.2-3.8 h each, ~14 h). Markers results/R8_<tag>_
{TRAINED,DUMPED}; resume after reboot: cd Alpamayo/code && tmux new -d -s r8b "bash
ar1/run_r8b.sh". Status Alpamayo/r8_status.log (ends R8B_DONE). Before each run the queue
stops with R8B_PAUSED_FOR_COC if AUDIT_SIGNOFF.md exists (then R7 Part B, then resume).
Then: python ar1/r8b_eval.py -> R8_REPORT.md Part B. R8 Part A done (bd8869c).
