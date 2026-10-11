# QUEUE_STATE -- R12 (2026-10-11)

Running: tmux r12b = Alpamayo/code/ar1/run_r12b.sh: CoC seed 42 on v3 traces (PROVISIONAL),
train (~3.6 h, started 18:29 UTC) then dumps holdout+val gen, val none / shuf / gt (~1.6 h).
Markers results/R12_COC3_<step>_DONE; status Alpamayo/r12_status.log (ends R12B_DONE).
Resume after reboot: cd Alpamayo/code && tmux new -d -s r12b "bash ar1/run_r12b.sh".
Then: python ar1/r11b_eval.py COC3 -> R12_REPORT Part B. Seeds 123 / 2024 only after
AUDIT_SIGNOFF.md. Short re-audit page audit/coc_audit_v3_short.html sent to Kushal.
