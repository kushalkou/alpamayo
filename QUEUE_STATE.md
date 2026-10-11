# QUEUE_STATE -- R11 (2026-10-11)

Running: tmux r11b = Alpamayo/code/ar1/run_r11b.sh: CoC arm seed 42 (PROVISIONAL) train
(~3.6 h) then dumps holdout+val gen, val none / shuf / gt (~1.6 h); ~42 GPU-h. Markers
results/R11_COC_<step>_DONE; status Alpamayo/r11_status.log (ends R11B_DONE). Resume after
reboot: cd Alpamayo/code && tmux new -d -s r11b "bash ar1/run_r11b.sh".
Seeds 123 / 2024 NOT launched until AUDIT_SIGNOFF.md exists. Audit page audit/coc_audit.html.
