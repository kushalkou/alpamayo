# QUEUE_STATE -- long autonomous queue (started 2026-09-30 13:14 UTC)

Resume rule after a reboot: tmux sessions die. Re-launch the incomplete step below; every
step's outputs are files, so completed steps are skipped.

## Track G (GPU)
- [running] w1: 3.5a retry (w1/run_retry.sh) -> Alpamayo/w1_overfit256_retry.log,
            dumps -> Alpamayo/w1_retry_dump.log (RETRY_QUEUE_DONE marker)
            resume: bash Alpamayo/code/w1/run_retry.sh (tmux w1)
- [queued]  G1 fix run + dumps + diagnostic-c probe: Alpamayo/code/w1/run_gq.sh (tmux fix),
            status Alpamayo/gq_status.log (GQ_DONE marker). No GPU was free alongside w1
            (8 x ~30/32 GB), so it waits for w1.
- [pending] G2 gate (w1/gate35.py on retry_* and fix_* dumps) -> G3 recipe decision
- [prepared] G4 Stage A: RECIPE=fix|retry bash Alpamayo/code/w1/run_stageA.sh (tmux stageA).
            Resumable: skips runs with results/STAGEA_<run>_TRAINED / _REPORTED markers.
            After each run it waits for results/STAGEA_<run>_REPORTED (touch after the
            report section is committed). Status: Alpamayo/stageA_status.log

## Track C (CPU)
- [running] C1 mini ladder (w1/ladder.py, tmux ladder, log Alpamayo/w1_ladder.log); resume: rerun it
- [partial] C2 figures: f1-f5 done (w1/figs.py); f6,f7 wait for C1 ladder (+A2)
- [pending] C3 REVIEW_RESULTS.md
