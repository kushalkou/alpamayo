# QUEUE_STATE -- long autonomous queue (started 2026-09-30 13:14 UTC)

Resume rule after a reboot: tmux sessions die. Re-launch the incomplete step below; every
step's outputs are files, so completed steps are skipped.

## Track G (GPU)
- [done] w1: 3.5a retry -- GATE FAIL (slot-0 0.598; median 0.447; shuffle ok) (w1/run_retry.sh) -> Alpamayo/w1_overfit256_retry.log,
            dumps -> Alpamayo/w1_retry_dump.log (RETRY_QUEUE_DONE marker)
            resume: bash Alpamayo/code/w1/run_retry.sh (tmux w1)
- [running] G1 fix run (started 15:29 UTC) + dumps + diagnostic-c probe: Alpamayo/code/w1/run_gq.sh (tmux fix),
            status Alpamayo/gq_status.log (GQ_DONE marker). No GPU was free alongside w1
            (8 x ~30/32 GB), so it waits for w1.
- [pending] G2 gate (w1/gate35.py on retry_* and fix_* dumps) -> G3 recipe decision
- [prepared] G4 Stage A: RECIPE=fix|retry bash Alpamayo/code/w1/run_stageA.sh (tmux stageA).
            Resumable: skips runs with results/STAGEA_<run>_TRAINED / _REPORTED markers.
            After each run it waits for results/STAGEA_<run>_REPORTED (touch after the
            report section is committed). Status: Alpamayo/stageA_status.log

## Track C (CPU)
- [done] C1 mini ladder (e0f7516)
- [partial] C2 figures f1-f7 done (f7 without the A2 line; rerun `python w1/figs.py f5 f7` after gating/A2)
- [partial] C3 REVIEW_RESULTS.md: T1-T6 written; T5 retry/fix and T7 Stage A pending
