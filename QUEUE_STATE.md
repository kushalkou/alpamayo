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
- [pending] G2 gate (gate35.py, both runs) -> G3 recipe decision -> G4 Stage A1, A2, A3, A0

## Track C (CPU)
- [pending] C1 mini ladder (w1/ladder.py)
- [pending] C2 figures (Alpamayo/viz/review/)
- [pending] C3 REVIEW_RESULTS.md
