# QUEUE_STATE -- REPRODUCTION PHASE R3 (started 2026-10-04 05:13 UTC)

Resume rule after a reboot: `cd Alpamayo/code && tmux new -d -s r3 "bash ar1/run_r3.sh"`.
It is resumable through results/R3_*_{TRAINED,DONE} and results/R33_*_{DONE,TRAINED,DUMPED,
EXPERT}; it calls ar1/run_r33.sh at the end. A training interrupted mid-run restarts from
scratch. Status log: Alpamayo/r3_status.log.

## GPU queue (tmux r3)
- [done] R3.0b train-subset dumps (A3_trsub, B1_trsub)
- [running] R3.2 B1_s123, then B1_s2024 (train + holdout / val dumps)
- [queued] R3.3 M1 smoke (20 steps) -> M1 seed 42 -> dumps -> M1 expert -> M1 s123, s2024
## CPU
- [done] R3.0 a/b (13c7f2f), R3.1 (13c7f2f), R3.4 a-d (4b4ebc7)
- after M1 seed 42 + expert: python ar1/r33_eval.py M1 -> gate report
