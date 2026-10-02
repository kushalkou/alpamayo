# QUEUE_STATE -- NEXT QUEUE v2 (started 2026-10-01 04:45 UTC)

Resume rule after a reboot: tmux sessions die. Relaunch `tmux new -s q2 'bash w1/run_q2.sh'`
from Alpamayo/code; every step writes results/Q2_<step>_DONE (training: Q2_<tag>_TRAINED)
and is skipped when present. Status: Alpamayo/q2_status.log. CPU C7:
`python w1/c7_bottleneck.py` (cached in results/w1_c7.pkl, recomputes only what is missing).
Previous queue: COMPLETE 2026-10-01 03:40 UTC (see git history, 32e40a5).

## GPU (tmux q2, w1/run_q2.sh), one 8-GPU job at a time
- [done] G5: LEN = 10 (G5 best -0.54% vs A3; G5 - A3 val n.s.)
- [done] G6 (i) G6C_s123, G6C_s2024; (ii) G6M_s42, G6M_s123, G6M_s2024 (T9)
- [done] G8 V8a, V8b (V8b val macro-F1 0.610; verdict: cameras add no decision information)
- [done] G7 (T11)
- [running] G9 A0 seed 123 (w1/q2_g9.sh)
## CPU
- [done] C7 (all predictors); C8 amendment (label ambiguity, per-class V8b vs V8a)
            once results/w1_intent_V8{a,b}.pkl exist
- [todo] T9/T10/T11 in REVIEW_RESULTS.md; f6/f7 rows; f8
