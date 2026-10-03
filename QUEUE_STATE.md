# QUEUE_STATE -- REPRODUCTION PHASE R2 (started 2026-10-03 07:00 UTC)

Resume rule after a reboot:
  - R2.3 expert: if results/ar1_expert_A3.pkl is missing, rerun
    `cd Alpamayo/code && CUDA_VISIBLE_DEVICES=1 python ar1/train_expert.py` (tmux expert;
    the A3 KV cache data/ar1_kv_A3_*.pt is already built).
  - R2.4: `tmux new -d -s r24 "bash ar1/run_r24.sh"` -- resumable via
    results/R24_<B1|B2>_{TRAINED,DUMPED,DONE}; status log Alpamayo/r24_status.log.
    (A crashed training restarts from scratch; finetune.py has no mid-run resume here.)

## Done
- [done] R2.0 patch-order confirm + quarantine (c75c7af)
- [done] R2.1 benchmark (c9dd833)
- [done] R2.2 vision cache, 94 GiB, unit test PASS (b8bc570)
- [done] R2.5 map expansion downloaded to nuscenes/maps/{expansion,basemap,prediction}
## Running / queued
- R2.3 expert on frozen A3: tmux expert, GPU 1 (log Alpamayo/ar1_expert_A3.log)
- R2.4 B1 then B2: tmux r24 (waits for the expert), 8 GPUs
## CPU
- R2.6 CoC v2 examples written; full-split distribution running (ar1_r26_coc_all.log)
