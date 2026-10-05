#!/bin/bash
# ar1/run_r5b2.sh -- R5 B2: M1-v2a seed 42 (M1 recipe; meta-action labels from 2 Hz GT, the ONLY
# change). Resumable via results/R5_M1v2a_{TRAINED,DUMPED}; status Alpamayo/r5_status.log.
# Resume after reboot: tmux new -d -s r5 "bash ar1/run_r5b2.sh"
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo; R=$L/results; S=$L/r5_status.log
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
if [ ! -f $R/R5_M1v2a_TRAINED ]; then
  st "M1v2a train start"
  $TR ar1/finetune_meta.py --tag M1v2a --seed 42 --labels 2hz > $L/ar1_r5_M1v2a.log 2>&1 || { st "R5_FAILED train"; exit 1; }
  st "M1v2a train done"; touch $R/R5_M1v2a_TRAINED
fi
if [ ! -f $R/R5_M1v2a_DUMPED ]; then
  $TR ar1/finetune_meta.py --tag M1v2a --labels 2hz --dump holdout,val > $L/ar1_r5_M1v2a_dump.log 2>&1 || { st "R5_FAILED dump"; exit 1; }
  st "M1v2a dumps done"; touch $R/R5_M1v2a_DUMPED
fi
st "R5_B2_DONE"
