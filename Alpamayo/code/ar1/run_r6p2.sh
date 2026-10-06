#!/bin/bash
# ar1/run_r6p2.sh -- R6 Part 2: M1-v2a seeds 123, 2024 (identical recipe) and no-camera M1-v2a s42.
# Resumable via results/R6_<tag>_{TRAINED,DUMPED}; status Alpamayo/r6_status.log.
# Resume after reboot: cd Alpamayo/code && tmux new -d -s r6 "bash ar1/run_r6p2.sh"
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo; R=$L/results; S=$L/r6_status.log
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
run(){ # tag seed extra
  local T=$1 SEED=$2; shift 2
  if [ ! -f $R/R6_${T}_TRAINED ]; then
    st "$T train start"
    $TR ar1/finetune_meta.py --tag $T --seed $SEED --labels 2hz "$@" > $L/ar1_r6_$T.log 2>&1 || { st "R6_FAILED $T train"; exit 1; }
    st "$T train done"; touch $R/R6_${T}_TRAINED
  fi
  if [ ! -f $R/R6_${T}_DUMPED ]; then
    $TR ar1/finetune_meta.py --tag $T --labels 2hz "$@" --dump holdout,val > $L/ar1_r6_${T}_dump.log 2>&1 || { st "R6_FAILED $T dump"; exit 1; }
    st "$T dumps done"; touch $R/R6_${T}_DUMPED
  fi
}
run M1v2a_nocam 42 --no_vision
run M1v2a_s123 123
run M1v2a_s2024 2024
st "R6_P2_DONE"
