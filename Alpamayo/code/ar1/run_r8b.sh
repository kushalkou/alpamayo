#!/bin/bash
# ar1/run_r8b.sh -- R8 Part B: no-route seeds (alternating pairs). Recipes identical to the
# s42 NR runs (ar1/run_r7a.sh). Before each run: if AUDIT_SIGNOFF.md exists, stop with
# R8B_PAUSED_FOR_COC (R7 Part B runs, then this queue is restarted).
# Resumable via results/R8_<tag>_{TRAINED,DUMPED}; status Alpamayo/r8_status.log.
# Resume after reboot: cd Alpamayo/code && tmux new -d -s r8b "bash ar1/run_r8b.sh"
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo; R=$L/results; S=$L/r8_status.log
SIGN=/home/dgx1user/Alpamayo-Kushal/AUDIT_SIGNOFF.md
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
REC="--fix --batch_size 3 --grad_accum_steps 1 --patience 5"
gate(){ if [ -f $SIGN ] && [ ! -f $R/R7B_COC_DONE ]; then st "R8B_PAUSED_FOR_COC (AUDIT_SIGNOFF.md present)"; exit 0; fi; }
m1(){ local T=$1 SEED=$2
  [ -f $R/R8_${T}_DUMPED ] && return 0
  gate
  if [ ! -f $R/R8_${T}_TRAINED ]; then
    st "$T train start"
    $TR ar1/finetune_meta.py --tag $T --seed $SEED --labels 2hz --no_cmd > $L/ar1_r8_$T.log 2>&1 || { st "R8_FAILED $T train"; exit 1; }
    st "$T train done"; touch $R/R8_${T}_TRAINED
  fi
  $TR ar1/finetune_meta.py --tag $T --labels 2hz --no_cmd --dump holdout,val > $L/ar1_r8_${T}_dump.log 2>&1 || { st "R8_FAILED $T dump"; exit 1; }
  st "$T dumps done"; touch $R/R8_${T}_DUMPED
}
b1(){ local T=$1 SEED=$2
  [ -f $R/R8_${T}_DUMPED ] && return 0
  gate
  if [ ! -f $R/R8_${T}_TRAINED ]; then
    st "$T train start"
    $TR w1/finetune_w1.py --tag $T --turn_weighted --seed $SEED --vcache t0 $REC --epochs 10 > $L/ar1_r8_$T.log 2>&1 || { st "R8_FAILED $T train"; exit 1; }
    st "$T train done"; touch $R/R8_${T}_TRAINED
  fi
  $TR w1/dump_w1.py --tag $T --vcache t0 --fix --splits holdout > $L/ar1_r8_${T}_dump.log 2>&1 || { st "R8_FAILED $T dump"; exit 1; }
  $TR w1/dump_w1.py --tag $T --vcache t0 --fix --splits val >> $L/ar1_r8_${T}_dump.log 2>&1 || { st "R8_FAILED $T dump"; exit 1; }
  st "$T dumps done"; touch $R/R8_${T}_DUMPED
}
m1 M1v2a_NR_s123 123
b1 B1_NR_s123 123
m1 M1v2a_NR_s2024 2024
b1 B1_NR_s2024 2024
st "R8B_DONE"
