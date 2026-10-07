#!/bin/bash
# ar1/run_r7a.sh -- R7 Part A: B1-NR s42, M1-v2a-NR s42 (nav command token removed), then
# Ego-MLP-NR (CPU). Resumable via results/R7_<tag>_{TRAINED,DUMPED}; status Alpamayo/r7_status.log.
# Resume after reboot: cd Alpamayo/code && tmux new -d -s r7a "bash ar1/run_r7a.sh"
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo; R=$L/results; S=$L/r7_status.log
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
REC="--fix --batch_size 3 --grad_accum_steps 1 --patience 5"
T=B1_NR
if [ ! -f $R/R7_${T}_TRAINED ]; then
  st "$T train start"
  $TR w1/finetune_w1.py --tag $T --turn_weighted --seed 42 --vcache t0 $REC --epochs 10 \
      > $L/ar1_r7_$T.log 2>&1 || { st "R7_FAILED $T train"; exit 1; }
  st "$T train done"; touch $R/R7_${T}_TRAINED
fi
if [ ! -f $R/R7_${T}_DUMPED ]; then
  $TR w1/dump_w1.py --tag $T --vcache t0 --fix --splits holdout > $L/ar1_r7_${T}_dump.log 2>&1 || { st "R7_FAILED $T dump"; exit 1; }
  $TR w1/dump_w1.py --tag $T --vcache t0 --fix --splits val >> $L/ar1_r7_${T}_dump.log 2>&1 || { st "R7_FAILED $T dump"; exit 1; }
  st "$T dumps done"; touch $R/R7_${T}_DUMPED
fi
T=M1v2a_NR
if [ ! -f $R/R7_${T}_TRAINED ]; then
  st "$T train start"
  $TR ar1/finetune_meta.py --tag $T --seed 42 --labels 2hz --no_cmd > $L/ar1_r7_$T.log 2>&1 || { st "R7_FAILED $T train"; exit 1; }
  st "$T train done"; touch $R/R7_${T}_TRAINED
fi
if [ ! -f $R/R7_${T}_DUMPED ]; then
  $TR ar1/finetune_meta.py --tag $T --labels 2hz --no_cmd --dump holdout,val > $L/ar1_r7_${T}_dump.log 2>&1 || { st "R7_FAILED $T dump"; exit 1; }
  st "$T dumps done"; touch $R/R7_${T}_DUMPED
fi
if [ ! -f $R/R7_egomlp_NR_DONE ]; then
  CUDA_VISIBLE_DEVICES= $PY ar1/r7_egomlp_nr.py > $L/ar1_r7_egomlp_nr.log 2>&1 || { st "R7_FAILED egomlp"; exit 1; }
  st "egomlp_NR done"; touch $R/R7_egomlp_NR_DONE
fi
st "R7A_DONE"
