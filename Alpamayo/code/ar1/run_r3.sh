#!/bin/bash
# ar1/run_r3.sh -- R3 GPU queue. Resumable via results/R3_<step>_DONE; status Alpamayo/r3_status.log.
#  1. R3.0b train-subset dumps (1,000 random n_fut=12 train samples, seed 0): A3 (no cam s42), B1
#  2. R3.2 B1 seeds 123, 2024 (A3 recipe, cached 3 cams t0): train + holdout/val dumps
#  3. R3.3 M1 (meta-action text + traj on B1 base) seed 42 -- run by ar1/run_r33.sh when ready
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo
R=$L/results
S=$L/r3_status.log
CKD=$L/models/checkpoints
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
die(){ st "R3_FAILED at $1 (see log)"; exit 1; }
REC="--fix --batch_size 3 --grad_accum_steps 1 --patience 5"
if [ ! -f $R/R3_trsub_DONE ]; then
  st "train-subset dumps start"
  $TR w1/dump_w1.py --tag A3_trsub --ckpt $CKD/_w1_A3/alpamayo_best.pt --cmd --no_vision --fix \
      --splits train --rand_subset 1000 > $L/ar1_r30_trsub.log 2>&1 || die "A3 trsub"
  $TR w1/dump_w1.py --tag B1_trsub --ckpt $CKD/_w1_B1/alpamayo_best.pt --cmd --vcache t0 --fix \
      --splits train --rand_subset 1000 >> $L/ar1_r30_trsub.log 2>&1 || die "B1 trsub"
  st "train-subset dumps done"; touch $R/R3_trsub_DONE
fi
run_b(){
  local T=$1 SEED=$2
  [ -f $R/R3_${T}_DONE ] && return 0
  if [ ! -f $R/R3_${T}_TRAINED ]; then
    st "$T train start"
    $TR w1/finetune_w1.py --tag $T --cmd --turn_weighted --seed $SEED --vcache t0 $REC --epochs 10 \
        > $L/ar1_r3_$T.log 2>&1 || die "$T train"
    st "$T train done"; touch $R/R3_${T}_TRAINED
  fi
  $TR w1/dump_w1.py --tag $T --cmd --vcache t0 --fix --splits holdout > $L/ar1_r3_${T}_dump.log 2>&1 || die "$T dump ho"
  $TR w1/dump_w1.py --tag $T --cmd --vcache t0 --fix --splits val >> $L/ar1_r3_${T}_dump.log 2>&1 || die "$T dump val"
  st "$T dumps done"; touch $R/R3_${T}_DONE
}
run_b B1_s123 123
run_b B1_s2024 2024
st "R32_DONE"
[ -x ar1/run_r33.sh ] && bash ar1/run_r33.sh
