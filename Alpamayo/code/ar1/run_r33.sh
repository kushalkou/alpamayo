#!/bin/bash
# ar1/run_r33.sh -- R3.3 GPU queue (called by run_r3.sh after R3.2, or by hand).
# M1 = meta-action words + trajectory on the B1 base. Smoke (20 steps) -> seed 42 train ->
# holdout / val dumps -> flow expert on frozen M1 -> seeds 123, 2024 (train + dumps).
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo
R=$L/results
S=$L/r3_status.log
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
die(){ st "R33_FAILED at $1 (see log)"; exit 1; }
if [ ! -f $R/R33_smoke_DONE ]; then
  st "M1 smoke start"
  $TR ar1/finetune_meta.py --tag M1smoke --seed 42 --max_steps 20 > $L/ar1_r33_smoke.log 2>&1 || die "M1 smoke"
  st "M1 smoke done"; touch $R/R33_smoke_DONE
fi
run_m(){
  local T=$1 SEED=$2
  if [ ! -f $R/R33_${T}_TRAINED ]; then
    st "$T train start"
    $TR ar1/finetune_meta.py --tag $T --seed $SEED > $L/ar1_r33_$T.log 2>&1 || die "$T train"
    st "$T train done"; touch $R/R33_${T}_TRAINED
  fi
  if [ ! -f $R/R33_${T}_DUMPED ]; then
    $TR ar1/finetune_meta.py --tag $T --dump holdout,val > $L/ar1_r33_${T}_dump.log 2>&1 || die "$T dump"
    st "$T dumps done"; touch $R/R33_${T}_DUMPED
  fi
}
run_m M1 42
if [ ! -f $R/R33_M1_EXPERT ]; then
  st "M1 expert start"
  $TR ar1/train_expert_meta.py --tag M1 > $L/ar1_r33_M1_expert.log 2>&1 || die "M1 expert"
  st "M1 expert done"; touch $R/R33_M1_EXPERT
fi
st "R33_SEED42_DONE"
run_m M1_s123 123
run_m M1_s2024 2024
st "R33_ALL_DONE"
