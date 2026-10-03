#!/bin/bash
# ar1/run_r24.sh -- R2.4 GPU queue: B1 (3 front cams, t0) then B2 (3 front cams x 4 keyframes).
# A3 recipe (fix, cmd [P], turn-weighted, plain CE, batch 3 x 8 GPUs, patience 5), 10 epochs,
# seed 42, cached native-order vision (ar1/vc_patch.py). Resumable: results/R24_<tag>_<step>.
# Waits for the R2.3 expert (tmux 'expert') to finish before using the GPUs.
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo
R=$L/results
S=$L/r24_status.log
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
die(){ st "R24_FAILED at $1 (see log)"; exit 1; }
REC="--fix --batch_size 3 --grad_accum_steps 1 --patience 5"
while pgrep -f ar1/train_expert.py > /dev/null; do sleep 60; done
run_b(){
  local T=$1 M=$2 CK=$L/models/checkpoints/_w1_$1
  [ -f $R/R24_${T}_DONE ] && return 0
  if [ ! -f $R/R24_${T}_TRAINED ]; then
    st "$T train start (vcache $M)"
    $TR w1/finetune_w1.py --tag $T --cmd --turn_weighted --seed 42 --vcache $M $REC --epochs 10 \
        > $L/ar1_r24_$T.log 2>&1 || die "$T train"
    st "$T train done"; touch $R/R24_${T}_TRAINED
  fi
  if [ ! -f $R/R24_${T}_DUMPED ]; then
    $TR w1/dump_w1.py --tag $T --cmd --vcache $M --fix --splits holdout > $L/ar1_r24_${T}_dump.log 2>&1 || die "$T dump ho"
    $TR w1/dump_w1.py --tag $T --cmd --vcache $M --fix --splits val >> $L/ar1_r24_${T}_dump.log 2>&1 || die "$T dump val"
    touch $R/R24_${T}_DUMPED
  fi
  $TR w1/dump_w1.py --tag ${T}_camshuf --ckpt $CK/alpamayo_best.pt --cmd --vcache $M --fix --splits val \
      --shuffle_cams >> $L/ar1_r24_${T}_dump.log 2>&1 || die "$T dump camshuf"
  st "$T dumps done"; touch $R/R24_${T}_DONE
}
run_b B1 t0
run_b B2 hist
st "R24_GPU_DONE"
