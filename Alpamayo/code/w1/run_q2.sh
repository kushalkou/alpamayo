#!/usr/bin/env bash
# NEXT QUEUE v2 (GPU part). One 8-GPU job at a time, resumable: every step writes a
# results/Q2_<step>_DONE marker and is skipped when the marker exists. A failing step
# stops the queue (Q2_FAILED in the status log); fix, then relaunch -- done steps skip.
#   tmux new -s q2 'bash w1/run_q2.sh'
# Order (pre-registered): G5 -> length rule -> G6 -> V8a -> V8b -> G7 -> G9.
# C7 (CPU) is run by the operator as soon as V8b's predictions exist, while G7 trains.
set -u
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo
R=$L/results
S=$L/q2_status.log
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
die(){ st "Q2_FAILED at $1 (see log)"; exit 1; }
REC="--fix --batch_size 3 --grad_accum_steps 1 --patience 5"

# trajectory run: $1 tag, $2 train flags, $3 dump flags, $4 epochs, $5 val flips,
# $6 vision (novis|vis)
run_traj(){
  local T=$1 CK=$L/models/checkpoints/_w1_$1 V="" Z=""
  [ "$6" = novis ] && { V="--no_vision"; Z="--zero_vision"; }
  [ -f $R/Q2_${T}_DONE ] && return 0
  if [ ! -f $R/Q2_${T}_TRAINED ]; then
    st "$T train start"
    $TR w1/finetune_w1.py --tag $T $2 $V $Z $REC --epochs $4 > $L/w1_q2_$T.log 2>&1 || die "$T train"
    st "$T train done"; touch $R/Q2_${T}_TRAINED
  fi
  $TR w1/dump_w1.py --tag $T $3 $V --fix --splits holdout > $L/w1_q2_${T}_dump.log 2>&1 || die "$T dump ho"
  $TR w1/dump_w1.py --tag $T $3 $V --fix --splits val --flips $5 >> $L/w1_q2_${T}_dump.log 2>&1 || die "$T dump val"
  if [ "$6" = novis ]; then
    $TR w1/dump_w1.py --tag ${T}_shuf --ckpt $CK/alpamayo_best.pt $3 $V --fix --splits val \
        --shuffle_inputs >> $L/w1_q2_${T}_dump.log 2>&1 || die "$T dump shuf"
  else
    $TR w1/dump_w1.py --tag ${T}_camshuf --ckpt $CK/alpamayo_best.pt $3 --fix --splits val \
        --shuffle_cams >> $L/w1_q2_${T}_dump.log 2>&1 || die "$T dump camshuf"
  fi
  st "$T dumps done"; touch $R/Q2_${T}_DONE
}

st "Q2 start"
# ---- G5: training length (A3 recipe, seed 42, 30 epochs, patience 5) ----
run_traj G5 "--cmd --turn_weighted --seed 42" "--cmd" 30 0 novis
[ -f $R/Q2_LEN ] || { $PY w1/q2_len.py > $L/w1_q2_len.txt 2>&1 || die "G5 length rule"; }
LEN=$(cat $R/Q2_LEN); st "length rule -> $LEN epochs"
# ---- G6: seeds (ego-only). (i) ego + cmd, A3 recipe; (ii) A2 design + turn weighting.
# (i) seed 42 = G5 if LEN=30, else A3. (ii) has no seed-42 run yet, so it is run here.
for s in 123 2024; do
  run_traj G6C_s$s "--cmd --turn_weighted --seed $s" "--cmd" $LEN 0 novis
done
for s in 42 123 2024; do
  run_traj G6M_s$s "--meta --meta_flip 0.1 --turn_weighted --seed $s" "--meta" $LEN 0,0.1,0.2,0.4 novis
done
# ---- G8: VLA intent predictor (lon, 4 classes), seed 42 ----
for v in V8a V8b; do
  if [ ! -f $R/Q2_${v}_DONE ]; then
    VX=""; [ $v = V8a ] && VX="--no_vision"
    st "$v start"
    $TR w1/finetune_intent.py --tag $v $VX --seed 42 --epochs 10 --patience 5 \
        > $L/w1_q2_$v.log 2>&1 || die "$v"
    st "$v done"; touch $R/Q2_${v}_DONE
  fi
done
# ---- G7: vision end-to-end trajectory (6 cameras + ego + cmd, A3 recipe) ----
run_traj G7 "--cmd --turn_weighted --seed 42" "--cmd" $LEN 0 vis
# ---- G9: A0 second seed (idle GPUs) ----
[ -f $R/Q2_G9_DONE ] || { bash w1/q2_g9.sh || die G9; touch $R/Q2_G9_DONE; }
st "Q2_GPU_DONE"
