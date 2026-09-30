#!/usr/bin/env bash
# QUEUE G4: Stage A, one run at a time: A1, A2, A3, A0 (seed 42, no visual tokens,
# official split for A1-A3, selection on the 400-sample holdout subset).
# Usage: RECIPE=fix|retry bash w1/run_stageA.sh
# After each run's evaluation the queue WAITS for results/STAGEA_<run>_REPORTED (written by
# the operator once the report section is committed) before launching the next run.
# Pre-registered early stop: if A1 fails the val input-shuffle test (shuffled ADE within 5%
# of unshuffled), A2 still runs, A3 and A0 are skipped.
set -u
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo
R=$L/results
S=$L/stageA_status.log
cd $L/code
st(){ echo "[$(date -u +%H:%M:%S)] $*" >> $S; }
case "${RECIPE:?set RECIPE=fix or retry}" in
  fix)   RX="--fix"; DX="--fix" ;;
  retry) RX="--lora_dropout 0 --min_lr_ratio 1.0"; DX="" ;;
esac
BASE="--no_vision --zero_vision --seed 42 --epochs 10 --patience 5 --batch_size 3 --grad_accum_steps 1"
st "Stage A start, RECIPE=$RECIPE"

wait_report(){ st "$1 evaluated; waiting for $R/STAGEA_$1_REPORTED"
  while [ ! -f $R/STAGEA_$1_REPORTED ]; do sleep 60; done; st "$1 reported"; }

run_w1(){ # $1 tag, $2 extra train flags, $3 dump flags, $4 flips
  local T=$1; local CK=$L/models/checkpoints/_w1_$T
  if [ ! -f $R/STAGEA_${T}_TRAINED ]; then
    st "$T train start"
    $TR w1/finetune_w1.py --tag $T $2 $RX $BASE > $L/w1_stageA_$T.log 2>&1
    st "$T train exit $?"; touch $R/STAGEA_${T}_TRAINED
  fi
  $TR w1/dump_w1.py --tag $T --ckpt $CK/alpamayo_best.pt $3 --no_vision $DX --splits holdout > $L/w1_stageA_${T}_dump.log 2>&1
  $TR w1/dump_w1.py --tag $T --ckpt $CK/alpamayo_best.pt $3 --no_vision $DX --splits val --flips $4 >> $L/w1_stageA_${T}_dump.log 2>&1
  $TR w1/dump_w1.py --tag ${T}_shuf --ckpt $CK/alpamayo_best.pt $3 --no_vision $DX --splits val --shuffle_inputs >> $L/w1_stageA_${T}_dump.log 2>&1
  st "$T dumps done"
}

# ---- A1 ----
[ -f $R/STAGEA_A1_REPORTED ] || { run_w1 A1 "--cmd" "--cmd" 0
  $PY w1/gate_a.py A1 > $L/w1_gateA_A1.txt 2>&1; wait_report A1; }
SKIP=0; grep -q "A1: .*FAILS shuffle test" $L/w1_gateA_A1.txt && SKIP=1 && st "A1 FAILS shuffle test -> skip A3, A0"
# ---- A2 ----
[ -f $R/STAGEA_A2_REPORTED ] || { run_w1 A2 "--meta --meta_flip 0.1" "--meta" 0,0.1,0.2,0.4
  $PY w1/gate_a.py A1 A2 > $L/w1_gateA_A2.txt 2>&1; wait_report A2; }
if [ $SKIP = 0 ]; then
  # ---- A3 ----
  [ -f $R/STAGEA_A3_REPORTED ] || { run_w1 A3 "--cmd --turn_weighted" "--cmd" 0
    $PY w1/gate_a.py A1 A2 A3 > $L/w1_gateA_A3.txt 2>&1; wait_report A3; }
  # ---- A0 (custom split, OLD recipe, causal ego, visual tokens REMOVED vs zeroed) ----
  if [ ! -f $R/STAGEA_A0_REPORTED ]; then
    if [ ! -f $R/STAGEA_A0_TRAINED ]; then
      st "A0 train start"
      $TR leak/finetune_causal.py --tag A0_ego_novis_s42 --no_vision --zero_vision --turn_weighted \
          --seed 42 --epochs 10 --patience 5 --batch_size 3 --grad_accum_steps 1 > $L/w1_stageA_A0.log 2>&1
      st "A0 train exit $?"; touch $R/STAGEA_A0_TRAINED
    fi
    for s in val test; do
      $TR leak/dump_ce.py --split $s --tag A0_causal_ego_novis --models A0_causal_ego_novis --causal \
          >> $L/w1_stageA_A0_dump.log 2>&1
    done
    $PY leak/a0_compare.py > $L/w1_gateA_A0.txt 2>&1; wait_report A0
  fi
fi
st "STAGE_A_DONE"
