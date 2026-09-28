#!/usr/bin/env bash
# OVERNIGHT2 GPU queue: item-5 free-running CE dump, then item-6 causal retrains + dumps.
set -u
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo
cd /home/dgx1user/Alpamayo-Kushal/Alpamayo/code
st(){ echo "[$(date -u +%H:%M:%S)] $*" >> $L/overnight2_gpu_status.log; }

st "5: free-running CE dump y1_ego,y1_full test"
$TR leak/dump_ce.py --split test --tag ce --models y1_ego,y1_full > $L/overnight2_dumpce.log 2>&1
grep -q DUMPCE_DONE $L/overnight2_dumpce.log && st "5 dump OK" || st "5 dump FAILED"

st "6: causal ego train"
$TR leak/finetune_causal.py --tag ego_s42 --zero_vision --turn_weighted --seed 42 \
    --epochs 10 --patience 5 --grad_accum_steps 3 > $L/overnight2_train_causal_ego.log 2>&1
st "6 causal ego train exit $?"
for s in val test; do
  $TR leak/dump_ce.py --split $s --tag causal_ego --models causal_ego --causal \
      > $L/overnight2_dump_causal_ego_$s.log 2>&1
  grep -q DUMPCE_DONE $L/overnight2_dump_causal_ego_$s.log && st "6 dump causal_ego $s OK" || st "6 dump causal_ego $s FAILED"
done
touch $L/results/CAUSAL_EGO_READY

st "6b: causal full train"
$TR leak/finetune_causal.py --tag full_s42 --turn_weighted --seed 42 \
    --epochs 10 --patience 5 --grad_accum_steps 3 > $L/overnight2_train_causal_full.log 2>&1
st "6b causal full train exit $?"
for s in val test; do
  $TR leak/dump_ce.py --split $s --tag causal_full --models causal_full --causal \
      > $L/overnight2_dump_causal_full_$s.log 2>&1
  grep -q DUMPCE_DONE $L/overnight2_dump_causal_full_$s.log && st "6b dump causal_full $s OK" || st "6b dump causal_full $s FAILED"
done
touch $L/results/CAUSAL_FULL_READY
st "GPU QUEUE DONE"
