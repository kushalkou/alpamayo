#!/usr/bin/env bash
# QUEUE v2 G9: A0 second seed (seed 123): visual tokens REMOVED, causal ego, custom split,
# old recipe -- identical to the A0 run of run_stageA.sh except the seed. Compared by
# w1/a0_seed2.py with A0 s42 (removed) and causal_ego s42 (zeroed).
set -u
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo; R=$L/results
cd $L/code
if [ ! -f $R/Q2_G9_TRAINED ]; then
  $TR leak/finetune_causal.py --tag A0_ego_novis_s123 --no_vision --zero_vision --turn_weighted \
      --seed 123 --epochs 10 --patience 5 --batch_size 3 --grad_accum_steps 1 > $L/w1_q2_G9.log 2>&1 || exit 1
  touch $R/Q2_G9_TRAINED
fi
for s in val test; do
  $TR leak/dump_ce.py --split $s --tag A0_causal_ego_novis_s123 --models A0_causal_ego_novis_s123 --causal \
      >> $L/w1_q2_G9_dump.log 2>&1 || exit 1
done
$PY w1/a0_seed2.py > $L/w1_q2_G9.txt 2>&1 || exit 1
