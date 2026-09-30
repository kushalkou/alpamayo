#!/usr/bin/env bash
# GPU queue G1: after w1 (retry) finishes -> FIX run + dumps -> diagnostic-c grad probe.
set -u
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo
CK=$L/models/checkpoints/_w1_overfit256_fix
S=$L/gq_status.log
cd $L/code
st(){ echo "[$(date -u +%H:%M:%S)] $*" >> $S; }
st "waiting for w1 (retry queue)"
while tmux has-session -t w1 2>/dev/null; do sleep 60; done
st "fix run start"
$TR w1/finetune_w1.py --tag overfit256_fix --cmd --no_vision --zero_vision --overfit 256 --fix \
    --seed 42 --epochs 150 --patience 150 --batch_size 3 --grad_accum_steps 1 > $L/w1_overfit256_fix.log 2>&1
st "fix run exit $?"
$TR w1/dump_w1.py --tag fix_latest --ckpt $CK/alpamayo_latest.pt --cmd --no_vision --fix --splits train --overfit 256 > $L/w1_fix_dump.log 2>&1
$TR w1/dump_w1.py --tag fix_latest_shuf --ckpt $CK/alpamayo_latest.pt --cmd --no_vision --fix --splits train --overfit 256 --shuffle_inputs >> $L/w1_fix_dump.log 2>&1
$TR w1/dump_w1.py --tag fix_best --ckpt $CK/alpamayo_best.pt --cmd --no_vision --fix --splits train --overfit 256 >> $L/w1_fix_dump.log 2>&1
st "fix dumps done"
$TR w1/finetune_w1.py --tag probe_grad --cmd --no_vision --zero_vision --overfit 256 --probe_grad \
    --seed 42 --epochs 150 --patience 150 --batch_size 3 --grad_accum_steps 1 --max_steps 200 > $L/w1_probe_grad.log 2>&1
st "probe exit $?"
st "GQ_DONE"
