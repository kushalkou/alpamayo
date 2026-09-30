#!/usr/bin/env bash
# 3.5a retry (LoRA dropout 0, constant LR) + final-checkpoint dumps incl. INPUT-SHUFFLE test
set -u
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo
CK=$L/models/checkpoints/_w1_overfit256_retry
cd $L/code
$TR w1/finetune_w1.py --tag overfit256_retry --cmd --no_vision --zero_vision --overfit 256 \
    --lora_dropout 0 --min_lr_ratio 1.0 --seed 42 --epochs 150 --patience 150 \
    --batch_size 3 --grad_accum_steps 1 > $L/w1_overfit256_retry.log 2>&1
echo "TRAIN_EXIT $?" >> $L/w1_overfit256_retry.log
$TR w1/dump_w1.py --tag retry_latest --ckpt $CK/alpamayo_latest.pt --cmd --no_vision --splits train --overfit 256 > $L/w1_retry_dump.log 2>&1
$TR w1/dump_w1.py --tag retry_latest_shuf --ckpt $CK/alpamayo_latest.pt --cmd --no_vision --splits train --overfit 256 --shuffle_inputs >> $L/w1_retry_dump.log 2>&1
$TR w1/dump_w1.py --tag retry_best --ckpt $CK/alpamayo_best.pt --cmd --no_vision --splits train --overfit 256 >> $L/w1_retry_dump.log 2>&1
echo RETRY_QUEUE_DONE >> $L/w1_retry_dump.log
