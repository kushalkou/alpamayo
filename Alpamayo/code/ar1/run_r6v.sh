#!/bin/bash
# R6 Part 1 verification decodes (batched, inference only). Ledger: Alpamayo/r6_gpu_ledger.txt
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo; cd $L/code
t0=$(date +%s)
/home/dgx1user/miniconda3/envs/alpamayo/bin/python -m torch.distributed.run --nproc_per_node=8 --master_port 29802 \
  ar1/meta_decode_batched.py --tag M1v2a --variants gen,cams,ego,cmd,force_gt,force_maj,force_s1 > $L/r6_verify.log 2>&1
echo "verify_all gpus=8 wall_s=$(( $(date +%s)-t0 )) rc=$?" >> $L/r6_gpu_ledger.txt
