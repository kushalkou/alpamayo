#!/bin/bash
# ar1/r4_gpu.sh -- R4-DIAG GPU inference (cap 3 GPU-h). Ledger: Alpamayo/r4_gpu_ledger.txt
# (wall seconds x GPUs per job). Batch 1: train 50-scene stationary dumps for B1 x3 seeds +
# A3 (2 GPUs each). Batch 2: G6C_s123 / s2024 train dumps (2 GPUs each) + B1 s42 val
# stationary camera shuffle among stationary samples (4 GPUs, perm seed 99).
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo; CK=$L/models/checkpoints; D=$L/data
cd $L/code
job(){ # gpus port tag ckpt flags...
  local G=$1 P=$2 T=$3 C=$4; shift 4
  local n=$(echo $G | tr ',' '\n' | wc -l) t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$G $PY -m torch.distributed.run --nproc_per_node=$n --master_port $P w1/dump_w1.py \
     --tag $T --ckpt $CK/$C/alpamayo_best.pt --fix "$@" > $L/ar1_r4_$T.log 2>&1
  local rc=$? t1=$(date +%s)
  echo "$T gpus=$n wall_s=$((t1-t0)) gpu_h=$(echo "scale=3; ($t1-$t0)*$n/3600" | bc) rc=$rc" >> $L/r4_gpu_ledger.txt
}
TR="--splits train --tokens_file $D/r4_train50_stationary.pkl"
job 0,1 29701 B1_tr50 _w1_B1 --cmd --vcache t0 $TR &
job 2,3 29702 B1_s123_tr50 _w1_B1_s123 --cmd --vcache t0 $TR &
job 4,5 29703 B1_s2024_tr50 _w1_B1_s2024 --cmd --vcache t0 $TR &
job 6,7 29704 A3_tr50 _w1_A3 --cmd --no_vision $TR &
wait
job 0,1 29705 G6C_s123_tr50 _w1_G6C_s123 --cmd --no_vision $TR &
job 2,3 29706 G6C_s2024_tr50 _w1_G6C_s2024 --cmd --no_vision $TR &
job 4,5,6,7 29707 B1_statshuf _w1_B1 --cmd --vcache t0 --splits val --tokens_file $D/r4_val_stationary.pkl --shuffle_cams &
wait
echo R4_GPU_DONE >> $L/r4_gpu_ledger.txt
