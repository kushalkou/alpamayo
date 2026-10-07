#!/bin/bash
# ar1/run_r7c3.sh -- R7 C3 blank-image control (inference, cap 2 GPU-h), AFTER Part A.
# Waits for R7A_DONE in Alpamayo/r7_status.log. Resumable via results/R7_C3_<step>_DONE.
# Resume after reboot: cd Alpamayo/code && tmux new -d -s r7c3 "bash ar1/run_r7c3.sh"
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo; R=$L/results; S=$L/r7_status.log; G=$L/r7_gpu_ledger.txt
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
until grep -q "R7A_DONE" $S; do grep -q "R7_FAILED" $S && exit 1; sleep 120; done
step(){ # name cmd...
  local N=$1; shift
  [ -f $R/R7_C3_${N}_DONE ] && return 0
  local t0=$(date +%s)
  "$@" > $L/ar1_r7_c3_$N.log 2>&1 || { st "R7_FAILED C3 $N"; exit 1; }
  echo "C3 $N gpus=8 wall_s=$(( $(date +%s) - t0 ))" >> $G; touch $R/R7_C3_${N}_DONE
}
st "C3 start"
step b1check $TR ar1/r7_c3_b1_batched.py --tag B1 --variants real --check 200
step b1 $TR ar1/r7_c3_b1_batched.py --tag B1 --variants real,cams,blank
step m1v2a $TR ar1/meta_decode_batched.py --tag M1v2a --variants blank
st "R7C3_DONE"
