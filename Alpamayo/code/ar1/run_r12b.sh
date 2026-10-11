#!/bin/bash
# ar1/run_r11b.sh -- R12 Part B: CoC arm seed 42 on v3 traces (PROVISIONAL) train + dumps (modes gen/none/shuf/gt).
# Resumable via results/R12_COC3_<step>_DONE; status Alpamayo/r12_status.log; ledger Alpamayo/r12_gpu_ledger.txt.
# Resume after reboot: cd Alpamayo/code && tmux new -d -s r12b "bash ar1/run_r12b.sh" (an interrupted step restarts).
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo; R=$L/results; S=$L/r12_status.log; G=$L/r12_gpu_ledger.txt
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
step(){ local N=$1; shift
  [ -f $R/R12_COC3_${N}_DONE ] && return 0
  st "$N start"; local t0=$(date +%s)
  "$@" > $L/ar1_r12_coc3_$N.log 2>&1 || { st "R12_FAILED $N"; exit 1; }
  echo "$N gpus=8 wall_s=$(( $(date +%s) - t0 ))" >> $G; touch $R/R12_COC3_${N}_DONE; st "$N done"
}
step train $TR ar1/finetune_coc.py --tag COC3 --coc ../data/ar1_coc_2hz_v3.pkl --seed 42
step dump_gen $TR ar1/finetune_coc.py --tag COC3 --coc ../data/ar1_coc_2hz_v3.pkl --dump holdout,val --mode gen
step dump_none $TR ar1/finetune_coc.py --tag COC3 --coc ../data/ar1_coc_2hz_v3.pkl --dump val --mode none
step dump_shuf $TR ar1/finetune_coc.py --tag COC3 --coc ../data/ar1_coc_2hz_v3.pkl --dump val --mode shuf
step dump_gt $TR ar1/finetune_coc.py --tag COC3 --coc ../data/ar1_coc_2hz_v3.pkl --dump val --mode gt
st "R12B_DONE"
