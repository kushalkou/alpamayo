#!/bin/bash
# ar1/run_r11b.sh -- R11 Part B: CoC arm seed 42 (PROVISIONAL) train + dumps (modes gen/none/shuf/gt).
# Resumable via results/R11_COC_<step>_DONE; status Alpamayo/r11_status.log; ledger Alpamayo/r11_gpu_ledger.txt.
# Resume after reboot: cd Alpamayo/code && tmux new -d -s r11b "bash ar1/run_r11b.sh" (an interrupted step restarts).
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo; R=$L/results; S=$L/r11_status.log; G=$L/r11_gpu_ledger.txt
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
step(){ local N=$1; shift
  [ -f $R/R11_COC_${N}_DONE ] && return 0
  st "$N start"; local t0=$(date +%s)
  "$@" > $L/ar1_r11_coc_$N.log 2>&1 || { st "R11_FAILED $N"; exit 1; }
  echo "$N gpus=8 wall_s=$(( $(date +%s) - t0 ))" >> $G; touch $R/R11_COC_${N}_DONE; st "$N done"
}
step train $TR ar1/finetune_coc.py --tag COC --seed 42
step dump_gen $TR ar1/finetune_coc.py --tag COC --dump holdout,val --mode gen
step dump_none $TR ar1/finetune_coc.py --tag COC --dump val --mode none
step dump_shuf $TR ar1/finetune_coc.py --tag COC --dump val --mode shuf
step dump_gt $TR ar1/finetune_coc.py --tag COC --dump val --mode gt
st "R11B_DONE"
