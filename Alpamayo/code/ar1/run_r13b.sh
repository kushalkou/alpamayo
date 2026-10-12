#!/bin/bash
# ar1/run_r13b.sh -- R13 Part B: CoC seeds 123, 2024 (v3 traces, recipe = seed 42) then CoC-AUX seed 42.
# Resumable via results/R13_<tag>_<step>_DONE; status Alpamayo/r13_status.log; ledger Alpamayo/r13_gpu_ledger.txt.
# Resume after reboot: cd Alpamayo/code && tmux new -d -s r13b "bash ar1/run_r13b.sh" (an interrupted step restarts).
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo; R=$L/results; S=$L/r13_status.log; G=$L/r13_gpu_ledger.txt
C="--coc ../data/ar1_coc_2hz_v3.pkl"
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
step(){ local T=$1 N=$2; shift 2
  [ -f $R/R13_${T}_${N}_DONE ] && return 0
  st "$T $N start"; local t0=$(date +%s)
  "$@" > $L/ar1_r13_${T}_$N.log 2>&1 || { st "R13_FAILED $T $N"; exit 1; }
  echo "$T $N gpus=8 wall_s=$(( $(date +%s) - t0 ))" >> $G; touch $R/R13_${T}_${N}_DONE; st "$T $N done"
}
for SD in 123 2024; do
  T=COC3_s$SD
  step $T train $TR ar1/finetune_coc.py --tag $T $C --seed $SD
  step $T dump_gen $TR ar1/finetune_coc.py --tag $T $C --dump holdout,val --mode gen
  step $T dump_shuf $TR ar1/finetune_coc.py --tag $T $C --dump val --mode shuf
  step $T dump_gt $TR ar1/finetune_coc.py --tag $T $C --dump val --mode gt
done
T=COC3AUX
step $T train $TR ar1/finetune_coc.py --tag $T $C --seed 42 --aux 0.5 --select_mode none
step $T dump_none $TR ar1/finetune_coc.py --tag $T $C --dump holdout,val --mode none
step $T dump_gen $TR ar1/finetune_coc.py --tag $T $C --dump holdout,val --mode gen
st "R13B_DONE"
