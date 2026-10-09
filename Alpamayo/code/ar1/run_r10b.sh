#!/bin/bash
# ar1/run_r10b.sh -- R10 B1: flow experts on frozen M1-v2a s123, s2024 (recipe = R9 s42 expert).
# Before each run: if AUDIT_SIGNOFF.md exists, stop (R10B_PAUSED_FOR_COC). Markers
# results/R10_EXPERT_<tag>_DONE. Resume after reboot: cd Alpamayo/code && tmux new -d -s r10b "bash ar1/run_r10b.sh"
# (a run interrupted mid-training restarts from scratch).
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo; R=$L/results; S=$L/r10_status.log
SIGN=/home/dgx1user/Alpamayo-Kushal/AUDIT_SIGNOFF.md
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
for T in M1v2a_s123 M1v2a_s2024; do
  [ -f $R/R10_EXPERT_${T}_DONE ] && continue
  if [ -f $SIGN ] && [ ! -f $R/R7B_COC_DONE ]; then st "R10B_PAUSED_FOR_COC (AUDIT_SIGNOFF.md present)"; exit 0; fi
  st "$T expert start"
  $TR ar1/train_expert_meta.py --tag $T --labels 2hz > $L/ar1_r10_${T}_expert.log 2>&1 || { st "R10_FAILED $T expert"; exit 1; }
  touch $R/R10_EXPERT_${T}_DONE; st "$T expert done"
done
st "R10B_DONE"
