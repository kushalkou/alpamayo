#!/bin/bash
# ar1/run_r9b.sh -- R9 Part B1: flow expert on frozen M1-v2a s42 (2 Hz training words).
# If AUDIT_SIGNOFF.md exists before start: stop (R9B_PAUSED_FOR_COC). Resumable: results/R9_EXPERT_DONE;
# Resume after reboot: cd Alpamayo/code && tmux new -d -s r9b "bash ar1/run_r9b.sh"
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo; R=$L/results; S=$L/r9_status.log; LOG=$L/ar1_r9_M1v2a_expert.log
SIGN=/home/dgx1user/Alpamayo-Kushal/AUDIT_SIGNOFF.md
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
[ -f $R/R9_EXPERT_DONE ] && { st "R9B_DONE"; exit 0; }
if [ -f $SIGN ] && [ ! -f $R/R7B_COC_DONE ]; then st "R9B_PAUSED_FOR_COC (AUDIT_SIGNOFF.md present)"; exit 0; fi
# Manual recovery if training finished but prediction failed: EXTRA="--predict_only --train_log $LOG"
EXTRA=""
st "M1v2a expert start $EXTRA"
$TR ar1/train_expert_meta.py --tag M1v2a --labels 2hz $EXTRA >> $LOG 2>&1 || { st "R9_FAILED expert (see log)"; exit 1; }
touch $R/R9_EXPERT_DONE
st "M1v2a expert done"; st "R9B_DONE"
