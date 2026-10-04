#!/bin/bash
# ar1/run_a3.sh -- R3.3 amendment A3 (GPU, after the M1 seeds): re-decode M1 seed 42 with
# words constrained to the valid label words: greedy (M1_G; must equal the original M1 dump,
# which used the same constrained greedy path) and sampling at T = 0.7 (M1_T07, seed 777+rank).
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
TR="$PY -m torch.distributed.run --nproc_per_node=8"
L=/home/dgx1user/Alpamayo-Kushal/Alpamayo
R=$L/results; S=$L/r3_status.log
cd $L/code
st(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> $S; }
until grep -q R33_ALL_DONE $S; do sleep 120; done
for X in "M1_G 0" "M1_T07 0.7"; do
  set -- $X
  [ -f $R/A3_$1_DONE ] && continue
  st "A3 $1 decode start"
  $TR ar1/finetune_meta.py --tag M1 --out_tag $1 --word_temp $2 --dump holdout,val > $L/ar1_a3_$1.log 2>&1 \
    || { st "A3_FAILED at $1"; exit 1; }
  st "A3 $1 decode done"; touch $R/A3_$1_DONE
done
st "A3_DONE"
