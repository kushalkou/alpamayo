#!/usr/bin/env bash
# Overnight queue driver: items 1-7. Commits after each. Item 1 is BLOCKING.
set -u
PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python
CODE=/home/dgx1user/Alpamayo-Kushal/Alpamayo/code
REPO=/home/dgx1user/Alpamayo-Kushal
OUT=$REPO/Alpamayo/overnight
mkdir -p "$OUT"
cd "$CODE" || exit 1

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

commit(){ # $1 = message
  cd "$REPO" || return 0
  git add -A Alpamayo/code Alpamayo/overnight OVERNIGHT_REPORT.md \
      Alpamayo/DECODE_RULE_RESULTS.md 2>/dev/null
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017yBf8aU6PNmPrJVpXzPj7p" 2>/dev/null \
    && timeout 180 git push -q 2>&1 | tail -2
  cd "$CODE" || return 0
}

# wait for dumps
log "waiting for dumps..."
while [ ! -f /tmp/claude-1000/dumps_complete ]; do
  if grep -qaE "ChildFailedError" /tmp/claude-1000/dump_val.log \
       /tmp/claude-1000/dump_test.log 2>/dev/null; then
    log "DUMP FAILED"; echo "DUMP FAILED" > "$OUT/FAILED"; exit 1
  fi
  sleep 60
done
log "dumps complete"

# ---- item 1: VERIFICATION (blocking) ----
log "item 1: verification"
$PY verify_dump.py 100 > "$OUT/01_verify.txt" 2>&1
if grep -q "\[verify\] PASS" "$OUT/01_verify.txt"; then
  log "item 1 PASS"
else
  log "item 1 FAILED - stopping queue"
  echo "VERIFICATION FAILED" > "$OUT/FAILED"
  commit "[overnight] 1. VERIFICATION FAILED - offline sweeps invalid, queue stopped"
  exit 1
fi
commit "[overnight] 1. verification passed: offline reconstruction matches slow decode"

# ---- item 2: zeroboth / CV divergence ----
log "item 2: zeroboth probe"
$PY zeroboth_probe.py > "$OUT/02_zeroboth.txt" 2>&1 || log "item 2 errored"
commit "[overnight] 2. zeroboth vs CV divergence decomposition"

# ---- items 3,4,5: hybrid + global shrinkage + per-stratum ----
log "items 3-5: shrink.py"
$PY shrink.py > "$OUT/03_05_shrink.txt" 2>&1 || log "items 3-5 errored"
commit "[overnight] 3-5. stationary hybrid, global shrinkage, per-stratum alpha"

# ---- item 6: adaptive ----
log "item 6: adaptive"
$PY adaptive.py > "$OUT/06_adaptive.txt" 2>&1 || log "item 6 errored"
commit "[overnight] 6. confidence-adaptive and per-horizon shrinkage"

# ---- item 7: subsume ----
log "item 7: subsume"
$PY subsume.py > "$OUT/07_subsume.txt" 2>&1 || log "item 7 errored"
commit "[overnight] 7. does shrinkage subsume the decode fix (2x2)"

log "ITEMS 1-7 COMPLETE"
echo done > "$OUT/ITEMS_1_7_DONE"
