#!/usr/bin/env bash
# Resumable batch collection driver.
#
# Collects EPISODES episodes in batches of BATCH, one shard file each, then
# merges and verifies. Safe to re-run: finished shards are skipped, so if a
# batch dies you just run the same command again.
#
#   ./scripts/collect_plan.sh                     # defaults: 1200 eps, batch 200
#   EPISODES=300 ./scripts/collect_plan.sh        # stage 1
#   EPISODES=30 BATCH=30 OUT=data/stage0.hdf5 ./scripts/collect_plan.sh
#
# Every run appends to logs/collect_<timestamp>.log as well as printing.

set -uo pipefail

EPISODES=${EPISODES:-1200}
BATCH=${BATCH:-200}
RES=${RES:-128}
START=${START:-0}
OUT=${OUT:-data/train.hdf5}
PARTDIR=${PARTDIR:-data/parts}
PY=${PY:-.venv/bin/python}

mkdir -p "$PARTDIR" logs
LOG="logs/collect_$(date +%Y%m%d_%H%M%S).log"

log() { echo "$*" | tee -a "$LOG"; }

# --- preflight -------------------------------------------------------------
# ~8.1 MB/episode at 128px, measured. The shards are KEPT alongside the merged
# file, so peak usage is two copies -- budget 2x plus a little headroom, or the
# run dies at the merge step with everything already collected.
PER_EP_MB=9
NEED_MB=$(( EPISODES * PER_EP_MB * 2 + 1024 ))
FREE_MB=$(df -m . | awk 'NR==2{print $4}')
log "=== SO-101 collection plan ==="
log "episodes   : $EPISODES  (batches of $BATCH, seeds from $START)"
log "resolution : ${RES}x${RES}"
log "output     : $OUT   (shards in $PARTDIR)"
log "disk       : need ~${NEED_MB} MB peak (shards + merged copy), have ${FREE_MB} MB free"
if [ "$FREE_MB" -lt "$NEED_MB" ]; then
  log "ABORT: not enough free disk space."
  exit 1
fi
log "log        : $LOG"
log ""

# --- collect ---------------------------------------------------------------
SHARDS=()
FAILED=0
seed=$START
end=$(( START + EPISODES ))
while [ "$seed" -lt "$end" ]; do
  n=$BATCH
  [ $(( seed + n )) -gt "$end" ] && n=$(( end - seed ))
  shard="$PARTDIR/part_$(printf '%06d' "$seed").hdf5"
  SHARDS+=("$shard")

  # File existence is NOT enough: a shard killed mid-run still opens cleanly and
  # reports a plausible episode count, so skipping on existence alone would leave
  # a silent hole in the dataset. Check it was actually finalized.
  if STATUS=$($PY scripts/shard_complete.py "$shard" "$n" 2>&1); then
    log "[$(date +%H:%M:%S)] shard $(basename "$shard") $STATUS -- skipping"
  else
    if [ -f "$shard" ]; then
      log "[$(date +%H:%M:%S)] shard $(basename "$shard") $STATUS -- discarding and recollecting"
      rm -f "$shard" "${shard%.hdf5}.json"
    fi
    log "[$(date +%H:%M:%S)] --- batch: $n episodes from seed $seed ---"
    # --plain-log keeps the tee'd log readable; per-shard report avoids the
    # collision you get from a single shared --report path.
    if ! $PY -m so101_sim.cli.collect \
          --num-episodes "$n" --start-seed "$seed" \
          --image-height "$RES" --image-width "$RES" \
          --out "$shard" --report "${shard%.hdf5}.json" \
          --plain-log 2>&1 | tee -a "$LOG"; then
      log "[$(date +%H:%M:%S)] batch at seed $seed FAILED -- re-run this script to retry"
      FAILED=1
    fi
  fi
  seed=$(( seed + n ))
done

if [ "$FAILED" -ne 0 ]; then
  log ""
  log "One or more batches failed. Completed shards are kept; re-run to resume."
  exit 1
fi

# --- merge and verify ------------------------------------------------------
log ""
log "[$(date +%H:%M:%S)] merging $(( ${#SHARDS[@]} )) shards -> $OUT"
$PY -m so101_sim.cli.merge "${SHARDS[@]}" --out "$OUT" --overwrite 2>&1 | tee -a "$LOG"

log ""
log "[$(date +%H:%M:%S)] verifying $OUT"
$PY scripts/verify_dataset.py "$OUT" 2>&1 | tee -a "$LOG"

log ""
log "=== done. dataset: $OUT ==="
log "shards kept in $PARTDIR -- once you trust the merge, reclaim ~$(( EPISODES * PER_EP_MB )) MB with:"
log "    rm -rf $PARTDIR"
