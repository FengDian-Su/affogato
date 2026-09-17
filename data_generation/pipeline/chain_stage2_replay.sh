#!/usr/bin/env bash
# Regenerate the whole stage2 release from FROZEN Molmo points (no vLLM).
#
#   chain_stage2_replay.sh <gpu> <shard> <nshards>
#
# Each category's stage1 index space is cut into NSHARDS contiguous ranges and
# this process takes range SHARD.  Several shards share a GPU on purpose: a
# worker alternates between CPU phases (scene/EXR load, projection, consensus,
# 3D vote) and GPU phases (SAM2) and saturates neither — measured 32% GPU util
# and 1.4 cores per worker — so concurrency, not a bigger SAM batch, is what
# fills the card.
#
# Small categories go first so a whole category completes early and any
# systematic problem surfaces before the 57k-object one starts.  run_stage2.sh
# keeps its own retry/--skip_existing resume, so a shard can be interrupted and
# relaunched with the identical command.
set -u
GPU=$1; SHARD=$2; NSHARDS=$3
DG=/home/michaellee/mclee/affogato/data_generation
cd "$DG" || exit 1
mkdir -p outputs/stage2
LOG=outputs/stage2/chain_gpu${GPU}s${SHARD}.log

run_shard () {
  local cat=$1 total=$2 map=$3 s e
  s=$(( total * SHARD / NSHARDS ))
  e=$(( total * (SHARD + 1) / NSHARDS ))
  [ "$s" -ge "$e" ] && return 0
  echo "[$(date '+%F %T')] gpu$GPU s$SHARD $cat [$s:$e] start" | tee -a "$LOG"
  REPLAY_POINTS="outputs/stage2_points_20260803/$cat" \
    bash pipeline/run_stage2.sh "$GPU" "$s" "$e" \
        "outputs/stage1/$cat/stage1_all.json" "dataset/$map" "outputs/stage2/$cat"
  echo "[$(date '+%F %T')] gpu$GPU s$SHARD $cat [$s:$e] end" | tee -a "$LOG"
}

run_shard electronics 5528  electronics_to_affogato.json
run_shard furnitures  8420  furnitures_to_affogato.json
run_shard daily_used  57441 daily_used_to_affogato.json
echo "[$(date '+%F %T')] CHAIN DONE gpu$GPU s$SHARD" | tee -a "$LOG"
