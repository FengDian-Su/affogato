#!/usr/bin/env bash
# Stage2 driver: run one object range on ONE gpu, resuming on crash.
# Usage: run_stage2.sh <gpu> <start> <end> <in_json> <mapping> <out_dir> [tries=8]
#
# Unlike stage1, stage2_v2 writes INDEPENDENT per-query files and skips finished
# objects with --skip_existing, so it is resumable at object granularity with no
# growing-file cost — no fine sharding needed. This just re-launches the same
# range on any non-zero exit (EngineCore death / OOM / transient), and
# --skip_existing picks up where it left off. stage2_v2 already swallows
# per-object errors internally; a non-zero exit means the engine itself died.
set -u
GPU=$1; START=$2; END=$3; IN=$4; MAP=$5; OUT=$6; TRIES=${7:-8}
DG=/home/michaellee/mclee/affogato/data_generation
PY=$HOME/miniconda3/envs/molmo/bin/python
cd "$DG" || exit 1
mkdir -p "$OUT"
LOG=$OUT/driver_gpu$GPU.log

for ((t = 1; t <= TRIES; t++)); do
  echo "[$(date '+%F %T')] stage2 gpu$GPU [$START:$END] attempt $t/$TRIES -> $OUT" | tee -a "$LOG"
  "$PY" pipeline/stage2_v2.py \
      --stage1 "$IN" --mapping "$MAP" --output_dir "$OUT" \
      --start "$START" --end "$END" --gpu "$GPU" \
      --sam_chunk 20 --skip_existing \
      >> "$OUT/run_gpu${GPU}_${START}_${END}.log" 2>&1 && {
    echo "[$(date '+%F %T')] DRIVER DONE gpu$GPU [$START:$END]" | tee -a "$LOG"
    exit 0
  }
  echo "[$(date '+%F %T')] gpu$GPU [$START:$END] attempt $t exited rc=$? — resuming in 60s" | tee -a "$LOG"
  sleep 60
done
echo "[$(date '+%F %T')] gpu$GPU [$START:$END] GAVE UP after $TRIES attempts" | tee -a "$LOG"
