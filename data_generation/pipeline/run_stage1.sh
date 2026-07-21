#!/usr/bin/env bash
# Stage1 driver: run a contiguous object range on ONE gpu, in fixed-size shards.
#
# Why shard at all: stage1_v2.py rewrites its ENTIRE output file once per window
# (8 objects), so a single 28k-object run would rewrite a ~290MB json ~3600
# times and slow down as it grows. Shards bound that cost, isolate a crash to
# one range, and each resumes independently (stage1_v2 skips object_ids already
# present in its --out file).
#
# Usage: run_stage1.sh <gpu> <start> <end> [step=5000]
#   run_stage1.sh 3 0 28720          # first half
#   run_stage1.sh 0 28720 57441      # second half, once that gpu frees up
set -u
GPU=$1; START=$2; END=$3; STEP=${4:-5000}

DG=/home/michaellee/mclee/affogato/data_generation
IN=outputs/stage0/daily_used/kept_all.json
OUT=$DG/outputs/stage1/daily_used
cd "$DG" || exit 1
mkdir -p "$OUT"

TRIES=${TRIES:-4}

for ((s = START; s < END; s += STEP)); do
  e=$(( s + STEP < END ? s + STEP : END ))
  tag=$(printf '%06d_%06d' "$s" "$e")
  f="$OUT/stage1_$tag.json"
  # Retry the SAME shard rather than moving on: a resumed shard skips the
  # object_ids already in its json, so a retry is cheap, and the monitor kills a
  # stalled python expecting this loop to pick it back up. Moving on instead
  # would silently leave a hole in the range.
  for ((try = 1; try <= TRIES; try++)); do
    echo "[$(date '+%F %T')] shard [$s:$e] try $try/$TRIES -> $f" | tee -a "$OUT/driver_gpu$GPU.log"
    GEMMA_BACKEND=vllm LD_LIBRARY_PATH=$HOME/miniconda3/envs/gemma4/lib \
      "$HOME/miniconda3/envs/gemma4/bin/python" pipeline/stage1_v2.py \
        --in "$IN" --out "$f" --start "$s" --end "$e" --gpu "$GPU" \
        --batch_size 8 --max_num_batched_tokens 40960 --gpu_mem 0.90 \
        >> "$OUT/shard_$tag.log" 2>&1 && break
    echo "[$(date '+%F %T')] SHARD [$s:$e] try $try FAILED rc=$?" | tee -a "$OUT/driver_gpu$GPU.log"
    sleep 60
  done
done
echo "[$(date '+%F %T')] DRIVER DONE gpu$GPU [$START:$END]" | tee -a "$OUT/driver_gpu$GPU.log"
