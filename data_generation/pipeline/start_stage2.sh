#!/usr/bin/env bash
# Launch stage2 for one dataset, splitting its object range evenly across a set
# of GPUs, each with its own driver + monitor window in a tmux session.
#
# Usage: start_stage2.sh <session> <in_json> <mapping> <out_dir> <total> <gpu>...
#   start_stage2.sh stage2_du outputs/stage1/daily_used/stage1_all.json \
#       dataset/daily_used_to_affogato.json outputs/stage2/daily_used 57441 0 1 3
#
# Idempotent-ish: if the session already exists it just adds windows, so don't
# re-run against a live session. Each GPU's [start,end] is disjoint; drivers
# resume via --skip_existing, so a re-launch of a range is safe.
set -u
SESSION=$1; IN=$2; MAP=$3; OUT=$4; TOTAL=$5; shift 5
GPUS=("$@"); NG=${#GPUS[@]}
P=/home/michaellee/mclee/affogato/data_generation/pipeline
mkdir -p "$OUT"
per=$(( (TOTAL + NG - 1) / NG ))     # ceil split

first=1
for ((i = 0; i < NG; i++)); do
  g=${GPUS[$i]}
  s=$(( i * per )); e=$(( s + per < TOTAL ? s + per : TOTAL ))
  [ "$s" -ge "$TOTAL" ] && break
  echo "GPU$g -> [$s:$e] ($((e-s)) objects)"
  DRV="bash $P/run_stage2.sh $g $s $e $IN $MAP $OUT"
  MON="MAP=$MAP bash $P/monitor_stage2.sh $g $s $e $IN $OUT"
  if [ "$first" = 1 ]; then
    tmux new-session -d -s "$SESSION" -n "drv$g" "$DRV"; first=0
  else
    tmux new-window -t "$SESSION" -n "drv$g" "$DRV"
  fi
  tmux new-window -t "$SESSION" -n "mon$g" "$MON"
done
echo "launched session $SESSION across GPUs: ${GPUS[*]}"
