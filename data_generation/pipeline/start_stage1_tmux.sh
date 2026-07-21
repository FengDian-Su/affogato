#!/usr/bin/env bash
# Bring up (or re-create) the stage1 daily_used tmux session: one driver + one
# monitor window per gpu half. Idempotent — exits quietly if the session exists,
# so the guardian can call it on a loop.
set -u
SESSION=stage1_daily
P=/home/michaellee/mclee/affogato/data_generation/pipeline
# gpu:start:end — the two halves of outputs/stage0/daily_used/kept_all.json (57,441 objects)
HALVES=("3:0:28720" "0:28720:57441")

tmux has-session -t "$SESSION" 2>/dev/null && exit 0

first=1
for h in "${HALVES[@]}"; do
  IFS=: read -r gpu s e <<<"$h"
  if [ "$first" = 1 ]; then
    tmux new-session -d -s "$SESSION" -n "drv_gpu$gpu" "bash $P/run_stage1.sh $gpu $s $e"
    first=0
  else
    tmux new-window -t "$SESSION" -n "drv_gpu$gpu" "bash $P/run_stage1.sh $gpu $s $e"
  fi
  tmux new-window -t "$SESSION" -n "mon_gpu$gpu" "bash $P/monitor_stage1.sh $gpu $s $e"
done
echo "started tmux session $SESSION"
