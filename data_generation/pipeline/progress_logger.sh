#!/usr/bin/env bash
# Append one progress line per interval to a durable file.
# Runs inside tmux, so the record survives any client/agent session ending;
# the push notifications do not, and this is what you read instead.
set -u
cd /home/michaellee/mclee/affogato/data_generation || exit 1
OUT=outputs/stage2/progress_history.log
INT=${INT:-3600}
while true; do
  printf '%s  %s\n' "$(date '+%F %T')" "$(bash pipeline/stage2_replay_progress.sh)" >> "$OUT"
  sleep "$INT"
done
