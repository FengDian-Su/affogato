#!/usr/bin/env bash
# tmux-independent guardian for stage1: re-creates the session if the tmux SERVER
# itself dies (this happened twice during stage0 — 07-18 reboot, 07-20 unexplained).
# The monitors handle everything inside a live session; this only covers the case
# where there is no session left to monitor from.
# Run detached:  setsid nohup guardian_stage1.sh &
set -u
P=/home/michaellee/mclee/affogato/data_generation/pipeline
OUT=/home/michaellee/mclee/affogato/data_generation/outputs/stage1/daily_used
SESSION=stage1_daily
log() { echo "$(date '+%F %T') $*" >> "$OUT/guardian.log"; }

log "guardian start (pid $$, session $SESSION)"
while true; do
  if grep -q "DRIVER DONE gpu3 \[0:28720\]" "$OUT/driver_gpu3.log" 2>/dev/null &&
     grep -q "DRIVER DONE gpu0 \[28720:57441\]" "$OUT/driver_gpu0.log" 2>/dev/null; then
    log "both halves complete; guardian exiting"; break
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    log "tmux session $SESSION MISSING -> recreating"
    bash "$P/start_stage1_tmux.sh" >/dev/null 2>&1 && log "recreated" || log "recreate FAILED"
  fi
  sleep 300
done
