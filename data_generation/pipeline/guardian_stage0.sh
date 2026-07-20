#!/usr/bin/env bash
# tmux-independent guardian: recreates the run's tmux session if the tmux SERVER dies (happened
# twice: 07-18 reboot, 07-20 unexplained). Run detached:  setsid nohup guardian_stage0.sh <cat> <map> [gpu] [part] &
set -u
CATEGORY=${1:?}; MAPPING=${2:?}; GPU=${3:-0}; PART=${4:-10000}
P=/home/michaellee/mclee/affogato/data_generation/pipeline
OUTDIR=/home/michaellee/mclee/affogato/data_generation/outputs/stage0/$CATEGORY
SESSION=stage0_$CATEGORY
log() { echo "$(date '+%F %T') $*" >> "$OUTDIR/guardian.log"; }
log "guardian start (pid $$, session $SESSION)"
while true; do
  grep -q "ALL PARTS DONE" "$OUTDIR/driver.log" 2>/dev/null && { log "complete; exiting"; break; }
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    log "tmux session $SESSION MISSING -> recreating"
    tmux new-session -d -s "$SESSION" -n driver "bash $P/run_stage0.sh $CATEGORY $MAPPING $GPU $PART" \
      && tmux new-window -t "$SESSION" -n monitor "bash $P/monitor_stage0.sh $CATEGORY $MAPPING $GPU $PART" \
      && log "recreated" || log "recreate FAILED"
  fi
  sleep 300
done
