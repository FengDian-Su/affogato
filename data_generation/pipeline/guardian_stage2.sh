#!/usr/bin/env bash
# tmux-independent guardian for stage2: re-creates the daily_used run if the
# tmux SERVER dies (happened during stage0/stage1). Only covers the case the
# per-GPU monitors cannot — no session left to monitor from. Run detached:
#   setsid nohup guardian_stage2.sh &
set -u
DG=/home/michaellee/mclee/affogato/data_generation
P=$DG/pipeline
SESSION=stage2
DU=$DG/outputs/stage2/daily_used
LOG=$DU/guardian.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "guardian start (pid $$)"
while true; do
  # stop once daily_used all three ranges are done (electronics chain takes over)
  d=1
  for r in "0 [0:19147]" "1 [19147:38294]" "3 [38294:57441]"; do
    g=${r%% *}; rng=${r#* }
    grep -q "DRIVER DONE gpu$g $rng" "$DU/driver_gpu$g.log" 2>/dev/null || d=0
  done
  [ "$d" = 1 ] && { log "daily_used complete; guardian exiting"; break; }
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    log "tmux session $SESSION MISSING -> recreating daily_used across GPUs 0 1 3"
    bash "$P/start_stage2.sh" "$SESSION" outputs/stage1/daily_used/stage1_all.json \
        dataset/daily_used_to_affogato.json "$DU" 57441 0 1 3 >/dev/null 2>&1 \
      && log "recreated" || log "recreate FAILED"
  fi
  sleep 300
done
