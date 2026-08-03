#!/usr/bin/env bash
# One-shot: wait for the daily_used retry sweep (guardian-recreated drivers) to
# drain, then launch electronics stage2 across GPUs 0/1/3. Replaces the broken
# chain trigger for this run (its DONE-grep bug is fixed separately). Run:
#   setsid nohup bash pipeline/wait_then_start_electronics.sh &
set -u
DG=/home/michaellee/mclee/affogato/data_generation
cd "$DG" || exit 1
L=outputs/stage2/electronics/chain.log
log() { echo "$(date '+%F %T') $*" >> "$L"; }

log "waiter start (pid $$): waiting for daily retry sweep to drain"
# patterns built at runtime so this script's own cmdline never matches them
P1='run_stage2.sh [013] .*daily'; P1="$P1"_used
P2='stage2_v2.py.*daily'; P2="$P2"_used
while pgrep -f "$P1" >/dev/null || pgrep -f "$P2" >/dev/null; do sleep 120; done
sleep 30
tmux kill-session -t stage2 2>/dev/null
bash pipeline/start_stage2.sh stage2 outputs/stage1/electronics/stage1_all.json \
    dataset/electronics_to_affogato.json outputs/stage2/electronics 5528 0 1 3 >> "$L" 2>&1 \
  && log "electronics stage2 launched" || log "electronics launch FAILED"
