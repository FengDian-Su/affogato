#!/usr/bin/env bash
# After daily_used stage2 finishes on all three GPUs, run electronics stage2
# split across the same three cards. Run detached:
#   setsid nohup chain_stage2_electronics.sh &
set -u
DG=/home/michaellee/mclee/affogato/data_generation
P=$DG/pipeline
DU=$DG/outputs/stage2/daily_used
EL_IN=outputs/stage1/electronics/stage1_all.json
EL_MAP=dataset/electronics_to_affogato.json
EL_OUT=$DG/outputs/stage2/electronics
GPUS=(0 1 3)
LOG=$EL_OUT/chain.log
mkdir -p "$EL_OUT"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "chain start (pid $$): waiting for all daily_used stage2 halves to finish"
while true; do
  done=1
  for r in "0 [0:19147]" "1 [19147:38294]" "3 [38294:57441]"; do
    g=${r%% *}; rng=${r#* }
    grep -q "DRIVER DONE gpu$g $rng" "$DU/driver_gpu$g.log" 2>/dev/null || done=0
  done
  [ "$done" = 1 ] && break
  sleep 300
done
log "daily_used stage2 complete; launching electronics on GPUs ${GPUS[*]}"
bash "$P/start_stage2.sh" stage2 "$EL_IN" "$EL_MAP" "$EL_OUT" 5528 "${GPUS[@]}" >> "$LOG" 2>&1 \
  && log "electronics stage2 launched" || log "electronics launch FAILED"
