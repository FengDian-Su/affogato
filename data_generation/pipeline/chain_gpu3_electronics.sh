#!/usr/bin/env bash
# One-shot chain: when GPU3 finishes the daily_used first half, hand GPU3 to
# electronics stage1 (its stage0 kept set is 5,528 objects) rather than let it
# sit idle while GPU0 finishes the daily_used second half (~10h longer).
# Run detached:  setsid nohup chain_gpu3_electronics.sh &
set -u
DG=/home/michaellee/mclee/affogato/data_generation
P=$DG/pipeline
SESSION=stage1_daily
GPU=3
DU_DRIVER_LOG=$DG/outputs/stage1/daily_used/driver_gpu3.log
ELEC_IN=outputs/stage0/electronics/stage0_part0.kept.json
ELEC_OUT=$DG/outputs/stage1/electronics
ELEC_N=5528
LOG=$ELEC_OUT/chain.log
mkdir -p "$ELEC_OUT"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "chain start (pid $$): waiting for daily_used gpu3 [0:28720] DRIVER DONE"
while ! grep -q "DRIVER DONE gpu3 \[0:28720\]" "$DU_DRIVER_LOG" 2>/dev/null; do
  sleep 120
done
log "daily_used gpu3 half DONE; waiting for gpu3 memory to free"

# Free GPU3 before launching (electronics needs the full 88GB). A CLEAN driver
# exit releases CUDA, but reap our own orphaned EngineCore if one lingers.
for w in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 0)
  [ "${used:-0}" -lt 5000 ] && break
  if [ "$w" -ge 6 ] && ! pgrep -f "stage1_v2.py.*--gpu 3" >/dev/null; then
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null); do
      if [ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" = "$(id -un | cut -c1-8)" ]; then
        log "reaping orphaned gpu3 process $pid (${used}MiB)"; kill -9 "$pid" 2>/dev/null || true
      fi
    done
    sleep 15
  fi
  log "gpu3 holds ${used}MiB; waiting"; sleep 20
done

DRV="bash $P/run_stage1.sh 3 0 $ELEC_N 5000 $ELEC_IN $ELEC_OUT"
MON="bash $P/monitor_stage1.sh 3 0 $ELEC_N 5000 $ELEC_IN $ELEC_OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-window -t "$SESSION" -n drv_elec3 "$DRV" \
    && tmux new-window -t "$SESSION" -n mon_elec3 "$MON" \
    && log "launched electronics stage1 on gpu3 ($ELEC_N objs) in session $SESSION"
else
  tmux new-session -d -s stage1_elec -n drv_elec3 "$DRV" \
    && tmux new-window -t stage1_elec -n mon_elec3 "$MON" \
    && log "session $SESSION gone; launched electronics in new session stage1_elec"
fi
