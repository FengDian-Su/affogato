#!/usr/bin/env bash
# Generic stage0 watchdog:  monitor_stage0.sh <category> <mapping.json> [gpu] [part_size]
# Every 5 min logs progress + GPU state; restarts a dead driver; kills a stalled stage0 python
# (>45 min with no output-json AND no part*.log activity, and only if the process itself has been
# alive that long - a fresh process is still loading the model, and after a reboot the newest json
# predates it). Exits once the driver logs ALL PARTS DONE.
set -u
CATEGORY=${1:?usage: monitor_stage0.sh <category> <mapping.json> [gpu] [part_size]}
MAPPING=${2:?usage: monitor_stage0.sh <category> <mapping.json> [gpu] [part_size]}
GPU=${3:-0}
PART=${4:-10000}
STALL_SEC=2700

PY=/home/michaellee/miniconda3/envs/gemma4/bin/python
REPO=/home/michaellee/mclee/affogato
DATA_GEN=$REPO/data_generation
OUTDIR=$DATA_GEN/outputs/stage0/$CATEGORY
DRIVER=$DATA_GEN/pipeline/run_stage0.sh
MAP_ABS=$([[ $MAPPING = /* ]] && echo "$MAPPING" || echo "$DATA_GEN/$MAPPING")
TOTAL=$("$PY" -c "import json; print(sum(1 for d in json.load(open('$MAP_ABS')) if d.get('dst')))")

log() { echo "$(date '+%F %T') $*" >> "$OUTDIR/monitor.log"; }
log "monitor start (category=$CATEGORY total=$TOTAL)"
while true; do
  if grep -q "ALL PARTS DONE" "$OUTDIR/driver.log" 2>/dev/null; then
    log "run complete; monitor exiting"; break
  fi
  drv=$(pgrep -fc "run_stage0.sh $CATEGORY" 2>/dev/null || true); drv=${drv:-0}
  s0=$(pgrep -fc "stage0_filter_components.py.*stage0/$CATEGORY" 2>/dev/null || true); s0=${s0:-0}
  prog=$("$PY" - "$OUTDIR" "$TOTAL" <<'PYEOF'
import json, glob, os, sys
d, tot_exp = sys.argv[1], sys.argv[2]
tot = err = 0
for f in sorted(glob.glob(os.path.join(d, "stage0_part*.json"))):
    if f.endswith(".kept.json"):
        continue
    try:
        rs = json.load(open(f))
    except Exception:
        continue
    tot += len(rs); err += sum(1 for r in rs if "error" in r)
print(f"records={tot}/{tot_exp} errors={err}")
PYEOF
)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i "$GPU" 2>/dev/null | tr -d ' ')
  newest=$(stat -c %Y "$OUTDIR"/stage0_part*.json "$OUTDIR"/part*.log 2>/dev/null | sort -n | tail -1)
  now=$(date +%s); age=$(( now - ${newest:-$now} ))
  log "driver=$drv stage0=$s0 $prog gpu$GPU=[$gpu] last_write_age=${age}s"
  if [ "$drv" -eq 0 ]; then
    if [ "$s0" -gt 0 ]; then
      log "driver dead but a stage0 python still runs; waiting"
    else
      log "driver dead -> RESTARTING"
      nohup bash "$DRIVER" "$CATEGORY" "$MAPPING" "$GPU" "$PART" >> "$OUTDIR/driver_console.log" 2>&1 &
    fi
  elif [ "$s0" -gt 0 ] && [ "$age" -gt "$STALL_SEC" ]; then
    pid=$(pgrep -f "stage0_filter_components.py.*stage0/$CATEGORY" | head -1)
    alive=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "$alive" ] && [ "$alive" -gt "$STALL_SEC" ]; then
      log "STALL: no activity ${age}s, pid $pid alive ${alive}s -> killing (driver resumes)"
      pkill -f "stage0_filter_components.py.*stage0/$CATEGORY" || true
      sleep 20
      nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    else
      log "no activity ${age}s but pid $pid alive only ${alive:-?}s (model load); not killing"
    fi
  fi
  sleep 300
done
