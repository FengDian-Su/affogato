#!/usr/bin/env bash
# Stage2 watchdog for ONE gpu's object range:
#   monitor_stage2.sh <gpu> <start> <end> <in_json> <out_dir>
# Every 5 min logs progress (objects in [start,end] that have output) + rate +
# ETA, restarts a dead driver, and kills a stalled molmo python (no new object
# for 45 min while the process is itself older than that — a fresh one is still
# loading Molmo+SAM2). run_stage2.sh resumes via --skip_existing. Exits once the
# driver logs DRIVER DONE for this range.
set -u
GPU=${1:?}; START=${2:?}; END=${3:?}; IN=${4:?}; OUT=${5:?}
STALL_SEC=2700
PY=/home/michaellee/miniconda3/envs/mm/bin/python
DG=/home/michaellee/mclee/affogato/data_generation
MAP=${MAP:-dataset/daily_used_to_affogato.json}
LOG=$OUT/monitor_gpu$GPU.log
TOTAL=$(( END - START ))
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

count() {   # objects in [START,END) that have >=1 written query dir (meta.json)
  "$PY" - "$IN" "$OUT" "$START" "$END" <<'PYEOF'
import json, os, sys
inp, out, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
recs = [r for r in json.load(open(inp)) if not r.get("error")]
oids = [r["object_id"] for r in recs[lo:hi]]
done = 0
for oid in oids:
    d = os.path.join(out, oid)
    if os.path.isdir(d) and any(
        os.path.exists(os.path.join(d, q, "meta.json")) for q in os.listdir(d)):
        done += 1
print(done)
PYEOF
}

t0=$(date +%s); n0=$(count); last_n=$n0; last_change=$t0
log "monitor start gpu=$GPU range=[$START:$END] total=$TOTAL already=$n0"

while true; do
  if grep -q "DRIVER DONE gpu$GPU \[$START:$END\]" "$OUT/driver_gpu$GPU.log" 2>/dev/null; then
    log "run complete ($(count)/$TOTAL); monitor exiting"; break
  fi
  n=$(count); now=$(date +%s)
  rate=$(awk -v a="$n" -v b="$n0" -v t="$((now-t0))" 'BEGIN{printf "%.1f",(a>b&&t>0)?t/(a-b):0}')
  eta=$(awk -v r="$rate" -v left="$((TOTAL-n))" 'BEGIN{printf "%.1f",(r>0)?r*left/3600:-1}')
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i "$GPU" 2>/dev/null | tr -d ' ')
  drv=$(pgrep -fc "run_stage2.sh $GPU $START $END" 2>/dev/null || true); drv=${drv:-0}
  py=$(pgrep -f "stage2_v2.py.*--gpu $GPU" 2>/dev/null | head -1 || true)
  log "progress=$n/$TOTAL rate=${rate}s/obj eta=${eta}h gpu=$gpu driver=$drv py=${py:-none}"
  if [ "$n" -gt "$last_n" ]; then last_n=$n; last_change=$now; fi
  if [ "$drv" -eq 0 ]; then
    log "DRIVER DEAD -> restarting [$START:$END] on gpu$GPU"
    nohup bash "$DG/pipeline/run_stage2.sh" "$GPU" "$START" "$END" "$IN" "$MAP" "$OUT" >/dev/null 2>&1 &
    last_change=$now
  elif [ -n "${py:-}" ] && [ $((now-last_change)) -gt "$STALL_SEC" ]; then
    age=$(( now - $(date -d "$(ps -o lstart= -p "$py")" +%s) ))
    if [ "$age" -gt "$STALL_SEC" ]; then
      log "STALLED: no new object for $((now-last_change))s, python $py age ${age}s -> kill (driver resumes)"
      kill -9 "$py" 2>/dev/null || true; last_change=$now
    fi
  fi
  sleep 300
done
