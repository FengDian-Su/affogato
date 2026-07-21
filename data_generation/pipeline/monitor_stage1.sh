#!/usr/bin/env bash
# Stage1 watchdog for ONE gpu's object range:  monitor_stage1.sh <gpu> <start> <end> [step]
# Every 5 min appends progress + rate + ETA to outputs/stage1/daily_used/monitor_gpu<gpu>.log,
# restarts a dead driver, and kills a stalled stage1 python (no new objects for 45 min while the
# process itself has been alive that long — a fresh process is still loading 48GB of weights, which
# takes ~90s, so a young process is never "stalled"). run_stage1.sh retries the same shard, which
# resumes from its own json. Exits once the driver logs DRIVER DONE for this range.
set -u
GPU=${1:?usage: monitor_stage1.sh <gpu> <start> <end> [step]}
START=${2:?}; END=${3:?}; STEP=${4:-5000}
STALL_SEC=2700
PY=/home/michaellee/miniconda3/envs/mm/bin/python
DG=/home/michaellee/mclee/affogato/data_generation
OUT=$DG/outputs/stage1/daily_used
LOG=$OUT/monitor_gpu$GPU.log
TOTAL=$(( END - START ))

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

count() {   # objects + errors written so far inside [START,END)
  "$PY" - "$OUT" "$START" "$END" <<'PYEOF'
import glob, json, os, re, sys, time
out, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
n = err = 0
for f in glob.glob(os.path.join(out, "stage1_*.json")):
    m = re.search(r"stage1_(\d{6})_(\d{6})\.json$", f)
    if not m or int(m.group(1)) < lo or int(m.group(2)) > hi:
        continue
    rs = None
    for _ in range(5):
        # stage1_v2 rewrites each shard json WHOLE every 8-object window, so a
        # read can land mid-write. Skipping the file instead would drop its
        # entire count and make progress appear to go BACKWARDS, which also
        # corrupts the rate/ETA. Retry until it parses.
        try:
            rs = json.load(open(f)); break
        except Exception:
            time.sleep(2)
    if rs is None:
        continue
    n += len(rs); err += sum(1 for r in rs if r.get("error"))
print(f"{n} {err}")
PYEOF
}

t0=$(date +%s); read -r n0 _ <<<"$(count)"
last_n=$n0; last_change=$t0
log "monitor start gpu=$GPU range=[$START:$END] total=$TOTAL already=$n0"

while true; do
  if grep -q "DRIVER DONE gpu$GPU \[$START:$END\]" "$OUT/driver_gpu$GPU.log" 2>/dev/null; then
    read -r n _ <<<"$(count)"; log "run complete ($n/$TOTAL); monitor exiting"; break
  fi

  read -r n err <<<"$(count)"
  now=$(date +%s)
  rate=$(awk -v a="$n" -v b="$n0" -v t="$((now - t0))" 'BEGIN{printf "%.2f", (a>b && t>0)? t/(a-b) : 0}')
  eta=$(awk -v r="$rate" -v left="$((TOTAL - n))" 'BEGIN{printf "%.1f", (r>0)? r*left/3600 : -1}')
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i "$GPU" 2>/dev/null | tr -d ' ')
  drv=$(pgrep -fc "run_stage1.sh $GPU $START $END" 2>/dev/null || true); drv=${drv:-0}
  py=$(pgrep -f "stage1_v2.py.*--gpu $GPU" 2>/dev/null | head -1 || true)
  log "progress=$n/$TOTAL errors=$err rate=${rate}s/obj eta=${eta}h gpu=$gpu driver=$drv py=${py:-none}"

  if [ "$n" -gt "$last_n" ]; then last_n=$n; last_change=$now; fi

  if [ "$drv" -eq 0 ]; then
    log "DRIVER DEAD -> restarting [$START:$END] on gpu$GPU"
    nohup bash "$DG/pipeline/run_stage1.sh" "$GPU" "$START" "$END" "$STEP" >/dev/null 2>&1 &
    last_change=$now
  elif [ -n "${py:-}" ] && [ $((now - last_change)) -gt "$STALL_SEC" ]; then
    # only if the process itself is older than the stall window
    age=$(( now - $(date -d "$(ps -o lstart= -p "$py")" +%s) ))
    if [ "$age" -gt "$STALL_SEC" ]; then
      log "STALLED: no new objects for $((now - last_change))s, python $py age ${age}s -> kill (driver retries the shard)"
      kill -9 "$py" 2>/dev/null || true
      last_change=$now
    fi
  fi
  sleep 300
done
