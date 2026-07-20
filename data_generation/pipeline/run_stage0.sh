#!/usr/bin/env bash
# Generic stage0 driver: run one gObjaverse category to completion, in parts, with resume-retry.
#
#   run_stage0.sh <category> <mapping.json> [gpu] [part_size]
#   run_stage0.sh electronics dataset/electronics_to_affogato.json 0 10000
#
# Output: outputs/stage0/<category>/stage0_part{N}.json (+ .kept.json), driver.log, partN.log.
# Re-invoking a part RESUMES it (done objects skipped, error stubs re-run), so crashes self-heal.
# A fast-exiting attempt (<300s with no new records = engine-init crash) backs off and retries
# instead of counting as a no-progress strike; only a full-length attempt with zero new records
# gives up on a part. "ALL PARTS DONE" is written only after every part verifies complete, so a
# watchdog that greps for it cannot be fooled by a wholesale failure (07-18 NFS-outage lesson).
set -u
CATEGORY=${1:?usage: run_stage0.sh <category> <mapping.json> [gpu] [part_size]}
MAPPING=${2:?usage: run_stage0.sh <category> <mapping.json> [gpu] [part_size]}
GPU=${3:-0}
PART=${4:-10000}
MAX_ATTEMPTS=12

PY=/home/michaellee/miniconda3/envs/gemma4/bin/python
export LD_LIBRARY_PATH=/home/michaellee/miniconda3/envs/gemma4/lib
REPO=/home/michaellee/mclee/affogato
DATA_GEN=$REPO/data_generation
STAGE0=$DATA_GEN/pipeline/stage0_filter_components.py
OUTDIR=$DATA_GEN/outputs/stage0/$CATEGORY
MAP_ABS=$([[ $MAPPING = /* ]] && echo "$MAPPING" || echo "$DATA_GEN/$MAPPING")

mkdir -p "$OUTDIR"
log() { echo "$(date '+%F %T') $*" >> "$OUTDIR/driver.log"; }
TOTAL=$("$PY" -c "import json; print(sum(1 for d in json.load(open('$MAP_ABS')) if d.get('dst')))")
NPARTS=$(( (TOTAL + PART - 1) / PART ))
log "driver start: category=$CATEGORY $TOTAL objects -> $NPARTS parts (size $PART), gpu $GPU"

status() {   # <file> <expected> -> "done=N err=N complete=0|1"
  "$PY" - "$1" "$2" <<'PYEOF'
import json, sys
try:
    rs = json.load(open(sys.argv[1]))
except Exception:
    rs = []
n = int(sys.argv[2])
err = sum(1 for r in rs if "error" in r)
print(f"done={len(rs)-err} err={err} complete={int(len(rs)==n and err==0)}")
PYEOF
}

wait_gpu_free() {
  for w in $(seq 1 16); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 0)
    [ "${used:-0}" -lt 5000 ] && return
    log "gpu$GPU still holds ${used}MiB; waiting"; sleep 15
  done
}

for i in $(seq 0 $((NPARTS-1))); do
  start=$((i*PART)); end=$(( (i+1)*PART )); [ "$end" -gt "$TOTAL" ] && end=$TOTAL
  out="$OUTDIR/stage0_part${i}.json"; expected=$((end-start))
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    st=$(status "$out" "$expected")
    case "$st" in *complete=1*) break;; esac
    wait_gpu_free
    log "part$i attempt $attempt [$start,$end) $st"
    t0=$(date +%s)
    "$PY" "$STAGE0" --gpu "$GPU" --mapping "$MAP_ABS" --start "$start" --end "$end" \
        --out "$out" >> "$OUTDIR/part${i}.log" 2>&1
    dur=$(( $(date +%s) - t0 ))
    st2=$(status "$out" "$expected")
    if [ "$st2" = "$st" ]; then
      if [ "$dur" -lt 300 ]; then
        log "part$i attempt $attempt CRASHED fast (${dur}s, no new records); backoff 60s"
        sleep 60
      else
        log "part$i NO PROGRESS after a full ${dur}s attempt ($st2); leaving for inspection"
        break
      fi
    fi
    sleep 10
  done
  log "part$i finished: $(status "$out" "$expected")"
done

incomplete=""
for i in $(seq 0 $((NPARTS-1))); do
  start=$((i*PART)); end=$(( (i+1)*PART )); [ "$end" -gt "$TOTAL" ] && end=$TOTAL
  case "$(status "$OUTDIR/stage0_part${i}.json" "$((end-start))")" in
    *complete=1*) ;;
    *) incomplete="$incomplete part$i";;
  esac
done
if [ -z "$incomplete" ]; then
  log "ALL PARTS DONE"
else
  log "PASS ENDED; INCOMPLETE:$incomplete (watchdog starts another pass)"
fi
