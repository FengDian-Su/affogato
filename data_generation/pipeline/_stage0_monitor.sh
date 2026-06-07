#!/bin/bash
# Auto-monitor the two stage0 shards: restart a shard if it crashes (resume),
# merge a+b when both reach 250, write a DONE marker. Logs to outputs/stage0_monitor.log.
cd /home/michaellee/mclee/affogato/data_generation || exit 1
GEM=/home/michaellee/miniconda3/envs/gemma4/bin/python
LOG=outputs/stage0_monitor.log
log(){ echo "$(date '+%m-%d %H:%M:%S')  $*" >> "$LOG"; }

declare -A CMD
CMD[a]="--gpu 3 --start 0 --end 250 --out outputs/stage0_a.json"
CMD[b]="--gpu 1 --start 250 --end 500 --out outputs/stage0_b.json"
declare -A RESTARTS=( [a]=0 [b]=0 )
MAXRESTART=10
TARGET=250

count(){ python3 -c "import json,os;p='outputs/stage0_$1.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)" 2>/dev/null || echo 0; }

rm -f outputs/stage0_MONITOR_DONE
log "monitor started (pid $$)"
while true; do
  da=0; db=0
  for s in a b; do
    cnt=$(count "$s")
    if tmux has-session -t "stage0$s" 2>/dev/null; then
      :   # still running
    elif [ "$cnt" -ge "$TARGET" ]; then
      [ "$s" = a ] && da=1 || db=1
    else
      if [ "${RESTARTS[$s]}" -ge "$MAXRESTART" ]; then
        log "shard $s exceeded $MAXRESTART restarts at $cnt/$TARGET; giving up"
        [ "$s" = a ] && da=1 || db=1
      else
        RESTARTS[$s]=$(( RESTARTS[$s] + 1 ))
        log "shard $s DOWN at $cnt/$TARGET -> restart #${RESTARTS[$s]} (resume)"
        tmux new-session -d -s "stage0$s" "$GEM pipeline/stage0_filter_components.py ${CMD[$s]} >> outputs/stage0_$s.log 2>&1; echo \"EXIT=\$? @ \$(date)\" >> outputs/stage0_$s.log"
        sleep 5
      fi
    fi
  done
  if [ "$da" = 1 ] && [ "$db" = 1 ]; then
    log "both shards complete -> merging (a restarts=${RESTARTS[a]}, b restarts=${RESTARTS[b]})"
    python3 - <<'PY' >> "$LOG" 2>&1
import json, os
load = lambda p: json.load(open(p)) if os.path.exists(p) else []
merged = load("outputs/stage0_a.json") + load("outputs/stage0_b.json")
json.dump(merged, open("outputs/stage0_filtered.json", "w"), indent=2, ensure_ascii=False)
kept = [r for r in merged if r.get("keep")]
json.dump(kept, open("outputs/stage0_filtered.kept.json", "w"), indent=2, ensure_ascii=False)
corrupt = sum(1 for r in merged if str(r.get("skip_reason", "")).startswith("corrupt"))
print(f"MERGED {len(merged)} records | {len(kept)} kept | {corrupt} corrupt-skipped")
PY
    log "DONE"
    touch outputs/stage0_MONITOR_DONE
    break
  fi
  sleep 45
done
