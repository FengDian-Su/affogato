#!/bin/bash
# Live status of the corpus heatmap run (tmux window "status" of session corpus_heatmap), refreshed
# every minute: per shard the heatmap rows, speed, ETA and failures; the supervisors, NFS and GPUs;
# and the monitor's recent events, including the score checks.
cd /home/michaellee/mclee/affogato || exit 1
ROOT=data_verification/outputs/quality_review/corpus
OUT=$ROOT/qwen38-27b
BASE=310234                   # task/role rows already in each shard file
TOTAL=155117                  # heatmap requests per shard
GPU=(0 3)
while true; do
  clear
  echo "Corpus heatmap judge (Qwen3.8-27B) | $(date '+%F %T') | refreshes every minute"
  echo
  for s in 0 1; do
    rows=$(( $(wc -l < $OUT/shard$s.jsonl) - BASE ))
    failed=$(grep -c '"axis": "heatmap", "score": null' $OUT/shard$s.jsonl)
    line=$(grep "s/request" $OUT/run_shard$s.log | tail -1)
    rate=$(echo "$line" | grep -o "[0-9.]* s/request")
    eta=$(echo "$line" | grep -o "eta [0-9.]*" | grep -o "[0-9]*" | head -1)
    finish=$([ -n "$eta" ] && date -d "+$eta min" '+%m-%d %H:%M' || echo "?")
    sup=$(pgrep -f "run_corpus.sh ${GPU[s]} $s 2" | head -1)
    printf "shard %s (gpu%s): %6d / %d  %3d%%  | %s | done ~%s | failed %s | supervisor %s\n" \
      $s ${GPU[s]} $rows $TOTAL $(( rows * 100 / TOTAL )) "${rate:-starting}" "$finish" $failed "${sup:-none}"
  done
  echo
  if timeout 20 ls /nfs_drive/gobjaverse > /dev/null 2>&1; then echo "NFS: up"; else echo "NFS: DOWN"; fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader -i 0,3 | sed 's/^/gpu /'
  echo
  echo "Monitor events (score checks in full: scores.log):"
  grep -v "^    gpu\|gpu[03]: [0-9]*/$TOTAL" $ROOT/monitor.log | tail -6 | cut -c1-200 | sed 's/^/  /'
  sleep 60
done
