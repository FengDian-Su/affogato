#!/bin/bash
# Health monitor for the whole-corpus heatmap judge run (tmux session `corpus_monitor`), independent
# of any Claude session. The heatmap rows are appended to the shard files after the task/role rows.
# Every 5 minutes it logs, per shard, the heatmap rows written, the judge's own speed
# and ETA, failed rows, attempts, and the GPU's memory and load. It also acts:
#   - an attempt that has written no rows for 20 minutes is killed (its whole process group, so no
#     EngineCore is left behind), and its supervisor restarts it from the shard file;
#   - a supervisor that is gone before its shard is complete is relaunched in its tmux window, at most
#     5 times per shard;
#   - while the NFS share holding the object views (/nfs_drive) cannot be read, both shards are paused
#     (supervisor and attempt killed) instead of burning their retries, and they are resumed when it is
#     back; a resume does not count as a relaunch;
#   - at every 10% of progress, and once more at the end, corpus_scores.py checks the scores so far
#     (scores.log, scores.jsonl): totals, the window since the last check, per dataset, and agreement
#     with the review run on the reviewed pairs; FLAG marks drift;
# Status: tail -f data_verification/outputs/quality_review/corpus/monitor.log, or tmux attach -t corpus_monitor
cd /home/michaellee/mclee/affogato || exit 1
ROOT=data_verification/outputs/quality_review/corpus        # data, logs, PAUSE_shard* flags
HERE=data_verification/quality_review/corpus_run            # these scripts
OUT=$ROOT/qwen38-27b
LOG=$ROOT/monitor.log
BASE=310234                   # task/role rows already in each shard file
TOTAL=155117                  # heatmap requests per shard: one per pair
AXES=heatmap
SESSION=corpus_heatmap        # tmux session: windows shard0, shard1, monitor, status (start_tmux.sh)
GPU=(0 3)                     # shard s runs on GPU ${GPU[s]}
INTERVAL=300; STALL=1200; MAX_RELAUNCH=5
PY=/home/michaellee/miniconda3/envs/mm/bin/python
declare -a last_rows changed_at relaunches paused

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
count_rows() { if [ -f "$1" ]; then echo $(( $(wc -l < "$1") - BASE )); else echo 0; fi; }
nfs_ok() { timeout 20 ls /nfs_drive/gobjaverse > /dev/null 2>&1; }
launch() {                    # (re)start shard $1's supervisor in its tmux window
  local cmd="exec bash $PWD/$HERE/run_corpus.sh ${GPU[$1]} $1 2 $AXES"
  tmux respawn-window -k -t "$SESSION:shard$1" "$cmd" 2>/dev/null || tmux new-window -d -t "$SESSION" -n "shard$1" "$cmd"
}

say "monitor started (pid $$)"
for s in 0 1; do last_rows[s]=$(count_rows $OUT/shard$s.jsonl); changed_at[s]=$(date +%s); relaunches[s]=0; paused[s]=0; done
next_check=$(( ( (last_rows[0] + last_rows[1]) * 100 / (2 * TOTAL) / 10 + 1 ) * 10 ))
while true; do
  now=$(date +%s); complete=0
  if nfs_ok; then nfs=up; else nfs=down; fi
  for s in 0 1; do
    rows=$(count_rows $OUT/shard$s.jsonl)
    failed=$(grep -c '"axis": "heatmap", "score": null' $OUT/shard$s.jsonl 2>/dev/null); failed=${failed:-0}
    attempts=$(grep -c " attempt " $OUT/run_shard$s.log 2>/dev/null); attempts=${attempts:-0}
    speed=$(grep "s/request" $OUT/run_shard$s.log 2>/dev/null | tail -1 | grep -o "[0-9.]* s/request, eta [0-9.]* min")
    sup=$(pgrep -f "run_corpus.sh ${GPU[s]} $s 2" | head -1)
    [ "$rows" -gt "${last_rows[s]}" ] && changed_at[s]=$now
    last_rows[s]=$rows
    say "shard $s gpu${GPU[s]}: $rows/$TOTAL ($(( rows * 100 / TOTAL ))%) | ${speed:-no progress line yet} | failed $failed | attempts $attempts | supervisor ${sup:-none} | nfs $nfs"
    if [ "$rows" -ge $TOTAL ]; then complete=$(( complete + 1 )); continue; fi
    if [ -e "$ROOT/PAUSE_shard$s" ]; then continue; fi   # paused by hand: never relaunch (delete the file to resume)
    if [ -n "$sup" ] && [ $nfs = down ]; then
      py=$(pgrep -P "$sup" -f qwen_judge | head -1)
      [ -z "$py" ] && py=$(pgrep -f "^[^ ]*python[^ ]* -m data_verification[.]quality_review[.]qwen_judge run .*--shard $s/2 " | head -1)
      say "shard $s: NFS unreadable; pausing (supervisor $sup, attempt ${py:-none}) until it is back"
      kill "$sup" 2>/dev/null
      [ -n "$py" ] && { kill -TERM -- -"$py" 2>/dev/null; sleep 30; kill -KILL -- -"$py" 2>/dev/null; }
      paused[s]=1
    elif [ -n "$sup" ]; then
      py=$(pgrep -P "$sup" -f qwen_judge | head -1)
      [ -z "$py" ] && py=$(pgrep -f "^[^ ]*python[^ ]* -m data_verification[.]quality_review[.]qwen_judge run .*--shard $s/2 " | head -1)   # an attempt the supervisor adopted
      [ -z "$py" ] && changed_at[s]=$now     # the supervisor is waiting for room: that is not a stall
      idle=$(( now - changed_at[s] ))
      if [ $idle -ge $STALL ] && [ -n "$py" ]; then
        say "shard $s: no new rows for $(( idle / 60 )) min; killing attempt $py (process group) so the supervisor restarts it"
        kill -TERM -- -"$py" 2>/dev/null; sleep 30; kill -KILL -- -"$py" 2>/dev/null
        changed_at[s]=$(date +%s)
      fi
    elif [ $nfs = down ]; then
      :                                   # paused: wait for NFS
    elif [ "${paused[s]}" -eq 1 ]; then
      paused[s]=0
      say "shard $s: NFS is back; resuming from $rows/$TOTAL"
      launch $s
      changed_at[s]=$now
    elif [ "${relaunches[s]}" -lt $MAX_RELAUNCH ]; then
      relaunches[s]=$(( relaunches[s] + 1 ))
      say "shard $s: supervisor gone at $rows/$TOTAL; relaunching (${relaunches[s]}/$MAX_RELAUNCH)"
      launch $s
      changed_at[s]=$now
    else
      say "shard $s: supervisor gone and all $MAX_RELAUNCH relaunches used; needs a person"
    fi
  done
  pct=$(( (last_rows[0] + last_rows[1]) * 100 / (2 * TOTAL) ))
  if [ $pct -ge $next_check ]; then
    say "$($PY $HERE/corpus_scores.py 2>&1 | tail -1)"
    next_check=$(( (pct / 10 + 1) * 10 ))
  fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 0,3 | sed 's/^/    gpu /' | tee -a "$LOG"
  if [ $complete -eq 2 ] && ! pgrep -f "run_corpus.sh" > /dev/null; then
    say "both shards complete: $(cat $OUT/shard*.jsonl | grep -c '"axis": "heatmap"') heatmap rows, $(cat $OUT/shard*.jsonl | grep -c '"axis": "heatmap", "score": null') failed"
    say "monitor done"
    exec sleep infinity
  fi
  sleep $INTERVAL
done
