#!/usr/bin/env bash
# Keep every stage2 replay shard alive inside its tmux window.
#
# WORKERS is "<gpu>:<shard>" per entry; NSHARDS must match what the chains were
# launched with or a restarted shard would cover a different range.
#
# Two independent failure modes are covered:
#   DEAD   the chain shell is gone (run_stage2.sh exhausted its 8 retries, the
#          tmux window was killed, the box rebooted) -> relaunch that shard.
#   WEDGED the chain lives but its run log has not grown for STALL_S (NFS render
#          read hanging, a SAM2 call that never returns) -> kill that shard's
#          python so run_stage2.sh's own retry loop starts a fresh one.
#
# Any restart is preceded by repair_stage2_partial.py, because --skip_existing
# accepts a truncated scores.npz that a killed worker left behind and would then
# skip that query for the rest of the run.
set -u
DG=/home/michaellee/mclee/affogato/data_generation
PY=$HOME/miniconda3/envs/molmo/bin/python
CHECK_S=${CHECK_S:-300}
STALL_S=${STALL_S:-1800}     # p99 object is 134s, so 30min without a line is wedged
SESSION=${SESSION:-stage2}
NSHARDS=${NSHARDS:-6}
WORKERS=${WORKERS:-"0:0 0:1 0:2 3:3 3:4 3:5"}
cd "$DG" || exit 1
LOG=outputs/stage2/guardian.log
mkdir -p outputs/stage2

# Anchored on the bash process: the tmux SERVER cmdline embeds the chain command
# it first spawned, so a loose match reports a dead chain alive forever.
alive () { pgrep -f "^bash .*chain_stage2_replay\.sh $1 $2 " >/dev/null 2>&1; }

# A shard's own log, not the GPU's: sibling shards share a card and would mask
# each other's stalls.
shard_log () { echo "outputs/stage2/chain_gpu${1}s${2}.log"; }

worker_range () {  # echo "<start> <end>" of the range this shard is running now
  grep -oE "\[[0-9]+:[0-9]+\] start" "$(shard_log "$1" "$2")" 2>/dev/null | tail -1 \
    | tr -d '[]' | sed 's/ start//' | tr ':' ' '
}

relaunch () {     # $1=gpu $2=shard
  # A chain can die while its run_stage2.sh / python children keep running.
  # Relaunching on top of those would put two workers on one range, racing for
  # the same output dirs, so reap this shard's orphans first.
  local r; r=$(worker_range "$1" "$2")
  if [ -n "$r" ]; then
    set -- "$1" "$2" $r
    pkill -f "run_stage2.sh $1 $3 $4 " 2>>"$LOG"
    pkill -f "stage2_v2.py .*--start $3 --end $4 --gpu $1" 2>>"$LOG"
    sleep 10
    pkill -9 -f "stage2_v2.py .*--start $3 --end $4 --gpu $1" 2>>"$LOG"
    sleep 2
  fi
  echo "[$(date '+%F %T')] repairing partial writes before restarting gpu$1 s$2" | tee -a "$LOG"
  "$PY" pipeline/repair_stage2_partial.py --root outputs/stage2 --since 180 >> "$LOG" 2>&1
  tmux respawn-window -k -t "${SESSION}:g$1s$2" \
      "bash $DG/pipeline/chain_stage2_replay.sh $1 $2 $NSHARDS" 2>>"$LOG" \
    || tmux new-window -d -t "$SESSION" -n "g$1s$2" \
        "bash $DG/pipeline/chain_stage2_replay.sh $1 $2 $NSHARDS" 2>>"$LOG"
  echo "[$(date '+%F %T')] relaunched gpu$1 s$2" | tee -a "$LOG"
}

declare -A LAST_SIZE LAST_MOVE
for w in $WORKERS; do LAST_SIZE[$w]=-1; LAST_MOVE[$w]=$(date +%s); done

echo "[$(date '+%F %T')] guardian up: [$WORKERS] check ${CHECK_S}s stall ${STALL_S}s" | tee -a "$LOG"
while true; do
  sleep "$CHECK_S"
  for w in $WORKERS; do
    g=${w%%:*}; s=${w##*:}
    grep -qF "CHAIN DONE gpu$g s$s" "$(shard_log "$g" "$s")" 2>/dev/null && continue
    if ! alive "$g" "$s"; then
      echo "[$(date '+%F %T')] gpu$g s$s chain DEAD" | tee -a "$LOG"
      relaunch "$g" "$s"
      LAST_MOVE[$w]=$(date +%s); LAST_SIZE[$w]=-1
      continue
    fi
    r=$(worker_range "$g" "$s"); [ -z "$r" ] && continue
    set -- $r
    f=$(ls -t outputs/stage2/*/run_gpu${g}_$1_$2.log 2>/dev/null | head -1)
    [ -z "$f" ] && continue
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0); now=$(date +%s)
    if [ "$sz" != "${LAST_SIZE[$w]}" ]; then
      LAST_SIZE[$w]=$sz; LAST_MOVE[$w]=$now
    elif [ $(( now - ${LAST_MOVE[$w]} )) -ge "$STALL_S" ]; then
      echo "[$(date '+%F %T')] gpu$g s$s WEDGED ($(( (now - ${LAST_MOVE[$w]}) / 60 ))m no growth) — killing worker" | tee -a "$LOG"
      pkill -f "stage2_v2.py .*--start $1 --end $2 --gpu $g" 2>>"$LOG"
      LAST_MOVE[$w]=$now
    fi
  done
done
