#!/bin/bash
# One shard of the LLM judge over the whole corpus, restarted (resuming from its shard file)
# if it fails. Each attempt runs in its own process group so a crashed parent cannot leave an
# orphaned vLLM EngineCore holding the GPU. Runs in a tmux window (start_tmux.sh), which shows the
# attempts and the judge's progress lines; everything also goes to the shard's log.
# The GPU is shared: an attempt uses the full setting (gpu-mem 0.8, batch 32) when the GPU is free,
# the shared one (0.64, batch 16; validated next to another user's ~28 GB job) when it is not, and
# waits, checking every second and without spending an attempt, when even that does not fit. A shared
# attempt is restarted at the full setting as soon as no other job has been on the GPU for about a
# second (checked every 0.5 s); if another job takes the GPU while it restarts, it falls back to the
# shared setting. Retries
# come after 5 s, so a crash leaves the GPU free only briefly. An attempt already running for this
# shard (from a supervisor that was replaced) is waited for first, not started twice.
#   usage: run_corpus.sh GPU SHARD NSHARDS [AXES]
GPU=$1; SHARD=$2; NSHARDS=$3; AXES=${4:-task,role}
cd /home/michaellee/mclee/affogato || exit 1
ROOT=data_verification/outputs/quality_review/corpus
OUT=$ROOT/qwen38-27b
LOG=$OUT/run_shard$SHARD.log
mkdir -p "$OUT"
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
foreign() {                   # is a process outside attempt $1's process group on this GPU?
  local p
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1 == u {print $2}'); do
    [ "$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')" != "$1" ] && return 0
  done
  return 1
}
watch_shared() {              # restart shared attempt $1 at the full setting once the GPU is ours alone
  # Switch back only after the GPU is clear of other jobs for CLEAR_S (10) continuous seconds. The other
  # user's scheduler runs jobs back-to-back with a few-second gap between them, so a short threshold
  # just thrashes (restart at 0.8 -> their next job is already there -> fall back). 30 s skips the
  # gaps and only fires when their whole schedule is actually done.
  ( CLEAR_S=10; clear_since=0
    while kill -0 "$1" 2>/dev/null; do
      if foreign "$1"; then clear_since=0
      else [ $clear_since -eq 0 ] && clear_since=$(date +%s)
      fi
      if [ $clear_since -ne 0 ] && [ $(( $(date +%s) - clear_since )) -ge $CLEAR_S ]; then
        echo "[$(date '+%F %T')] gpu$GPU: no other job for ${CLEAR_S}s; restarting at the full setting" | tee -a "$LOG"
        kill -TERM -- -"$1" 2>/dev/null; break
      fi
      sleep 1
    done ) &
}
PAUSE=$ROOT/PAUSE_shard$SHARD
REQUEST=/home/michaellee/mclee/robocasa_rc/REQUEST_GPU0
RELEASED=/home/michaellee/mclee/robocasa_rc/GPU0_RELEASED
paused() {                    # paused by hand or by a GPU0 hand-over: never start an attempt
  if [ -e "$PAUSE" ]; then
    echo "[$(date '+%F %T')] gpu$GPU shard $SHARD paused ($PAUSE); not restarting" | tee -a "$LOG"
    exec sleep infinity
  fi
}
if [ "$GPU" = 0 ]; then       # hand GPU0 over when robocasa asks (user-approved, 2026-10-14)
  ( while [ ! -e "$REQUEST" ]; do sleep 1; done
    touch "$PAUSE"
    echo "[$(date '+%F %T')] gpu0: $REQUEST seen; pausing shard $SHARD and releasing GPU0" | tee -a "$LOG"
    a=$(pgrep -f "^[^ ]*python[^ ]* -m data_verification[.]quality_review[.]qwen_judge run .*--shard $SHARD/$NSHARDS " | head -1)
    [ -n "$a" ] && kill -TERM -- -"$a" 2>/dev/null
    for i in $(seq 1 60); do [ -n "$a" ] && pgrep -g "$a" > /dev/null || break; sleep 1; done
    [ -n "$a" ] && kill -KILL -- -"$a" 2>/dev/null
    while nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "$uuid"; do
      mine=0; for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1 == u {print $2}'); do
        [ "$(ps -o uid= -p "$p" 2>/dev/null | tr -d ' ')" = "$(id -u)" ] && mine=1; done
      [ $mine -eq 0 ] && break; sleep 1
    done
    touch "$RELEASED"
    echo "[$(date '+%F %T')] gpu0 released; wrote $RELEASED" | tee -a "$LOG" ) &
fi
paused
existing=$(pgrep -f "^[^ ]*python[^ ]* -m data_verification[.]quality_review[.]qwen_judge run .*--shard $SHARD/$NSHARDS " | head -1)   # the judge's python only, never a shell quoting this pattern
if [ -n "$existing" ]; then
  echo "[$(date '+%F %T')] gpu$GPU shard $SHARD: waiting for the running attempt $existing" | tee -a "$LOG"
  tail -n 0 -F --pid=$existing "$LOG" 2>/dev/null | grep --line-buffered -E "requests to run|s/request|Error" &
  ps -o args= -p "$existing" | grep -q -- "--gpu-mem 0.64" && watch_shared "$existing"
  while kill -0 "$existing" 2>/dev/null; do sleep 1; done
  while pgrep -g "$existing" > /dev/null; do sleep 1; done
fi
attempt=0; waiting=0; stable=0
while [ $attempt -lt 20 ]; do
  paused
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
  if [ "$free" -ge 79000 ]; then MEM=0.8; BATCH=32
  elif [ "$free" -ge 63000 ]; then MEM=0.64; BATCH=16
  else
    [ $waiting -eq 0 ] && echo "[$(date '+%F %T')] gpu$GPU has $free MiB free; waiting for room" | tee -a "$LOG"
    waiting=1; stable=0; sleep 1; continue
  fi
  if [ $waiting -eq 1 ]; then   # another job had the GPU: start only once the room has lasted 10 s
    stable=$(( stable + 1 ))    # (a scheduler that runs jobs back to back leaves only seconds between them)
    if [ $stable -lt 10 ]; then sleep 1; continue; fi
  fi
  waiting=0; stable=0; attempt=$(( attempt + 1 ))
  echo "[$(date '+%F %T')] gpu$GPU shard $SHARD/$NSHARDS $AXES attempt $attempt (gpu-mem $MEM, batch $BATCH)" | tee -a "$LOG"
  setsid -w env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
      LD_LIBRARY_PATH=/home/michaellee/miniconda3/envs/gemma4/lib \
      /home/michaellee/miniconda3/envs/gemma4/bin/python -m data_verification.quality_review.qwen_judge run \
      --manifest $ROOT/manifest.jsonl --split all --axes $AXES --shard $SHARD/$NSHARDS \
      --batch $BATCH --gpu-mem $MEM --out $OUT >> "$LOG" 2>&1 &
  pid=$!
  tail -n 0 -F --pid=$pid "$LOG" 2>/dev/null | grep --line-buffered -E "requests to run|s/request|Error" &
  [ "$MEM" != 0.8 ] && watch_shared $pid
  wait $pid; status=$?
  kill -- -$pid 2>/dev/null
  while pgrep -g $pid > /dev/null; do sleep 1; done     # our memory is free before the next check
  if [ $status -eq 0 ]; then
    echo "[$(date '+%F %T')] gpu$GPU shard $SHARD done" | tee -a "$LOG"; exit 0
  fi
  echo "[$(date '+%F %T')] gpu$GPU shard $SHARD exited $status; retrying in 5 s" | tee -a "$LOG"
  sleep 5
done
exit 1
