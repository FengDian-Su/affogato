#!/usr/bin/env bash
# One line of stage2-replay progress, for the hourly heartbeat.
# Targets are what the frozen point archive can actually regenerate, not the
# stage1 totals: the 1,480 known_exclusions queries have no saved points.
set -u
DG=/home/michaellee/mclee/affogato/data_generation
cd "$DG" || exit 1
TOTAL=310234

done_n=$(find outputs/stage2 -name scores.npz 2>/dev/null | wc -l)
pct=$(awk -v d="$done_n" -v t="$TOTAL" 'BEGIN{printf "%.1f", 100*d/t}')

# rate over every run log written so far, per GPU
rate_line=""
tot_s=0; tot_o=0
for g in 0 3; do
  s=0; o=0
  for f in outputs/stage2/*/run_gpu${g}_*.log; do
    [ -e "$f" ] || continue
    read -r fs fo < <(grep -oE "queries  [0-9]+s" "$f" | grep -oE "[0-9]+" \
                      | awk '{s+=$1;n++} END {print s+0, n+0}')
    s=$((s+fs)); o=$((o+fo))
  done
  tot_s=$((tot_s+s)); tot_o=$((tot_o+o))
  if [ "$o" -gt 0 ]; then
    rate_line="$rate_line gpu$g=$(awk -v s="$s" -v o="$o" 'BEGIN{printf "%.1f", s/o}')s/obj($o)"
  else
    rate_line="$rate_line gpu$g=idle"
  fi
done

# health
bad=""
# unquoted default on purpose: quotes here would suppress word splitting and
# collapse the list into one token (g=0, sh=5 -> a phantom "g0s5-DEAD")
for w in ${WORKERS:-0:0 0:1 0:2 0:3 3:4 3:5 3:6 3:7}; do
  g=${w%%:*}; sh=${w##*:}
  grep -qF "CHAIN DONE gpu$g s$sh" outputs/stage2/chain_gpu${g}s${sh}.log 2>/dev/null && continue
  pgrep -f "^bash .*chain_stage2_replay\.sh $g $sh " >/dev/null || bad="$bad g${g}s${sh}-DEAD"
done
pgrep -f "[g]uardian_stage2_replay.sh" >/dev/null || bad="$bad GUARDIAN-DEAD"
fails=$(grep -ch "FAILED" outputs/stage2/*/run_gpu*.log 2>/dev/null | awk '{s+=$1} END {print s+0}')
gaveup=$(grep -lF "GAVE UP" outputs/stage2/*/driver_gpu*.log 2>/dev/null | wc -l)
[ "$gaveup" -gt 0 ] && bad="$bad GAVE-UP:$gaveup"

# ETA from the observed two-GPU throughput
# Per-object seconds RISES as workers are added (they contend); only the
# aggregate matters, so divide by the number of workers actually running.
nw=$(pgrep -cf "^bash .*chain_stage2_replay\.sh " 2>/dev/null || echo 1)
[ "${nw:-0}" -lt 1 ] && nw=1
eta="?"
if [ "$tot_o" -gt 20 ] && [ "$tot_s" -gt 0 ]; then
  eta=$(awk -v d="$done_n" -v t="$TOTAL" -v s="$tot_s" -v o="$tot_o" -v w="$nw" \
        'BEGIN{qpo=(d>0&&o>0)?d/o:5; spq=(s/o)/qpo/w; printf "%.1fd", (t-d)*spq/86400}')
fi

# Lead with the real clock: these lines are read hours later, out of order,
# and a timestamp inferred by adding an interval drifts without anyone noticing.
printf "%s  stage2-replay %s/%s (%s%%) |%s | ETA %s | fails %s%s\n" \
       "$(date '+%m-%d %H:%M')" "$done_n" "$TOTAL" "$pct" "$rate_line" "$eta" "$fails" \
       "${bad:+ | ALERT:$bad}"
