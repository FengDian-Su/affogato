#!/bin/bash
# Re-run stage1 ONLY on the existing 40-object subset (components unchanged), after prompt fixes.
# Sharded GPU3 0-20 / GPU1 20-40, then merge + gallery + review-inputs + DONE marker.
cd /home/michaellee/mclee/affogato/data_generation || exit 1
GEM=/home/michaellee/miniconda3/envs/gemma4/bin/python
MM=/home/michaellee/miniconda3/envs/mm/bin/python
LOG=outputs/stage1_rerun.log
log(){ echo "$(date '+%m-%d %H:%M:%S')  $*" >> "$LOG"; }
: > "$LOG"
rm -f outputs/STAGE1_RERUN_DONE outputs/stage1_a.json outputs/stage1_b.json outputs/stage1_a.log outputs/stage1_b.log

SUB=outputs/stage0_gallery_subset.json
log "stage1 re-run launched on $SUB (prompt fixes: honest-justification + rigid-body + region-grounding + body-injection)"
tmux new-session -d -s s1ra "$GEM pipeline/stage1_task_role_assemble.py --gpu 3 --in $SUB --start 0  --end 20 --out outputs/stage1_a.json > outputs/stage1_a.log 2>&1; echo EXIT=\$? >> outputs/stage1_a.log"
tmux new-session -d -s s1rb "$GEM pipeline/stage1_task_role_assemble.py --gpu 3 --in $SUB --start 20 --end 40 --out outputs/stage1_b.json > outputs/stage1_b.log 2>&1; echo EXIT=\$? >> outputs/stage1_b.log"
sleep 15
while tmux has-session -t s1ra 2>/dev/null || tmux has-session -t s1rb 2>/dev/null; do sleep 30; done
log "stage1 shards done; merging"
$MM -c "import json,os
a=json.load(open('outputs/stage1_a.json')) if os.path.exists('outputs/stage1_a.json') else []
b=json.load(open('outputs/stage1_b.json')) if os.path.exists('outputs/stage1_b.json') else []
json.dump(a+b,open('outputs/stage1_dataset.json','w'),indent=2,ensure_ascii=False)
print('merged',len(a)+len(b))" >> "$LOG" 2>&1
$MM viz/build_stage1_gallery.py --in outputs/stage1_dataset.json --out galleries/stage1_gallery.html >> "$LOG" 2>&1
$MM viz/build_review_inputs.py --in outputs/stage1_dataset.json --out review/inputs >> "$LOG" 2>&1
log "DONE (gallery + review inputs rebuilt)"
touch outputs/STAGE1_RERUN_DONE
