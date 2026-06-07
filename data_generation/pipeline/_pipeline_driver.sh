#!/bin/bash
cd /home/michaellee/mclee/affogato/data_generation || exit 1
GEM=/home/michaellee/miniconda3/envs/gemma4/bin/python
LOG=outputs/pipeline_driver.log
log(){ echo "$(date '+%m-%d %H:%M:%S')  $*" >> "$LOG"; }
rm -f outputs/STAGE1_PIPELINE_DONE
log "driver started; waiting for stage0 monitor..."
while [ ! -f outputs/stage0_MONITOR_DONE ]; do sleep 30; done
KEPT=$(python3 -c "import json;print(len(json.load(open('outputs/stage0_filtered.kept.json'))))" 2>/dev/null)
log "stage0 DONE (kept=$KEPT); building 40-object part-rich subset"
python3 -c "import json;d=json.load(open('outputs/stage0_filtered.kept.json'));d.sort(key=lambda r:-len(r.get('components',[])));json.dump(d[:40],open('outputs/stage0_gallery_subset.json','w'),indent=2,ensure_ascii=False)"
rm -f outputs/stage1_a.json outputs/stage1_b.json outputs/stage1_a.log outputs/stage1_b.log
tmux new-session -d -s stage1a "$GEM pipeline/stage1_task_role_assemble.py --gpu 3 --in outputs/stage0_gallery_subset.json --start 0 --end 20 --out outputs/stage1_a.json > outputs/stage1_a.log 2>&1; echo EXIT=\$? >> outputs/stage1_a.log"
tmux new-session -d -s stage1b "$GEM pipeline/stage1_task_role_assemble.py --gpu 1 --in outputs/stage0_gallery_subset.json --start 20 --end 40 --out outputs/stage1_b.json > outputs/stage1_b.log 2>&1; echo EXIT=\$? >> outputs/stage1_b.log"
log "stage1 launched (multi-image + decomposition fixes), sharded"
sleep 15
while tmux has-session -t stage1a 2>/dev/null || tmux has-session -t stage1b 2>/dev/null; do sleep 30; done
log "stage1 DONE; merging + gallery"
python3 -c "import json,os
a=json.load(open('outputs/stage1_a.json')) if os.path.exists('outputs/stage1_a.json') else []
b=json.load(open('outputs/stage1_b.json')) if os.path.exists('outputs/stage1_b.json') else []
json.dump(a+b,open('outputs/stage1_dataset.json','w'),indent=2,ensure_ascii=False)
print('merged',len(a)+len(b))" >> "$LOG" 2>&1
$GEM viz/build_stage1_gallery.py --in outputs/stage1_dataset.json --out galleries/stage1_gallery.html >> "$LOG" 2>&1
log "DONE"
touch outputs/STAGE1_PIPELINE_DONE
