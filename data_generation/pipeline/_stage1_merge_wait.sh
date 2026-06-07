#!/bin/bash
# Robust merge-waiter: runs INSIDE its own tmux session (survives harness background-task kills).
# Waits for the two full-run shards (s1fa/s1fb) to finish, then merges + builds the gallery + DONE marker.
cd /home/michaellee/mclee/affogato/data_generation || exit 1
MM=/home/michaellee/miniconda3/envs/mm/bin/python
LOG=outputs/stage1_full.log
log(){ echo "$(date '+%m-%d %H:%M:%S')  $*" >> "$LOG"; }
rm -f outputs/STAGE1_FULL_DONE
log "merge-waiter (tmux) watching s1fa/s1fb"
while tmux has-session -t s1fa 2>/dev/null || tmux has-session -t s1fb 2>/dev/null; do sleep 30; done
log "shards done; merging"
$MM -c "import json,os
a=json.load(open('outputs/stage1_full_a.json')) if os.path.exists('outputs/stage1_full_a.json') else []
b=json.load(open('outputs/stage1_full_b.json')) if os.path.exists('outputs/stage1_full_b.json') else []
m=a+b
json.dump(m,open('outputs/stage1_dataset_full.json','w'),indent=2,ensure_ascii=False)
print('merged',len(m),'objects',sum(len(o.get('queries',[])) for o in m),'queries')" >> "$LOG" 2>&1
$MM viz/build_stage1_gallery.py --in outputs/stage1_dataset_full.json --out galleries/stage1_gallery_full.html >> "$LOG" 2>&1
log "DONE"
touch outputs/STAGE1_FULL_DONE
