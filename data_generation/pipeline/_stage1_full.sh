#!/bin/bash
# Scale stage1 to ALL 109 kept objects. Both shards on GPU 3 (97GB fits two 12B; GPU 1 is in use by
# another user). Resume-safe: stage1 skips objects already in its out file, so re-running continues.
cd /home/michaellee/mclee/affogato/data_generation || exit 1
GEM=/home/michaellee/miniconda3/envs/gemma4/bin/python
MM=/home/michaellee/miniconda3/envs/mm/bin/python
LOG=outputs/stage1_full.log
log(){ echo "$(date '+%m-%d %H:%M:%S')  $*" >> "$LOG"; }
: > "$LOG"
rm -f outputs/STAGE1_FULL_DONE
IN=outputs/stage0_filtered.kept.json   # 109 kept objects (stage1 filters keep+components internally)

log "stage1 FULL launched on $IN (109 objects, 2 shards on GPU 3, resume-safe)"
# shard a: objects 0-55, shard b: 55-109; both GPU 3
tmux new-session -d -s s1fa "$GEM pipeline/stage1_task_role_assemble.py --gpu 3 --in $IN --start 0  --end 55  --out outputs/stage1_full_a.json >> outputs/stage1_full_a.log 2>&1; echo EXIT=\$? >> outputs/stage1_full_a.log"
tmux new-session -d -s s1fb "$GEM pipeline/stage1_task_role_assemble.py --gpu 3 --in $IN --start 55 --end 109 --out outputs/stage1_full_b.json >> outputs/stage1_full_b.log 2>&1; echo EXIT=\$? >> outputs/stage1_full_b.log"
sleep 15
while tmux has-session -t s1fa 2>/dev/null || tmux has-session -t s1fb 2>/dev/null; do sleep 30; done
log "shards done; merging"
$MM -c "import json,os
a=json.load(open('outputs/stage1_full_a.json')) if os.path.exists('outputs/stage1_full_a.json') else []
b=json.load(open('outputs/stage1_full_b.json')) if os.path.exists('outputs/stage1_full_b.json') else []
m=a+b
json.dump(m,open('outputs/stage1_dataset_full.json','w'),indent=2,ensure_ascii=False)
nq=sum(len(o.get('queries',[])) for o in m)
print('merged',len(m),'objects',nq,'queries')" >> "$LOG" 2>&1
$MM viz/build_stage1_gallery.py --in outputs/stage1_dataset_full.json --out galleries/stage1_gallery_full.html >> "$LOG" 2>&1
log "DONE"
touch outputs/STAGE1_FULL_DONE
