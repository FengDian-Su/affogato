#!/bin/bash
# Starts the corpus heatmap run under tmux session corpus_heatmap, independent of any Claude session:
#   shard0, shard1  the supervisors (run_corpus.sh) on GPU0 and GPU3, resuming from the shard files
#   monitor         monitor.sh: stalls, relaunches, NFS pause / resume, score checks every 10%
#   status          status.sh: live overview, refreshed every minute
# Panes stay open after their command exits, so a finished or failed window can still be read.
#   usage: start_tmux.sh        attach: tmux attach -t corpus_heatmap
cd /home/michaellee/mclee/affogato || exit 1
HERE=$PWD/data_verification/quality_review/corpus_run    # these scripts; data and logs: outputs/quality_review/corpus
S=corpus_heatmap
tmux new-session -d -s $S -n shard0 "exec bash $HERE/run_corpus.sh 0 0 2 heatmap"
tmux new-window -d -t $S -n shard1 "exec bash $HERE/run_corpus.sh 3 1 2 heatmap"
tmux new-window -d -t $S -n monitor "exec bash $HERE/monitor.sh"
tmux new-window -d -t $S -n status "exec bash $HERE/status.sh"
for window in shard0 shard1 monitor status; do
  tmux set-option -w -t "$S:$window" remain-on-exit on
done
