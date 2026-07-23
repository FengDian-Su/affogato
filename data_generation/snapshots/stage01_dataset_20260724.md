# Stage0/Stage1 dataset snapshot — 2026-07-24

Compressed canonical artifacts of the full daily_used + electronics rerun
(stage1 v13.2 recipe). Extract with: `tar xzf stage01_dataset_20260724.tar.gz`.

Contents (paths relative to data_generation/outputs/):
- stage0/daily_used/kept_all.json         — 57,441 kept objects (stage0 filter)
- stage0/electronics/stage0_part0.kept.json — 5,528 kept objects
- stage1/daily_used/stage1_all.json       — 57,441 objects / 251,390 queries / mean 4.38
- stage1/electronics/stage1_all.json      — 5,528 objects / 28,674 queries / mean 5.19

Verified per-record: 100% coverage, 0 dup, 0 gap, 0 errors; every query has
exactly 2 roles + 2 molmo_queries; 100% stage2-usable. 13 daily objects have
0 queries (mostly correct: pliers/scissors/bolt-cutters = one-handed tools).

Full uncompressed backup (incl. per-shard files): /nfs_drive/mclee/affogato_backup/20260724/
