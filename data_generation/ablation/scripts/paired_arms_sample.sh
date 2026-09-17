#!/usr/bin/env bash
# Paired aggregation ablation on a stratified sample, three arms.
#
#   arm1  conditional_mean  (the adopted method)
#   arm2  iou_group         (the historical baseline)
#   arm3  conditional_mean  (a REPEAT of arm1)
#
# arm3 is the control: per-point SAM masks are never persisted, so each arm
# re-runs SAM2 and picks up its nondeterminism.  |arm1-arm2| is then only
# interpretable against |arm1-arm3|, which is that noise floor measured on the
# very same objects.  Without arm3 the comparison would rest on the pilot's
# 12-object floor estimate instead of this sample's own.
#
#   paired_arms_sample.sh <category> <start> <end> <gpu>
set -u
CAT=$1; S=$2; E=$3; GPU=$4
# Halved from the production 20: this shares a card with the main run, and a
# smaller view window caps the SAM encoder peak so the ablation cannot OOM a
# production worker out of the remaining VRAM.
CHUNK=${SAM_CHUNK:-10}
DG=/home/michaellee/mclee/affogato/data_generation
PY=$HOME/miniconda3/envs/molmo/bin/python
cd "$DG" || exit 1
OUT=ablation/results/paired_arms_$CAT
mkdir -p "$OUT"

for arm in arm1_conditional:conditional_mean arm2_iou:iou_group arm3_conditional_repeat:conditional_mean; do
  name=${arm%%:*}; method=${arm##*:}
  echo "[$(date '+%F %T')] $name ($method) $CAT [$S:$E] on gpu$GPU"
  "$PY" pipeline/stage2_v2.py \
      --stage1 "outputs/stage1/$CAT/stage1_all.json" \
      --mapping "dataset/${CAT}_to_affogato.json" \
      --output_dir "$OUT/$name" \
      --replay_points "outputs/stage2_points_20260803/$CAT" \
      --start "$S" --end "$E" --gpu "$GPU" --sam_chunk "$CHUNK" \
      --mask_consolidation "$method" \
      > "$OUT/${name}.log" 2>&1
  echo "[$(date '+%F %T')] $name done: $(find "$OUT/$name" -name scores.npz | wc -l) queries"
done
echo "[$(date '+%F %T')] PAIRED ARMS DONE $CAT [$S:$E]"
