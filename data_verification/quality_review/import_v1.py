#!/usr/bin/env python3
"""Carry the pilot's human heatmap ratings (stage2-human-review-v1.0) into a quality-review database.

Pilot samples are linked to review samples by (object_id, task). A pilot heatmap rating is carried
over only where the released heatmap still looks like the one that was rated: for both hands, the
hand-coloured region of the released render must overlap the rated render with IoU >= --min-iou.
The release was regenerated (conditional-mean mask consolidation, 2026-09-14..17) after the pilot
renders were made; the few pairs that changed visibly keep no pilot rating and are re-rated by
everyone. On carried-over pairs the pilot raters (blind raters and adjudicators) skip the heatmap
step, and the carried finals must reproduce the pilot gold exactly or nothing is written.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image

from data_verification.quality_review.app import (
    DEFAULT_DB, REPO, connect, file_sha256, get_metadata, recompute_finals, set_metadata,
    transaction, utcnow, validate_rating,
)
from data_verification.quality_review.rubric import AXIS_TAGS

DEFAULT_V1_DB = REPO / "data_verification/outputs/human_review/stage2_pilot1000.sqlite3"
AXIS = "heatmap"


def coloured(path: str) -> np.ndarray:
    """Pixels painted with a hand's orange/teal ramp; the grey background and black points are not."""
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    return (image.max(-1) - image.min(-1)) > 60


def region_iou(pair: tuple[tuple[str, str], tuple[str, str]]) -> float:
    """The smaller of the two hands' IoU between the rated and the released coloured regions."""
    ious = []
    for rated, released in zip(*pair):
        a, b = coloured(rated), coloured(released)
        union = (a | b).sum()
        ious.append(1.0 if union == 0 else float((a & b).sum() / union))
    return min(ious)


def import_v1(db_path: Path, v1_db: Path, min_iou: float = 0.7, workers: int = 16) -> dict:
    v1 = sqlite3.connect(f"file:{v1_db}?mode=ro", uri=True)
    v1.row_factory = sqlite3.Row
    connection = connect(db_path)
    try:
        if connection.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]:
            raise SystemExit("this database already has ratings; carry the pilot over before anyone rates")
        n = int(get_metadata(connection, "raters_per_sample"))

        def by_object_and_task(rows) -> dict:
            linked = {}
            for row in rows:
                key = (row["object_id"], row["task"].strip())
                if key in linked:
                    raise SystemExit(f"(object_id, task) is not unique: {key}")
                linked[key] = row
            return linked
        review = by_object_and_task(connection.execute("SELECT * FROM samples"))
        pilot = by_object_and_task(v1.execute("SELECT * FROM samples"))
        linked = {key: (pilot[key], review[key]) for key in pilot if key in review}

        renders = [((p["heat_a_path"], p["heat_b_path"]), (r["heat_a_path"], r["heat_b_path"]))
                   for p, r in linked.values()]
        with ProcessPoolExecutor(workers) as pool:
            ious = dict(zip(linked, pool.map(region_iou, renders, chunksize=8)))
        carry = {pilot[key]["sample_id"]: review[key]["sample_id"] for key in linked if ious[key] >= min_iou}

        gold = {row["sample_id"]: row["label"] for row in v1.execute("SELECT sample_id, label FROM final_labels")}
        schema = json.loads(v1.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0])
        axis_tags = {AXIS: AXIS_TAGS[AXIS]}
        raters: Counter = Counter()
        with transaction(connection):
            for row in v1.execute("SELECT * FROM annotations ORDER BY id"):
                if row["sample_id"] not in carry:
                    continue
                labels, tags, comment = validate_rating(
                    {AXIS: row["label"]}, {AXIS: json.loads(row["defect_tags_json"])}, row["comment"],
                    axis_tags, (AXIS,))
                source = f"{schema}:annotations/{row['id']}"
                connection.execute(
                    """INSERT INTO annotations(sample_id,username,kind,heatmap,tags_json,comment,
                                               idempotency_key,source,created_at)
                       VALUES(?,?,'imported',?,?,?,?,?,?)""",
                    (carry[row["sample_id"]], row["username"], labels[AXIS], json.dumps(tags), comment,
                     source, source, row["created_at"]))
                raters[row["username"]] += 1
            adjudicated = 0
            for row in v1.execute("SELECT * FROM adjudications ORDER BY id"):
                if row["sample_id"] not in carry:
                    continue
                labels, tags, comment = validate_rating(
                    {AXIS: row["label"]}, {AXIS: json.loads(row["defect_tags_json"])}, row["comment"],
                    axis_tags, (AXIS,))
                cursor = connection.execute(
                    """INSERT INTO adjudications(sample_id,username,labels_json,tags_json,comment,source,created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (carry[row["sample_id"]], row["username"], json.dumps(labels), json.dumps(tags), comment,
                     f"{schema}:adjudications/{row['id']}", row["created_at"]))
                connection.execute(
                    """INSERT INTO final_labels(sample_id,axis,label,resolution,resolved_by,tags_json,
                                                source_ids_json,created_at)
                       VALUES(?,?,?,'adjudicated',?,?,?,?)""",
                    (carry[row["sample_id"]], AXIS, labels[AXIS], row["username"], json.dumps(tags[AXIS]),
                     json.dumps([cursor.lastrowid]), row["created_at"]))
                raters[row["username"]] += 1
                adjudicated += 1
            for sample_id in carry.values():
                connection.execute("UPDATE samples SET carried_json=? WHERE sample_id=?",
                                   (json.dumps([AXIS]), sample_id))
                recompute_finals(connection, sample_id, n)

            finals = {row["sample_id"]: row["label"] for row in connection.execute(
                "SELECT sample_id, label FROM final_labels WHERE axis=?", (AXIS,))}
            wrong = [p for p, r in carry.items() if finals.get(r) != gold.get(p)]
            if wrong:
                raise SystemExit(f"carried finals differ from the pilot gold on {len(wrong)} pairs, e.g. {wrong[:3]}")

            values = np.array(list(ious.values()))
            rerated = sorted((review[key]["sample_id"] for key in linked if ious[key] < min_iou))
            summary = {
                "source": str(v1_db.resolve()),
                "source_sha256": file_sha256(v1_db),
                "source_schema": schema,
                "linked_by": "object_id + task",
                "pilot_samples": len(pilot),
                "linked": len(linked),
                "min_iou": min_iou,
                "carried": len(carry),
                "rerated": len(rerated),
                "iou_percentiles": {str(p): round(float(np.percentile(values, p)), 4) for p in (1, 5, 10, 25, 50)},
                "carried_finals": {str(k): v for k, v in sorted(Counter(finals[r] for r in carry.values()).items())},
                "imported_ratings": sum(raters.values()) - adjudicated,
                "imported_adjudications": adjudicated,
                "raters": sorted(raters),
                "imported_at": utcnow(),
            }
            set_metadata(connection, "carried_raters", {AXIS: sorted(raters)})
            set_metadata(connection, "import_v1", {
                "summary": summary,
                "rerated": [{"sample_id": sid, "iou": round(ious[key], 4), "pilot_label": gold.get(pilot[key]["sample_id"])}
                            for key in linked for sid in [review[key]["sample_id"]] if sid in rerated],
                "iou": {review[key]["sample_id"]: round(ious[key], 4) for key in linked},
            })
    finally:
        connection.close()
        v1.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--v1-db", type=Path, default=DEFAULT_V1_DB)
    parser.add_argument("--min-iou", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(import_v1(args.db, args.v1_db, args.min_iou, args.workers), indent=2))


if __name__ == "__main__":
    main()
