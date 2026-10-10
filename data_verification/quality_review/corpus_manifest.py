#!/usr/bin/env python3
"""Manifest of every released pair, for running the LLM judge on the whole corpus.

Rows have the review manifest's fields (prepare.py): the task and roles come from each pair's
stage-2 meta.json, and the views are the ones stage 1 used (`views_used`), which are the views the
raters saw. A view that cannot be decoded is left out, as stage 0 leaves out missing views, and a
pair needs 5 to 8 views, as in the review. Nothing is rendered: the heatmap criterion needs the
review sheets, which this script does not make.

  python -m data_verification.quality_review.corpus_manifest [--out DIR] [--workers 32]
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from data_verification.quality_review.prepare import REPO, ROLE_FIELDS, view_label

STAGE1 = REPO / "data_generation/outputs/stage1"
STAGE2 = REPO / "data_generation/outputs/stage2"
DATASETS = ("daily_used", "electronics", "furnitures")
DEFAULT_OUT = REPO / "data_verification/outputs/quality_review/corpus"


def readable(path: str) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except OSError:
        return False


def object_rows(job: tuple[str, str, str, list[str]]) -> tuple[list[dict[str, Any]], Counter, int]:
    dataset, object_id, stage1_name, views_used = job
    views = [view for view in views_used if readable(view)]
    rows, skipped = [], Counter()
    for qdir in sorted(os.listdir(STAGE2 / dataset / object_id)):
        meta_path = STAGE2 / dataset / object_id / qdir / "meta.json"
        scores_path = meta_path.with_name("scores.npz")
        if not (meta_path.is_file() and scores_path.is_file()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        roles = [{field: role.get(field) for field in ROLE_FIELDS} for role in meta.get("roles") or []]
        if [role["id"] for role in roles] != ["A", "B"] or not str(meta.get("task") or "").strip():
            skipped["task or roles malformed"] += 1
            continue
        if not 5 <= len(views) <= 8:
            skipped[f"{len(views)} readable views"] += 1
            continue
        rows.append({
            "sample_id": f"{object_id}/{qdir}",
            "object_id": object_id,
            "qdir": qdir,
            "dataset": dataset,
            "object_name": meta.get("object_name") or stage1_name,
            "task": meta["task"],
            "roles": roles,
            "category": meta.get("category"),
            "coordination": meta.get("coordination"),
            "pattern": meta.get("pattern"),
            "views": views,
            "view_labels": [view_label(view) for view in views],
            "meta_path": str(meta_path),
            "scores_path": str(scores_path),
        })
    return rows, skipped, len(views_used) - len(views)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    jobs, skipped = [], Counter()
    for dataset in DATASETS:
        with (STAGE1 / dataset / "stage1_all.json").open(encoding="utf-8") as stream:
            stage1 = {record["object_id"]: (record["object_name"], record["views_used"])
                      for record in json.load(stream)}
        for object_id in sorted(os.listdir(STAGE2 / dataset)):
            if not (STAGE2 / dataset / object_id).is_dir():
                continue
            if object_id not in stage1:
                skipped["object not in stage 1"] += 1
                continue
            jobs.append((dataset, object_id, *stage1[object_id]))
        print(f"{dataset}: {sum(job[0] == dataset for job in jobs)} objects", flush=True)

    rows, unreadable = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (object_rows_, object_skipped, object_unreadable) in enumerate(
                pool.map(object_rows, jobs, chunksize=16), 1):
            rows.extend(object_rows_)
            skipped.update(object_skipped)
            unreadable += object_unreadable
            if done % 5000 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] objects, {len(rows)} pairs", flush=True)
    for index, row in enumerate(rows):
        row["sample_index"] = index

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.jsonl"
    temporary = manifest.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, manifest)
    summary = {
        "pairs": len(rows),
        "objects": len({row["object_id"] for row in rows}),
        "per_dataset": {dataset: {"objects": len({r["object_id"] for r in rows if r["dataset"] == dataset}),
                                  "pairs": sum(r["dataset"] == dataset for r in rows)}
                        for dataset in DATASETS},
        "views_per_pair": dict(sorted(Counter(len(row["views"]) for row in rows).items())),
        "unreadable_views": unreadable,
        "skipped": dict(skipped),
        "manifest": str(manifest),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
