#!/usr/bin/env python3
"""Build the release-quality review set.

The sample frame is the 1,000 pilot human-gold samples. Their task and role text are re-read
from the CURRENT stage-2 release, and their heatmaps are re-rendered from the release
scores.npz with the validated camera-aligned renderer, so raters judge exactly what ships.
(The release was regenerated with conditional-mean mask consolidation on 2026-09-14..17, after
the pilot renders were made.) Sheets are lossless WebP: the renderer's exact pixels, about half
the disk of PNG.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
RENDERER = REPO / "data_verification/pipeline/render_aligned_heatmaps.py"
sys.path.insert(0, str(RENDERER.parent))
from render_aligned_heatmaps import BG, _heat_panel, _sheet  # noqa: E402

DEFAULT_GOLD = REPO / "data_verification/outputs/human_review/pilot1000_export/stage2_human_gold.jsonl"
DEFAULT_PILOT_MANIFEST = (REPO / "data_verification/quality_evaluation/claude_pilot_1000/"
                          "manifest.aligned_v18_7.full1000.jsonl")
DEFAULT_OUT = REPO / "data_verification/outputs/quality_review/release1000"
ROLE_FIELDS = ("id", "role", "target", "contact_region", "function")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def view_label(path: str) -> str:
    stem = Path(path).stem
    return {"00025": "top-down", "00026": "underside"}.get(stem, f"view {stem}")


def save_webp(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.webp")
    image.save(temporary, format="WEBP", lossless=True, method=4)
    os.replace(temporary, path)


def render(row: dict[str, Any], render_dir: str, force: bool) -> tuple[str, dict[str, str]]:
    base = Path(render_dir) / row["object_id"]
    paths = {kind: base / f"{row['qdir']}.{kind}.webp" for kind in ("rgb", "A", "B")}
    if force or not all(path.exists() for path in paths.values()):
        with np.load(row["scores_path"], allow_pickle=False) as data:
            xyz = data["xyz"].astype(np.float32)
            scores = {"A": data["scoreA"].astype(np.float32), "B": data["scoreB"].astype(np.float32)}
        views, labels = row["views"], row["view_labels"]
        rgb = [Image.open(view).convert("RGB") for view in views]
        save_webp(_sheet(rgb, labels, (255, 255, 255)), paths["rgb"])
        for hand in ("A", "B"):
            panels = [_heat_panel(xyz, scores[hand], view, hand) for view in views]
            save_webp(_sheet(panels, labels, BG), paths[hand])
    return row["sample_id"], {kind: str(path) for kind, path in paths.items()}


def build_rows(gold_path: Path, pilot_manifest: Path, limit: int) -> list[dict[str, Any]]:
    gold = {row["sample_id"]: row for row in read_jsonl(gold_path)}
    pilot = sorted(read_jsonl(pilot_manifest), key=lambda row: row["sample_index"])
    pilot = [row for row in pilot if row["sample_id"] in gold]
    if len(pilot) != len(gold):
        raise SystemExit(f"gold has {len(gold)} samples but only {len(pilot)} are in the pilot manifest")
    rows = []
    for source in pilot[:limit or None]:
        meta_path, scores_path = Path(source["meta_path"]), Path(source["scores_path"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        roles = [{field: role.get(field) for field in ROLE_FIELDS} for role in meta["roles"]]
        if len(roles) != 2 or [role["id"] for role in roles] != ["A", "B"]:
            raise SystemExit(f"{source['sample_id']}: expected roles A and B, got {roles}")
        if not 5 <= len(source["views"]) <= 8:
            raise SystemExit(f"{source['sample_id']}: expected 5..8 views, got {len(source['views'])}")
        reference = gold[source["sample_id"]]
        rows.append({
            "sample_id": source["sample_id"],
            "sample_index": len(rows),
            "object_id": source["object_id"],
            "qdir": source["qdir"],
            "dataset": reference.get("dataset") or source.get("dataset"),
            "split": reference.get("split"),
            "object_name": meta.get("object_name") or source.get("object_name"),
            "task": meta["task"],
            "roles": roles,
            "category": meta.get("category"),
            "coordination": meta.get("coordination"),
            "pattern": meta.get("pattern"),
            "views": source["views"],
            "view_labels": [view_label(view) for view in source["views"]],
            "meta_path": str(meta_path),
            "scores_path": str(scores_path),
            "meta_sha256": sha256(meta_path),
            "scores_sha256": sha256(scores_path),
            "engine": meta.get("engine") or {},
            # Hidden from raters (the API never serves manifest fields beyond the public sample).
            "pilot_heatmap_gold": {"label": reference["overall_score"],
                                   "resolution": reference["resolution"]},
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="re-render existing sheets")
    args = parser.parse_args()

    rows = build_rows(args.gold, args.pilot_manifest, args.limit)
    render_dir = args.out / "renders"
    rendered: dict[str, dict[str, str]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(render, row, str(render_dir), args.force) for row in rows]
        for done, future in enumerate(as_completed(futures), 1):
            sample_id, paths = future.result()
            rendered[sample_id] = paths
            if done % 50 == 0 or done == len(rows):
                print(f"[{done}/{len(rows)}] rendered", flush=True)
    for row in rows:
        paths = rendered[row["sample_id"]]
        row.update({"rgb_path": paths["rgb"], "heat_a_path": paths["A"], "heat_b_path": paths["B"]})

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.jsonl"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                         encoding="utf-8")
    os.replace(temporary, manifest)
    summary = {
        "samples": len(rows),
        "objects": len({row["object_id"] for row in rows}),
        "splits": dict(Counter(row["split"] for row in rows)),
        "datasets": dict(Counter(row["dataset"] for row in rows)),
        "consolidation": dict(Counter(str(row["engine"].get("consolidation")) for row in rows)),
        "gold": {"path": str(args.gold.resolve()), "sha256": sha256(args.gold)},
        "pilot_manifest": {"path": str(args.pilot_manifest.resolve()),
                           "sha256": sha256(args.pilot_manifest)},
        "renderer": {"path": str(RENDERER), "sha256": sha256(RENDERER)},
        "manifest": {"path": str(manifest.resolve()), "sha256": sha256(manifest)},
    }
    (args.out / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
