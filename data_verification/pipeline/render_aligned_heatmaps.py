#!/usr/bin/env python3
"""Render Stage-2 heatmaps with the exact cameras and panel order of their RGB references.

The older review render used a synthetic Matplotlib turntable.  It covered the object well, but its
panels were not camera-aligned with the GObjaverse RGB scan.  This renderer projects the scored
Affogato points through each selected RGB view's own extrinsics and intrinsics, then lays RGB, hand
A, and hand B out with one shared panel order.  It changes presentation only; it never derives or
changes a heatmap score.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from judge_stage2_simple import VISUAL_ANCHORS


THRESHOLD = 0.15
BG = (142, 142, 147)
BLACK = (13, 13, 15)
RAMPS = {
    "A": ((122, 66, 0), (196, 106, 0), (245, 166, 35), (255, 184, 77)),
    "B": ((10, 90, 85), (10, 156, 153), (46, 216, 206), (82, 245, 232)),
}
CELL = 256
LABEL_H = 18


def _camera(json_path: Path) -> np.ndarray:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = np.asarray(data["x"])
    c2w[:3, 1] = -np.asarray(data["y"])
    c2w[:3, 2] = -np.asarray(data["z"])
    c2w[:3, 3] = np.asarray(data["origin"])
    flip = np.eye(4, dtype=np.float32)
    flip[1, 1] = flip[2, 2] = -1
    return np.linalg.inv(c2w @ flip)


def _project(xyz: np.ndarray, rgb_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    p = Path(rgb_path)
    stem = p.stem
    w2c = _camera(p.with_name(stem + ".json"))
    with Image.open(p) as image:
        width, height = image.size
    aligned = xyz[:, [0, 2, 1]].copy()
    aligned[:, 1] *= -1
    homo = np.column_stack((aligned, np.ones(len(aligned), dtype=np.float32)))
    camera_xyz = (w2c @ homo.T).T[:, :3]
    focal = 1422.222 * height / 1024.0
    u = focal * camera_xyz[:, 0] / camera_xyz[:, 2] + width / 2.0
    v = focal * camera_xyz[:, 1] / camera_xyz[:, 2] + height / 2.0
    valid = ((camera_xyz[:, 2] > 0) & (u >= 0) & (u < width) &
             (v >= 0) & (v < height))
    # Painter's order supplies self-occlusion without needing EXR/OpenCV: far points first.
    order = np.flatnonzero(valid)[np.argsort(camera_xyz[valid, 2])[::-1]]
    return u, v, order, width, height


def _colour(score: float, ramp: tuple[tuple[int, int, int], ...]) -> tuple[int, int, int]:
    x = float(np.clip(score, 0.0, 1.0)) * (len(ramp) - 1)
    lo = min(int(x), len(ramp) - 2)
    t = x - lo
    return tuple(round(ramp[lo][i] * (1 - t) + ramp[lo + 1][i] * t) for i in range(3))


def _heat_panel(xyz: np.ndarray, scores: np.ndarray, rgb_path: str, hand: str) -> Image.Image:
    u, v, order, width, height = _project(xyz, rgb_path)
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    active = scores >= THRESHOLD
    active_count = int(active.sum())
    active_radius = int(np.clip(round(2.0 * np.sqrt(1200 / max(active_count, 1))), 2, 8))
    if active_count:
        values = scores[active]
        hi = max(float(np.percentile(values, 98)), THRESHOLD + 1e-3)
    else:
        hi = THRESHOLD + 1e-3
    ramp = RAMPS[hand]
    for index in order:
        x, y = int(round(u[index])), int(round(v[index]))
        if active[index]:
            normalized = float(np.clip((scores[index] - THRESHOLD) / (hi - THRESHOLD), 0, 1))
            radius = active_radius
            fill = _colour(normalized, ramp)
        else:
            radius = 1
            fill = BLACK
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    return canvas.resize((CELL, CELL), Image.Resampling.LANCZOS)


def _sheet(panels: list[Image.Image], labels: list[str], background: tuple[int, int, int]) -> Image.Image:
    sheet = Image.new("RGB", (CELL * 4, (CELL + LABEL_H) * 2), background)
    draw = ImageDraw.Draw(sheet)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        row, col = divmod(i, 4)
        x, y = col * CELL, row * (CELL + LABEL_H)
        draw.text((x + 4, y + 2), label, fill=(20, 20, 20))
        sheet.paste(panel.resize((CELL, CELL), Image.Resampling.LANCZOS), (x, y + LABEL_H))
    return sheet


def _atomic_save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    image.save(tmp, format="PNG")
    os.replace(tmp, path)


def render_row(row: dict[str, Any], out_dir: str, force: bool = False) -> tuple[str, dict[str, str]]:
    safe = row["sample_id"].replace("/", "__")
    base = Path(out_dir) / safe
    paths = {kind: str(base.with_suffix(f".{kind}.png"))
             for kind in ("rgb", "A", "B", "composite")}
    if not force and all(Path(path).exists() for path in paths.values()):
        return row["sample_id"], paths

    with np.load(row["scores_path"], allow_pickle=False) as data:
        xyz = data["xyz"].astype(np.float32)
        scores_a = data["scoreA"].astype(np.float32)
        scores_b = data["scoreB"].astype(np.float32)
    views = row["views"]
    if not 5 <= len(views) <= 8:
        raise ValueError(f"{row['sample_id']}: expected 5..8 selected views, got {len(views)}")
    labels = []
    for path in views:
        index = Path(path).stem
        labels.append({"00025": "top-down", "00026": "underside"}.get(index, f"view {index}"))

    rgb_panels = [Image.open(path).convert("RGB") for path in views]
    sheet_rgb = _sheet(rgb_panels, labels, (255, 255, 255))
    sheet_a = _sheet([_heat_panel(xyz, scores_a, path, "A") for path in views], labels, BG)
    sheet_b = _sheet([_heat_panel(xyz, scores_b, path, "B") for path in views], labels, BG)
    composite = Image.new("RGB", (sheet_a.width, sheet_a.height * 2 + LABEL_H * 2), BG)
    draw = ImageDraw.Draw(composite)
    draw.text((6, 2), "HAND A (orange)", fill=(20, 20, 20))
    composite.paste(sheet_a, (0, LABEL_H))
    y_b = LABEL_H + sheet_a.height
    draw.text((6, y_b + 2), "HAND B (teal)", fill=(20, 20, 20))
    composite.paste(sheet_b, (0, y_b + LABEL_H))

    for key, image in (("rgb", sheet_rgb), ("A", sheet_a), ("B", sheet_b),
                       ("composite", composite)):
        _atomic_save(image, Path(paths[key]))
    return row["sample_id"], paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-visual-anchors", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda row: row["sample_index"])
    selected = rows[args.offset:]
    if args.limit:
        selected = selected[:args.limit]
    wanted = {row["sample_id"] for row in selected}
    if args.include_visual_anchors:
        wanted.update(spec["sample_id"] for spec in VISUAL_ANCHORS)
    jobs = [row for row in rows if row["sample_id"] in wanted]

    rendered: dict[str, dict[str, str]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_row, row, args.out_dir, args.force): row for row in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            sid, paths = future.result()
            rendered[sid] = paths
            if done % 10 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] aligned renders", flush=True)

    with Path(args.out_manifest).open("w", encoding="utf-8") as output:
        for row in rows:
            copy = dict(row)
            if row["sample_id"] in rendered:
                paths = rendered[row["sample_id"]]
                copy.update({"rgb_png_aligned": paths["rgb"],
                             "heat_png_A_aligned": paths["A"],
                             "heat_png_B_aligned": paths["B"],
                             "heat_png_aligned": paths["composite"]})
            output.write(json.dumps(copy, ensure_ascii=False) + "\n")
    print(f"manifest -> {args.out_manifest} ({len(rendered)} aligned samples)")


if __name__ == "__main__":
    main()
