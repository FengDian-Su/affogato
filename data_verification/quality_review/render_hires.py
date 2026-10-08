#!/usr/bin/env python3
"""Full-resolution copies of the review sheets for the LLM judge.

The rating UI's sheets downscale each 512 px view to 256 px; raters can open a view at full size, the
judge cannot. This re-renders the same three sheets (RGB, Hand A, Hand B) with the same renderer and
layout at 512 px per view (2048 x 1060), into release1000/renders_2x/, mirroring renders/.

  python -m data_verification.quality_review.render_hires [--split dev|test|all]
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from data_verification.quality_review import prepare

CELL = 512
OUT = prepare.DEFAULT_OUT / "renders_2x"


def render(row: dict) -> tuple[str, dict[str, str]]:
    import render_aligned_heatmaps as renderer      # importable once prepare has set up sys.path
    renderer.CELL = CELL                            # read at call time by _heat_panel and _sheet
    return prepare.render(row, str(OUT), False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (prepare.DEFAULT_OUT / "manifest.jsonl").open(encoding="utf-8")]
    rows = [row for row in rows if args.split in ("all", row["split"])]
    with Pool(args.workers) as pool:
        done = pool.map(render, rows, chunksize=4)
    print(f"{len(done)} pairs rendered to {OUT}")


if __name__ == "__main__":
    main()
