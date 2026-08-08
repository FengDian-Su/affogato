#!/usr/bin/env python3
"""Run the Stage-2 contact rubric with Codex vision and cache every completed batch.

This is intentionally separate from ``judge_stage2_simple.py``'s local-vLLM runner.  It uses the
same prompt and item builder, but sends small groups of A/B render pairs to ``codex exec``.  Batch
responses are durable, so a rate limit, timeout, or interrupted shell can be resumed without
repeating successful calls.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

from judge_stage2_simple import MULTI_AXES, PROMPTS, VERSION, VISUAL_ANCHORS, build_item
from simple_verifier_common import read_jsonl


DEFAULT_MODEL = "gpt-5.6-sol"
ANCHOR_CALIBRATION = "visual-anchors-v7-camera-aligned"


def image_paths(item: Any) -> tuple[str, ...]:
    return item.image_path if isinstance(item.image_path, tuple) else (item.image_path,)


def result_schema(sample_ids: list[str]) -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sample_id": {"type": "string", "enum": sample_ids},
            "orange_present": {"type": "boolean"},
            "teal_present": {"type": "boolean"},
            "orange_shape": {"type": "string", "enum": ["complete", "incomplete", "absent"]},
            "teal_shape": {"type": "string", "enum": ["complete", "incomplete", "absent"]},
            "orange_alignment": {"type": "string",
                                 "enum": ["clear_hit", "partial_hit", "severe_miss"]},
            "teal_alignment": {"type": "string",
                               "enum": ["clear_hit", "partial_hit", "severe_miss"]},
            "hand_A": {"type": "integer", "enum": [0, 1, 2]},
            "hand_B": {"type": "integer", "enum": [0, 1, 2]},
            "reason": {"type": "string"},
        },
        "required": ["sample_id", "orange_present", "teal_present", "orange_shape", "teal_shape",
                     "orange_alignment", "teal_alignment", "hand_A", "hand_B",
                     "reason"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": item,
                "minItems": len(sample_ids),
                "maxItems": len(sample_ids),
            }
        },
        "required": ["results"],
    }


def batch_key(sample_ids: list[str], input_format: str, anchored: bool = False) -> str:
    digest = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()[:12]
    stem = f"{sample_ids[0].split('/')[0]}__n{len(sample_ids)}__{digest}"
    # Preserve the original split-run key so the completed two-image calls remain resumable.  The
    # composite prefix makes the controlled-comparison cache a distinct namespace.
    if input_format != "split":
        stem = f"{input_format.replace('-', '_')}__{stem}"
    return f"{ANCHOR_CALIBRATION}__{stem}" if anchored else stem


def validate_results(value: Any, sample_ids: list[str]) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise ValueError("response must contain a results array")
    results = value["results"]
    expected = set(sample_ids)
    got = [row.get("sample_id") for row in results if isinstance(row, dict)]
    if len(results) != len(sample_ids) or set(got) != expected or len(set(got)) != len(got):
        raise ValueError(f"sample ids differ: expected {len(expected)} unique, got {got!r}")
    by_id = {row["sample_id"]: row for row in results}
    ordered = []
    for sid in sample_ids:
        row = by_id[sid]
        if type(row.get("orange_present")) is not bool or type(row.get("teal_present")) is not bool:
            raise ValueError(f"{sid}: invalid presence booleans")
        for field in ("orange_alignment", "teal_alignment"):
            if row.get(field) not in ("clear_hit", "partial_hit", "severe_miss"):
                raise ValueError(f"{sid}: invalid {field}={row.get(field)!r}")
        for field in ("orange_shape", "teal_shape"):
            if row.get(field) not in ("complete", "incomplete", "absent"):
                raise ValueError(f"{sid}: invalid {field}={row.get(field)!r}")
        for key in MULTI_AXES:
            if type(row.get(key)) is not int or row[key] not in (0, 1, 2):
                raise ValueError(f"{sid}: invalid {key}={row.get(key)!r}")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError(f"{sid}: missing reason")
        if not row["orange_present"] and row["hand_A"] != 0:
            raise ValueError(f"{sid}: orange absent but hand_A is nonzero")
        if not row["teal_present"] and row["hand_B"] != 0:
            raise ValueError(f"{sid}: teal absent but hand_B is nonzero")
        def derived(present, alignment, shape):
            if not present or alignment == "severe_miss" or shape == "absent":
                return 0
            return 2 if alignment == "clear_hit" and shape == "complete" else 1
        if row["hand_A"] != derived(row["orange_present"], row["orange_alignment"],
                                    row["orange_shape"]):
            raise ValueError(f"{sid}: hand_A disagrees with alignment/shape")
        if row["hand_B"] != derived(row["teal_present"], row["teal_alignment"],
                                    row["teal_shape"]):
            raise ValueError(f"{sid}: hand_B disagrees with alignment/shape")
        ordered.append(row)
    return ordered


def composite_rubric() -> str:
    """Adapt only the image-layout words; the scoring rubric stays byte-for-byte otherwise."""
    prompt = PROMPTS["multi"]
    prompt = prompt.replace(
        "You are shown the same object twice, as two pictures. The FIRST picture is hand A's "
        "predicted contact map in orange, the SECOND is hand B's in teal, each from the same eight "
        "viewpoints. Score the two hands independently.",
        "You are shown one split-composite picture of an object. Its TOP half is hand A's predicted "
        "contact map in orange and its BOTTOM half is hand B's in teal; each half contains the same "
        "eight viewpoints. Score the two hands independently.",
    )
    return prompt.replace("FIRST picture", "TOP half").replace("SECOND picture", "BOTTOM half")


def rgb_composite_rubric(aligned: bool) -> str:
    reference = """For each sample, the RGB reference and heatmap composite contain the same selected
views in the same positions. Use RGB only to understand geometry, physical-part identity,
occlusion, and cross-view correspondence. Judge prediction presence, strength, and quality only
from the heatmap. Apparent separation caused by viewpoint, self-occlusion, thin geometry, or the
contact sheet layout is not fragmentation. A field is incomplete only when correspondence across
the matched views shows that its probability mass does not form one physical contact region.
""" if aligned else """The RGB reference identifies object geometry and parts, but its camera
poses are not panel-aligned with the heatmap. Do not compare panels position by position or infer
fragmentation from their apparent view correspondence. Judge prediction quality from the heatmap.
"""
    base = composite_rubric()
    if aligned:
        base = (base.replace("same eight viewpoints", "same selected viewpoints")
                    .replace("The eight views together", "The selected views together")
                    .replace("in any of its eight views", "in any shown view")
                    .replace("shown from eight\nviewpoints", "shown from the selected matched\nviewpoints"))
    return reference + "\n" + base


def make_prompt(items: list[Any], input_format: str,
                anchors: list[tuple[Any, dict[str, Any]]] | None = None) -> str:
    if input_format == "composite":
        layout = (
            f"Evaluate {len(items)} independent samples. Each attached image is one sample: its TOP "
            "half is orange hand A and its BOTTOM half is teal hand B."
        )
        rubric = composite_rubric()
    elif input_format in ("rgb-composite", "rgb-aligned-composite"):
        aligned = input_format == "rgb-aligned-composite"
        layout = (
            f"Evaluate {len(items)} independent samples. Each sample has two images: first an RGB "
            "multi-view geometry reference, then a heatmap composite with hand A orange on TOP and "
            f"hand B teal on BOTTOM. The panel positions {'correspond exactly' if aligned else 'are not camera-aligned'}."
        )
        rubric = rgb_composite_rubric(aligned)
    else:
        layout = (
            f"Evaluate {len(items)} independent samples. The attached images are ordered as A then B "
            "for each sample."
        )
        rubric = PROMPTS["multi"]
    lines = [
        "Act only as the visual heatmap judge below. Do not inspect files, run tools, or discuss the repository.",
    ]
    anchor_offset = sum(len(image_paths(anchor)) for anchor, _spec in (anchors or []))
    if anchors:
        anchor_layout = (
            f"The first {anchor_offset} attached images are {len(anchors)} labeled calibration "
            "pairs, each ordered RGB reference then A/B heatmap composite."
            if input_format in ("rgb-composite", "rgb-aligned-composite") else
            f"The first {anchor_offset} attached images are {len(anchors)} labeled visual calibration anchors."
        )
        lines.extend([
            anchor_layout,
            "Use them only to calibrate the visual meaning of good, ok, and bad. Do not return results for anchors.",
            "No numerical property is supplied or implied by these anchors; compare visual probability-field structure.",
            "",
            "VISUAL CALIBRATION ANCHORS:",
        ])
        next_image = 1
        for index, (anchor, spec) in enumerate(anchors, 1):
            count = len(image_paths(anchor))
            image_label = (
                f"images {next_image} and {next_image + 1} (RGB, then heatmap)"
                if count == 2 else f"image {next_image}"
            )
            lines.append(
                f"Anchor {index}, {image_label}: overall {spec['score']} ({['bad', 'ok', 'good'][spec['score']]}); "
                f"{spec['note']}; sample_id={anchor.sample_id}"
            )
            next_image += count
        lines.append("")
    lines.extend([
        layout,
        "Keep candidate samples separate and return one result for every exact candidate sample_id.",
        "",
        "RUBRIC:",
        rubric,
        "",
        "CANDIDATE SAMPLES AND IMAGE ORDER:",
    ])
    for i, item in enumerate(items, 1):
        start = anchor_offset + sum(len(image_paths(previous)) for previous in items[:i - 1]) + 1
        if input_format == "composite":
            image_order = f"attached image {start} (A on top, B on bottom)"
        elif input_format in ("rgb-composite", "rgb-aligned-composite"):
            image_order = f"attached images {start} and {start + 1} (RGB, then A/B heatmap)"
        else:
            image_order = f"attached images {start} and {start + 1} (A, then B)"
        lines.extend([
            f"Sample {i}; {image_order}; sample_id={item.sample_id}",
            item.user_text,
            "",
        ])
    lines.append("Return only the JSON object required by the supplied output schema.")
    return "\n".join(lines)


def load_cached(path: Path, items: list[Any]) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        sample_ids = [item.sample_id for item in items]
        return validate_results(json.loads(path.read_text(encoding="utf-8")), sample_ids)
    except Exception:
        return None


def run_batch(
    codex: str,
    repo: Path,
    model: str,
    batch_dir: Path,
    items: list[Any],
    input_format: str,
    anchors: list[tuple[Any, dict[str, Any]]],
    retries: int,
    timeout: int,
) -> tuple[str, list[dict[str, Any]], bool]:
    sample_ids = [item.sample_id for item in items]
    key = batch_key(sample_ids, input_format, bool(anchors))
    response_path = batch_dir / f"{key}.json"
    cached = load_cached(response_path, items)
    if cached is not None:
        return key, cached, True

    schema = result_schema(sample_ids)
    prompt = make_prompt(items, input_format, anchors)
    last_error = "not attempted"
    for attempt in range(1, retries + 1):
        with tempfile.TemporaryDirectory(prefix="codex-stage2-") as tmp:
            tmpdir = Path(tmp)
            schema_path = tmpdir / "schema.json"
            answer_path = tmpdir / "answer.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            cmd = [
                codex, "exec", "--ephemeral", "--color", "never", "--model", model,
                "--sandbox", "read-only", "--output-schema", str(schema_path),
                "--output-last-message", str(answer_path), "--cd", str(repo),
            ]
            for anchor, _spec in anchors:
                for path in image_paths(anchor):
                    cmd.extend(["--image", path])
            for item in items:
                for path in image_paths(item):
                    cmd.extend(["--image", path])
            cmd.append("-")
            try:
                proc = subprocess.run(
                    cmd, input=prompt, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=timeout, cwd=repo,
                )
                if proc.returncode != 0:
                    last_error = f"exit {proc.returncode}: {proc.stdout[-3000:]}"
                elif not answer_path.exists():
                    last_error = "codex did not create --output-last-message"
                else:
                    value = json.loads(answer_path.read_text(encoding="utf-8"))
                    rows = validate_results(value, sample_ids)
                    tmp_response = response_path.with_suffix(".json.tmp")
                    tmp_response.write_text(json.dumps({"results": rows}, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
                    os.replace(tmp_response, response_path)
                    return key, rows, False
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        (batch_dir / f"{key}.attempt{attempt}.log").write_text(last_error + "\n", encoding="utf-8")
        if attempt < retries:
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"{key} failed after {retries} attempts: {last_error}")


def write_labels(path: Path, manifest_rows: list[dict[str, Any]], results: dict[str, dict[str, Any]],
                 model: str, input_format: str, anchored: bool = False) -> None:
    lines = []
    for manifest in manifest_rows:
        sid = manifest["sample_id"]
        row = results[sid]
        a, b = row["hand_A"], row["hand_B"]
        score = min(a, b)
        fault = "none" if score == 2 else ("hand_A" if a <= b else "hand_B")
        record = {
            "sample_id": sid,
            "score_A": a,
            "score_B": b,
            "overall_score": score,
            "reason": row["reason"].strip(),
            "fault": fault,
            "orange_present": row["orange_present"],
            "teal_present": row["teal_present"],
            "orange_alignment": row["orange_alignment"],
            "teal_alignment": row["teal_alignment"],
            "orange_shape": row["orange_shape"],
            "teal_shape": row["teal_shape"],
            "model": model,
            "prompt_version": VERSION,
            "prompt_calibration": ANCHOR_CALIBRATION if anchored else "none",
            "render_scheme": ({
                "composite": "split-composite-bright-color-tau0.15",
                "rgb-composite": "rgb-reference-plus-split-composite-bright-color-tau0.15",
                "rgb-aligned-composite": "camera-aligned-rgb-plus-projected-split-composite-bright-color-tau0.15",
                "split": "split-per-hand-bright-color-tau0.15",
            }[input_format]),
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="new JSONL label file; existing gold is never read or changed")
    parser.add_argument("--run-dir", required=True, help="durable batch response/cache directory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-format",
                        choices=["composite", "rgb-composite", "rgb-aligned-composite", "split"],
                        default="split", help="rgb-aligned-composite requires exact-camera aligned fields")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--offset", type=int, default=0,
                        help="skip this many sorted manifest rows before applying --limit")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--visual-anchors", action="store_true",
                        help="attach the human-labeled visual anchors declared by the judge prompt")
    parser.add_argument("--sample-id", action="append", default=[],
                        help="judge only this exact sample_id; repeat for a targeted recheck")
    args = parser.parse_args()

    if args.batch_size < 1 or args.batch_size > 5:
        raise SystemExit("--batch-size must be 1..5; larger batches make image/sample association unreliable")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    codex = shutil.which("codex")
    if not codex:
        raise SystemExit("codex CLI not found")

    repo = Path(__file__).resolve().parents[2]
    all_manifests = read_jsonl(args.manifest)
    all_manifests.sort(key=lambda row: row["sample_index"])
    manifest_by_id = {row["sample_id"]: row for row in all_manifests}
    manifests = all_manifests
    if args.sample_id:
        wanted = set(args.sample_id)
        manifests = [row for row in manifests if row["sample_id"] in wanted]
        found = {row["sample_id"] for row in manifests}
        if found != wanted:
            raise SystemExit(f"unknown --sample-id values: {sorted(wanted - found)}")
    if args.offset:
        manifests = manifests[args.offset:]
    if args.limit:
        manifests = manifests[:args.limit]
    anchors: list[tuple[Any, dict[str, Any]]] = []
    if args.visual_anchors:
        missing = [spec["sample_id"] for spec in VISUAL_ANCHORS
                   if spec["sample_id"] not in manifest_by_id]
        if missing:
            raise SystemExit(f"visual anchors missing from manifest: {missing}")
        for spec in VISUAL_ANCHORS:
            row = manifest_by_id[spec["sample_id"]]
            if args.input_format == "rgb-aligned-composite":
                paths = (row["rgb_png_aligned"], row["heat_png_aligned"])
            elif args.input_format == "rgb-composite":
                paths = (row["rgb_png"], row["heat_png"])
            else:
                paths = row["heat_png"]
            anchor = replace(build_item(row), image_path=paths)
            anchors.append((anchor, spec))
    items = [build_item(row) for row in manifests]
    if args.input_format == "composite":
        # WorkItem is frozen; keep the exact claims from build_item and replace only its image input.
        items = [replace(item, image_path=row["heat_png"]) for item, row in zip(items, manifests)]
    elif args.input_format == "rgb-composite":
        items = [replace(item, image_path=(row["rgb_png"], row["heat_png"]))
                 for item, row in zip(items, manifests)]
    elif args.input_format == "rgb-aligned-composite":
        missing_aligned = [row["sample_id"] for row in manifests
                           if not row.get("rgb_png_aligned") or not row.get("heat_png_aligned")]
        if missing_aligned:
            raise SystemExit(f"aligned renders missing for {len(missing_aligned)} candidates")
        items = [replace(item, image_path=(row["rgb_png_aligned"], row["heat_png_aligned"]))
                 for item, row in zip(items, manifests)]
    batches = [items[i:i + args.batch_size] for i in range(0, len(items), args.batch_size)]
    batch_dir = Path(args.run_dir) / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, Any]] = {}
    completed = 0
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_batch, codex, repo, args.model, batch_dir, batch, args.input_format,
                        anchors, args.retries, args.timeout): batch
            for batch in batches
        }
        for future in concurrent.futures.as_completed(futures):
            batch = futures[future]
            try:
                key, rows, cached = future.result()
                for row in rows:
                    all_results[row["sample_id"]] = row
                completed += len(rows)
                print(f"[{completed}/{len(items)}] {key} {'cached' if cached else 'judged'}", flush=True)
            except Exception as exc:
                ids = ", ".join(item.sample_id for item in batch)
                failures.append(f"{ids}: {exc}")
                print(f"FAILED {ids}: {exc}", flush=True)

    if failures:
        failure_path = Path(args.run_dir) / "failures.txt"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(failures)} batches failed; resume the same command after checking {failure_path}")
    if set(all_results) != {item.sample_id for item in items}:
        raise SystemExit("internal error: final result ids do not match requested ids")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_labels(out, manifests, all_results, args.model, args.input_format, bool(anchors))
    summary = {
        "model": args.model,
        "prompt_version": VERSION,
        "input_format": args.input_format,
        "prompt_calibration": ANCHOR_CALIBRATION if anchors else "none",
        "samples": len(items),
        "batch_size": args.batch_size,
        "score_distribution": {
            str(score): sum(min(row["hand_A"], row["hand_B"]) == score for row in all_results.values())
            for score in (0, 1, 2)
        },
        "labels": str(out),
    }
    (Path(args.run_dir) / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
