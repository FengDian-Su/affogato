#!/usr/bin/env python3
"""Score check for the corpus judge run, called by monitor.sh at every 10% of progress (of the heatmap
phase once it has started; the task/role rows come first in each shard file).

Per criterion, over all rows so far ("total") and over the rows added since the previous check
("window"), overall and per dataset: pairs, score (mean label / 2 x 100), % graded Good, failed rows.
For the reviewed pairs already reached, the share of labels identical to the review run. Appends a
JSON line to scores.jsonl and a readable block to scores.log, and prints one summary line. FLAG marks
anything worth a look:
  - a daily_used window more than 3 points from the judge's score on the 1,000 reviewed pairs, which
    are daily_used (task 93.3, role 96.5, heatmap 90.3);
  - a dataset more than 5 points from daily_used;
  - fewer than 97% identical labels on the reviewed pairs (two runs of the same prompt agree on 99%);
  - more than 0.5% failed rows.
"""
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "data_verification/outputs/quality_review/corpus"   # the run's data
RUN = next(path for path in (ROOT / "qwen38-27b-v23", ROOT / "qwen38-27b") if path.exists())
REVIEW_RUN = ROOT.parent / "release1000/qwen38_27b/qwen38-27b-v23"
REFERENCE = {"task": 93.3, "role": 96.5, "heatmap": 90.3}
AXES = ("task", "role", "heatmap")
PAIRS = 310234


def dataset_of() -> dict[str, str]:
    cache = ROOT / "sample_dataset.tsv"
    if not cache.exists():
        temporary = ROOT / "sample_dataset.tmp"
        with (ROOT / "manifest.jsonl").open(encoding="utf-8") as src, temporary.open("w") as dst:
            for line in src:
                row = json.loads(line)
                dst.write(f"{row['sample_id']}\t{row['dataset']}\n")
        temporary.rename(cache)
    return dict(line.rstrip("\n").split("\t") for line in cache.open())


def summary(counts: Counter) -> dict:
    n = counts[0] + counts[1] + counts[2]
    return {"n": n, "score": round(100 * (counts[1] + 2 * counts[2]) / (2 * n), 2) if n else None,
            "good": round(100 * counts[2] / n, 2) if n else None,
            "dist": [counts[0], counts[1], counts[2]], "failed": counts[None]}


def main() -> None:
    datasets = dataset_of()
    history_path = ROOT / "scores.jsonl"
    history = [json.loads(line) for line in history_path.open()] if history_path.exists() else []
    before = history[-1]["rows"] if history else {}
    review = {(r["sample_id"], r["axis"]): r["score"] for path in REVIEW_RUN.glob("shard*.jsonl")
              for r in map(json.loads, path.open(encoding="utf-8")) if r["axis"] in AXES}
    total, window = defaultdict(Counter), defaultdict(Counter)        # (axis, dataset) -> label counts
    reached, same, rows = Counter(), Counter(), {}
    for path in sorted(RUN.glob("shard*.jsonl")):
        done = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):                          # a chunk being written right now
                    break
                done += 1
                r = json.loads(line)
                key = (r["axis"], datasets[r["sample_id"]])
                total[key][r["score"]] += 1
                if done > before.get(path.name, 0):
                    window[key][r["score"]] += 1
                reference = review.get((r["sample_id"], r["axis"]))
                if reference is not None and r["score"] is not None:
                    reached[r["axis"]] += 1
                    same[r["axis"]] += reference == r["score"]
        rows[path.name] = done

    names = sorted({name for _, name in total})

    def merged(table: dict, axis: str) -> dict:
        out = {"all": summary(sum((c for (a, _), c in table.items() if a == axis), Counter()))}
        out.update({name: summary(table[(axis, name)]) for name in names})
        return out

    heatmap_rows = sum(sum(c.values()) for (axis, _), c in total.items() if axis == "heatmap")
    record = {"time": time.strftime("%F %T"), "rows": rows,
              "progress": round(100 * (heatmap_rows or sum(rows.values()) / 2) / PAIRS, 1),
              "total": {axis: merged(total, axis) for axis in AXES},
              "window": {axis: merged(window, axis) for axis in AXES},
              "reviewed": {axis: {"n": reached[axis], "identical": round(100 * same[axis] / reached[axis], 1)
                                  if reached[axis] else None} for axis in AXES}}
    flags = []
    for axis in AXES:
        piece = record["window"][axis]["daily_used"]
        if piece["n"] >= 2000 and abs(piece["score"] - REFERENCE[axis]) > 3:
            flags.append(f"{axis}: daily_used window {piece['score']} vs {REFERENCE[axis]} on the reviewed pairs")
        base = record["total"][axis]["daily_used"]
        for name in names:
            other = record["total"][axis][name]
            if name != "daily_used" and other["n"] >= 2000 and base["n"] and abs(other["score"] - base["score"]) > 5:
                flags.append(f"{axis}: {name} {other['score']} vs daily_used {base['score']}")
        reviewed = record["reviewed"][axis]
        if reviewed["n"] >= 100 and reviewed["identical"] < 97:
            flags.append(f"{axis}: reviewed pairs {reviewed['identical']}% identical")
        everything = record["total"][axis]["all"]
        if everything["failed"] > 0.005 * (everything["n"] + everything["failed"]):
            flags.append(f"{axis}: {everything['failed']} failed rows")
    record["flags"] = flags
    with history_path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")

    def show(entry: dict) -> str:
        return "--" if not entry["n"] else f"{entry['score']:.1f} ({entry['good']:.1f}% G, n={entry['n']:,})"

    lines = [f"== {record['time']} | {record['progress']}% | "
             + " ".join(f"{name} {count:,}" for name, count in rows.items()) + " =="]
    for axis in AXES:
        lines.append(f"{axis:4s} total {show(record['total'][axis]['all'])} | window {show(record['window'][axis]['all'])}"
                     f" | failed {record['total'][axis]['all']['failed']}")
        lines.append("     by dataset: " + " | ".join(f"{name} {show(record['total'][axis][name])}" for name in names))
    lines.append("reviewed pairs reached: " + ", ".join(
        f"{axis} {record['reviewed'][axis]['n']} ({record['reviewed'][axis]['identical']}% identical)" for axis in AXES))
    lines.append("FLAG: " + "; ".join(flags) if flags else "flags: none")
    with (ROOT / "scores.log").open("a") as stream:
        stream.write("\n".join(lines) + "\n\n")
    print(f"score check {record['progress']}%: "
          + " ".join(f"{axis} {record['total'][axis]['all']['score']} (window {record['window'][axis]['all']['score']})"
                     for axis in AXES)
          + " | reviewed identical " + "/".join(f"{record['reviewed'][axis]['identical']}" for axis in AXES)
          + "% | " + ("FLAG: " + "; ".join(flags) if flags else "flags: none"))


if __name__ == "__main__":
    main()
