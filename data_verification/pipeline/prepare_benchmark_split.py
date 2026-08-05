#!/usr/bin/env python3
"""Build the dev/test split over the re-labelled Claude gold set.

Replaces prepare_simple_benchmark.py, which also rebuilt the gold from the old strict labels and
the small-pickup re-adjudication; those inputs are gone and the gold is now produced directly by
the re-labelling pass. This script only splits.

Guarantees: no object appears in both splits; the (metadata, heatmap) score pair is stratified to
match the full set as closely as an object-disjoint split allows; the size is exact; the same seed
reproduces the same split byte for byte.

  python pipeline/prepare_benchmark_split.py --n-dev 700
"""
import json
import math
import random
import hashlib
import argparse
import collections
from pathlib import Path

SEED = 20260805
ROOT = Path(__file__).resolve().parents[1] / "quality_evaluation/claude_pilot_1000"


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def target_counts(pair_counts, n_dev, total):
    raw = {k: n_dev * v / total for k, v in pair_counts.items()}
    targets = {k: math.floor(v) for k, v in raw.items()}
    for key in sorted(raw, key=lambda k: (raw[k] - targets[k], pair_counts[k], k),
                      reverse=True)[:n_dev - sum(targets.values())]:
        targets[key] += 1
    return targets


def choose_dev(manifest, pair_by_id, n_dev):
    by_object = collections.defaultdict(list)
    for row in manifest:
        by_object[row["object_id"]].append(row)
    totals = collections.Counter(pair_by_id.values())
    targets = target_counts(totals, n_dev, len(manifest))
    groups = list(by_object.items())
    random.Random(SEED).shuffle(groups)
    selected, selected_n, current = set(), 0, collections.Counter()

    while selected_n < n_dev:
        capacity = n_dev - selected_n
        cands = [(o, r) for o, r in groups if o not in selected and len(r) <= capacity]
        if not cands:
            raise RuntimeError("could not reach the exact dev size without splitting an object")

        def cost(c):
            oid, rows = c
            proposed = current.copy()
            for row in rows:
                proposed[pair_by_id[row["sample_id"]]] += 1
            return (sum((proposed[k] - targets[k]) ** 2 for k in totals),
                    sum(max(0, proposed[k] - targets[k]) for k in totals), -len(rows), oid)

        oid, rows = min(cands, key=cost)
        selected.add(oid)
        selected_n += len(rows)
        for row in rows:
            current[pair_by_id[row["sample_id"]]] += 1
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--n-dev", type=int, default=700)
    a = ap.parse_args()
    root = Path(a.root)

    manifest = read_jsonl(root / "manifest.jsonl")
    meta = {r["sample_id"]: r["overall_score"] for r in read_jsonl(root / "metadata_labels.gold.jsonl")}
    heat = {r["sample_id"]: r["overall_score"] for r in read_jsonl(root / "heatmap_labels.gold.jsonl")}
    ids = {r["sample_id"] for r in manifest}
    if set(meta) != ids or set(heat) != ids:
        raise ValueError("gold and manifest sample ids do not match")

    pair = {sid: (meta[sid], heat[sid]) for sid in ids}
    dev_objects = choose_dev(manifest, pair, a.n_dev)
    rows = [{"sample_id": r["sample_id"], "object_id": r["object_id"],
             "split": "dev" if r["object_id"] in dev_objects else "test",
             "stratum": f"m{meta[r['sample_id']]}_h{heat[r['sample_id']]}"}
            for r in manifest]

    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    if len(dev) != a.n_dev or ({r["object_id"] for r in dev} & {r["object_id"] for r in test}):
        raise ValueError("split failed its own invariants")

    out = root / "simple_verifier_1000"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "benchmark_split.jsonl", rows)
    summary = {
        "schema_version": "benchmark-split-v3", "seed": SEED,
        "gold": "metadata_labels.gold.jsonl + heatmap_labels.gold.jsonl (re-labelled 2026-08-05)",
        "dev": len(dev), "test": len(test), "object_overlap": 0,
        "dev_objects": len(dev_objects), "test_objects": len({r["object_id"] for r in test}),
        "dev_metadata_dist": dict(sorted(collections.Counter(meta[r["sample_id"]] for r in dev).items())),
        "test_metadata_dist": dict(sorted(collections.Counter(meta[r["sample_id"]] for r in test).items())),
        "dev_heatmap_dist": dict(sorted(collections.Counter(heat[r["sample_id"]] for r in dev).items())),
        "test_heatmap_dist": dict(sorted(collections.Counter(heat[r["sample_id"]] for r in test).items())),
        "dev_strata": dict(sorted(collections.Counter(r["stratum"] for r in dev).items())),
        "split_sha256": hashlib.sha256(open(out / "benchmark_split.jsonl", "rb").read()).hexdigest(),
    }
    json.dump(summary, open(out / "benchmark_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
