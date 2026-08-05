"""Build a reproducible human-verification sample from completed Stage 2 results.

This is a prevalence sample: it estimates the quality distribution in the generated
data.  It is deliberately separate from synthetic-corruption/challenge sets, which
measure whether a judge can detect known errors.
"""
import argparse
import hashlib
import json
import os
import random
from collections import Counter

import common as C


SCHEMA_VERSION = "human-verification-sample-v1.0"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_dataset(dataset, n, seed):
    records = C.load_stage1(dataset)
    samples, _ = C.expected_samples(records, C.STAGE2_ROOT.format(ds=dataset))
    rng = random.Random(seed)
    rng.shuffle(samples)
    present = [sm for sm in samples if sm["state"] == "present"]
    picks = present[:n]
    if len(picks) != n:
        raise RuntimeError("%s: requested %d completed results, found %d" %
                           (dataset, n, len(picks)))
    return picks, len(present)


def annotation_record(dataset, sm, sample_index, population_n, sample_n):
    meta_path = os.path.join(sm["qdir"], "meta.json")
    score_path = os.path.join(sm["qdir"], "scores.npz")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        meta = {}
    roles = meta.get("roles") or []
    role_pair = sorted(str(r.get("role", "")) for r in roles)
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_index": sample_index,
        "dataset": dataset,
        "sample_id": sm["sample_id"],
        "object_id": sm["object_id"],
        "qdir": sm["qdir"],
        "meta_path": meta_path,
        "scores_path": score_path,
        "analysis_weight": population_n / sample_n,
        "object_name": meta.get("object_name"),
        "task": meta.get("task", sm.get("task")),
        "category": meta.get("category", sm.get("category")),
        "coordination": meta.get("coordination", sm.get("coordination")),
        "role_pair": role_pair,
        "metadata_annotation": {
            "task_score": None,
            "bimanual_score": None,
            "hand_A_score": None,
            "hand_B_score": None,
            "overall_score": None,
            "error_tags": [],
            "reason": None,
        },
        "heatmap_annotation": {
            "evaluable": None,
            "hand_A_score": None,
            "hand_B_score": None,
            "dual_score": None,
            "overall_score": None,
            "error_tags": [],
            "evidence_views": [],
            "reason": None,
        },
        "final_decision": None,
        "annotator_id": None,
        "annotation_round": None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-dataset", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260804)
    p.add_argument("--out", default=os.path.join(
        C.REPO, "data_verification/quality_evaluation/human_sample_1000"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    population_counts = {}
    # Dataset-specific derived seeds make each stratum reproducible by itself.
    for di, dataset in enumerate(C.DATASETS):
        picks, population_n = sample_dataset(dataset, a.per_dataset, a.seed + di)
        population_counts[dataset] = population_n
        rows.extend(annotation_record(dataset, sm, len(rows), population_n, len(picks))
                    for sm in picks)

    # Shuffle presentation order so annotators do not see one dataset in a block.
    random.Random(a.seed).shuffle(rows)
    for i, row in enumerate(rows):
        row["sample_index"] = i

    manifest = os.path.join(a.out, "manifest.jsonl")
    with open(manifest, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Separate forms enforce the two-pass protocol and avoid leaking a semantic verdict
    # into the spatial judgment.  They are templates; raw annotator copies must be kept.
    metadata_form = os.path.join(a.out, "metadata_labels.template.jsonl")
    heatmap_form = os.path.join(a.out, "heatmap_labels.template.jsonl")
    with open(metadata_form, "w") as fm, open(heatmap_form, "w") as fh:
        for row in rows:
            key = {k: row[k] for k in ("schema_version", "sample_index", "dataset", "sample_id")}
            fm.write(json.dumps({**key, **row["metadata_annotation"]}, ensure_ascii=False) + "\n")
            fh.write(json.dumps({**key, **row["heatmap_annotation"]}, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": a.seed,
        "sampling_design": "equal_allocation_by_dataset_then_uniform_over_completed_results",
        "eligibility": "all Stage 2 query directories mapped to current Stage 1 (no quality prefilter)",
        "n": len(rows),
        "population_counts": population_counts,
        "dataset_counts": dict(Counter(r["dataset"] for r in rows)),
        "unique_objects": len({(r["dataset"], r["object_id"]) for r in rows}),
        "category_counts": dict(Counter(str(r["category"]) for r in rows)),
        "role_pair_counts": dict(Counter(" + ".join(r["role_pair"]) for r in rows)),
        "manifest_sha256": file_sha256(manifest),
    }
    with open(os.path.join(a.out, "sample_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
