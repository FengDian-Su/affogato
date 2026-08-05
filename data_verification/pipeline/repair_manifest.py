"""Stage 0 unified repair manifest -- merge all-zero heatmaps and metadata flags.

Input  : data_verification/outputs/stage0/{ds}/{hard_failures,metadata_corrections}.jsonl
Output : data_verification/outputs/repair_stage0/

Listing and counting only. Nothing here regenerates a heatmap, edits stage1, or
touches warnings/diagnostics.

Excluded on purpose: every sample whose state is not `present`. Those belong to
the objects Generation Stage 2 never reached; they are a generation backlog, not
a repair job, and are reported as `excluded_not_generated`. This matters because
audit.py judges roles/enums straight from stage1 even when stage2 emitted no
directory, so metadata_corrections.jsonl DOES contain such rows.

Usage:
    python repair_manifest.py
"""
import collections
import json
import os

import common as C

STAGE0 = os.path.join(C.REPO, "data_verification/outputs/stage0")
OUT_DIR = os.path.join(C.REPO, "data_verification/outputs/repair_stage0")
ALL_ZERO = ("all_zero_A", "all_zero_B")
# out of scope for this repair round (2026-08-04): a coordination enum fix does not
# invalidate a heatmap, so these are tracked but never a reason to repair a sample
DROP_METADATA = ("coordination_invalid", "meta_missing_coordination")
FIELDS = ("dataset", "object_id", "sample_id", "hard_failures",
          "metadata_correction_flags", "source_stage1_path", "source_stage2_path",
          "repair_group")


def _read(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    merged, excluded, dup_hits, anomalies = {}, collections.Counter(), collections.Counter(), []
    dropped = collections.Counter()
    per_ds_excluded_objs = {}

    for ds in C.DATASETS:
        s1 = C.STAGE1_JSON.format(ds=ds)
        root = C.STAGE2_ROOT.format(ds=ds)
        skipped_objs = set()

        for fname, key, keep in (("hard_failures.jsonl", "hard_failures", ALL_ZERO),
                                 ("metadata_corrections.jsonl", "metadata_correction_flags",
                                  DROP_METADATA)):
            for row in _read(os.path.join(STAGE0, ds, fname)):
                # state first, so the two exclusion counters never double-count a row
                if row["state"] != "present":            # generation backlog, not repair
                    excluded["%s:%s" % (ds, key)] += 1
                    skipped_objs.add(row["sample_id"].split("/")[0])
                    continue
                flags = ([t for t in row["flags"] if t in keep] if keep is ALL_ZERO
                         else [t for t in row["flags"] if t not in keep])
                if not flags:                            # nothing left in scope
                    dropped["%s:%s" % (ds, key)] += 1
                    continue
                oid, uid = row["sample_id"].split("/")[0], (ds, row["sample_id"])
                if uid in merged and merged[uid][key]:
                    dup_hits[uid] += 1                   # same key twice in one source file
                rec = merged.setdefault(uid, {
                    "dataset": ds, "object_id": oid, "sample_id": row["sample_id"],
                    "hard_failures": [], "metadata_correction_flags": [],
                    "source_stage1_path": s1,
                    "source_stage2_path": os.path.join(root, row["sample_id"]),
                    "repair_group": None,
                })
                rec[key] = sorted(set(rec[key]) | set(flags))
        per_ds_excluded_objs[ds] = len(skipped_objs)

    # --- group + integrity
    groups = {"heatmap_only": [], "metadata_only": [], "metadata_and_heatmap": []}
    for uid, r in merged.items():
        h, m = bool(r["hard_failures"]), bool(r["metadata_correction_flags"])
        r["repair_group"] = ("metadata_and_heatmap" if h and m
                             else "heatmap_only" if h else "metadata_only")
        groups[r["repair_group"]].append(r)
        if not (h or m):
            anomalies.append({"sample_id": r["sample_id"], "issue": "empty_flags", "detail": None})
        missing = [f for f in FIELDS if r.get(f) in (None, "")]
        if missing:
            anomalies.append({"sample_id": r["sample_id"], "issue": "missing_fields",
                              "detail": missing})
        if not os.path.isdir(r["source_stage2_path"]):
            anomalies.append({"sample_id": r["sample_id"], "issue": "stage2_path_missing",
                              "detail": r["source_stage2_path"]})
    for p in {r["source_stage1_path"] for r in merged.values()}:
        if not os.path.isfile(p):
            anomalies.append({"sample_id": None, "issue": "stage1_path_missing", "detail": p})

    rows = sorted(merged.values(), key=lambda r: (r["dataset"], r["sample_id"]))
    with open(os.path.join(OUT_DIR, "repair_manifest.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for g, gr in groups.items():
        with open(os.path.join(OUT_DIR, "%s.jsonl" % g), "w") as f:
            for r in sorted(gr, key=lambda r: (r["dataset"], r["sample_id"])):
                f.write(json.dumps(r) + "\n")

    def _cnt(pred, field=None):
        c = collections.Counter()
        for r in rows:
            if pred(r):
                c["%s" % r["dataset"]] += 1
                for t in (r[field] if field else []):
                    c["%s:%s" % (r["dataset"], t)] += 1
        return dict(c)

    per_ds = collections.Counter(r["dataset"] for r in rows)
    summary = {
        "schema_version": "repair-stage0-v1.0",
        "source": STAGE0,
        "n_unique_samples": len(rows),
        "per_dataset_unique_samples": dict(per_ds),
        "repair_groups": {g: len(v) for g, v in groups.items()},
        "repair_groups_per_dataset": {
            g: dict(collections.Counter(r["dataset"] for r in v)) for g, v in groups.items()},
        "n_intersection_both_problems": len(groups["metadata_and_heatmap"]),
        "hard_failure_flags": _cnt(lambda r: r["hard_failures"], "hard_failures"),
        "metadata_correction_flags": _cnt(
            lambda r: r["metadata_correction_flags"], "metadata_correction_flags"),
        # README 3.4: correcting role/target/contact_region/function forces a heatmap
        # regeneration too, so metadata_only is not automatically heatmap-safe
        "metadata_only_touching_role_fields": sum(
            1 for r in groups["metadata_only"]
            if any(t.startswith(("role_", "missing_")) for t in r["metadata_correction_flags"])),
        "excluded_metadata_flags": {
            "flags": list(DROP_METADATA),
            "n_samples_dropped_entirely": dict(dropped),
            "note": "Out of scope this round; a coordination fix leaves the heatmap valid. "
                    "Samples keeping a role_not_core8_* or all_zero_* flag stay in, "
                    "with the coordination flags stripped.",
        },
        "excluded_not_generated": dict(
            _excluded(),
            n_objects_touched_by_dropped_rows=sum(per_ds_excluded_objs.values()),
            n_rows_dropped_from_sources=dict(excluded),
            note="Generation Stage 2 never produced these; backlog, not repair.",
        ),
        "anomalies": {"n_duplicate_ids": len(dup_hits),
                      "n_missing_field_rows": sum(a["issue"] == "missing_fields" for a in anomalies),
                      "n_missing_source_paths": sum(
                          a["issue"] in ("stage2_path_missing", "stage1_path_missing")
                          for a in anomalies),
                      "detail": anomalies[:20]},
    }
    with open(os.path.join(OUT_DIR, "repair_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print("[repair] wrote %s" % OUT_DIR)


def _excluded():
    """The whole not_generated population: objects stage2 never created a dir for.

    Samples come from the audit summaries; objects are recomputed as
    stage1 - on_disk, which is exactly how audit.py assigns `not_generated`.
    """
    objs, smp, zeroq = {}, {}, {}
    for ds in C.DATASETS:
        with open(os.path.join(STAGE0, ds, "audit_summary.json")) as f:
            smp[ds] = json.load(f)["state_counts"].get("not_generated", 0)
        root = C.STAGE2_ROOT.format(ds=ds)
        on_disk = set(os.listdir(root))
        missing = [r for r in C.load_stage1(ds) if r["object_id"] not in on_disk]
        # an object with no stage1 queries owes no sample, so it is not part of the
        # not_generated population even though it has no stage2 directory either
        objs[ds] = sum(1 for r in missing if r.get("queries"))
        zeroq[ds] = sum(1 for r in missing if not r.get("queries"))
    return {"n_objects": dict(objs, total=sum(objs.values())),
            "n_samples": dict(smp, total=sum(smp.values())),
            "n_stage1_objects_with_zero_queries": dict(zeroq, total=sum(zeroq.values()))}


if __name__ == "__main__":
    main()
