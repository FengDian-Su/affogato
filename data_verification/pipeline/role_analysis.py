"""Repair step 2 -- diagnose every `role_not_core8_A/B` sample. Read-only.

For each manifest row carrying a role flag, put the stage1 record and the stage2
meta.json side by side and decide WHERE the illegal verb lives, so the repair can
target the right file. Nothing is rewritten and no mapping (`stabilize` -> `hold`
etc.) is applied or even suggested: the raw values are only counted.

Input  : data_verification/outputs/repair_stage0/{metadata_only,metadata_and_heatmap}.jsonl
Output : data_verification/outputs/repair_stage0/role_analysis/

Usage:
    python role_analysis.py
"""
import collections
import json
import os

import common as C

REPAIR = os.path.join(C.REPO, "data_verification/outputs/repair_stage0")
OUT_DIR = os.path.join(REPAIR, "role_analysis")
SRC = ("metadata_only.jsonl", "metadata_and_heatmap.jsonl")
GROUPS = ("stage2_metadata_only", "stage1_role_invalid", "unresolved")
# a stage1 fix that changes any of these invalidates the heatmap (README 3.4)
REGEN_FIELDS = ("role", "target", "contact_region", "function")


def _index_stage1(ds):
    return {r["object_id"]: r.get("queries", []) for r in C.load_stage1(ds)}


def _role_of(roles, hand):
    """The role verb for hand A/B, or (None, reason) when it cannot be resolved."""
    if not isinstance(roles, list) or len(roles) != 2:
        return None, "roles_not_two"
    by_id = {str(r.get("id")): r for r in roles if isinstance(r, dict)}
    if set(by_id) == {"A", "B"}:
        r = by_id[hand]
    else:                                   # ids missing/garbled -> fall back to position
        r = roles[0 if hand == "A" else 1]
        if not isinstance(r, dict):
            return None, "role_not_dict"
    v = r.get("role")
    return (v if isinstance(v, str) else None), (None if isinstance(v, str) else "role_missing")


def analyse(row, queries):
    out = {"dataset": row["dataset"], "sample_id": row["sample_id"],
           "object_id": row["object_id"], "repair_group": row["repair_group"],
           "role_flags": [t for t in row["metadata_correction_flags"] if t.startswith("role_")],
           "hands": {}, "group": None, "reasons": []}

    qi = int(row["sample_id"].split("/")[-1].split("_")[0][1:])
    if qi >= len(queries):
        out["group"] = "unresolved"
        out["reasons"].append("stage1_query_index_out_of_range")
        return out
    q = queries[qi]
    out["stage1_task"] = q.get("task")

    meta_p = os.path.join(row["source_stage2_path"], "meta.json")
    try:
        meta = C.load_meta(meta_p)
    except Exception as e:
        meta = None
        out["reasons"].append("stage2_meta_unreadable:%s" % type(e).__name__)
    if meta is not None and str(meta.get("task")) != str(q.get("task")):
        out["reasons"].append("task_mismatch_stage1_vs_stage2")

    verdicts = []
    for hand in [t[-1] for t in out["role_flags"]]:
        s1, why1 = _role_of(q.get("roles"), hand)
        s2, why2 = (_role_of(meta.get("roles"), hand) if meta is not None else (None, "no_meta"))
        h = {"stage1_role": s1, "stage2_role": s2,
             "stage1_in_core8": s1 in C.ROLE_VERBS, "stage2_in_core8": s2 in C.ROLE_VERBS,
             "agree": s1 == s2}
        if why1 or why2:
            h["note"] = why1 or why2
        if s1 is None or (s2 is None and meta is not None):
            v = "unresolved"
        elif not h["stage1_in_core8"]:
            v = "stage1_role_invalid"
        elif not h["stage2_in_core8"] or not h["agree"]:
            v = "stage2_metadata_only"
        else:
            # audit flagged it, yet both sides look legal now -> stage1 changed under us
            v = "unresolved"
            out["reasons"].append("flag_not_reproducible_%s" % hand)
        h["verdict"] = v
        out["hands"][hand] = h
        verdicts.append(v)

        # does the stage2 heatmap depend on this hand's other fields staying put?
        src = q.get("roles")
        if isinstance(src, list) and len(src) == 2 and meta is not None:
            m = _pair(meta.get("roles"), hand)
            s = _pair(src, hand)
            if m and s:
                diff = [f for f in REGEN_FIELDS if str(s.get(f)) != str(m.get(f))]
                if diff:
                    h["fields_differing_stage1_vs_stage2"] = diff

    if not verdicts:
        out["group"] = "unresolved"
        out["reasons"].append("no_role_flag_on_row")
    elif "unresolved" in verdicts or "task_mismatch_stage1_vs_stage2" in out["reasons"]:
        out["group"] = "unresolved"
    elif "stage1_role_invalid" in verdicts:
        out["group"] = "stage1_role_invalid"      # dominates: stage1 is the upstream truth
    else:
        out["group"] = "stage2_metadata_only"
    return out


def _pair(roles, hand):
    if not isinstance(roles, list) or len(roles) != 2:
        return None
    by_id = {str(r.get("id")): r for r in roles if isinstance(r, dict)}
    return by_id.get(hand) if set(by_id) == {"A", "B"} else roles[0 if hand == "A" else 1]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for f in SRC:
        with open(os.path.join(REPAIR, f)) as fh:
            rows += [json.loads(l) for l in fh]
    rows = [r for r in rows
            if any(t.startswith("role_") for t in r["metadata_correction_flags"])]

    results = []
    for ds in C.DATASETS:
        idx = _index_stage1(ds)
        for r in rows:
            if r["dataset"] == ds:
                results.append(analyse(r, idx.get(r["object_id"], [])))

    by_group = collections.defaultdict(list)
    for r in results:
        by_group[r["group"]].append(r)

    results.sort(key=lambda r: (r["dataset"], r["sample_id"]))
    with open(os.path.join(OUT_DIR, "role_analysis.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    for g in GROUPS:
        with open(os.path.join(OUT_DIR, "%s.jsonl" % g), "w") as f:
            for r in sorted(by_group[g], key=lambda r: (r["dataset"], r["sample_id"])):
                f.write(json.dumps(r) + "\n")

    # --- illegal verbs, counted verbatim. No normalisation, no mapping.
    s1_vals, s2_vals, per_ds = collections.Counter(), collections.Counter(), collections.Counter()
    hand_verdict, agree, regen = collections.Counter(), collections.Counter(), collections.Counter()
    for r in results:
        for hand, h in r["hands"].items():
            hand_verdict[h["verdict"]] += 1
            agree["agree" if h["agree"] else "differ"] += 1
            if not h["stage1_in_core8"] and h["stage1_role"] is not None:
                s1_vals[h["stage1_role"]] += 1
                per_ds["%s:%s" % (r["dataset"], h["stage1_role"])] += 1
            if not h["stage2_in_core8"] and h["stage2_role"] is not None:
                s2_vals[h["stage2_role"]] += 1
            for f in h.get("fields_differing_stage1_vs_stage2", []):
                regen[f] += 1

    with open(os.path.join(OUT_DIR, "illegal_role_values.json"), "w") as f:
        json.dump({"note": "raw values only -- no mapping applied or implied",
                   "stage1": dict(s1_vals.most_common()),
                   "stage2_meta": dict(s2_vals.most_common()),
                   "stage1_per_dataset": dict(per_ds.most_common())}, f, indent=1)

    summary = {
        "schema_version": "role-analysis-v1.0",
        "n_samples": len(results),
        "n_flagged_hands": sum(hand_verdict.values()),
        "groups": {g: len(by_group[g]) for g in GROUPS},
        "groups_per_dataset": {
            g: dict(collections.Counter(r["dataset"] for r in by_group[g])) for g in GROUPS},
        "hand_level_verdicts": dict(hand_verdict),
        "stage1_vs_stage2_role_agreement": dict(agree),
        "n_unique_illegal_stage1_values": len(s1_vals),
        "illegal_stage1_values": dict(s1_vals.most_common()),
        "illegal_stage2_values": dict(s2_vals.most_common()),
        "repair_implication": {
            "stage2_meta_sync_only": len(by_group["stage2_metadata_only"]),
            "stage1_fix_needed": len(by_group["stage1_role_invalid"]),
            "stage1_fix_also_forces_stage2_regen": sum(
                1 for r in by_group["stage1_role_invalid"]
                if r["repair_group"] == "metadata_and_heatmap"
                or any(h["verdict"] == "stage1_role_invalid" for h in r["hands"].values())),
            "note": "Any stage1 edit that changes role/target/contact_region/function or the "
                    "Molmo query invalidates the heatmap (README 3.4); the verb itself is one "
                    "of those fields, so a stage1 role fix always implies stage2 regeneration.",
        },
        "stage1_stage2_field_divergence": dict(regen),
        "unresolved_reasons": dict(collections.Counter(
            x for r in by_group["unresolved"] for x in r["reasons"]).most_common()),
    }
    with open(os.path.join(OUT_DIR, "role_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print("[role] wrote %s" % OUT_DIR)


if __name__ == "__main__":
    main()
