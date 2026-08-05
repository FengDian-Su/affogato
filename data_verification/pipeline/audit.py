"""Verification Stage 0 -- Programmatic Audit (README section 3).

Runs on 100% of samples. Answers only: are the files usable, is the schema and
the enum set correct, are the heatmap values valid, and what are the two dual
geometry statistics. It does NOT judge whether a task is sensible, whether the
contact region matches the object, or whether a heatmap sits on the right part.

Per README L1 there is no geometry-usability judgement of any kind here; `xyz`
is only checked for readability (shape / finite / length match).

Usage:
    python audit.py --dataset daily_used           # -> ../outputs/stage0/daily_used
    python audit.py --example --limit 200          # plumbing check on the old fixture
"""
import argparse
import collections
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as C


def _check_roles(roles, hard, corr):
    if not isinstance(roles, list) or len(roles) != 2:
        hard.append("roles_not_exactly_two")
        return None, None
    if {str(r.get("id")) for r in roles} != {"A", "B"}:
        corr.append("role_ids_not_AB")
    verdict = []
    for r, tag in zip(roles, "AB"):
        for f in C.ROLE_FIELDS:
            if not str(r.get(f) or "").strip():
                corr.append("missing_%s_%s" % (f, tag))
        ok = r.get("role") in C.ROLE_VERBS
        if not ok:
            corr.append("role_not_core8_%s" % tag)
        verdict.append(ok)
    return verdict[0], verdict[1]


def _check_heatmap(s, tag, n_pts, hard):
    """Returns True if the hand's heatmap is numerically usable."""
    if s.shape[0] != n_pts:
        hard.append("shape_mismatch_%s" % tag)
        return False
    if not np.isfinite(s).all():
        hard.append("nan_inf_%s" % tag)
        return False
    if s.min() < 0.0 or s.max() > 1.0:
        hard.append("out_of_range_%s" % tag)
        return False
    if not (s > 0).any():
        hard.append("all_zero_%s" % tag)
        return False
    return True


def audit_sample(sm, tau):
    """One object-task pair -> the Stage 0 record (README section 3.7)."""
    out = {
        "schema_version": C.SCHEMA_VERSION,
        "sample_id": sm["sample_id"],
        "schema_pass": False,
        "heatmap_A_valid": False, "heatmap_B_valid": False,
        "role_A_valid": False, "role_B_valid": False,
        "raw_soft_overlap": None,
        "normalized_center_distance": None,
        "hard_failures": [], "metadata_correction_flags": [], "warning_flags": [],
        "stats": {},
    }
    hard, corr, warn = out["hard_failures"], out["metadata_correction_flags"], out["warning_flags"]
    roles = sm["roles"]

    # --- roles / enums: judged from stage1 even when stage2 never emitted a dir,
    # so that generator-skipped queries still get their real diagnosis.
    out["role_A_valid"], out["role_B_valid"] = _check_roles(roles, hard, corr)
    if sm["category"] not in C.CATEGORY_ENUM:
        corr.append("category_invalid")
    if sm["coordination"] not in C.COORDINATION_ENUM:
        corr.append("coordination_invalid")

    # --- presence
    out["stats"]["state"] = sm["state"]
    if sm["state"] != "present":
        if sm["state"] == "missing_qdir":
            # Reproduce the generator's own filter (stage2_v2.py:264) instead of
            # inferring from "the object dir exists": a directory can also be
            # absent because this stage2 run came from a different stage1
            # version, which is drift (see orphans), not a generator skip.
            if not (isinstance(roles, list) and len(roles) == 2 and sm["n_molmo"] == 2):
                out["stats"]["state"] = "skipped_by_generator"
                hard.append("skipped_by_generator")
            else:
                hard.append("missing_qdir")
        return out

    npz_p, meta_p = os.path.join(sm["qdir"], "scores.npz"), os.path.join(sm["qdir"], "meta.json")
    for p, tag in ((npz_p, "scores_npz"), (meta_p, "meta_json")):
        if not os.path.isfile(p):
            hard.append("missing_%s" % tag)
    if hard:
        out["stats"]["state"] = "missing_file"
        return out

    try:
        z = C.load_npz(npz_p)
        keys = set(z.files)
        meta = C.load_meta(meta_p)
    except Exception as e:                                    # unreadable / truncated
        hard.append("unreadable:%s" % type(e).__name__)
        out["stats"]["state"] = "unreadable"
        return out
    out["stats"]["state"] = "present"

    missing_req = [k for k in C.NPZ_REQUIRED if k not in keys]
    if missing_req:
        hard += ["missing_key_%s" % k for k in missing_req]
    for k in C.META_REQUIRED:
        if meta.get(k) in (None, "", []):
            corr.append("meta_missing_%s" % k)
    unknown = keys - set(C.NPZ_REQUIRED) - set(C.NPZ_EXPECTED) - set(C.NPZ_OPTIONAL_KNOWN)
    if unknown:
        warn.append("unknown_npz_keys:%s" % ",".join(sorted(unknown)))
    if missing_req:
        return out

    # --- mapping consistency
    if meta.get("object_id") != sm["object_id"]:
        corr.append("object_id_mismatch")
    if str(meta.get("task")) != str(sm["task"]):
        corr.append("task_mismatch_meta_vs_stage1")

    # --- xyz readability (the point cloud IS scores.npz:xyz -- no separate file)
    xyz = z["xyz"]
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        hard.append("xyz_bad_shape")
        return out
    if not np.isfinite(xyz).all():
        hard.append("xyz_nan_inf")
        return out
    n_pts = xyz.shape[0]

    sa, sb = z["scoreA"].astype(np.float64), z["scoreB"].astype(np.float64)
    out["heatmap_A_valid"] = _check_heatmap(sa, "A", n_pts, hard)
    out["heatmap_B_valid"] = _check_heatmap(sb, "B", n_pts, hard)

    # --- section 3.6 dual geometry (2 metrics only)
    if all(k in keys for k in C.NPZ_EXPECTED):
        ra, rb = z["scoreA_raw"].astype(np.float64), z["scoreB_raw"].astype(np.float64)
        if ra.shape[0] == n_pts and rb.shape[0] == n_pts and np.isfinite(ra).all() and np.isfinite(rb).all():
            ov = C.raw_soft_overlap(ra, rb)
            out["raw_soft_overlap"] = ov
            if ov >= C.WARN["raw_overlap_high"]:
                warn.append("raw_overlap_high")
        else:
            warn.append("raw_scores_unusable")
    else:
        warn.append("missing_raw_scores")

    if out["heatmap_A_valid"] and out["heatmap_B_valid"]:
        d = C.normalized_center_distance(xyz, sa, sb)
        out["normalized_center_distance"] = d
        if d <= C.WARN["center_dist_low"]:
            warn.append("center_distance_low")
        elif d >= C.WARN["center_dist_high"]:
            warn.append("center_distance_high")
        if np.array_equal(sa, sb):
            warn.append("AB_identical")

        fa, fb = C.active_frac(sa, tau), C.active_frac(sb, tau)
        for f, tag in ((fa, "A"), (fb, "B")):
            if f <= C.WARN["active_frac_low"]:
                warn.append("active_region_small_%s" % tag)
            elif f >= C.WARN["active_frac_high"]:
                warn.append("active_region_large_%s" % tag)
        # raw numbers kept so Phase 2 can re-derive tau_vis without re-running
        out["stats"].update(n_points=n_pts, tau_vis=tau,
                            active_frac_A=fa, active_frac_B=fb,
                            scoreA_max=float(sa.max()), scoreB_max=float(sb.max()))

    for k, lo in (("hitA", C.WARN["hit_low"]), ("hitB", C.WARN["hit_low"]),
                  ("coverage", C.WARN["coverage_low"])):
        v = meta.get(k)
        if isinstance(v, (int, float)) and v < lo:
            warn.append("%s_low" % k)
        out["stats"][k] = v

    out["schema_pass"] = not hard and not corr
    return out


def _worker(args):
    return audit_sample(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="daily_used", choices=list(C.DATASETS))
    ap.add_argument("--example", action="store_true",
                    help="audit the old 1000-object fixture (plumbing only)")
    ap.add_argument("--stage2_root", default=None)
    ap.add_argument("--tau_vis", type=float, default=C.TAU_VIS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = a.stage2_root or (C.EXAMPLE_ROOT if a.example else C.STAGE2_ROOT.format(ds=a.dataset))
    out_dir = a.out or os.path.join(C.REPO, "data_verification/outputs/stage0",
                                    "example" if a.example else a.dataset)
    os.makedirs(out_dir, exist_ok=True)

    recs = C.load_stage1(a.dataset)
    if a.example:                       # fixture holds a subset of daily_used objects
        have = set(os.listdir(root))
        recs = [r for r in recs if r["object_id"] in have]
    samples, orphans = C.expected_samples(recs, root)
    orphan_objs = [] if a.example else C.orphan_objects(recs, root)
    if a.limit:
        samples = samples[:a.limit]
    print("[stage0] %s | %d objects | %d expected samples | stage1=%s | root=%s"
          % (a.dataset, len(recs), len(samples), C.STAGE1_JSON.format(ds=a.dataset), root))
    if orphans:
        print("[stage0] WARNING: %d objects have %d directories stage1 does not expect "
              "-- this stage2 run may come from a different stage1 version; mapping "
              "verdicts for them are meaningless (e.g. %s)"
              % (len(orphans), sum(len(v) for v in orphans.values()),
                 list(orphans.items())[0]))
    if orphan_objs:
        print("[stage0] WARNING: %d object directories on disk are absent from stage1 "
              "(e.g. %s) -- leftovers from an earlier stage1 version, not audited"
              % (len(orphan_objs), orphan_objs[:3]))

    jobs = [(s, a.tau_vis) for s in samples]
    t0 = time.time()
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            results = list(ex.map(_worker, jobs, chunksize=64))
    else:
        results = [_worker(j) for j in jobs]
    elapsed = time.time() - t0

    res_p = os.path.join(out_dir, "audit_results.jsonl")
    with open(res_p, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    state = collections.Counter(r["stats"].get("state") for r in results)
    hard = collections.Counter(t for r in results for t in r["hard_failures"])
    corr = collections.Counter(t for r in results for t in r["metadata_correction_flags"])
    warn = collections.Counter(t.split(":")[0] for r in results for t in r["warning_flags"])
    diag = {k: v for k, v in warn.items() if k in C.DIAGNOSTIC_WARNINGS}
    real_warn = {k: v for k, v in warn.items() if k not in C.DIAGNOSTIC_WARNINGS}
    present = [r for r in results if r["stats"].get("state") == "present"]

    # full lists of everything that needs action (README 3.3 / 3.4)
    def _dump(name, key):
        rows = [r for r in results if r[key]]
        with open(os.path.join(out_dir, name), "w") as f:
            for r in rows:
                f.write(json.dumps({"sample_id": r["sample_id"], "flags": r[key],
                                    "state": r["stats"].get("state")}) + "\n")
        return len(rows)

    n_hard_rows = _dump("hard_failures.jsonl", "hard_failures")
    n_corr_rows = _dump("metadata_corrections.jsonl", "metadata_correction_flags")
    with open(os.path.join(out_dir, "orphans.json"), "w") as f:
        json.dump({"orphan_query_dirs": orphans, "orphan_object_dirs": orphan_objs}, f, indent=1)
    ov = [r["raw_soft_overlap"] for r in present if r["raw_soft_overlap"] is not None]
    cd = [r["normalized_center_distance"] for r in present if r["normalized_center_distance"] is not None]

    n_pass = sum(r["schema_pass"] for r in results)
    summary = {
        "schema_version": C.SCHEMA_VERSION,
        "dataset": a.dataset,
        "stage1_json": C.STAGE1_JSON.format(ds=a.dataset), "stage2_root": root,
        "tau_vis": a.tau_vis,
        "run": {"workers": a.workers, "seconds": round(elapsed, 1),
                "samples_per_second": round(len(results) / elapsed, 1)},
        "n_objects": len(recs), "n_expected_samples": len(results),
        "state_counts": dict(state),
        "stage1_drift": {"n_objects_with_orphan_query_dirs": len(orphans),
                         "n_orphan_query_dirs": sum(len(v) for v in orphans.values()),
                         "n_orphan_object_dirs": len(orphan_objs)},
        "n_schema_pass": n_pass,
        "schema_pass_rate": round(n_pass / max(len(results), 1), 6),
        # unique affected samples
        "n_hard_failed": n_hard_rows,
        "n_metadata_correction": n_corr_rows,
        "n_warned": sum(bool(r["warning_flags"]) for r in results),
        "n_warned_excluding_diagnostic": sum(
            any(t.split(":")[0] not in C.DIAGNOSTIC_WARNINGS for t in r["warning_flags"])
            for r in results),
        # flag occurrences (a sample can carry several)
        "hard_failures": dict(hard.most_common()),
        "metadata_correction_flags": dict(corr.most_common()),
        "warning_flags": dict(sorted(real_warn.items(), key=lambda kv: -kv[1])),
        "diagnostic_flags": dict(sorted(diag.items(), key=lambda kv: -kv[1])),
        "n_flag_occurrences": {"hard": sum(hard.values()), "correction": sum(corr.values()),
                               "warning": sum(real_warn.values()), "diagnostic": sum(diag.values())},
        "raw_soft_overlap": _dist(ov),
        "normalized_center_distance": _dist(cd),
    }
    with open(os.path.join(out_dir, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print(json.dumps(summary, indent=1))
    print("[stage0] wrote %s" % res_p)


def _dist(v):
    if not v:
        return None
    v = np.asarray(v)
    return {"n": int(v.size), "mean": float(v.mean()),
            "p05": float(np.percentile(v, 5)), "p25": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
            "p95": float(np.percentile(v, 95))}


if __name__ == "__main__":
    main()
