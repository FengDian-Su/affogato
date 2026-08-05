"""Shared constants, loaders and metrics for the verification pipeline.

Spec: data_verification/pipeline/README.md (v2.0, 2026-07-27).
Everything here is the *contract* between verification stages; per-stage logic
lives in audit.py / render.py / judge_stage1.py / judge_stage2.py.
"""
import json
import os

import numpy as np

SCHEMA_VERSION = "verification-stage0-v1.0"
EPS = 1e-8

# ---------------------------------------------------------------- paths

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = ("daily_used", "electronics")

# read the live generation output directly -- no snapshot, so an audit can never
# be run against a stale copy (the data_verification/data/stage1/ snapshot was
# md5-identical on 2026-08-03, but nothing keeps it that way)
STAGE1_JSON = os.path.join(REPO, "data_generation/outputs/stage1/{ds}/stage1_all.json")
STAGE2_ROOT = os.path.join(REPO, "data_generation/outputs/stage2/{ds}")
# plumbing-only fixture (OLD batch: different npz keys AND different score scale;
# never freeze tau_vis on it -- README section 1.2)
EXAMPLE_ROOT = os.path.join(REPO, "data_verification/data/stage2_example_data")

# ---------------------------------------------------------------- frozen enums

# core-8, from data_generation/pipeline/stage1_v2.py:60
ROLE_VERBS = ("hold", "lift", "push", "pull", "press", "slide", "rotate", "squeeze")
CATEGORY_ENUM = ("inter", "intra", "pose")
COORDINATION_ENUM = ("whole-body", "stabilize+actuate", "co-actuate")
ROLE_FIELDS = ("role", "target", "contact_region", "function")

# ---------------------------------------------------------------- npz/meta contract

NPZ_REQUIRED = ("xyz", "scoreA", "scoreB")
# needed by section 3.6 raw soft overlap; absent -> warning, not a hard failure
NPZ_EXPECTED = ("scoreA_raw", "scoreB_raw")
NPZ_OPTIONAL_KNOWN = (
    "ptsA", "ptsB", "ptsA_multi", "ptsB_multi", "pexistA", "pexistB",
    "molmo_queries", "task",
    "gt", "counts", "sel_version", "partition_axis",  # old-batch only (example data)
)
META_REQUIRED = ("object_id", "object_name", "task", "query", "category", "coordination", "roles")

# README 3.5: known data property, diagnostic only -- never a hard failure and
# never counted in the formal error rate
DIAGNOSTIC_WARNINGS = ("active_region_large_A", "active_region_large_B")

TAU_VIS = 0.20      # FROZEN 2026-07-28, shared by A and B (README 4.1.2)
# warning thresholds below are still PROVISIONAL (README section 16)
WARN = {
    "raw_overlap_high": 0.60,
    "center_dist_low": 0.05,
    "center_dist_high": 0.80,
    "hit_low": 4,
    "coverage_low": 0.50,
    "active_frac_low": 0.002,
    "active_frac_high": 0.50,
}


def slugify(text, n=40):
    """EXACT copy of data_generation/pipeline/stage2_v2.py:95.

    The stage2 output directory is f"q{qi}_{slugify(task)}"; any divergence here
    turns every sample into a false mapping-mismatch. Do not "improve" it.
    """
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:n].strip("_")


# ---------------------------------------------------------------- loading

def load_stage1(dataset):
    with open(STAGE1_JSON.format(ds=dataset)) as f:
        return json.load(f)


def expected_samples(records, stage2_root):
    """Every sample stage1 *says* should exist -> (samples, orphans).

    stage1 is authoritative on purpose: stage2_v2.py:264 skips queries whose
    roles or molmo_queries are not exactly 2, so those never produce a
    directory. Walking stage2/ instead would silently drop them and report a
    falsely clean audit.

    qi is the index into the object's FULL query list (stage2 enumerates before
    filtering), so qN maps straight back into stage1_all.json.

    `orphans` = directories on disk that stage1 does not expect. Non-empty means
    the stage2 run came from a DIFFERENT stage1 version, and every mapping-based
    verdict for that object is meaningless -- 2026-07-27: the example fixture
    scores ~49% "missing" purely from this drift (it predates the 07-24 stage1
    rerun and its task lists differ), so this must be surfaced, never silently
    charged to the generator.

    Only the fields the audit needs are carried, to keep the ProcessPool pickle
    small at 280k samples.
    """
    samples, orphans = [], {}
    for rec in records:
        oid = rec["object_id"]
        obj_dir = os.path.join(stage2_root, oid)
        try:
            on_disk = set(os.listdir(obj_dir))
        except OSError:
            on_disk = None                       # generation has not reached this object
        queries = rec.get("queries", [])
        names = ["q%d_%s" % (i, slugify(q.get("task", ""))) for i, q in enumerate(queries)]
        if on_disk is not None:
            extra = on_disk - set(names)
            if extra:
                orphans[oid] = sorted(extra)
        for qi, (q, name) in enumerate(zip(queries, names)):
            samples.append({
                "sample_id": "%s/%s" % (oid, name),
                "object_id": oid,
                "qi": qi,
                "state": ("not_generated" if on_disk is None
                          else "present" if name in on_disk else "missing_qdir"),
                "qdir": os.path.join(obj_dir, name),
                "task": q.get("task"),
                "roles": q.get("roles"),
                "category": q.get("category"),
                "coordination": q.get("coordination"),
                "n_molmo": len(q.get("molmo_queries") or []),
            })
    return samples, orphans


def orphan_objects(records, stage2_root):
    """Object directories on disk that stage1 does not list at all.

    expected_samples() only sees orphan *query* dirs inside objects stage1 knows
    about; a whole object directory left over from an earlier stage1 version is
    invisible to it. Same drift signal, one level up.
    """
    try:
        on_disk = {d for d in os.listdir(stage2_root)
                   if os.path.isdir(os.path.join(stage2_root, d))}
    except OSError:
        return []
    return sorted(on_disk - {r["object_id"] for r in records})


def load_npz(path):
    # allow_pickle=False is safe: the only object array (molmo_queries) is never read here
    return np.load(path, allow_pickle=False)


def load_meta(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- metrics

def raw_soft_overlap(a, b):
    """O_AB = sum min / sum max, on the PRE-partition raw votes (README 3.6)."""
    return float(np.minimum(a, b).sum() / (np.maximum(a, b).sum() + EPS))


def soft_centroid(xyz, s):
    w = float(s.sum())
    return (s[:, None] * xyz).sum(0) / (w + EPS)


def bbox_diag(xyz):
    return float(np.linalg.norm(xyz.max(0) - xyz.min(0)))


def normalized_center_distance(xyz, sa, sb):
    """D_AB on the FINAL scores, normalized by the object bbox diagonal."""
    d = np.linalg.norm(soft_centroid(xyz, sa) - soft_centroid(xyz, sb))
    return float(d / (bbox_diag(xyz) + EPS))


def active_frac(s, tau=TAU_VIS):
    return float((s >= tau).mean())
