"""Stage 2 synthetic spatial corruption test (README section 9.2 C).

Four gate types. Unlike Stage 1 these edit the HEATMAP ARRAYS, not the metadata, so
each corrupted sample is materialised as its own scores.npz and rendered through the
identical frozen protocol.

A programmatic edit is NOT automatically a valid corruption, so every type carries an
eligibility filter and every built entry records the geometry that justifies it:

  wrong_region      one hand's region is transplanted to the far side of the object.
                    Rejected unless SoftIoU(original, moved) is near zero, the moved
                    region is non-empty, its centroid is far from the original, AND the
                    sample is not symmetric / hand_agnostic -- on a symmetric object the
                    "wrong" side is often equally valid.
  over_expanded     one hand's region is inflated to cover most of the object. Only used
                    where the metadata names a LOCAL contact region (a specific side,
                    height, end or feature) and the region is not already broad, so a
                    legitimate whole-body contact is never mislabelled as expansion.
  ab_collapse       hand B's map is replaced by hand A's. Excluded when the sample is
                    symmetric or hand_agnostic, when both hands share role or target, or
                    when A and B already overlap heavily -- in those cases a collapsed
                    map is close to the truth.
  only_one_correct  exactly one hand is moved to a wrong region, the other left untouched.
                    Restricted at scoring time to samples whose CLEAN verdict accepted
                    both hands, so "only one is correct" is actually true of the pair.
                    Construction overlaps with wrong_region by design; what differs is the
                    eligibility and the metric it targets (dual_coordination, not per-hand).

    python corrupt_stage2.py build --per_type 20     # -> manifest + review.html (renders)
    # human spot-checks review.html, bad ids -> rejected.txt
    python judge_stage2.py judge --manifest <manifest.jsonl>
    python corrupt_stage2.py score
"""
import argparse
import collections
import json
import os
import random
import re
import shutil

import numpy as np
from scipy.spatial import cKDTree

import common as C
import render as R

OUT_DIR = os.path.join(C.REPO, "data_verification/quality_evaluation/corruption_stage2")
SAMPLES = os.path.join(OUT_DIR, "samples")
MANIFEST = "manifest.jsonl"
TAU = R.TAU_VIS

TYPES = ["wrong_region", "over_expanded", "ab_collapse", "only_one_correct"]

# diagnostic only
EXPECTED_TAGS = {
    "wrong_region": {"wrong_region"},
    "over_expanded": {"over_expanded"},
    "ab_collapse": {"ab_collapse"},
    "only_one_correct": {"only_one_correct", "only_A_valid", "only_B_valid"},
}

# a contact_region that pins down a specific place on a part
LOCAL_RE = re.compile(
    r"mid-height|midheight|top|bottom|upper|lower|edge|rim|side|end\b|tip|corner|opposite|"
    r"outer|inner|base|neck|mouth|left|right|front|back|underside|centre|center|near the|"
    r"just above|just below|along the", re.I)
GENERIC_RE = re.compile(r"^\s*(the\s+)?(whole|entire|outer\s+)?(body|surface|object|exterior)"
                        r"\s*$", re.I)

MAX_IOU_MOVED = 0.05        # moved region must barely touch the original
MIN_SHIFT = 0.30            # centroid shift, as a fraction of the bbox diagonal
# The point farthest from a hand's region is usually where the PARTNER hand sits, so a naive
# "move it far away" lands on the partner's own contact area -- which is a legitimate place to
# grip and turns the edit into a partial ab_collapse. Measured before this guard: A/B SoftIoU
# rose 0.006 -> 0.341. The moved region must avoid the partner too.
MAX_PARTNER_IOU = 0.10
MAX_AB_IOU = 0.30           # ab_collapse: original hands must already be well separated
MAX_ACT_FOR_EXPAND = 0.50   # over_expanded: region must not already be broad
EXPAND_TO = 0.85            # target coverage after inflation


def soft_iou(a, b):
    return float(np.minimum(a, b).sum() / (np.maximum(a, b).sum() + C.EPS))


def centroid(xyz, s):
    w = s.sum()
    return (s[:, None] * xyz).sum(0) / (w + C.EPS)


def stats(xyz, a, b):
    return {"active_A": float((a >= TAU).mean()), "active_B": float((b >= TAU).mean()),
            "ab_soft_iou": soft_iou(a, b),
            "mass_A": float(a.sum()), "mass_B": float(b.sum())}


# ---------------------------------------------------------------- constructions

def move_region(xyz, s, partner, kdt, diag, n_try=24):
    """Transplant the region elsewhere, preserving its size and score profile.

    Tries the farthest candidate seeds in turn and keeps the first placement that clears
    the original AND the partner hand -- see MAX_PARTNER_IOU.
    """
    m = s >= TAU
    n_act = int(m.sum())
    if n_act < 20:
        return None
    c = centroid(xyz, s)
    prof = np.sort(s[m])[::-1]
    far_order = np.argsort(-((xyz - c) ** 2).sum(1))[:n_try]
    for seed in far_order:
        idx = np.atleast_1d(kdt.query(xyz[int(seed)], k=min(n_act, len(xyz)))[1])
        new = np.zeros_like(s)
        new[idx] = prof[:len(idx)]           # nearest-to-seed gets the highest score
        if not (new >= TAU).any():
            continue
        if soft_iou(s, new) > MAX_IOU_MOVED:
            continue
        if soft_iou(new, partner) > MAX_PARTNER_IOU:
            continue                          # landed on the partner's own contact area
        if np.linalg.norm(centroid(xyz, new) - c) / (diag + C.EPS) < MIN_SHIFT:
            continue
        return new
    return None


def inflate_region(xyz, s):
    """Inflate to EXPAND_TO coverage with a falloff, keeping the original peak."""
    c = centroid(xyz, s)
    d = np.linalg.norm(xyz - c, axis=1)
    k = int(EXPAND_TO * len(xyz))
    cut = np.partition(d, k - 1)[k - 1]
    ramp = np.clip(0.90 - 0.62 * (d / (cut + C.EPS)), 0.0, 0.90)   # 0.90 at centre -> 0.28
    new = np.maximum(s, np.where(d <= cut, ramp, 0.0))
    if (new >= TAU).mean() < 0.60:
        return None
    return new


# ---------------------------------------------------------------- eligibility

def _local_contact(role):
    cr = str(role.get("contact_region") or "")
    if GENERIC_RE.match(cr.strip()):
        return False
    return bool(LOCAL_RE.search(cr))


def eligible(ctype, meta, xyz, a, b, hand):
    """Returns (ok, reason_if_not)."""
    ra, rb = meta["roles"][0], meta["roles"][1]
    sym = bool(meta.get("symmetric")) or bool(meta.get("hand_agnostic"))
    if ctype in ("wrong_region", "only_one_correct"):
        if sym:
            return False, "symmetric/hand_agnostic: the far side may be equally valid"
    if ctype == "over_expanded":
        h = ra if hand == "A" else rb
        if not _local_contact(h):
            return False, "contact_region is not local; a broad contact may be legitimate"
        act = float(((a if hand == "A" else b) >= TAU).mean())
        if act >= MAX_ACT_FOR_EXPAND:
            return False, "region already broad (active %.2f)" % act
    if ctype == "ab_collapse":
        if sym:
            return False, "symmetric/hand_agnostic: a shared region may be near-correct"
        if str(ra.get("role")) == str(rb.get("role")):
            return False, "same role on both hands"
        if str(ra.get("target")).strip().lower() == str(rb.get("target")).strip().lower():
            return False, "same target on both hands"
        iou = soft_iou(a, b)
        if iou > MAX_AB_IOU:
            return False, "A/B already overlap (SoftIoU %.2f)" % iou
    return True, None


# ---------------------------------------------------------------- build

def build(a):
    os.makedirs(SAMPLES, exist_ok=True)
    samples, _ = C.expected_samples(C.load_stage1(a.dataset), C.STAGE2_ROOT.format(ds=a.dataset))
    present = [s for s in samples if s["state"] == "present"]
    print("[s2corrupt] present pool: %d" % len(present))

    made = collections.defaultdict(list)
    skipped = collections.Counter()
    for ctype in TYPES:
        rng = random.Random("%s|%d" % (ctype, a.seed))   # per-type RNG, see corrupt.py
        order = list(present)
        rng.shuffle(order)
        for sm in order:
            if len(made[ctype]) >= a.per_type:
                break
            try:
                with np.load(os.path.join(sm["qdir"], "scores.npz"), allow_pickle=False) as z:
                    xyz = z["xyz"].astype(np.float64)
                    A = z["scoreA"].astype(np.float64)
                    B = z["scoreB"].astype(np.float64)
                meta = C.load_meta(os.path.join(sm["qdir"], "meta.json"))
            except Exception:
                continue
            if not (isinstance(meta.get("roles"), list) and len(meta["roles"]) == 2):
                continue
            if not (A >= TAU).any() or not (B >= TAU).any():
                skipped["empty_hand"] += 1
                continue

            hand = rng.choice(["A", "B"]) if ctype != "ab_collapse" else "B"
            ok, why = eligible(ctype, meta, xyz, A, B, hand)
            if not ok:
                skipped["%s:%s" % (ctype, why.split(":")[0].split("(")[0].strip())] += 1
                continue

            diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
            kdt = cKDTree(xyz)
            nA, nB = A.copy(), B.copy()
            if ctype in ("wrong_region", "only_one_correct"):
                mv = move_region(xyz, A if hand == "A" else B,
                                 B if hand == "A" else A, kdt, diag)
                if mv is None:
                    skipped["%s:move_failed" % ctype] += 1
                    continue
                if hand == "A":
                    nA = mv
                else:
                    nB = mv
                how = "hand %s region transplanted to the far side" % hand
            elif ctype == "over_expanded":
                ex = inflate_region(xyz, A if hand == "A" else B)
                if ex is None:
                    skipped["over_expanded:inflate_failed"] += 1
                    continue
                if hand == "A":
                    nA = ex
                else:
                    nB = ex
                how = "hand %s region inflated to cover most of the object" % hand
            else:
                nB = A.copy()
                how = "hand B map replaced by hand A's (collapse)"

            cid = "%s__%s__%s" % (ctype, sm["object_id"][:12], sm["sample_id"].split("/")[1][:24])
            d = os.path.join(SAMPLES, cid)
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, "scores.npz"), xyz=xyz.astype(np.float32),
                                scoreA=nA.astype(np.float32), scoreB=nB.astype(np.float32))
            shutil.copy(os.path.join(sm["qdir"], "meta.json"), os.path.join(d, "meta.json"))

            before, after = stats(xyz, A, B), stats(xyz, nA, nB)
            made[ctype].append({
                "corruption_id": cid, "corruption_type": ctype, "hand": hand,
                "base_sample_id": sm["sample_id"], "object_id": sm["object_id"],
                "object_name": meta.get("object_name"), "task": meta.get("task"),
                "clean_qdir": sm["qdir"], "corrupt_qdir": d, "how": how,
                "expected_tags": sorted(EXPECTED_TAGS[ctype]),
                "contact_A": meta["roles"][0].get("contact_region"),
                "contact_B": meta["roles"][1].get("contact_region"),
                "role_A": meta["roles"][0].get("role"), "role_B": meta["roles"][1].get("role"),
                "symmetric": meta.get("symmetric"), "hand_agnostic": meta.get("hand_agnostic"),
                "before": before, "after": after,
                "soft_iou_hand": soft_iou(A if hand == "A" else B, nA if hand == "A" else nB),
                "human_confirmed": None,
            })
        print("  %-18s %d" % (ctype, len(made[ctype])))

    rows = [r for t in TYPES for r in made[t]]
    with open(os.path.join(OUT_DIR, MANIFEST), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[s2corrupt] top skip reasons: %s" % dict(skipped.most_common(8)))
    review(rows, a.review_per_type)
    print("[s2corrupt] %d corruptions -> %s" % (len(rows), os.path.join(OUT_DIR, MANIFEST)))


def review(rows, per_type):
    """Render clean vs corrupted six-view pairs for a few cases per type."""
    img_dir = os.path.join(OUT_DIR, "img")
    os.makedirs(img_dir, exist_ok=True)
    picked = []
    for t in TYPES:
        picked += [r for r in rows if r["corruption_type"] == t][:per_type]
    h = ["<meta charset='utf-8'><title>stage2 corruption review</title><style>"
         "body{font-family:system-ui;margin:16px}h3{font-size:14px;margin:20px 0 2px}"
         "img{width:150px;border:1px solid #ddd}.r{display:flex;gap:3px;align-items:flex-start}"
         ".lab{width:74px;flex:0 0 74px;font-size:12px;color:#0a58ca;padding-top:56px}"
         ".t{color:#666;font-size:12px}table{border-collapse:collapse;font-size:12px;"
         "margin:4px 0}td,th{border:1px solid #ddd;padding:1px 6px}</style>",
         "<h1 style='font-size:16px'>Stage 2 spatial corruption review</h1>"
         "<div class=t>Top row = clean, bottom row = corrupted. Confirm the corrupted "
         "version is genuinely wrong for the stated contact regions. "
         "Bad ids &rarr; <code>rejected.txt</code>.</div>"]
    for r in picked:
        h.append("<h3>%s <span class=t>[%s, hand %s] %s</span></h3>"
                 % (r["corruption_id"], r["corruption_type"], r["hand"], r["object_name"]))
        h.append("<div class=t>task: %s<br>A (%s): %s<br>B (%s): %s<br>%s</div>"
                 % (r["task"], r["role_A"], r["contact_A"], r["role_B"], r["contact_B"], r["how"]))
        b, af = r["before"], r["after"]
        h.append("<table><tr><th></th><th>active A</th><th>active B</th><th>A/B SoftIoU</th>"
                 "<th>mass A</th><th>mass B</th></tr>"
                 "<tr><td>clean</td><td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%.0f</td>"
                 "<td>%.0f</td></tr>"
                 "<tr><td>corrupt</td><td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%.0f</td>"
                 "<td>%.0f</td></tr></table>"
                 "<div class=t>SoftIoU(clean hand %s, corrupted hand %s) = %.4f</div>"
                 % (b["active_A"], b["active_B"], b["ab_soft_iou"], b["mass_A"], b["mass_B"],
                    af["active_A"], af["active_B"], af["ab_soft_iou"], af["mass_A"], af["mass_B"],
                    r["hand"], r["hand"], r["soft_iou_hand"]))
        for label, qdir in (("clean", r["clean_qdir"]), ("corrupt", r["corrupt_qdir"])):
            d = os.path.join(img_dir, r["corruption_id"], label)
            paths, _, _ = R.render_views(qdir, d, TAU, "v", "linear")
            h.append("<div class=r><div class=lab>%s</div>%s</div>"
                     % (label, "".join("<img src='%s'>"
                                       % os.path.relpath(paths[v], OUT_DIR) for v in R.VIEW_ORDER)))
    with open(os.path.join(OUT_DIR, "review.html"), "w") as f:
        f.write("\n".join(h))
    print("[s2corrupt] review renders: %d cases -> %s"
          % (len(picked), os.path.join(OUT_DIR, "review.html")))


# ---------------------------------------------------------------- scoring

def _core(j):
    return [j["hand_A_grounding"]["score"], j["hand_B_grounding"]["score"],
            j["dual_coordination"]["score"]]


def _flagged(j, lenient):
    if j.get("judge_failure"):
        return False
    sc = _core(j)
    return min(sc) <= 1 if lenient else min(sc) == 0


def _auto_accepts(j):
    return (not j.get("judge_failure")) and min(_core(j)) == 2


def _tags(j):
    return {t for k in ("hand_A_grounding", "hand_B_grounding", "dual_coordination")
            for t in j.get(k, {}).get("error_tags", [])}


def score(a):
    man = {r["corruption_id"]: r for r in
           (json.loads(l) for l in open(os.path.join(OUT_DIR, MANIFEST)))}
    rej_p = os.path.join(OUT_DIR, "rejected.txt")
    rejected = set()
    if os.path.exists(rej_p):
        rejected = {l.strip() for l in open(rej_p) if l.strip() and not l.startswith("#")}
        print("[s2corrupt] %d excluded by human review" % len(rejected))
    else:
        print("[s2corrupt] NOTE: no rejected.txt -- PRE human confirmation")

    V = {}
    for l in open(os.path.join(OUT_DIR, "judged.jsonl")):
        j = json.loads(l)
        V[j["sample_id"]] = j

    # only_one_correct is only meaningful where the clean pair was accepted on BOTH hands
    dropped_ooc = 0
    print("\n%-18s %4s | %6s %6s | %9s | %7s %7s %7s"
          % ("type", "n", "strict", "lenien", "AA escape", "dropA", "dropB", "dropDual"))
    tot = collections.Counter()
    tagrow = {}
    for ctype in TYPES:
        ids = [i for i, r in man.items() if r["corruption_type"] == ctype and i not in rejected]
        n = ds = dl = esc = ct = 0
        dA, dB, dD = [], [], []
        for i in ids:
            jc, jo = V.get("CORRUPT::" + i), V.get("CLEAN::" + i)
            if not jc or jc.get("judge_failure"):
                continue
            if ctype == "only_one_correct":
                if not jo or jo.get("judge_failure"):
                    continue
                if min(jo["hand_A_grounding"]["score"], jo["hand_B_grounding"]["score"]) < 2:
                    dropped_ooc += 1
                    continue
            n += 1
            ds += _flagged(jc, False); dl += _flagged(jc, True); esc += _auto_accepts(jc)
            ct += bool(_tags(jc) & set(man[i]["expected_tags"]))
            if jo and not jo.get("judge_failure"):
                co, cc = _core(jo), _core(jc)
                dA.append(co[0] - cc[0]); dB.append(co[1] - cc[1]); dD.append(co[2] - cc[2])
        if not n:
            continue
        tagrow[ctype] = ct / n
        for k, v in (("n", n), ("ds", ds), ("dl", dl), ("esc", esc)):
            tot[k] += v
        mean = lambda v: (sum(v) / len(v)) if v else float("nan")   # noqa: E731
        print("%-18s %4d | %6.2f %6.2f | %9.2f | %+7.2f %+7.2f %+7.2f"
              % (ctype, n, ds / n, dl / n, esc / n, mean(dA), mean(dB), mean(dD)))
    if tot["n"]:
        print("%-18s %4d | %6.2f %6.2f | %9.2f |"
              % ("OVERALL", tot["n"], tot["ds"] / tot["n"], tot["dl"] / tot["n"],
                 tot["esc"] / tot["n"]))
    if dropped_ooc:
        print("  (only_one_correct: %d dropped because the clean pair was not accepted on "
              "both hands)" % dropped_ooc)

    clean = [v for k, v in V.items()
             if k.startswith("CLEAN::") and k[7:] not in rejected and not v.get("judge_failure")]
    if clean:
        print("\nclean originals (n=%d): false-positive strict %.2f | lenient %.2f | "
              "auto-accepted %.2f"
              % (len(clean), sum(_flagged(v, False) for v in clean) / len(clean),
                 sum(_flagged(v, True) for v in clean) / len(clean),
                 sum(_auto_accepts(v) for v in clean) / len(clean)))
    print("\ndiagnostic only -- error-tag hit rate:")
    for k, v in tagrow.items():
        print("  %-18s %.2f" % (k, v))
    print("judge_failure: %d (never counted as interception)"
          % sum(1 for v in V.values() if v.get("judge_failure")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "score"])
    ap.add_argument("--dataset", default="daily_used")
    ap.add_argument("--per_type", type=int, default=20)
    ap.add_argument("--review_per_type", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    (build if a.mode == "build" else score)(a)


if __name__ == "__main__":
    main()
