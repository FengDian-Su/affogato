"""Stage 1 synthetic visual-semantic corruption test (README section 9.2 B).

Taxonomy v3 (2026-07-31, aligned to judge schema v2.0) -- four categories, each with ONE
structured field it is supposed to move (its TARGET AXIS):

  implausible_task        task rewritten into an operation this object cannot perform.
                          target axis: task_plausibility.score == 0
  role_function_conflict  primitive role edited so it contradicts its own function. Every
                          other field left untouched, so the only thing a judge can react
                          to is the role. Renamed from role_function_task_conflict: the
                          criterion is now role<->function text consistency, not the task.
                          target axis: role_function_consistency == "conflict" (edited hand)
  contact_conflict        contact region moved onto a large part the object lacks.
                          target axis: contact_region_evidence == "absent" (edited hand)
  bimanual_conflict       a two-hand configuration that genuinely cannot support the task
                          -- NEVER a merely redundant one.
                          target axis: bimanual_validity.label == "invalid"

Because each type owns one axis, a corruption that moves the OTHER three axes is measured
as cross-axis spillover: the judge noticed something is wrong but cannot localise it.

Small and high quality on purpose (~25/type). Each corruption is wrong BY CONSTRUCTION
(bases are pre-filtered so the edit cannot land on a still-plausible case), and each is
still put in front of a human in review.html before scoring.

Every corruption is paired with its untouched original and BOTH are judged: a detection
rate measured only on corrupted samples cannot separate a judge that detects errors from
one that just scores everything low.

    python corrupt.py build --per_type 25       # -> manifest.jsonl + review.html
    # human reviews; ids that are still plausible OR carry several error classes at once
    # go in rejected.txt
    python judge_stage1.py --manifest <manifest.jsonl>
    python corrupt.py score
"""
import argparse
import collections
import html
import json
import os
import random
import re

import common as C

MANIFEST = "manifest.jsonl"
# v2 directory: the pre-2026-07-31 run under quality_evaluation/corruption/ used the old
# judge schema (per-item model-emitted error_tags, no role_function_consistency) and must
# never be mixed with these numbers. It is left untouched.
OUT_DIR = os.path.join(C.REPO, "data_verification/quality_evaluation/corruption_v2")

# duplicated from judge_stage1.JUDGE_VERSION on purpose -- importing that module pulls in
# vLLM, which costs ~20 s and a CUDA probe for what is a pure string comparison.
# Override with --judge_version to score an earlier prompt version's file for comparison;
# v2.0 and v2.1 share the schema, so only the SYSTEM prompt differs between them.
REQUIRED_JUDGE_VERSION = "qwen3vl-visual-semantic-v2.0"

TYPES = ["implausible_task", "role_function_conflict",
         "contact_conflict", "bimanual_conflict"]

# The one structured field each corruption is built to move. Everything else is off-target.
TARGET_AXIS = {"implausible_task": "task", "role_function_conflict": "role_function",
               "contact_conflict": "contact", "bimanual_conflict": "bimanual"}
AXES = ("task", "role_function", "contact", "bimanual")

# Diagnostic only. Since 2026-07-31 tags are DERIVED from the structured verdict rather than
# emitted by the model, so this row no longer measures the judge's tagging ability -- it is
# now a near-restatement of target-axis detection. Kept only to prove the derivation wiring
# fires; it is NOT one of the primary metrics.
EXPECTED_TAGS = {
    "implausible_task": {"implausible_task"},
    "role_function_conflict": {"role_function_conflict"},
    "contact_conflict": {"contact_part_absent"},
    "bimanual_conflict": {"bimanual_conflict"},
}

# Soft / deformable goods fold and often zip even when nothing in their text says so
# ("fold the Gym Bag flat" is perfectly possible), so they are excluded by name from both
# the fold and unzip templates.
SOFT = (r"fabric|cloth|textile|leather|canvas|bag|sack|pouch|towel|blanket|cushion|pillow|"
        r"curtain|mat|rug|clothing|garment|shirt|jacket|glove|hat|sock|paper|cardboard|"
        r"tarp|flexib|soft|shoe|sneaker|boot|sandal|slipper|dress|skirt|scarf|belt|strap")

# Each template names ONE specific mechanism. It is only used on an object whose text
# mentions none of that mechanism's words, so the named mechanism is provably absent.
# Exclusions also cover object CLASSES that plausibly carry the mechanism without naming
# it (a barrel has a threaded bung; a chest may be a chest of drawers; a break-action gun
# hinges open) -- those are exactly the cases that would survive as "still reasonable".
IMPLAUSIBLE_TEMPLATES = [
    ("swing the hinged door of the {name} open",
     r"hinge|door|lid|flap|cover|panel|cabinet|fridge|freezer|oven|microwave|locker|"
     r"wardrobe|safe|shotgun|rifle|gun|clamshell|case"),
    ("unscrew the threaded cap of the {name}",
     r"cap|screw|thread|lid|cork|neck|spout|barrel|keg|drum|canister|tank|jar|bottle|"
     r"flask|tube|jug|flagon|cylinder|valve|nozzle"),
    ("pull the sliding drawer of the {name} out",
     r"drawer|slide|track|rail|shelf|compartment|chest|dresser|cabinet|desk|nightstand|"
     r"storage|box|unit|console|sideboard|cart|tray"),
    ("unzip the {name} along its zipper", r"zip|pocket|" + SOFT),
    ("fold the {name} flat along its hinge", r"fold|hinge|joint|collaps|chair|table|stand|"
     + SOFT),
]
FOREIGN_PARTS = ["the hinged lid", "the screw cap", "the pull-out drawer", "the zipper pull",
                 "the folding handle", "the spout", "the wheel", "the shoulder strap"]

# ONLY genuinely force-opposing pairs. `hold` is deliberately absent: "one hand rotates or
# slides while the other holds" is the CANONICAL CORRECT bimanual pattern, so pairing
# anything with `hold` produces a valid sample, not a corruption.
OPPOSING = {"push": "pull", "pull": "push", "press": "lift", "lift": "press"}

# role -> the direction of force that primitive actually produces
UP = re.compile(r"lift|raise|upward|hoist|elevate|pick up", re.I)
DOWN = re.compile(r"press|push down|depress|downward|compress|close|seat", re.I)
STEADY = re.compile(r"stabil|steady|hold|counter|brace|support|anchor|prevent", re.I)


def _wellformed(q):
    r = q.get("roles")
    return (isinstance(r, list) and len(r) == 2
            and all(str(h.get(f) or "").strip() for h in r for f in C.ROLE_FIELDS)
            and all(h.get("role") in C.ROLE_VERBS for h in r)
            and q.get("category") in C.CATEGORY_ENUM
            and q.get("coordination") in C.COORDINATION_ENUM)


def _obj_text(rec):
    return " ".join([str(rec.get("object_name") or "")]
                    + [str(c.get("name", "")) + " " + str(c.get("interaction", ""))
                       for c in (rec.get("components") or [])])


def _clone(q):
    return json.loads(json.dumps(q))


# ---------------------------------------------------------------- builders

"""Builders return (corrupted_query, variant, how, hand) where `hand` is the index of the
hand that was edited, or None when the edit is not attributable to a single hand (the task
text, or both hands at once). Scoring needs it to read the target axis on the RIGHT hand --
checking "any hand" would let a spillover onto the untouched hand count as a detection."""


def c_implausible_task(rec, q, rng, _pool):
    """Original definition (restored 2026-07-28): the task is rewritten into an operation
    THIS object cannot perform, keeping the object itself. Transplanting another object's
    task is a different corruption (wrong_object_task) and is not tested here."""
    return _v_articulation(rec, q, rng)


def _v_articulation(rec, q, rng):
    txt = _obj_text(rec)
    usable = [t for t, kw in IMPLAUSIBLE_TEMPLATES if not re.search(kw, txt, re.I)]
    if not usable:
        return None                       # not provably impossible; refuse to fabricate
    n = rec.get("object_name")
    q2 = _clone(q)
    t = rng.choice(usable).format(name=n)
    q2["task"] = t
    q2["query"] = "Point to the regions of the object you would use to %s." % t
    q2["goal"] = "%s closed -> %s opened via the mechanism named in the task" % (n, n)
    return q2, "articulation", "task requires a mechanism this object provably lacks", None


def c_role_function_conflict(rec, q, rng, _pool):
    """Edit ONLY the role, so the role contradicts its own function text.

    Nothing else is touched -- no target, contact_region, function or task text changes --
    so a judge that fails here is demonstrably not reading the role field. This is exactly
    what role_function_consistency was added to the schema to measure.
    """
    goal = "%s %s" % (q.get("task", ""), q.get("goal", ""))
    i = rng.choice([0, 1])
    h = q["roles"][i]
    fn = str(h.get("function", ""))
    if UP.search(fn) and not DOWN.search(fn):
        new, why = "press", "function/goal describe an upward force; press produces the opposite"
    elif DOWN.search(fn) and not UP.search(fn):
        new, why = "lift", "function/goal describe a downward force; lift produces the opposite"
    elif STEADY.search(fn):
        new, why = "slide", "function is to hold the object still; slide is active displacement"
    else:
        return None                        # cannot guarantee a contradiction; skip
    if new == h.get("role"):
        return None
    q2 = _clone(q)
    old = q2["roles"][i]["role"]
    q2["roles"][i]["role"] = new           # every other field intact
    q2["pattern"] = "%s + %s" % (q2["roles"][0]["role"], q2["roles"][1]["role"])
    return (q2, "%s->%s" % (old, new),
            "hand %s role %s -> %s; %s (task goal: %s)"
            % ("AB"[i], old, new, why, goal.strip()[:70]), i)


def c_contact_conflict(rec, q, rng, _pool):
    mine = _obj_text(rec).lower()
    cands = [p for p in FOREIGN_PARTS if p.split()[-1] not in mine]
    if not cands:
        return None
    i = rng.choice([0, 1])
    part = rng.choice(cands)
    q2 = _clone(q)
    q2["roles"][i]["target"] = part.replace("the ", "")
    q2["roles"][i]["contact_region"] = part
    return (q2, "foreign_part",
            "hand %s contact moved to '%s', a large part that is absent from this object"
            % ("AB"[i], part), i)


def c_bimanual_conflict(rec, q, rng, _pool):
    """Both hands on the SAME region with directly opposing primitives, so their
    contributions cancel. Deliberately NOT 'one hand would suffice'."""
    q2 = _clone(q)
    a, b = q2["roles"][0], q2["roles"][1]
    region, target = a.get("contact_region"), a.get("target")
    ra = a.get("role")
    if ra in OPPOSING:
        rb = OPPOSING[ra]
    else:
        ra, rb = "push", "pull"            # force a genuinely opposing pair rather than skip
        a["role"] = ra
    b["role"], b["target"], b["contact_region"] = rb, target, region
    a["function"] = "apply force to move the object in one direction"
    b["function"] = "apply an equal and opposite force at the very same place, cancelling hand A"
    q2["relation"] = ("A and B act on the identical region in directly opposing directions, "
                      "so their forces cancel")
    q2["pattern"] = "%s + %s" % (ra, rb)
    return (q2, "%s_vs_%s" % (ra, rb),
            "both hands on the same region '%s' with opposing primitives %s vs %s"
            % (region, ra, rb), None)          # both hands edited -> not one hand's fault


BUILDERS = {"implausible_task": c_implausible_task,
            "role_function_conflict": c_role_function_conflict,
            "contact_conflict": c_contact_conflict,
            "bimanual_conflict": c_bimanual_conflict}


def _multi_error_risk(ctype, variant):
    """Flag edits that plausibly introduce more than one error class, so the human can
    exclude them before scoring (they make per-type attribution ambiguous)."""
    return None


def build(a):
    recs = C.load_stage1(a.dataset)
    random.Random(a.seed).shuffle(recs)
    recs = recs[:a.pool_objects]
    pool = [(r, q) for r in recs for q in r.get("queries", []) if _wellformed(q)]
    print("[corrupt] candidate pool: %d objects / %d queries" % (len(recs), len(pool)))

    made = collections.defaultdict(list)
    used = set()
    for ctype in TYPES:
        # per-type RNG: editing one category's builder must not perturb which samples the
        # other categories draw, so already-judged entries stay reusable across iterations
        rng = random.Random("%s|%d" % (ctype, a.seed))
        order = list(pool)
        rng.shuffle(order)
        for rec, q in order:
            if len(made[ctype]) >= a.per_type:
                break
            qi = rec["queries"].index(q)
            key = (ctype, rec["object_id"], qi)
            if key in used:
                continue
            if not (rec.get("views_used") and all(os.path.exists(p) for p in rec["views_used"])):
                continue
            res = BUILDERS[ctype](rec, q, rng, pool)
            if not res:
                continue
            q2, variant, how, hand = res
            used.add(key)
            made[ctype].append({
                "corruption_id": "%s__%s__%d" % (ctype, rec["object_id"][:12], qi),
                "corruption_type": ctype, "variant": variant,
                "target_axis": TARGET_AXIS[ctype],
                "hand": None if hand is None else "AB"[hand],
                "base_sample_id": "%s/q%d_%s" % (rec["object_id"], qi, C.slugify(q.get("task", ""))),
                "object_id": rec["object_id"], "object_name": rec.get("object_name"),
                "qi": qi, "how": how, "expected_tags": sorted(EXPECTED_TAGS[ctype]),
                "multi_error_risk": _multi_error_risk(ctype, variant),
                "original": q, "corrupted": q2, "human_confirmed": None,
            })
        print("  %-28s %d  %s" % (ctype, len(made[ctype]),
                                  dict(collections.Counter(r["variant"] for r in made[ctype]))))

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = [r for t in TYPES for r in made[t]]
    with open(os.path.join(OUT_DIR, MANIFEST), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    review(rows, recs)
    print("[corrupt] %d corruptions over %d objects -> %s"
          % (len(rows), len({r["object_id"] for r in rows}), os.path.join(OUT_DIR, MANIFEST)))
    print("[corrupt] REVIEW %s ; list ids that are still plausible OR mix several error "
          "classes in rejected.txt" % os.path.join(OUT_DIR, "review.html"))


def review(rows, recs):
    by_id = {r["object_id"]: r for r in recs}
    h = ["<meta charset='utf-8'><title>corruption review</title><style>"
         "body{font-family:system-ui;margin:16px}h3{font-size:14px;margin:18px 0 3px}"
         "img{width:120px;border:1px solid #ddd;margin-right:2px}"
         "table{border-collapse:collapse;font-size:12px;margin-top:4px}"
         "td,th{border:1px solid #ddd;padding:2px 6px;vertical-align:top;max-width:520px}"
         "ins{background:#e0ffe0;text-decoration:none}.t{color:#888;font-size:12px}"
         ".risk{background:#fff3cd;padding:2px 6px;font-size:12px}"
         "code{background:#f4f4f4;padding:0 3px}</style>",
         "<h1 style='font-size:16px'>Corruption review</h1>"
         "<div class=t>Confirm each edit is genuinely impossible for THIS object. "
         "Reject an id if it could still be reasonable, or if it carries more than one "
         "error class at once. Put rejected ids in <code>rejected.txt</code>.</div>"]
    for r in rows:
        rec = by_id.get(r["object_id"], {})
        h.append("<h3>%s <span class=t>[%s / %s] %s</span></h3>"
                 % (r["corruption_id"], r["corruption_type"], r["variant"], r["object_name"]))
        h.append("<div class=t>%s</div>" % r["how"])
        h.append("<div class=t>target axis: <b>%s</b> &nbsp;|&nbsp; edited hand: <b>%s</b> "
                 "&nbsp;|&nbsp; expected tag: <b>%s</b></div>"
                 % (r["target_axis"], r["hand"] or "both/none",
                    ", ".join(r["expected_tags"])))
        if r["multi_error_risk"]:
            h.append("<div class=risk>multi-error risk: %s</div>" % r["multi_error_risk"])
        h.append("".join("<img src='file://%s'>" % p for p in (rec.get("views_used") or [])[:6]))
        o, c = r["original"], r["corrupted"]
        rows_ = [("task", o.get("task"), c.get("task")), ("goal", o.get("goal"), c.get("goal")),
                 ("relation", o.get("relation"), c.get("relation"))]
        for i, tag in enumerate("AB"):
            for fld in C.ROLE_FIELDS:
                rows_.append(("%s.%s" % (tag, fld), o["roles"][i].get(fld), c["roles"][i].get(fld)))
        h.append("<table><tr><th>field</th><th>original</th><th>corrupted</th></tr>")
        for k, x, y in rows_:
            h.append("<tr><td>%s</td><td>%s</td><td>%s</td></tr>"
                     % (k, x, y if str(x) == str(y) else "<ins>%s</ins>" % y))
        h.append("</table>")
    with open(os.path.join(OUT_DIR, "review.html"), "w") as f:
        f.write("\n".join(h))


# ---------------------------------------------------------------- scoring

HANDS = ("hand_A_visual_semantic_consistency", "hand_B_visual_semantic_consistency")


def _core(j):
    return [j["task_plausibility"]["score"]] + [j[h]["score"] for h in HANDS]


def _axis_state(j, axis, hand=None):
    """One axis of one judgement -> "bad" | "uncertain" | "ok".

    hand is the index of the hand to read for the per-hand axes; None means "either hand",
    which is what spillover wants (a corruption that damages the untouched hand still
    spilled) but NOT what target-axis detection wants.
    """
    if axis == "task":
        s = j["task_plausibility"]["score"]
        return "bad" if s == 0 else "uncertain" if s == 1 else "ok"
    if axis == "bimanual":
        lab = j["bimanual_validity"]["label"]
        return "bad" if lab == "invalid" else "uncertain" if lab == "acceptable" else "ok"
    keys = [HANDS[hand]] if hand in (0, 1) else list(HANDS)
    field, bad, unc = (("role_function_consistency", "conflict", "uncertain")
                       if axis == "role_function"
                       else ("contact_region_evidence", "absent", "uncertain"))
    vals = [j[k][field] for k in keys]
    return "bad" if bad in vals else "uncertain" if unc in vals else "ok"


def _flagged(j, lenient):
    if j.get("judge_failure"):
        return False                       # a parse failure is NOT a successful interception
    sc, lab = _core(j), j["bimanual_validity"]["label"]
    return (min(sc) <= 1 or lab != "valid") if lenient else (min(sc) == 0 or lab == "invalid")


def _auto_accepts(j):
    """Would README section 7.1 Auto-Accept this verdict? ('acceptable' alone does not block.)"""
    if j.get("judge_failure"):
        return False
    return min(_core(j)) == 2 and j["bimanual_validity"]["label"] != "invalid"


def _tags(j):
    """Top-level, rule-derived (judge_stage1.derive_error_tags). Records written before
    2026-07-31 carry per-item model-emitted tags instead and will read as empty here --
    that is intended: those tags came from a different (removed) mechanism."""
    return set(j.get("error_tags") or [])


def score(a):
    man = {r["corruption_id"]: r for r in
           (json.loads(l) for l in open(os.path.join(OUT_DIR, MANIFEST)))}
    rej_p = os.path.join(OUT_DIR, "rejected.txt")
    rejected = set()
    if os.path.exists(rej_p):
        rejected = {l.strip() for l in open(rej_p) if l.strip() and not l.startswith("#")}
        print("[corrupt] %d corruptions excluded by human review" % len(rejected))
    else:
        print("[corrupt] NOTE: no rejected.txt -- these numbers are PRE human confirmation")

    want = a.judge_version or REQUIRED_JUDGE_VERSION
    judged_p = a.judged if os.path.isabs(a.judged) else os.path.join(OUT_DIR, a.judged)
    V, wrong_ver = {}, collections.Counter()
    for l in open(judged_p):
        j = json.loads(l)
        # v1 records carry per-item model-emitted tags and no role_function_consistency;
        # scoring them here would silently read every target axis as missing
        if j.get("judge_version") != want:
            wrong_ver[j.get("judge_version")] += 1
            continue
        V[j["sample_id"]] = j
    print("[corrupt] %s | judge_version %s | %d judgements" % (judged_p, want, len(V)))
    if wrong_ver:
        print("[corrupt] DROPPED %d judgements with another judge_version: %s"
              % (sum(wrong_ver.values()), dict(wrong_ver)))
    if not V:
        raise SystemExit("[corrupt] no judgements at %s -- rerun judge_stage1.py --manifest"
                         % want)

    print("\n%-24s %4s | %6s %6s | %6s %6s | %8s | %6s | %6s"
          % ("corruption type", "n", "strict", "lenien", "tgt-st", "tgt-le",
             "AA escape", "coreD", "spill"))
    tot = collections.Counter()
    tagrow, spill_axis = {}, {}
    for ctype in TYPES:
        ids = [i for i, r in man.items() if r["corruption_type"] == ctype and i not in rejected]
        axis = TARGET_AXIS[ctype]
        n = ds = dl = ts = tl = esc = ct = sp = 0
        drops, per_axis = [], collections.Counter()
        for i in ids:
            jc, jo = V.get("CORRUPT::" + i), V.get("CLEAN::" + i)
            if not jc or jc.get("judge_failure"):
                continue
            n += 1
            hand = {"A": 0, "B": 1}.get(man[i].get("hand"))
            ds += _flagged(jc, False); dl += _flagged(jc, True)
            # target-axis detection: read the axis this corruption was built to move, on the
            # hand it actually edited. "uncertain" counts lenient only.
            st = _axis_state(jc, axis, hand)
            ts += st == "bad"; tl += st in ("bad", "uncertain")
            # live EXPECTED_TAGS, not man[i]["expected_tags"]: the manifest bakes the tag
            # vocabulary in at build time and goes stale the moment the vocabulary changes
            esc += _auto_accepts(jc); ct += bool(_tags(jc) & EXPECTED_TAGS[ctype])
            if jo and not jo.get("judge_failure"):
                drops.append(sum(_core(jo)) / 3.0 - sum(_core(jc)) / 3.0)
                # spillover is measured PAIRED: an off-target axis counts only if it was ok
                # on the untouched original and went bad on the corrupted variant, otherwise
                # pre-existing weakness in the sample would be charged to the corruption
                hit = [ax for ax in AXES if ax != axis
                       and _axis_state(jo, ax) == "ok" and _axis_state(jc, ax) == "bad"]
                sp += bool(hit)
                per_axis.update(hit)
        if not n:
            continue
        tagrow[ctype] = ct / n
        spill_axis[ctype] = (dict(per_axis), len(drops))
        for k, v in (("n", n), ("ds", ds), ("dl", dl), ("ts", ts), ("tl", tl),
                     ("esc", esc), ("sp", sp), ("npair", len(drops))):
            tot[k] += v
        print("%-24s %4d | %6.2f %6.2f | %6.2f %6.2f | %8.2f | %+5.2f | %6s"
              % (ctype, n, ds / n, dl / n, ts / n, tl / n, esc / n,
                 sum(drops) / len(drops) if drops else float("nan"),
                 "%.2f" % (sp / len(drops)) if drops else "n/a"))
    if tot["n"]:
        print("%-24s %4d | %6.2f %6.2f | %6.2f %6.2f | %8.2f |       | %6s"
              % ("OVERALL", tot["n"], tot["ds"] / tot["n"], tot["dl"] / tot["n"],
                 tot["ts"] / tot["n"], tot["tl"] / tot["n"], tot["esc"] / tot["n"],
                 "%.2f" % (tot["sp"] / tot["npair"]) if tot["npair"] else "n/a"))
    print("  strict/lenient = ANY axis flagged; tgt-st/tgt-le = the axis this corruption "
          "targets, on the edited hand ('uncertain' counts lenient only);")
    print("  spill = fraction of pairs where an OFF-target axis went ok(clean) -> bad(corrupt)")

    print("\ncross-axis spillover, per off-target axis (paired clean -> corrupt):")
    for ctype, (per_axis, npair) in spill_axis.items():
        print("  %-24s target=%-13s %s"
              % (ctype, TARGET_AXIS[ctype],
                 ", ".join("%s %d/%d" % (k, v, npair) for k, v in sorted(per_axis.items()))
                 or "none"))

    clean = [(k, v) for k, v in V.items()
             if k.startswith("CLEAN::") and k[7:] not in rejected and not v.get("judge_failure")]
    if clean:
        print("\nclean originals (n=%d): false-positive strict %.2f | lenient %.2f | "
              "auto-accepted %.2f" % (len(clean),
                                      sum(_flagged(v, False) for _, v in clean) / len(clean),
                                      sum(_flagged(v, True) for _, v in clean) / len(clean),
                                      sum(_auto_accepts(v) for _, v in clean) / len(clean)))
        print("  -> read detection against this; a judge that flags everything scores high "
              "on detection and terribly here.")

    print("\ndiagnostic only -- error-tag hit rate (tags are rule-derived from the axes "
          "above, so this is NOT an independent signal):")
    for k, v in tagrow.items():
        print("  %-24s %.2f" % (k, v))
    print("judge_failure: %d (never counted as interception)"
          % sum(1 for v in V.values() if v.get("judge_failure")))



RCSS = """body{font-family:system-ui,sans-serif;margin:14px;color:#222;font-size:13px}
h3{font-size:13px;margin:20px 0 3px;border-top:1px solid #ddd;padding-top:9px}
table{border-collapse:collapse;font-size:12px;margin:4px 0}
td,th{border:1px solid #ddd;padding:2px 6px;vertical-align:top;max-width:430px;text-align:left}
.t{color:#777;font-size:12px}.k{font-weight:600}
.hit{background:#2e7d32;color:#fff;padding:1px 6px;border-radius:3px}
.miss{background:#c62828;color:#fff;padding:1px 6px;border-radius:3px}
.sp{background:#ef6c00;color:#fff;padding:1px 6px;border-radius:3px}
.bad{background:#ffe0e0}.unc{background:#fff6d8}.tag{background:#c62828;color:#fff;
padding:1px 5px;border-radius:3px;font-size:11px;margin-right:3px}
ins{background:#e0ffe0;text-decoration:none}"""


def review_judged(a):
    """Paired clean-vs-corrupt HTML for one judged file (README section 9.2 B).

    Separate from review.html, which reviews the CORRUPTIONS before any GPU time. This one
    reviews the JUDGEMENTS: it marks whether the target axis was hit and which off-target
    axes spilled, so a disagreement between two models can be read case by case.
    """
    man = {r["corruption_id"]: r for r in
           (json.loads(l) for l in open(os.path.join(OUT_DIR, MANIFEST)))}
    judged_p = a.judged if os.path.isabs(a.judged) else os.path.join(OUT_DIR, a.judged)
    want = a.judge_version or REQUIRED_JUDGE_VERSION
    V = {j["sample_id"]: j for l in open(judged_p)
         for j in [json.loads(l)] if j.get("judge_version") == want}
    out_p = a.review_out or os.path.join(OUT_DIR, "review_judged.html")

    h = ["<meta charset='utf-8'><title>judged corruptions</title><style>%s</style>" % RCSS,
         "<h1 style='font-size:16px'>Corruption judgements &mdash; %s</h1>"
         "<div class=t>%s | %d judgements</div>" % (want, os.path.basename(judged_p), len(V))]
    for ctype in TYPES:
        axis = TARGET_AXIS[ctype]
        for cid in sorted(i for i, r in man.items() if r["corruption_type"] == ctype):
            jc, jo = V.get("CORRUPT::" + cid), V.get("CLEAN::" + cid)
            if not jc:
                continue
            m = man[cid]
            hand = {"A": 0, "B": 1}.get(m.get("hand"))
            h.append("<h3>%s <span class=t>[target: %s | hand %s] %s</span></h3>"
                     % (cid, axis, m.get("hand") or "-", html.escape(str(m["object_name"]))))
            h.append("<div class=t>%s</div>" % html.escape(m["how"]))
            if jc.get("judge_failure"):
                h.append("<div class=miss>judge_failure</div>")
                continue
            st = _axis_state(jc, axis, hand)
            spilled = ([ax for ax in AXES if ax != axis and _axis_state(jo, ax) == "ok"
                        and _axis_state(jc, ax) == "bad"] if jo and not jo.get("judge_failure")
                       else [])
            h.append("<div><span class=%s>target %s: %s</span> %s</div>"
                     % ("hit" if st == "bad" else "miss", axis, st,
                        "".join("<span class=sp>spill: %s</span> " % x for x in spilled)))
            h.append("<table><tr><th>axis</th><th>clean</th><th>corrupt</th></tr>")
            for ax in AXES:
                cv = _axis_state(jo, ax) if jo and not jo.get("judge_failure") else "?"
                vv = _axis_state(jc, ax)
                cls = "bad" if vv == "bad" else "unc" if vv == "uncertain" else ""
                h.append("<tr><td class=k>%s%s</td><td>%s</td><td class=%s>%s</td></tr>"
                         % (ax, " *" if ax == axis else "", cv, cls, vv))
            h.append("</table>")
            h.append("<div>%s</div>"
                     % ("".join("<span class=tag>%s</span>" % t for t in jc.get("error_tags") or [])
                        or "<span class=t>no error tags</span>"))
            for hh, tg in zip(HANDS, "AB"):
                h.append("<div class=t><b>hand %s</b> (%s): %s</div>"
                         % (tg, jc[hh]["role_function_consistency"],
                            html.escape(jc[hh]["reason"])))
    with open(out_p, "w") as f:
        f.write("\n".join(h))
    print("[corrupt] wrote %s" % out_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "score", "review_judged"])
    ap.add_argument("--dataset", default="daily_used")
    ap.add_argument("--per_type", type=int, default=25)
    ap.add_argument("--pool_objects", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judged", default="judged.jsonl",
                    help="score mode: judged file inside OUT_DIR (or an absolute path)")
    ap.add_argument("--judge_version", default=None,
                    help="score mode: only accept records at this judge_version")
    ap.add_argument("--out_dir", default=None, help="override the corruption output dir")
    ap.add_argument("--review_out", default=None, help="review_judged mode: html path")
    a = ap.parse_args()
    if a.out_dir:
        global OUT_DIR
        OUT_DIR = a.out_dir
    {"build": build, "score": score, "review_judged": review_judged}[a.mode](a)


if __name__ == "__main__":
    main()
