"""Summarise + visualise a Verification Stage 1 run (README section 5).

Reads a judged.jsonl produced by judge_stage1.py, joins it back to the stage1 records so
the HTML can show the metadata the judge was actually reacting to, and writes
summary.json + index.html next to it.

Distributions are computed over SUCCESSFUL records only; judge_failure rows carry none of
the verdict fields, so folding them into a denominator would silently deflate every rate.

    python report_stage1.py --judged ../outputs/stage1_test/judged.jsonl --dataset daily_used
"""
import argparse
import collections
import html
import json
import os

import common as C
from judge_stage1 import ERROR_TAGS, HANDS

ITEMS = ("task_plausibility", "bimanual_validity") + HANDS


def _pct(c, n):
    return {k: [v, round(v / n, 3)] for k, v in sorted(c.items(), key=lambda kv: -kv[1])} if n else {}


def summarize(rows, meta, n_obj_scope):
    ok = [r for r in rows if not r.get("judge_failure")]
    fail = [r for r in rows if r.get("judge_failure")]
    n = len(ok)
    dist = {
        "task_plausibility_score": _pct(collections.Counter(r["task_plausibility"]["score"] for r in ok), n),
        "bimanual_validity_label": _pct(collections.Counter(r["bimanual_validity"]["label"] for r in ok), n),
    }
    for h, tag in zip(HANDS, "AB"):
        dist["hand_%s_score" % tag] = _pct(collections.Counter(r[h]["score"] for r in ok), n)
        dist["hand_%s_contact_region_evidence" % tag] = _pct(
            collections.Counter(r[h]["contact_region_evidence"] for r in ok), n)
        dist["hand_%s_role_function_consistency" % tag] = _pct(
            collections.Counter(r[h]["role_function_consistency"] for r in ok), n)

    tags = collections.Counter(t for r in ok for t in r.get("error_tags") or [])
    # Auto-Accept per README section 7.1: all three scores 2 and bimanual not invalid.
    aa = sum(1 for r in ok
             if min(r["task_plausibility"]["score"], *[r[h]["score"] for h in HANDS]) == 2
             and r["bimanual_validity"]["label"] != "invalid")
    # "uncertain" fires no tag but is a Human Review signal (README 5.4) -- counted so the
    # untagged rows are not mistaken for clean ones.
    unc = sum(1 for r in ok if any(r[h]["role_function_consistency"] == "uncertain"
                                   or r[h]["contact_region_evidence"] == "uncertain" for h in HANDS))
    return {
        "judge_version": meta.get("judge_version"), "model": meta.get("model"),
        "dataset": meta.get("dataset"),
        "n_objects_requested": meta.get("limit_objects"),
        "n_objects_in_scope": n_obj_scope,
        "n_objects_judged": len({r["object_id"] for r in rows}),
        "n_queries_judged": len(rows), "n_ok": n, "n_judge_failure": len(fail),
        "timing": {"load_seconds": meta.get("load_seconds"),
                   "judge_seconds": meta.get("judge_seconds"),
                   "seconds_per_query": meta.get("seconds_per_sample")},
        "views": {"samples_with_missing_views": sum(1 for r in rows if r.get("views_missing")),
                  "n_views_shown_median": sorted(len(r["views_shown"]) for r in rows)[len(rows) // 2]
                  if rows else None},
        "distributions": dist,
        "error_tags": {t: [tags.get(t, 0), round(tags.get(t, 0) / n, 3) if n else 0] for t in ERROR_TAGS},
        "n_samples_with_any_tag": sum(1 for r in ok if r.get("error_tags")),
        "auto_accept": [aa, round(aa / n, 3) if n else 0],
        "human_review_uncertain_only": [unc, round(unc / n, 3) if n else 0],
        "judge_failure_ids": [r["sample_id"] for r in fail],
    }


CSS = """body{font-family:system-ui,sans-serif;margin:16px;color:#222}
h1{font-size:17px}h3{font-size:13px;margin:22px 0 4px;border-top:1px solid #ddd;padding-top:10px}
img{height:104px;border:1px solid #ddd;margin:0 2px 2px 0}
table{border-collapse:collapse;font-size:12px;margin:4px 0}
td,th{border:1px solid #ddd;padding:2px 6px;vertical-align:top;max-width:560px;text-align:left}
.t{color:#777;font-size:12px}.k{font-weight:600}
.s2{background:#e6f6e6}.s1{background:#fff6d8}.s0{background:#ffe0e0}
.tag{background:#c62828;color:#fff;padding:1px 6px;border-radius:3px;font-size:11px;margin-right:4px}
.ok{background:#2e7d32;color:#fff;padding:1px 6px;border-radius:3px;font-size:11px}
.miss{color:#c62828}.fail{background:#ffe0e0;padding:6px}
pre{background:#f6f6f6;padding:6px;font-size:11px;white-space:pre-wrap;max-height:300px;overflow:auto}
.sum td{font-size:12px}"""


def _sc(v):
    return "<td class=s%s>%s</td>" % (v, v) if v in (0, 1, 2) else "<td>%s</td>" % v


def render(rows, recs_by_id, summary, out_p):
    h = ["<meta charset='utf-8'><title>Stage 1 test run</title><style>%s</style>" % CSS,
         "<h1>Verification Stage 1 &mdash; %s (%s)</h1>" % (summary["judge_version"], summary["model"]),
         "<table class=sum>"]
    for k in ("n_objects_judged", "n_queries_judged", "n_ok", "n_judge_failure",
              "auto_accept", "human_review_uncertain_only"):
        h.append("<tr><td class=k>%s</td><td>%s</td></tr>" % (k, summary[k]))
    h.append("<tr><td class=k>timing</td><td>%s</td></tr></table>" % summary["timing"])
    h.append("<pre>%s</pre>" % html.escape(json.dumps(
        {"distributions": summary["distributions"], "error_tags": summary["error_tags"]},
        indent=1, ensure_ascii=False)))

    for r in rows:
        rec = recs_by_id.get(r["object_id"], {})
        q = None
        for qi, cand in enumerate(rec.get("queries") or []):
            if r["sample_id"].endswith("/q%d_%s" % (qi, C.slugify(cand.get("task", "")))):
                q = cand
                break
        h.append("<h3>%s</h3>" % html.escape(r["sample_id"]))
        h.append("<div class=t>object: <b>%s</b> &nbsp;|&nbsp; task: <b>%s</b></div>"
                 % (html.escape(str(rec.get("object_name"))),
                    html.escape(str((q or {}).get("task", "?")))))
        h.append("<div class=t>views shown (%d): %s%s</div>"
                 % (len(r["views_shown"]), ", ".join(r["views_shown"]),
                    "<span class=miss> | MISSING: %s</span>" % ", ".join(r["views_missing"])
                    if r["views_missing"] else ""))
        for p in (rec.get("views_used") or []):
            h.append("<img src='file://%s' title='%s'>" % (p, os.path.basename(p)))

        if q:
            h.append("<table><tr><th>hand</th><th>role</th><th>target</th>"
                     "<th>contact_region</th><th>function</th></tr>")
            for i, tag in enumerate("AB"):
                hd = q["roles"][i]
                h.append("<tr><td class=k>%s</td>%s</tr>"
                         % (tag, "".join("<td>%s</td>" % html.escape(str(hd.get(f)))
                                         for f in C.ROLE_FIELDS)))
            h.append("</table>")

        if r.get("judge_failure"):
            h.append("<div class=fail><b>judge_failure</b> &mdash; raw output below</div>")
        else:
            h.append("<table><tr><th>item</th><th>verdict</th><th>contact_region_evidence</th>"
                     "<th>role_function_consistency</th><th>reason</th></tr>")
            tp = r["task_plausibility"]
            h.append("<tr><td class=k>task_plausibility</td>%s<td></td><td></td><td>%s</td></tr>"
                     % (_sc(tp["score"]), html.escape(tp["reason"])))
            bv = r["bimanual_validity"]
            h.append("<tr><td class=k>bimanual_validity</td><td>%s</td><td></td><td></td>"
                     "<td>%s</td></tr>" % (bv["label"], html.escape(bv["reason"])))
            for hh, tag in zip(HANDS, "AB"):
                d = r[hh]
                h.append("<tr><td class=k>hand_%s</td>%s<td>%s</td><td>%s</td><td>%s</td></tr>"
                         % (tag, _sc(d["score"]), d["contact_region_evidence"],
                            d["role_function_consistency"], html.escape(d["reason"])))
            h.append("</table>")
            tags = r.get("error_tags") or []
            h.append("<div>%s &nbsp;<span class=t>evidence_views: %s | confidence: %s</span></div>"
                     % ("".join("<span class=tag>%s</span>" % t for t in tags)
                        or "<span class=ok>no error tags</span>",
                        ", ".join(r.get("evidence_views") or []),
                        r.get("self_reported_confidence")))
        h.append("<details><summary class=t>raw output</summary><pre>%s</pre></details>"
                 % html.escape(r.get("raw", "")))

    with open(out_p, "w") as f:
        f.write("\n".join(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--dataset", default="daily_used")
    a = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(a.judged))
    rows = [json.loads(l) for l in open(a.judged) if l.strip()]
    meta_p = os.path.join(out_dir, "run_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}

    n_scope = meta.get("n_objects_in_scope")
    recs = C.load_stage1(a.dataset)
    ids = {r["object_id"] for r in rows}
    recs_by_id = {r["object_id"]: r for r in recs if r["object_id"] in ids}

    summary = summarize(rows, meta, n_scope)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    render(rows, recs_by_id, summary, os.path.join(out_dir, "index.html"))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("[report] wrote %s/summary.json and index.html" % out_dir)


if __name__ == "__main__":
    main()
