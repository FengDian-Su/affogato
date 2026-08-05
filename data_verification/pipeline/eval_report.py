#!/usr/bin/env python
"""Package the stage2 ground-truth labels + print the distribution report.

Reads labels_axis1.json / labels_axis2.json (Claude ground truth) + labeling_records.json, writes
ground_truth.json (one row per result with both axis scores + reasons + meta + render paths), and
prints the per-axis distribution, the task/role x heatmap cross-tab, and per-coordination/category
breakdowns. This ground_truth.json is the reference an LLM judge is later scored against."""
import json
from collections import Counter

D = "/home/michaellee/mclee/affogato/data_generation/outputs/stage2_eval/daily_used"
recs = {r["idx"]: r for r in json.load(open(f"{D}/labeling_records.json"))}
a1 = {int(k): v for k, v in json.load(open(f"{D}/labels_axis1.json")).items()}
a2 = {int(k): v for k, v in json.load(open(f"{D}/labels_axis2.json")).items()}


def dist(d):
    c = Counter(v["score"] for v in d.values()); n = max(len(d), 1)
    return f"bad {c[0]} ({c[0]/n*100:.0f}%) / ok {c[1]} ({c[1]/n*100:.0f}%) / good {c[2]} ({c[2]/n*100:.0f}%)  [n={len(d)}]"


print("=" * 68)
print("AXIS 1  task/role :", dist(a1))
print("AXIS 2  heatmap   :", dist(a2))

both = [(a1[i]["score"], a2[i]["score"]) for i in a2 if i in a1]
print(f"\ncross-tab (task/role rows x heatmap cols), n={len(both)}:")
print("          heat0  heat1  heat2")
for t in (0, 1, 2):
    row = [sum(1 for x, y in both if x == t and y == h) for h in (0, 1, 2)]
    print(f"task={t}  " + "".join(f"{v:>7}" for v in row))

for key in ("coordination", "category"):
    print(f"\naxis2 heatmap by {key}:")
    for val in sorted({str(recs[i][key]) for i in a2}):
        sub = [a2[i]["score"] for i in a2 if str(recs[i][key]) == val]
        c = Counter(sub); n = len(sub)
        print(f"  {val:20} bad {c[0]/n*100:>3.0f}%  ok {c[1]/n*100:>3.0f}%  good {c[2]/n*100:>3.0f}%   (n={n})")

# ---- package ground truth ----
rows = []
for i in sorted(recs):
    r = recs[i]
    l1, l2 = a1.get(i), a2.get(i)
    rows.append({
        "idx": i, "object_id": r["object_id"], "object_name": r["object_name"],
        "task": r["task"], "category": r["category"], "coordination": r["coordination"],
        "query": r["query"], "roles": r["roles"], "coverage": r["coverage"],
        "axis1_taskrole": (l1 or {}).get("score"), "axis1_reason": (l1 or {}).get("reason"),
        "axis2_heatmap": (l2 or {}).get("score"), "axis2_reason": (l2 or {}).get("reason"),
        "rgb_png": r["rgb_png"], "heat_png": r["heat_png"],
    })
json.dump(rows, open(f"{D}/ground_truth.json", "w"), ensure_ascii=False, indent=1)
done = sum(1 for r in rows if r["axis1_taskrole"] is not None and r["axis2_heatmap"] is not None)
print(f"\nground_truth.json written: {len(rows)} rows, {done} with BOTH axes labeled")
