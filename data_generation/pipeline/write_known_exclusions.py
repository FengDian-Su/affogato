#!/usr/bin/env python
"""Regenerate known_exclusions.json from the run that actually produced the release.

The old lists were assembled by hand across several runs and some entries carry
reason "unknown"; this rebuilds them from the runner's own FAILED/SKIP lines, so
every excluded object states why it was excluded and which stage1 index it was.

Every stage1 object with no output is accounted for: a runner line explains it,
or it lands in "unexplained" for a human to look at rather than being dropped.

  python pipeline/write_known_exclusions.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

CATS = ("daily_used", "electronics", "furnitures")
FAILED = re.compile(r"\[(\d+)\] (.{1,28}?)\s+FAILED \((.+?)\)")
SKIPPED = re.compile(r"\[(\d+)\] (.{1,28}?)\s+SKIP \((.+?)\)")


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = {}
    for cat in CATS:
        recs = [r for r in json.load(open(f"outputs/stage1/{cat}/stage1_all.json"))
                if not r.get("error")]
        produced = {o for o in os.listdir(f"outputs/stage2/{cat}")
                    if os.path.isdir(f"outputs/stage2/{cat}/{o}")}
        # the runner logs an index, not an id; index into the same filtered list
        # the runner itself sliced with --start/--end
        reasons = {}
        for log in glob.glob(f"outputs/stage2/{cat}/run_gpu*.log"):
            for line in open(log, errors="ignore"):
                for pat, kind in ((FAILED, "FAILED"), (SKIPPED, "SKIP")):
                    m = pat.match(line)
                    if m:
                        reasons.setdefault(int(m.group(1)), f"{kind}: {m.group(3)}")

        missing = []
        for i, rec in enumerate(recs):
            if rec["object_id"] in produced:
                continue
            n_q = len([q for q in rec.get("queries", [])
                       if len(q.get("roles", [])) == 2 and len(q.get("molmo_queries", [])) == 2])
            missing.append(dict(index=i, object_id=rec["object_id"],
                                object_name=rec["object_name"], n_queries=n_q,
                                reason=reasons.get(i, "zero stage1 queries" if n_q == 0
                                                   else "unexplained")))
        unexplained = [m for m in missing if m["reason"] == "unexplained"]
        out[cat] = dict(total=len(recs), completed=len(produced), missing=len(missing),
                        unexplained=len(unexplained), objects=missing)
        print(f"{cat:12} stage1 {len(recs):6}  produced {len(produced):6}  "
              f"excluded {len(missing):4}  unexplained {len(unexplained):3}")
        for u in unexplained[:5]:
            print(f"    UNEXPLAINED: {u}")

    path = "outputs/stage2/known_exclusions.json"
    with open(path + ".tmp", "w") as s:
        json.dump(out, s, indent=1, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    print(f"\n-> {path}")
    return sum(v["unexplained"] for v in out.values())


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
