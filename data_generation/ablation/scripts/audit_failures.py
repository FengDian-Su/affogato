#!/usr/bin/env python
"""Check every FAILED object against the shipped release's exclusion lists.

The point is to separate "this source file was already broken" from "the replay
introduced a failure".  Only the latter needs action.

Matching is by stage1 INDEX plus a name PREFIX: the runner's log pads the object
name into a 28-character column, so any longer name arrives truncated and an
exact comparison reports a false new failure ("Plastic Ring with a Connector").
"""
import collections
import glob
import json
import re

EXCL = (("outputs/stage2_points_20260803/known_exclusions.json", None),
        ("outputs/stage2_points_20260803/furnitures_known_exclusions.json", "furnitures"))
LINE = re.compile(r"\[(\d+)\] (.{1,28}?)\s+FAILED \((.+?)\)")


def load_exclusions():
    ex = {}
    for path, forced in EXCL:
        data = json.load(open(path))
        if forced:
            items = data if isinstance(data, list) else data.get("objects", [])
            for o in items:
                ex[(forced, o["index"])] = (o["object_name"], o.get("reason", ""))
        else:
            for cat, v in data.items():
                for o in v["objects"]:
                    ex[(cat, o["index"])] = (o["object_name"], o.get("reason", ""))
    return ex


def main():
    ex = load_exclusions()
    known, fresh = [], []
    kinds, percat = collections.Counter(), collections.Counter()
    seen = set()
    for f in glob.glob("outputs/stage2/*/run_gpu*.log"):
        cat = f.split("/")[2]
        for line in open(f, errors="ignore"):
            m = LINE.match(line)
            if not m:
                continue
            idx, name, reason = int(m.group(1)), m.group(2).strip(), m.group(3)
            if (cat, idx) in seen:
                continue
            seen.add((cat, idx))
            percat[cat] += 1
            kinds[reason.split(":")[0] + (":" + reason.split(":")[1][:24] if ":" in reason else "")] += 1
            hit = ex.get((cat, idx))
            # truncated-column safe: the logged name is a prefix of the real one
            if hit and hit[0].startswith(name):
                known.append((cat, idx, name))
            else:
                fresh.append((cat, idx, name, reason[:70], hit))
    print(f"failures {len(seen)}   known source defects {len(known)}   NEW {len(fresh)}")
    print("by category:", dict(percat))
    print("kinds:")
    for k, v in kinds.most_common():
        print(f"  {v:4}  {k}")
    if fresh:
        print("\nNEW failures needing attention:")
        for c, i, n, r, hit in fresh:
            print(f"  [{c} {i}] {n}: {r}   (exclusion entry: {hit})")
    return len(fresh)


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
