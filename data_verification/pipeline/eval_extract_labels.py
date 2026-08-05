#!/usr/bin/env python
"""Pull completed axis1/axis2 labels out of the labeling workflow's agent transcripts and write
labels_axis1.json / labels_axis2.json ({ "<idx>": {"score","reason"} }). Works on a partially-
finished run (only agents that produced a labels array are harvested). Then rebuild the gallery."""
import os, re, json, glob, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "daily_used")
# scan every labeling workflow (axis1 ran in one, axis2 in another after the render change)
WF_ROOT = ("/home/michaellee/.claude/projects/-home-michaellee-mclee-affogato/"
           "fe626019-4ae4-4733-beca-c6303cf44040/subagents/workflows")


def find_labels(obj):
    """Recursively locate a {'labels':[{idx,score,reason}...]} anywhere in a json blob."""
    if isinstance(obj, dict):
        if isinstance(obj.get("labels"), list) and obj["labels"] and isinstance(obj["labels"][0], dict) \
           and "idx" in obj["labels"][0] and "score" in obj["labels"][0]:
            return obj["labels"]
        for v in obj.values():
            r = find_labels(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_labels(v)
            if r:
                return r
    return None


a1, a2 = {}, {}
n_agents = 0
for f in glob.glob(f"{WF_ROOT}/wf_*/agent-*.jsonl"):
    try:
        lines = [json.loads(l) for l in open(f)]
    except Exception:
        continue
    blob = json.dumps(lines)
    axis = 1 if "AXIS 1" in blob else (2 if "AXIS 2" in blob else None)
    if axis is None:
        continue
    labels = None
    for l in reversed(lines):                       # the structured result is near the end
        labels = find_labels(l)
        if labels:
            break
    if not labels:
        continue
    n_agents += 1
    tgt = a1 if axis == 1 else a2
    for lb in labels:
        try:
            tgt[int(lb["idx"])] = {"score": int(lb["score"]), "reason": str(lb.get("reason", ""))}
        except Exception:
            pass

json.dump({str(k): v for k, v in a1.items()}, open(f"{D}/labels_axis1.json", "w"), ensure_ascii=False)
json.dump({str(k): v for k, v in a2.items()}, open(f"{D}/labels_axis2.json", "w"), ensure_ascii=False)
print(f"harvested {n_agents} finished agents -> axis1 {len(a1)}/1000, axis2 {len(a2)}/1000")
subprocess.run([sys.executable, os.path.join(HERE, "build_gallery.py")], check=True)
