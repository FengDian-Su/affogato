#!/usr/bin/env python
"""
Score a semantic-judge output against a labeled ground-truth file.

Ground truth: verify/groundtruth/*.json — [{object_id, label: ok|junk|borderline, ...}]
(labels produced by a Claude image-verified audit + adversarial skeptic pass; 'borderline' is
excluded from scoring). Judge output: judge_semantic.py records with a 'verdict' field.

  python verify/calibrate.py --judge <judge_output.json> \
      [--truth verify/groundtruth/claude_labels_0720.json]

A judge is fit to run a full scan when junk RECALL is high (it is a SCREEN - misses are permanent,
false flags are cheap because a Claude verification pass arbitrates every flag downstream).
"""
import os
import json
import argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", required=True, help="judge_semantic.py output json")
    ap.add_argument("--truth", default=os.path.join(HERE, "groundtruth", "claude_labels_0720.json"))
    ap.add_argument("--show", type=int, default=10, help="max examples to print per error type")
    args = ap.parse_args()

    truth = {g["object_id"]: g for g in json.load(open(args.truth))}
    judge = {r["object_id"]: r for r in json.load(open(args.judge))}

    tp = fn = fp = tn = 0
    missed, false_alarms = [], []
    per_stratum = Counter()
    for oid, g in truth.items():
        j = judge.get(oid, {})
        v = j.get("verdict")
        if g["label"] == "borderline" or v not in ("ok", "junk"):
            continue
        if g["label"] == "junk" and v == "junk":
            tp += 1
        elif g["label"] == "junk":
            fn += 1
            missed.append((j.get("object_name"), str(j.get("notes", ""))[:90]))
        elif v == "junk":
            fp += 1
            false_alarms.append((j.get("object_name"), j.get("junk_types"), str(j.get("notes", ""))[:70]))
            per_stratum[g.get("stratum", "?")] += 1
        else:
            tn += 1

    n_junk, n_ok = tp + fn, fp + tn
    print(f"scored {n_junk + n_ok} (truth: {n_junk} junk, {n_ok} ok; borderline excluded)")
    print(f"TP={tp} FN={fn} FP={fp} TN={tn}")
    if n_junk:
        print(f"junk recall:    {tp / n_junk * 100:.0f}%")
    if tp + fp:
        print(f"junk precision: {tp / (tp + fp) * 100:.0f}%")
    if n_ok:
        print(f"ok false-flag:  {fp / n_ok * 100:.1f}%   (by stratum: {dict(per_stratum)})")
    bj = [judge.get(o, {}).get("verdict") for o, g in truth.items() if g["label"] == "borderline"]
    print(f"borderline -> junk {bj.count('junk')}, ok {bj.count('ok')}")
    print(f"\nMISSED junk ({fn}):")
    for m in missed[:args.show]:
        print("  -", m)
    print(f"\nFALSE ALARMS ({fp}):")
    for f in false_alarms[:args.show]:
        print("  -", f)


if __name__ == "__main__":
    main()
