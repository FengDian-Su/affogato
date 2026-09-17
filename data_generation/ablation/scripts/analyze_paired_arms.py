#!/usr/bin/env python
"""Separate the aggregation effect from the SAM2 noise floor.

arm1 = conditional_mean, arm2 = iou_group, arm3 = conditional_mean AGAIN.
Because per-point SAM masks are never persisted, every arm re-ran SAM2 and
carries its nondeterminism.  arm1-vs-arm3 is that noise measured on these exact
objects; arm1-vs-arm2 is noise PLUS the method change.  The ratio between them
is what says whether the method change is real, and it needs no appeal to the
12-object pilot's floor estimate.
"""
import json
import os
import sys

import numpy as np

R = "ablation/results/paired_arms_electronics"
A1, A2, A3 = f"{R}/arm1_conditional", f"{R}/arm2_iou", f"{R}/arm3_conditional_repeat"
SUPPORT_THR = 0.15


def load(root, oid, q, role):
    with np.load(f"{root}/{oid}/{q}/scores.npz", allow_pickle=False) as z:
        return z[f"score{role}"].astype(np.float64), z[f"pts{role}_multi"]


rows = []
for oid in sorted(os.listdir(A1)):
    if not os.path.isdir(f"{A1}/{oid}"):
        continue
    for q in sorted(os.listdir(f"{A1}/{oid}")):
        if not all(os.path.isfile(f"{r}/{oid}/{q}/scores.npz") for r in (A1, A2, A3)):
            continue
        for role in "AB":
            a1, multi = load(A1, oid, q, role)
            a2, _ = load(A2, oid, q, role)
            a3, _ = load(A3, oid, q, role)
            k = np.bincount(multi[:, 0].astype(int)) if len(multi) else np.array([0])
            rows.append(dict(
                multi=int((k > 1).sum()),
                method=float(np.abs(a1 - a2).mean()),      # aggregation + noise
                noise=float(np.abs(a1 - a3).mean()),       # noise alone
                method_moved=float((np.abs(a1 - a2) > 1e-3).mean()),
                noise_moved=float((np.abs(a1 - a3) > 1e-3).mean()),
                method_flip=float(((a1 >= SUPPORT_THR) != (a2 >= SUPPORT_THR)).mean()),
                noise_flip=float(((a1 >= SUPPORT_THR) != (a3 >= SUPPORT_THR)).mean()),
            ))

if not rows:
    sys.exit("no paired rows")
single = [r for r in rows if r["multi"] == 0]
multi = [r for r in rows if r["multi"] > 0]


def show(label, rs):
    if not rs:
        return
    m = np.array([r["method"] for r in rs]); n = np.array([r["noise"] for r in rs])
    mm = np.array([r["method_moved"] for r in rs]); nm = np.array([r["noise_moved"] for r in rs])
    mf = np.array([r["method_flip"] for r in rs]); nf = np.array([r["noise_flip"] for r in rs])
    # With the points frozen the whole chain is deterministic, so the noise
    # column is normally exactly 0.  A ratio against it is meaningless (it just
    # reports 1/epsilon), so state determinism as a fact and only ratio when
    # some noise actually exists.
    def line(name, mv, nv, fmt):
        r = f"   ratio {mv/nv:7.1f}x" if nv > 0 else "   noise is EXACTLY 0 (bit-identical reruns)"
        print(f"    {name:8} method {mv:{fmt}}   noise {nv:{fmt}}{r}")
    print(f"\n  [{label}]  n={len(rs)} role-heatmaps")
    line("MAE", m.mean(), n.mean(), ".6f")
    line("moved", mm.mean(), nm.mean(), ".6f")
    line("flipped", mf.mean(), nf.mean(), ".6f")
    print(f"    per-heatmap method>noise: {100*np.mean(m > n):.1f}%")


print(f"=== paired arms, electronics [0:80] — {len(rows)} role-heatmaps ===")
show("ALL", rows); show("single-point", single); show("multi-point", multi)
json.dump(rows, open("ablation/results/paired_arms_electronics/summary.json", "w"))
print("\n-> ablation/results/paired_arms_electronics/summary.json")
