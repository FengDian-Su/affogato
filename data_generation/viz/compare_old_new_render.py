#!/usr/bin/env python
"""Old-vs-new segmentation comparison renders.

For (oid, qdir) pairs present in BOTH outputs/bimanual_grounding (old prod,
July 4) and outputs/stage2_full (new prod), render a 4-row figure with the
SAME renderer/cameras: OLD raw, OLD final, NEW raw, NEW final; 4 azimuths.
Run: mm python comp_render.py [n_samples] [seed]
"""
import os, sys, json, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

O = "/home/michaellee/mclee/affogato/data_generation/outputs"
OLD, NEW = f"{O}/bimanual_grounding", f"{O}/stage2_full"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comp_renders")
THR = 0.15
CA, CB, CO = "#F28E2B", "#0ABAB5", "#d612d6"  # A orange, B teal, overlap magenta


def draw_row(fig, row, nrows, xyz, sA, sB, label):
    a = sA > THR
    b = sB > THR
    both = a & b
    x, y, z = xyz[:, 0], xyz[:, 2], xyz[:, 1]  # axis1 = up
    for col, az in enumerate((45, 135, 225, 315)):
        ax = fig.add_subplot(nrows, 4, row * 4 + col + 1, projection="3d")
        ax.scatter(x, y, z, s=0.5, c="#c8c8c8", alpha=0.25, linewidths=0)
        for m, c in ((a & ~both, CA), (b & ~both, CB), (both, CO)):
            if m.any():
                ax.scatter(x[m], y[m], z[m], s=2.0, c=c, alpha=0.9, linewidths=0)
        ax.view_init(elev=18, azim=az)
        ax.set_axis_off()
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        if col == 0:
            ax.text2D(-0.12, 0.5, label, transform=ax.transAxes, rotation=90,
                      va="center", ha="center", fontsize=11, fontweight="bold")


def render_pair(job):
    idx, oid, qd = job
    out = f"{OUT}/cmp_{idx:03d}_{oid[:8]}_{qd[:40]}.png"
    if os.path.exists(out):
        return out
    try:
        do = np.load(f"{OLD}/{oid}/{qd}/scores.npz", allow_pickle=True)
        dn = np.load(f"{NEW}/{oid}/{qd}/scores.npz", allow_pickle=True)
        mo = json.load(open(f"{OLD}/{oid}/{qd}/meta.json"))
        mn = json.load(open(f"{NEW}/{oid}/{qd}/meta.json"))
    except Exception as e:
        return f"SKIP {oid}/{qd}: {e}"
    fig = plt.figure(figsize=(15, 14.5))
    rows = [
        (do["xyz"], do["scoreA_raw"], do["scoreB_raw"], "OLD raw"),
        (do["xyz"], do["scoreA"], do["scoreB"], "OLD final"),
        (dn["xyz"], dn["scoreA_raw"], dn["scoreB_raw"], "NEW raw"),
        (dn["xyz"], dn["scoreA"], dn["scoreB"], "NEW final"),
    ]
    for r, (xyz, sA, sB, lab) in enumerate(rows):
        draw_row(fig, r, 4, xyz.astype(np.float32), sA, sB, lab)
    ro, rn = mo["roles"], mn["roles"]
    ttl = (f"{mn.get('object_name','?')} — {mn['task']}\n"
           f"OLD  A:{ro[0].get('region')} | B:{ro[1].get('region')}\n"
           f"NEW  A:{rn[0].get('region')} ({rn[0].get('target')}) | "
           f"B:{rn[1].get('region')} ({rn[1].get('target')})")
    fig.suptitle(ttl, fontsize=10)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.01,
                        wspace=0.0, hspace=0.02)
    fig.savefig(out, dpi=78)
    plt.close(fig)
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    os.makedirs(OUT, exist_ok=True)
    pairs = []
    for oid in sorted(os.listdir(OLD)):
        nd = os.path.join(NEW, oid)
        if not os.path.isdir(nd):
            continue
        for qd in sorted(os.listdir(os.path.join(OLD, oid))):
            if (os.path.exists(f"{OLD}/{oid}/{qd}/scores.npz")
                    and os.path.exists(f"{NEW}/{oid}/{qd}/scores.npz")):
                pairs.append((oid, qd))
    random.seed(seed)
    random.shuffle(pairs)
    # at most one query per object for breadth
    seen, sample = set(), []
    for oid, qd in pairs:
        if oid in seen:
            continue
        seen.add(oid)
        sample.append((oid, qd))
        if len(sample) >= n:
            break
    jobs = [(i, oid, qd) for i, (oid, qd) in enumerate(sample)]
    with Pool(8) as p:
        for i, r in enumerate(p.imap_unordered(render_pair, jobs)):
            if r.startswith("SKIP") or (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(jobs)}] {r}", flush=True)
    print("DONE", len(jobs))


if __name__ == "__main__":
    main()
