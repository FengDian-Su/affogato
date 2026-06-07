#!/usr/bin/env python
"""
compare_to_affogato.py
======================

Quantify how well our reproduced single-region affordance heatmaps match the
ORIGINAL AFFOGATO ground-truth heatmaps, per (object, query) and aggregated.

Input: the `affordance_pred.npz` files written by `single_region_affordance.py`,
each containing (on the SAME 16384 GT points):
    pred   (N, K) float32   our heatmap, column j = query j
    gt     (N, K) float32   AFFOGATO GT heatmap, column j (= xyzc.npy col 3+j)
    counts (N,)   int32      multi-view visibility count (0 = never seen)
    queries(K,)              the K query strings

Metrics (definitions follow the AFFOGATO paper -> LASO/IAGNet -> 3D-AffordanceNet
and the saliency literature; see notes/affogato_validation_methodology.md):

  PRIMARY (scale-robust — the right lens since SAM2/MobileSAM + Gaussian differences
           mean P and G may differ in absolute scale):
    spearman   rank correlation (monotonic-invariant)         [-1,1] up
    cc         Pearson linear correlation (affine-invariant)  [-1,1] up
    auc        ROC-AUC, GT binarized at 0.5 (monotonic-inv.)  [0,1]  up   (=AUC-Judd for continuous P)
    sim        Σ min(P,G) after sum-to-1 normalization        [0,1]  up

  SECONDARY (scale-sensitive — reported for paper comparability, not for ranking fidelity):
    aiou       mean IoU over threshold sweep 0..0.99,
               P min-max normalized, GT binarized at 0.5       [0,1]  up
    mae        mean |P-G| (raw)                                [0,1]  down
    mae_affine mean |aP+b-G| after best least-squares affine   [0,1]  down

Each metric is computed in two point sets and you should report both:
    all      = all N points (unseen points contribute their structural 0)
    visible  = only points seen by >=1 view (counts>0); the "coverage-conditioned" view

Also computes an ALIGNMENT diagnostic: the KxK Spearman matrix between pred col i
and gt col j. The diagonal should dominate; if not, the query<->channel order is
wrong (or that object reproduces poorly).

Usage:
  python compare_to_affogato.py --pred_dir output_single_region --out compare_results
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------------ metrics ---

def _safe(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)


def sim_metric(P, G, eps=1e-12):
    """Similarity / histogram intersection after sum-to-1 normalization."""
    sP, sG = P.sum(), G.sum()
    if sP <= eps or sG <= eps:
        return np.nan
    return float(np.minimum(P / sP, G / sG).sum())


def auc_metric(P, G, gt_thresh=0.5):
    """ROC-AUC with GT binarized at gt_thresh. Needs both classes present."""
    y = (G >= gt_thresh).astype(np.int32)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    if P.std() == 0:
        return 0.5
    return float(roc_auc_score(y, P))


def aiou_metric(P, G, gt_thresh=0.5, step=0.01):
    """Mean IoU over threshold sweep 0..0.99; P min-max normalized, GT binarized."""
    y = G >= gt_thresh
    if y.sum() == 0:
        return np.nan
    pmin, pmax = P.min(), P.max()
    Pn = (P - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(P)
    thr = np.arange(0.0, 1.0, step)
    ious = []
    for t in thr:
        pb = Pn >= t
        inter = np.logical_and(pb, y).sum()
        union = np.logical_or(pb, y).sum()
        ious.append(inter / union if union > 0 else 0.0)
    return float(np.mean(ious))


def corr_metrics(P, G):
    """Spearman + Pearson; guard against zero-variance vectors."""
    if P.std() == 0 or G.std() == 0:
        return np.nan, np.nan
    sp = spearmanr(P, G).correlation
    cc = pearsonr(P, G)[0]
    return float(sp), float(cc)


def mae_metrics(P, G):
    """Raw MAE and MAE after best least-squares affine map aP+b -> G."""
    mae = float(np.mean(np.abs(P - G)))
    # affine fit
    A = np.vstack([P, np.ones_like(P)]).T
    try:
        (a, b), *_ = np.linalg.lstsq(A, G, rcond=None)
        mae_aff = float(np.mean(np.abs(a * P + b - G)))
    except Exception:
        mae_aff = np.nan
    return mae, mae_aff


def channel_metrics(P, G):
    """All metrics for one (object, query) heatmap pair on a given point set."""
    sp, cc = corr_metrics(P, G)
    mae, mae_aff = mae_metrics(P, G)
    return {
        "spearman": _safe(sp),
        "cc": _safe(cc),
        "auc": _safe(auc_metric(P, G)),
        "sim": _safe(sim_metric(P, G)),
        "aiou": _safe(aiou_metric(P, G)),
        "mae": _safe(mae),
        "mae_affine": _safe(mae_aff),
    }


# ------------------------------------------------------------- per object ----

def alignment_matrix(pred, gt):
    """KxK Spearman(pred[:,i], gt[:,j]); diagonal dominance => correct query order."""
    K = pred.shape[1]
    M = np.full((K, K), np.nan)
    for i in range(K):
        if pred[:, i].std() == 0:
            continue
        for j in range(K):
            if gt[:, j].std() == 0:
                continue
            M[i, j] = spearmanr(pred[:, i], gt[:, j]).correlation
    # diagonal dominance: for each predicted channel i, is gt col i its argmax?
    diag_is_best = []
    for i in range(K):
        row = M[i]
        if np.all(np.isnan(row)):
            diag_is_best.append(None)
        else:
            diag_is_best.append(bool(np.nanargmax(row) == i))
    return M, diag_is_best


def evaluate_object(npz_path):
    """Return per-(query, mode) metric rows + alignment info for one object."""
    d = np.load(npz_path, allow_pickle=True)
    pred, gt, counts = d["pred"], d["gt"].astype(np.float32), d["counts"]
    queries = list(d["queries"]) if "queries" in d else [f"q{j}" for j in range(pred.shape[1])]
    object_id = os.path.basename(os.path.dirname(npz_path))

    Kp, Kg = pred.shape[1], gt.shape[1]
    K = min(Kp, Kg)
    if Kp != Kg:
        print(f"  [warn] {object_id}: pred has {Kp} channels but gt has {Kg}; comparing first {K}")

    visible = counts > 0
    coverage = float(visible.mean())

    rows = []
    for j in range(K):
        P_all, G_all = pred[:, j].astype(np.float64), gt[:, j].astype(np.float64)
        for mode, mask in (("all", np.ones_like(visible)), ("visible", visible)):
            if mask.sum() < 10:
                continue
            m = channel_metrics(P_all[mask], G_all[mask])
            m.update(object_id=object_id, query_idx=j,
                     query=str(queries[j]) if j < len(queries) else f"q{j}",
                     mode=mode, coverage=coverage, n_points=int(mask.sum()))
            rows.append(m)

    align_M, diag_best = alignment_matrix(pred[:, :K], gt[:, :K])
    return rows, {"object_id": object_id, "coverage": coverage,
                  "alignment_matrix": align_M.tolist(),
                  "diag_is_best": diag_best}


# --------------------------------------------------------------- aggregate ---

METRIC_KEYS = ["spearman", "cc", "auc", "sim", "aiou", "mae", "mae_affine"]


def aggregate(rows, mode):
    sub = [r for r in rows if r["mode"] == mode]
    out = {}
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in sub if r[k] is not None], dtype=np.float64)
        out[k] = {
            "mean": float(np.mean(vals)) if len(vals) else None,
            "median": float(np.median(vals)) if len(vals) else None,
            "std": float(np.std(vals)) if len(vals) else None,
            "n": int(len(vals)),
        }
    return out


def print_summary(rows, summaries):
    n_obj = len({r["object_id"] for r in rows})
    covs = [s["coverage"] for s in summaries]
    print(f"\n{'='*72}\nAFFOGATO reproduction-fidelity summary")
    print(f"  objects: {n_obj} | mean coverage: {np.mean(covs)*100:.1f}% "
          f"(min {np.min(covs)*100:.1f}%, max {np.max(covs)*100:.1f}%)")

    # alignment health
    diag_ok = [b for s in summaries for b in s["diag_is_best"] if b is not None]
    if diag_ok:
        print(f"  query<->channel alignment: {sum(diag_ok)}/{len(diag_ok)} predicted "
              f"channels best-match their own GT channel ({np.mean(diag_ok)*100:.0f}%)")

    for mode in ("all", "visible"):
        agg = aggregate(rows, mode)
        print(f"\n  --- mode = {mode} ---")
        print(f"    {'metric':<11}{'mean':>9}{'median':>9}{'std':>8}{'n':>6}   (direction)")
        dirs = {"spearman": "↑", "cc": "↑", "auc": "↑", "sim": "↑",
                "aiou": "↑", "mae": "↓", "mae_affine": "↓"}
        for k in METRIC_KEYS:
            a = agg[k]
            if a["mean"] is None:
                print(f"    {k:<11}{'n/a':>9}"); continue
            print(f"    {k:<11}{a['mean']:>9.4f}{a['median']:>9.4f}{a['std']:>8.4f}{a['n']:>6}   {dirs[k]}")
    print(f"{'='*72}")
    print("Read primary metrics (spearman/cc/auc/sim) for fidelity; mae is scale-sensitive.")
    print("'visible' mode excludes never-projected points (structural 0) — usually the fairer view.")


# -------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description="Compare reproduced heatmaps to AFFOGATO GT")
    ap.add_argument("--pred_dir", default="outputs/output_single_region",
                    help="dir containing <object_id>/affordance_pred.npz")
    ap.add_argument("--out", default="results/compare_results", help="output dir for csv/json")
    args = ap.parse_args()

    npz_paths = sorted(glob.glob(os.path.join(args.pred_dir, "*", "affordance_pred.npz")))
    if not npz_paths:
        print(f"No affordance_pred.npz under {args.pred_dir}"); sys.exit(1)
    print(f"Found {len(npz_paths)} objects under {args.pred_dir}")

    all_rows, summaries = [], []
    for p in npz_paths:
        rows, summ = evaluate_object(p)
        all_rows.extend(rows)
        summaries.append(summ)
        cov = summ["coverage"]
        # quick per-object line (visible-mode primary metrics, mean over queries)
        vis = [r for r in rows if r["mode"] == "visible"]
        def mean_of(k):
            v = [r[k] for r in vis if r[k] is not None]
            return np.mean(v) if v else float("nan")
        print(f"  {summ['object_id']}  cov={cov*100:4.1f}%  "
              f"ρ={mean_of('spearman'):.3f} cc={mean_of('cc'):.3f} "
              f"auc={mean_of('auc'):.3f} sim={mean_of('sim'):.3f}")

    os.makedirs(args.out, exist_ok=True)
    # CSV
    import csv
    csv_path = os.path.join(args.out, "per_query_metrics.csv")
    fields = ["object_id", "query_idx", "mode", "coverage", "n_points",
              *METRIC_KEYS, "query"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fields})
    # JSON aggregate + alignment
    agg_path = os.path.join(args.out, "summary.json")
    with open(agg_path, "w") as f:
        json.dump({
            "n_objects": len(summaries),
            "aggregate_all": aggregate(all_rows, "all"),
            "aggregate_visible": aggregate(all_rows, "visible"),
            "per_object": summaries,
        }, f, indent=2)

    print_summary(all_rows, summaries)
    print(f"\n[INFO] wrote {csv_path}")
    print(f"[INFO] wrote {agg_path}")


if __name__ == "__main__":
    main()
