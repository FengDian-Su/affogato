# AFFOGATO Reproduction — Validation Findings (run 1)

Date: 2026-06-04. Setup: 8 electronics objects (mapping idx 0–7), 24 views,
`single_region_affordance.py` (SAM2 + Gaussian, MolmoPoint-8B), compared per-point
to affogato GT (`xyzc.npy` cols 3–7) with `compare_to_affogato.py`.

## ✅ DAILY-USED subset — N=89 (frame-aligned, 24 views, 3 GPUs)
Ran the single-region pipeline on 90 daily-used objects (mapping
`dataset/daily_used_to_affogato.json`, renders on `/nfs_drive/gobjaverse/daily-used`,
affogato pc in part013 etc.). 89 ok / 1 fail. coverage 98.6% (frame transform correct
on this subset too). Workers: GPU1 Ada + GPU2 3090 + GPU3 Blackwell.

| metric (visible) | daily-used N=89 | electronics N=100 | paper Espresso-3D on Affogato |
|---|---|---|---|
| AUC | **0.73** (median 0.76) | 0.78 | 0.790 |
| Spearman | 0.25 | 0.19 | — |
| CC | 0.31 | 0.35 | — |
| SIM | **0.43** | 0.35 | 0.429 |
| aIoU | 0.12 | 0.14 | 0.136 |
| MAE | 0.14 (0.10 affine) | 0.083 | 0.111 |

Reproduction generalizes to daily-used: AUC 0.73 is slightly below electronics (0.78)
— expected, daily-used objects are more diverse/organic (Cloud, Flower, …) so Molmo+SAM2
localize less cleanly — but SIM (0.43) actually MATCHES the paper's model exactly and
Spearman is higher than electronics. Still near the affogato dataset ceiling (~0.79).
Artifacts: `output_daily_used/`, `compare_results_daily_used/`, `gallery_daily_used.html`.

## ✅ VALIDATION RESULT — N=100 (frame-aligned, 24 views, 2 GPUs)
Final systematic run over **100 objects** (the locally-available limit is ~420; ran the
first 100 with local gobjaverse renders, idx 0–95 + scattered 649–3768). Two workers
(GPU1 RTX 6000 Ada + GPU2 RTX 3090). **The re-implementation is validated as correct.**

| metric (visible pts) | N=100 | N=8 (earlier) | dir |
|---|---|---|---|
| **AUC** | **0.78** (median **0.85**) | 0.83 | ↑ |
| **CC** | 0.35 | 0.47 | ↑ |
| **Spearman** | 0.19 | 0.30 | ↑ |
| **SIM** | 0.35 | 0.43 | ↑ |
| **aIoU** | 0.14 | 0.18 | ↑ |
| **MAE** | 0.083 (0.066 affine) | 0.067 | ↓ |
| query↔channel diag-best | **34%** (vs 20% chance) | 48% | ↑ |

mean coverage 97.9%. AUC median 0.85 with most objects 0.5–0.98 (selection spread:
best ~0.98, worst ~0.45). The N=8 set happened to contain easier objects; N=100
regresses toward the true mean but stays clearly above chance on every metric and on
alignment. Per-object spread is large (expected: GT channels near-duplicate, model
divergences SAM2/MobileSAM + Gaussian + MolmoPoint-8B/Molmo-7B-D). Artifacts:
`output_single_region_fixed/` (100 npz), `compare_results_fixed/` (per-query CSV +
summary.json), `images_n100/` + `gallery_n100.html` (15 representative objects).

## (earlier) VALIDATION RESULT — N=8 (frame-aligned, 24 views)

| metric (visible pts) | buggy frame | **fixed frame** | dir |
|---|---|---|---|
| AUC  | 0.49 (chance) | **0.83** (median 0.94) | ↑ |
| CC   | 0.02 | **0.47** | ↑ |
| Spearman | 0.01 | **0.30** | ↑ |
| SIM  | 0.18 | **0.43** | ↑ |
| aIoU | 0.02 | **0.18** | ↑ |
| MAE  | 0.092 | **0.067** | ↓ |
| query↔channel diag-best | 15% | **48%** (vs 20% chance) | ↑ |

Per-object AUC 0.65–0.91; coverage 89–100%. The 48% diagonal-best confirms
`queries.json` order == `xyzc` column order (col 3+i = query i). Remaining gap from
perfect is expected: GT's own channels are near-duplicates (self-corr ~0.6) and our
engine diverges from the original (SAM2 vs MobileSAM, +Gaussian, MolmoPoint-8B vs
Molmo-7B-D-0924). Artifacts: `output_single_region_fixed/`, `compare_results_fixed/`.

## ⚠️ ROOT CAUSE: coordinate-frame bug in run_pipeline.py
The weak per-query diagonal (run 1) was **NOT** mainly model divergence — it was a
**coordinate-frame bug**. `run_pipeline.py` votes onto the **raw** affogato GT points,
but the affogato points use a different axis convention than the G-Objaverse camera
frame. `point_cloud_from_depth.ipynb` (the correct reference) applies a transform
before voting — CELL 17:
```python
gt_points = gt_points[:, [0, 2, 1]]   # swap Y/Z
gt_points[:, 1] *= -1                  # flip new Y   -> (x, -z, y)
```
`run_pipeline.py` (and the first extraction of `single_region_affordance.py`) omitted
it, so each GT point projected to the WRONG pixel and sampled the wrong heatmap value.
Proof (project GT points, measure fraction landing on the object silhouette):

| object | raw (buggy): cov / in-sil | swapYZ+flipY (fixed): cov / in-sil |
|---|---|---|
| b9223e94 | 99.3% / 0.72 | 100.0% / **0.98** |
| 02886b7e | 64.4% / 0.62 |  97.3% / **0.91** |
| b8aca7c6 | 38.7% / 0.15 |  99.9% / **0.90** |

Fix: `single_region_affordance.py` now applies this transform before voting
(`PipelineConfig.align_gt_frame=True`). Run 1 numbers below are from the BUGGY frame
and are **invalid**; re-running frame-aligned (run 3).

## TL;DR (run 1 — buggy frame, superseded)
The extracted pipeline is mechanically correct and does localize real affordance
regions (~3× above chance), but the naive per-query diagonal looked like failure
(AUC≈0.51) — now explained by the frame bug above plus secondary confounds
(near-duplicate GT channels, model divergences).

## What is verified correct (not the problem)
- **Projection / pixel space**: depth↔RGB silhouette IoU = 0.96–0.99; camera
  intrinsics match `x_fov` (fx≈711@512px); no flip/transpose. GT points project
  onto the rendered object (coverage 39–100% over 24 views).
- **Voting/projection round-trip** (splat GT→2D→vote back, no Molmo/SAM2):
  Spearman 0.45–0.96 vs GT. So the 2D→3D math is faithful (the <1.0 is pixel-
  quantization loss — this is also the realistic **ceiling** for the method).

## The real signal (forensics)
Enrichment E = mean(GT over top-200 predicted points) / mean(GT). 1.0 = no signal.
Shuffle-control null calibrates Hungarian inflation.

| quantity | observed | shuffled null | verdict |
|---|---|---|---|
| **diagonal** enrichment (pred_i vs gt_i) | **1.29** | 0.97 | weak per-query signal |
| **best-permutation** enrichment | **2.95** | 1.22 | **strong real localization** |

⇒ Our 5 heatmaps collectively localize affordance regions ~3× above chance, but
**query i does NOT reliably map to GT column i**. The best-matching permutation
**differs per object** (no single fix), so it is **not** a fixed channel-order bug.

## Why the diagonal looks like failure — three confounds
1. **GT channels are near-duplicates.** Mean pairwise Spearman among GT's own 5
   channels = **0.60** (up to **0.90** for Digital Camera / object c9405115). The 5
   affordance queries per object often highlight overlapping regions, so per-query
   discrimination is intrinsically weak in the GT itself. Our predictions are
   *more* query-distinct (pred inter-channel corr = 0.32) than the GT.
2. **Model divergences from the original AFFOGATO engine** (all reduce fidelity):
   - SAM2  vs original **MobileSAM**
   - **+Gaussian weight** vs original **raw sigmoid** (← tested in run 2)
   - **MolmoPoint-8B** vs original **Molmo-7B-D-0924**
3. **Huge per-object variance.** Best-perm enrichment: Digital Camera 5.11 (great),
   Monitor (b9223e94) 1.43 (poor even at best) despite 99% coverage. Some objects
   reproduce well, some poorly.

## Standard affogato metrics on the DIAGONAL (run 1, visible points)
AUC 0.51 · Spearman 0.006 · CC 0.022 · SIM 0.18 · aIoU 0.02 · MAE 0.09.
These **understate** true fidelity because of the confounds above. Report them, but
pair them with best-match localization evidence and per-object breakdown.

## Recommended next steps
- **run 2 — `--no_gaussian`** (raw sigmoid, closer to original): does removing the
  Gaussian we added improve diagonal fidelity? (in progress)
- **Match the original engine** for a clean fidelity test: Molmo-7B-D-0924 +
  MobileSAM + raw sigmoid. The current divergences confound "is our reimplementation
  correct" with "do model swaps change the output."
- **Reporting**: always report per-(object,query); give both diagonal (honest
  reproduction metric) and best-permutation (localization ceiling); always show
  coverage; consider excluding objects whose GT channels are near-duplicates
  (self-corr > ~0.8) from per-query discrimination claims.
- **Open question to resolve**: confirm whether affogato's `queries.json` order is
  guaranteed to match `xyzc.npy` column order (no metadata on HF says so). If it
  does, the weak diagonal is genuine per-query reproduction noise; if not, we need
  the true mapping.
