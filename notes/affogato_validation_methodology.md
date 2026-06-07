# AFFOGATO Reproduction-Fidelity Validation — Methodology

> Purpose: validate that our re-implemented **Molmo → SAM2 → multi-view-voting**
> single-region affordance pipeline reproduces the original `project-affogato/affogato`
> per-point heatmaps. `VERIFIED` = confirmed against primary sources (arXiv 2506.12009,
> HF, benchmark papers) or local data. `UNVERIFIED` = flagged inline.
>
> Research compiled 2026-06-04 (fan-out web research + adversarial verification).

---

## A. AFFOGATO paper: identity, metrics, GT definition

**Identity (VERIFIED)**
- Title: *"Affogato: Learning Open-Vocabulary Affordance Grounding with Automated Data Generation at Scale"*
- Authors: Junha Lee, Eunha Park, Chunghyun Park, Dahyun Kang, Minsu Cho (POSTECH, RLWRLD)
- arXiv **2506.12009** (v1, 13 Jun 2025) — [abs](https://arxiv.org/abs/2506.12009) · [html](https://arxiv.org/html/2506.12009v1) · [pdf](https://arxiv.org/pdf/2506.12009)
- Project page: https://junha-l.github.io/affogato/ · HF dataset: https://huggingface.co/datasets/project-affogato/affogato
- **No code release** — project page links `github.com/junha-l/affordance_segmentation` but it 404s. Reimplement from paper text only.

**Evaluation metrics (VERIFIED, §5.1)** — paper: *"For 3D evaluation, following prior work [12], we use average Intersection over Union (aIoU), Area Under the ROC Curve (AUC), Similarity (SIM), and Mean Absolute Error (MAE)... we primarily report aIoU, AUC, and SIM."*
- **3D: aIoU↑, AUC↑, SIM↑ (primary) + MAE↓ (de-emphasized** — paper calls MAE *"sensitive to annotation scale and less reliable for fair comparison"*).
- 2D (not used for 3D-point comparison): KLD↓, SIM↑, NSS↑.
- No explicit formulas in-paper; 3D set inherited from **LASO** (CVPR 2024) ← **IAGNet** (ICCV 2023). aIoU threshold-sweep originates in **3D AffordanceNet** (CVPR 2021): sweep threshold 0→0.99 step 0.01, average IoU; GT binarized at 0.5.

**GT heatmap definition (VERIFIED, §3.2 "Affogato-Engine")** — 3 stages:
1. **Queries**: Gemma3 generates 5 free-form NL affordance queries/object from G-Objaverse renders.
2. **Points**: Molmo (`allenai/Molmo-7B-D-0924`) grounds each query → a pixel coordinate/view.
3. **Mask→heatmap→3D**: point prompts a segmenter; *"segmentation logits → sigmoid → probabilistic heatmaps in [0,1]"*; per-view heatmaps *"projected onto the 3D surface using camera params + depth"* with *"voting-based aggregation."*

⇒ **GT = sigmoid(mask logits) → [0,1]/view → camera/depth reprojection → multi-view voting.** Values are **soft probabilities in [0,1], not binary.** Counts (VERIFIED): 5 queries/obj · 750,520 pairs · 150,104 objects · **16,384 points/object** · partial = 2,048 FPS points. Matches local `xyzc.npy` (16384, 8) float16.

**Divergences original ↔ our reproduction — flag in the writeup:**
- **Segmenter**: original appendix A.1 = **MobileSAM** (main text says "SAM"); ours = **SAM2**.
- **Heatmap shaping**: original = **raw sigmoid, no Gaussian**; ours multiplies by a **Gaussian kernel** around the Molmo point (`use_gaussian`).
- **Point model**: original `allenai/Molmo-7B-D-0924`; ours `allenai/MolmoPoint-8B`.
- **(UNVERIFIED)** whether original applies per-object renormalization after voting — assume scales may differ (drives metric choice).

---

## B. Recommended metric set (this fidelity check)

Constraint: P (repro) and G (GT) share the **same 16,384 points** but may differ in **absolute scale / nonlinear shaping**. ⇒ **scale-invariance is the top criterion.** Treat the **5 queries as 5 independent maps** (cols 3–7); compute every metric **per (object, query)** then aggregate.

### Primary (report all four)
1. **Spearman ρ** — invariant to ANY monotonic rescaling (the exact failure mode). Pearson on ranks; **use tie-corrected estimator** (huge blocks of exact-0 points). [−1,1]↑, no normalization.
2. **CC (Pearson)** — `cov(P,G)/(σ_P σ_G)`. [−1,1]↑. Invariant to affine `aP+b, a>0`. High ρ + low CC ⇒ nonlinear scale mismatch.
3. **AUC (Judd & Borji)** — treat P as classifier of binarized G (binarize **G at 0.5**); ROC area. [0,1]↑, 0.5=chance. Invariant to monotonic transforms of P. Judd≈Borji is a sanity check.
4. **SIM** — `Σ min(P_i,G_i)` after **normalizing each map to sum=1** (divide by sum, NOT min-max). [0,1]↑. Invariant to positive global scalar; sensitive to spread. One of AFFOGATO's primary 3.

### Secondary (report, don't rank fidelity on)
5. **aIoU** — sweep t=0→0.99 step 0.01, GT binarized at 0.5; `IoU(t)=TP/(TP+FP+FN)`. AFFOGATO primary metric BUT same numeric t on P ⇒ **confounded by scale**; **min-max normalize P (and ideally G) first** and say so. [0,1]↑.
6. **MAE/RMSE** — absolute-calibration only; no invariance, looks bad even with perfect localization. Compute **after** removing best least-squares affine P→G. AFFOGATO itself demotes MAE.

### Avoid as primary
- **KLD** explodes on zero/unseen points (intrinsic to voting). **NSS** unbounded/binary-GT/uncommon. **sAUC** meaningless per-object.

### Reading the pattern (more diagnostic than any single number)
- High ρ, low CC → right ordering, wrong nonlinear scaling.
- High CC, low SIM → right linear scaling, wrong spatial spread.
- **Expected signature of success** given our divergences (SAM2 vs MobileSAM, +Gaussian): **moderate-but-not-perfect ρ/CC with high AUC** = "same localization, different score shaping" = reproduction success, not failure.

---

## C. Methodology cautions specific to this setup

**(i) Re-run with AFFOGATO's own 5 queries — NOT the bimanual role queries.** Hold *content* constant: read the 5 queries from each object's `queries.json`, in GT order, and generate channel i from the same prompt as GT col 3+i. → done by [single_region_affordance.py](../data_generation/single_region_affordance.py).

**(ii) Unseen/never-projected points get a structural 0 — handle explicitly.** Voting assigns 0 to points not visible in any view, while GT may be >0 there. These (a) explode KLD, (b) bias uncorrected Spearman via tied ranks, (c) inflate low-threshold IoU. **Decide + report**: (a) all 16,384 points, or (b) **coverage-conditioned** on the visible subset. **Report coverage (% points seen by ≥1 view) with every metric.** With 24 views, observed coverage ≈ 64% on a sample object → consider more views. (Coverage-conditioning is sound but UNVERIFIED as a benchmark convention.)

**(iii) Per-query channel↔query alignment.** `xyzc.npy` cols 3–7 are the 5 heatmaps in the **same order** as `queries.json`. Compare channel i ↔ GT col 3+i one-to-one. **Never concatenate the 5 maps before correlating.** Break out per query/class.

**Normalization discipline (document per metric):** SIM/KLD → sum-to-1; CC/Spearman/AUC → none; aIoU/MAE → explicit min-max or affine alignment.

---

## D. HuggingFace download (only if a fresh/full copy is wanted)

repo_id `project-affogato/affogato`, repo_type **dataset**, public. 16 parts `affogato_all_part000…015.tar.gz` (000–014 ≈2.02–2.03 GB / 10,000 objects each; 015 ≈20.3 MB / ~100). Total ≈30.4 GB. **Viewer is broken** (`SplitsNotFoundError`) → do NOT use `load_dataset`/streaming; download raw tarballs + extract.

```bash
# one part (subset)
hf download project-affogato/affogato --repo-type dataset \
  --include "affogato_all_part000.tar.gz" --local-dir /home/michaellee/mclee/affogato/dataset
# full: drop --include   (Python equiv: huggingface_hub.snapshot_download with allow_patterns)
cd /home/michaellee/mclee/affogato/dataset && mkdir -p affogato
tar -xzf affogato_all_part000.tar.gz -C affogato/   # -> affogato/affogato_all_part000/{object_id}/{xyzc.npy,queries.json}
```

**This machine:** all 16 parts are **already extracted** at `/home/michaellee/mclee/affogato/dataset/affogato/` (raw `.tar.gz` deleted; a fresh `snapshot_download` would re-pull them). For validation just use the extracted tree — no re-download needed.

---

**Sources:** arXiv:2506.12009 · LASO (CVPR 2024) · IAGNet (arXiv:2303.10437) · 3D AffordanceNet (arXiv:2103.16397) · Bylinskii et al. TPAMI 2019 (arXiv:1604.03605) · Tübingen saliency benchmark · HF dataset page.

**Top UNVERIFIED flags:** (1) exact in-paper metric formulas (deferred to LASO/IAGNet field-standard forms); (2) whether GT is per-object renormalized after voting; (3) Spearman + coverage-conditioning are general diagnostics, not benchmark protocol; (4) original = MobileSAM + raw sigmoid (no Gaussian), differs from our SAM2 + Gaussian.
