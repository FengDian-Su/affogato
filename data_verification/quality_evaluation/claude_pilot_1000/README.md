# Claude pilot annotation set (1,000 daily_used samples)

Migrated 2026-08-05 from `data_generation/outputs/stage2_eval/daily_used/`, where it was built
before this verification pipeline was noticed. It is kept as a **pilot annotation set**, which is
what `human_sample_1000/RUBRIC.md` permits: *"Assistant/LLM labels may be kept as pilot
annotations, but are not human gold labels."*

**This is NOT the official human sample.** The official frame is `../human_sample_1000/`
(500 daily_used + 500 electronics, stratified, `analysis_weight`, sha256). This one is a separate
uniform-random draw over daily_used only. The two frames share **exactly 1 sample_id**, so they are
statistically independent draws, not a re-annotation of the official frame.

## Files

| file | contents |
|---|---|
| `manifest.jsonl` | 1,000 rows: `sample_id`, stage2 `meta_path`/`scores_path` (all verified to exist), the 8 source views, and the render paths the labels were actually made from |
| `metadata_labels.claude_pilot.jsonl` | Task/role judgment, 0/1/2 |
| `heatmap_labels.claude_pilot.jsonl` | Heatmap judgment, 0/1/2 |
| `route_labels_axis1_bad.jsonl` | Root-cause labels for the 176 metadata-score-0 samples |
| `sample_summary.json` | distributions + manifest sha256 |

Score distribution: metadata 2/1/0 = 410 / 414 / **176**; heatmap 2/1/0 = 583 / 310 / **107**.
The two axes were scored independently and came out statistically independent (Cramér's V 0.074).

## Three caveats that limit how these labels may be used

**1. The scores are holistic, not per-axis.** The official schema wants
`task_score`, `bimanual_score`, `hand_A_score`, `hand_B_score` with
`overall_score = min(...)`. This pilot produced ONE integrated task/role score, so it is written to
`overall_score` and the four sub-scores are left `null`. They are not back-filled, because a single
holistic score cannot be decomposed after the fact. Same for the heatmap file: `overall_score` only,
with `hand_A_score` / `hand_B_score` / `dual_score` / `evaluable` all `null`.

**2. ~~The heatmap labels were made on NON-canonical renders.~~ RESOLVED 2026-08-05 — the protocol
these labels used IS now the canonical one.** Per the user's decision (spec §4 note **L4**),
`pipeline/render_eval.py` — **8** views (6 orbit azimuths + top-down + underside), threshold
**0.15**, orange = A / teal-green = B with dark = high on a light-grey object — replaces
`render.py`'s frozen 6-view τ=0.20 red/blue/purple protocol. So these 1,000 heatmap labels are
directly usable as the calibration gold for the Stage 2 judge.
The rows still carry `render_protocol: "stage2_eval-8view-tau0.15-nonstandard"` from before the
decision; read it as *the now-canonical protocol*, not as a warning.
⚠ The debt moved to the other side: the existing Stage 2 probe (29), judge smoke (24) and the 160
`corrupt_stage2.py` corruption judgments were all produced under the OLD protocol and must be
re-run before they can be compared with anything scored here. The Stage 2 prompt's "red = Hand A,
blue = Hand B" wording also has to change to orange/teal.

**3. Error tags are mapped, and one mapping is imprecise.** The pilot used its own 5-label route
taxonomy; `error_tags` holds the official names and `source_routes` keeps the original verbatim:

| pilot route | official tag | mapping |
|---|---|---|
| `invalid_task` | `implausible_task` | exact |
| `small_pickup_review` | `forced_bimanual` | exact |
| `hallucinated_component` | `contact_part_absent` | exact |
| `viewer_relative_contact` | `too_vague_to_verify` | close (a camera-relative region cannot be re-found from another view) |
| `role_physics` | `role_function_conflict` | **imprecise** — the pilot route spans both `role_function_conflict` and `role_task_conflict`; read `reason` to separate them |

Tag counts over the 176: role_function_conflict 113, forced_bimanual 59, contact_part_absent 54,
implausible_task 53, too_vague_to_verify 6.

## Why these labels are worth keeping

They are the only scored samples in the project drawn from **real** generation output rather than
synthetic corruptions, so they are directly usable for §9.1 judge-reliability numbers
(Auto-Accept Precision, Bad-Sample Recall) once a judge is run over the same 1,000 — subject to
caveat 2 for anything spatial.

They also independently reproduced the corruption test's central finding. Running Qwen3.5 and
Qwen3.6-35B-A3B as a blind root-cause router over the 176 bad samples: the model agreed the sample
was bad **176/176 (100%)**, but its root-cause labels matched only 24-26% exactly (Jaccard 0.56)
against a Claude-vs-Claude ceiling of 89% / 0.94, tagging `role_physics` on 175/176. That is the
same "detects but cannot localise" result as the 8B/30B spillover measurements in
`pipeline/README.md` §corruption — now confirmed on real failures, not injected ones. Evidence:
`data_generation/refinement/task_role_v1/outputs/route_agreement/`.
