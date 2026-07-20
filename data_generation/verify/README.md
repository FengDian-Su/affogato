# verify/ — systematic dataset verification suite

Reusable QC for stage0-format outputs (and the pattern generalizes to later stages).
Three layers, run in order; each layer's output feeds the next.

**One entry point for layers 2-3 + applying the result: `refine_stage0.py`**

```bash
python verify/refine_stage0.py screen --gpu 0     # both screens -> qc_flags.json
#   ... arbitrate qc_flags.json (Claude image review; see layer 3) -> junk list ...
python verify/refine_stage0.py apply --junk refinement_junk_list.json [--dry-run]
python verify/refine_stage0.py status | restore
```

`apply` edits the ORIGINAL `stage0_part{N}.json` in place: keep -> false plus
`skip_reason`/`qc_junk_type`/`qc_evidence`/`qc_note` on each removed record (nothing is deleted -
the audit trail stays), then regenerates every `.kept.json` so stage1 reads clean input. The
pre-apply files are copied to `pre_apply_backup/`; `restore` undoes it.

## 1. Structural check (deterministic, full coverage, seconds)

```bash
python verify/check_structural.py            # paths at top of file default to outputs/stage0/<category>
```

Scans EVERY record for: schema/field completeness, filter-dict shape, keep-logic coherence
(keep == filter-pass AND >=1 component), kept.json == keep-true subset, component text sanity
(empty names/interactions, dup names, extreme counts), skip/error record shape, id/slice alignment
with the mapping. Emits a flag file (`audit_flags.json`) + histograms. Anything non-minor here
blocks delivery.

## 2. Semantic judge (model screen, full coverage, GPU, ~1-2s/object)

```bash
python verify/judge_semantic.py --gpu 0 --parts 0-12          # default judge: Qwen3.5-35B-A3B
```

Every KEPT object's renders are re-examined by a judge from a DIFFERENT model family than the
filter (cross-family rule: a family must never verify its own decisions - gemma option exists for
ablation only). Checks: single-object, geometry integrity, identity-vs-name, two-hand
manipulability, component groundedness. Output `judge_qwen_part{N}.json` per part; resume-safe.
The judge is a SCREEN: tune/accept it for high junk-recall; precision is delegated to layer 3.
Long runs: use `outputs/stage0/<category>/run_judge_full.sh` (driver) + `monitor_judge.sh` (watchdog)
in tmux.

## 3. Calibration + arbitration (Claude)

```bash
python verify/calibrate.py --judge <judge output>             # score vs groundtruth/ labels
```

`groundtruth/claude_labels_0720.json`: 200 kept objects labeled ok/junk/borderline by a Claude
image-verified audit (21 agents, adversarial skeptic pass, 2026-07-20). Use it to qualify any new
judge/prompt before a full scan (gate: junk recall ~>=85%; false-flag rate only affects layer-3
volume). After a full scan, EVERY judge flag goes to Claude subagents for image-verified
arbitration (workflow pattern: reviewer per ~25 objects -> skeptic re-check of claimed defects),
plus a sample of judge-passed objects to measure the residual miss rate.

## Calibration history (stage0_full, 2026-07-20)

| judge | junk recall | ok false-flag | note |
|---|---|---|---|
| gemma-4-26B-A4B v1 (strict) | 91% | 38% | filter's own family - retired per policy |
| gemma-4-26B-A4B v2 (lenient) | 32% | 12% | over-corrected |
| gemma-4-26B-A4B v3 (unbundled) | 91% | 32% | prompt iteration converged; oscillates |
| Qwen3.5-35B-A3B v1 | 50% | 4% | high precision (69%), name-anchored misses |
| Qwen3.5-35B-A3B v2 (geometry-first) | 50% | 7% | same 11 misses - perception-level, not prompt |

Qwen's 11 misses were re-adjudicated by a Claude agent reading ALL 8 views: all 11 confirmed junk
(shell fragments/solid primitives/flat sprites read as functional from oblique views + name).
Qwen = high-precision LOW-RECALL screen; its flags are trustworthy, its passes are not clean.
Coverage gaps are filled by (a) deterministic image-stats detectors (flat/sliver, ground-plane
slab) and (b) direct Claude review of the high-junk-density strata (single-body keeps).
