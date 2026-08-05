#!/usr/bin/env python3
"""Evaluate simple verifier outputs and select one dev winner per axis."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

from simple_verifier_common import VARIANT_KEYS, read_jsonl, write_json


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def macro_f1(gold: list[int], pred: list[int]) -> float:
    scores = []
    for label in (0, 1, 2):
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision, recall = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
        scores.append(safe_div(2 * precision * recall, precision + recall))
    return sum(scores) / 3


def quadratic_weighted_kappa(gold: list[int], pred: list[int]) -> float:
    n = len(gold)
    if not n:
        return 0.0
    observed = [[0] * 3 for _ in range(3)]
    gh, ph = [0] * 3, [0] * 3
    for g, p in zip(gold, pred):
        observed[g][p] += 1
        gh[g] += 1
        ph[p] += 1
    num = den = 0.0
    for i in range(3):
        for j in range(3):
            weight = ((i - j) / 2) ** 2
            num += weight * observed[i][j] / n
            den += weight * (gh[i] * ph[j]) / (n * n)
    return 1.0 - safe_div(num, den) if den else 0.0


def evaluate(rows: list[dict], gold_by_id: dict[str, int], scope: set[str]) -> tuple[dict, list[dict]]:
    by_id = {r["sample_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("judged output contains duplicate sample_id values")
    extra = set(by_id) - scope
    if extra:
        raise ValueError(f"judged output contains {len(extra)} samples outside requested split")

    gold, pred, failures = [], [], []
    for sid in sorted(scope):
        row = by_id.get(sid)
        if row is None or row.get("judge_failure") or row.get("score") not in (0, 1, 2):
            effective = 1  # operationally safe: a failed verifier must route to review
            failures.append({"sample_id": sid, "kind": "missing" if row is None else "judge_failure",
                             "record": row})
        else:
            effective = row["score"]
        gold.append(gold_by_id[sid])
        pred.append(effective)

    matrix = [[0] * 3 for _ in range(3)]
    for g, p in zip(gold, pred):
        matrix[g][p] += 1
    n_bad = sum(g == 0 for g in gold)
    n_accept = sum(p == 2 for p in pred)
    unsafe = sum(g == 0 and p == 2 for g, p in zip(gold, pred))
    strict_bad = sum(g == 0 and p == 0 for g, p in zip(gold, pred))
    true_good_accept = sum(g == 2 and p == 2 for g, p in zip(gold, pred))
    metrics = {
        "n": len(gold), "judge_failures": len(failures),
        "judge_failure_rate": safe_div(len(failures), len(gold)),
        "confusion_gold_rows_pred_columns": matrix,
        "gold_distribution": dict(sorted(collections.Counter(gold).items())),
        "prediction_distribution": dict(sorted(collections.Counter(pred).items())),
        "unsafe_accept_count": unsafe,
        "unsafe_accept_rate_among_gold_bad": safe_div(unsafe, n_bad),
        "strict_bad_recall": safe_div(strict_bad, n_bad),
        "auto_accept_count": n_accept,
        "auto_accept_precision_gold2": safe_div(true_good_accept, n_accept),
        "review_rate": safe_div(sum(p == 1 for p in pred), len(pred)),
        "macro_f1": macro_f1(gold, pred),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(gold, pred),
    }
    metrics["passes_targets"] = (
        metrics["strict_bad_recall"] >= 0.90
        and metrics["auto_accept_precision_gold2"] >= 0.98
        and metrics["auto_accept_count"] > 0
        and metrics["prediction_distribution"] != {1: len(pred)}
    )
    return metrics, failures


def candidate_key(metrics: dict) -> tuple:
    # A candidate with broken I/O or no auto-accept is not eligible to become a winner.
    ineligible = (metrics["judge_failure_rate"] > 0.01
                  or metrics["auto_accept_count"] == 0
                  or metrics["prediction_distribution"] == {1: metrics["n"]})
    return (
        int(ineligible),
        metrics["unsafe_accept_rate_among_gold_bad"],
        -metrics["strict_bad_recall"],
        -metrics["auto_accept_precision_gold2"],
        -metrics["macro_f1"],
        -metrics["quadratic_weighted_kappa"],
        metrics["review_rate"],
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--judged", action="append", required=True,
                   help="judged.jsonl; repeat for each candidate")
    p.add_argument("--gold-metadata", required=True)
    p.add_argument("--gold-heatmap", required=True)
    p.add_argument("--split-file", required=True)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--out", required=True)
    p.add_argument("--select-winners", action="store_true")
    args = p.parse_args()
    if args.select_winners and args.split != "dev":
        raise ValueError("winners may only be selected on the dev split")

    split_rows = read_jsonl(args.split_file)
    scope = {r["sample_id"] for r in split_rows if r["split"] == args.split}
    meta_gold = {r["sample_id"]: r["overall_score"] for r in read_jsonl(args.gold_metadata)}
    heat_gold = {r["sample_id"]: r["overall_score"] for r in read_jsonl(args.gold_heatmap)}
    if not scope or not scope <= set(meta_gold) or not scope <= set(heat_gold):
        raise ValueError("split and gold files do not cover the same samples")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reports, failures_all = {}, []
    seen_candidates = set()
    for path in args.judged:
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"empty judged output: {path}")
        axis, variant = rows[0].get("axis"), rows[0].get("variant")
        if axis not in ("stage1", "stage2") or variant not in VARIANT_KEYS:
            raise ValueError(f"invalid axis/variant in {path}")
        if any(r.get("axis") != axis or r.get("variant") != variant for r in rows):
            raise ValueError(f"mixed axis/variant records in {path}")
        key = f"{axis}:{variant}"
        if key in seen_candidates:
            raise ValueError(f"duplicate candidate {key}")
        seen_candidates.add(key)
        gold = meta_gold if axis == "stage1" else heat_gold
        metrics, failures = evaluate(rows, gold, scope)
        reports[key] = {"axis": axis, "variant": variant, "judged_path": path, **metrics}
        failures_all.extend({"candidate": key, **r} for r in failures)

    winners = {}
    if args.select_winners:
        for axis in ("stage1", "stage2"):
            candidates = [r for r in reports.values() if r["axis"] == axis]
            variants = {r["variant"] for r in candidates}
            if not variants <= set(VARIANT_KEYS) or len(variants) < 2:
                raise ValueError(f"winner selection needs >=2 known variants for {axis}, "
                                 f"got {sorted(variants)}")
            winner = min(candidates, key=candidate_key)
            if candidate_key(winner)[0]:
                raise RuntimeError(f"all {axis} candidates are ineligible")
            winners[axis] = {
                "variant": winner["variant"], "selected_on": "dev",
                "selection_order": ["unsafe_accept_rate", "strict_bad_recall",
                                    "auto_accept_precision", "macro_f1", "qwk", "review_rate"],
                "metrics": {k: v for k, v in winner.items()
                            if k not in ("axis", "variant", "judged_path")},
            }
        write_json(out / "winners.json", winners)

    report = {"split": args.split, "scope_size": len(scope), "candidates": reports,
              "winners": winners}
    write_json(out / "metrics.json", report)
    with open(out / "failures.jsonl", "w", encoding="utf-8") as f:
        for row in failures_all:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [f"# Simple verifier report ({args.split})", ""]
    for key in sorted(reports):
        r = reports[key]
        lines += [f"## {key}", "",
                  f"- n={r['n']}, failures={r['judge_failures']}",
                  f"- unsafe accept={r['unsafe_accept_count']} "
                  f"({r['unsafe_accept_rate_among_gold_bad']:.3f} of gold bad)",
                  f"- strict bad recall={r['strict_bad_recall']:.3f}",
                  f"- auto-accept precision={r['auto_accept_precision_gold2']:.3f} "
                  f"at n={r['auto_accept_count']}",
                  f"- review rate={r['review_rate']:.3f}",
                  f"- macro-F1={r['macro_f1']:.3f}, QWK={r['quadratic_weighted_kappa']:.3f}",
                  f"- confusion (gold rows, prediction columns): `{r['confusion_gold_rows_pred_columns']}`",
                  ""]
    if winners:
        lines += ["## Selected winners", ""] + [f"- {a}: {v['variant']}" for a, v in winners.items()]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

