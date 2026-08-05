#!/usr/bin/env python3
"""CPU-only tests for the simple verifier benchmark."""

import json
import tempfile
import unittest
from pathlib import Path

import judge_stage1_simple as s1
import judge_stage2_simple as s2
from evaluate_simple_verifier import evaluate
from simple_verifier_common import (MCQ_GOOD, MCQ_OK, VARIANT_KEYS, _parse, cascade_score, common_parser,
                                    mcq_schema, mcq_score, read_jsonl, validate_locked_test)


ROOT = Path(__file__).resolve().parents[1] / "quality_evaluation/claude_pilot_1000"
BENCH = ROOT / "simple_verifier_1000"


class SimpleVerifierTests(unittest.TestCase):
    def test_parse_and_cascade_mapping(self):
        self.assertEqual(_parse('{"score":2,"reason":"ok"}', "direct")["score"], 2)
        self.assertEqual(cascade_score({"clear_bad": True, "reason": "bad"}, None)[0], 0)
        self.assertEqual(cascade_score({"clear_bad": False, "reason": "ok"},
                                       {"fully_good": False, "reason": "minor"})[0], 1)
        self.assertEqual(cascade_score({"clear_bad": False, "reason": "ok"},
                                       {"fully_good": True, "reason": "good"})[0], 2)

    def test_minimal_prompt_inputs(self):
        row = read_jsonl(ROOT / "manifest.jsonl")[0]
        one = s1.build_item(row).user_text.lower()
        two = s2.build_item(row).user_text.lower()
        for forbidden in ("coverage", "component", "mechanism", "confidence", "category"):
            self.assertNotIn(forbidden, one)
        for forbidden in ("function:", "action:", "coverage", "coordination:"):
            self.assertNotIn(forbidden, two)

    def test_benchmark_split(self):
        rows = read_jsonl(BENCH / "benchmark_split.jsonl")
        self.assertEqual(len(rows), 1000)
        self.assertEqual(len({r["sample_id"] for r in rows}), 1000)
        dev = [r for r in rows if r["split"] == "dev"]
        test = [r for r in rows if r["split"] == "test"]
        self.assertEqual((len(dev), len(test)), (700, 300))
        self.assertFalse({r["object_id"] for r in dev} & {r["object_id"] for r in test})

    def test_gold_covers_the_manifest_on_both_axes(self):
        ids = {r["sample_id"] for r in read_jsonl(ROOT / "manifest.jsonl")}
        for name in ("metadata_labels.gold.jsonl", "heatmap_labels.gold.jsonl"):
            rows = read_jsonl(ROOT / name)
            self.assertEqual(len(rows), 1000, name)
            self.assertEqual({r["sample_id"] for r in rows}, ids, name)
            self.assertTrue(all(r["overall_score"] in (0, 1, 2) for r in rows), name)

    def test_mcq_is_one_call_and_maps_deterministically(self):
        # the argparse gate is what makes the branch live at all
        args = common_parser("stage1").parse_args(
            ["--manifest", "m", "--split-file", "s", "--variant", "mcq", "--out", "o"])
        self.assertEqual(args.variant, "mcq")
        self.assertEqual(VARIANT_KEYS["mcq"], ["mcq"], "mcq must be a single call")
        for axis in (s1, s2):
            self.assertTrue(set(VARIANT_KEYS["mcq"]) <= set(axis.PROMPTS), axis.AXIS)
            self.assertTrue(axis.FINDINGS)
            enum = mcq_schema(axis.FINDINGS)["properties"]["finding"]["enum"]
            self.assertEqual(enum, list(axis.FINDINGS) + [MCQ_OK, MCQ_GOOD])
            for answer in enum:            # every answer must appear in the prompt it is offered by
                self.assertIn(answer, axis.PROMPTS["mcq"], f"{axis.AXIS}: {answer} not offered")
        # one answer -> score AND error type
        self.assertEqual(mcq_score({"finding": "forces_cancel", "reason": "r"}),
                         (0, "r", "forces_cancel"))
        self.assertEqual(mcq_score({"finding": MCQ_OK, "reason": "loose"}), (1, "loose", "none"))
        self.assertEqual(mcq_score({"finding": MCQ_GOOD, "reason": "ok"}), (2, "ok", "none"))
        # an id outside the answer space must not parse
        self.assertEqual(_parse(f'{{"finding":"{MCQ_GOOD}","reason":"r"}}', "mcq",
                                s1.FINDINGS)["finding"], MCQ_GOOD)
        with self.assertRaises(ValueError):
            _parse('{"finding":"made_up","reason":"r"}', "mcq", s1.FINDINGS)

    def test_token_budget_is_generous_and_failures_are_loud(self):
        # a generation that hits max_tokens leaves unclosed JSON; that reads as a per-sample judge
        # failure and can void a whole run silently (v10: 662/700, v11: 696/700). The budget must be
        # ample and the runner must shout when the failure rate is non-trivial.
        args = common_parser("stage1").parse_args(
            ["--manifest", "m", "--split-file", "s", "--variant", "multi", "--out", "o"])
        self.assertGreaterEqual(args.max_tokens, 512)
        src = (Path(__file__).parent / "simple_verifier_common.py").read_text()
        self.assertIn("NOT usable", src)
        self.assertIn('"judge_failures"', src)

    def test_mcq_findings_cover_every_fatal_gold_fault(self):
        # pass 1 only decides "is this a 0", so the invariant is over score-0 rows. The heatmap gold
        # also tags score-1 rows (faint_or_drifting, borderline breadth); mcq reports those as
        # `none` by design, so tag agreement is only measurable on the 0s.
        for name, axis in (("metadata_labels.gold.jsonl", s1), ("heatmap_labels.gold.jsonl", s2)):
            rows = read_jsonl(ROOT / name)
            fatal = {r["fault"] for r in rows if r["overall_score"] == 0} - {"none"}
            self.assertTrue(fatal <= set(axis.FINDINGS),
                            f"{axis.AXIS}: gold scores 0 with faults the mcq answer space lacks: "
                            f"{fatal - set(axis.FINDINGS)}")
            self.assertTrue(fatal, f"{axis.AXIS}: no fatal faults found in gold")

    def test_locked_test_requires_matching_winner(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "winners.json"
            p.write_text(json.dumps({"stage1": {"variant": "cascade"}}))
            validate_locked_test("stage1", "cascade", "test", str(p))
            with self.assertRaises(ValueError):
                validate_locked_test("stage1", "direct", "test", str(p))
            with self.assertRaises(ValueError):
                validate_locked_test("stage2", "direct", "test", str(p))

    def test_failure_routes_to_review(self):
        scope = {"a", "b", "c"}
        gold = {"a": 0, "b": 1, "c": 2}
        rows = [
            {"sample_id": "a", "score": 0, "judge_failure": False},
            {"sample_id": "b", "score": None, "judge_failure": True},
            {"sample_id": "c", "score": 2, "judge_failure": False},
        ]
        metrics, failures = evaluate(rows, gold, scope)
        self.assertEqual(metrics["prediction_distribution"], {0: 1, 1: 1, 2: 1})
        self.assertEqual(len(failures), 1)


if __name__ == "__main__":
    unittest.main()

