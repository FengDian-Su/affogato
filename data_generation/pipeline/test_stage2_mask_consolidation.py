"""CPU-only tests for the production conditional-mean rollout.

Run: python data_generation/pipeline/test_stage2_mask_consolidation.py -v
"""
import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import stage2_v2 as stage2
from single_region.mask_consolidation import conditional_mean_masks as mean
from single_region import single_region_affordance as sra


class ConditionalMeanTests(unittest.TestCase):
    def test_pixel_examples_and_strict_cutoff(self):
        for values, expected in (([.9, .7, 0], .8), ([.9, .05, .05], .9),
                                 ([.9, .15], .525), ([.9, .1], .9), ([.05], 0)):
            with self.subTest(values=values):
                masks = [np.full((1, 1), v, np.float32) for v in values]
                self.assertAlmostEqual(float(mean(masks, (1, 1))[0, 0]), expected, places=6)

    def test_empty_single_and_outputs_do_not_alias(self):
        a, b = mean([], (1, 3)), mean([], (1, 3))
        a[:] = 1
        self.assertFalse(b.any())
        x = np.array([[.05, .1, .9]], np.float32)
        out = mean([x], x.shape)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out, np.array([[0, 0, .9]], np.float32))
        out[:] = 0
        np.testing.assert_array_equal(x, np.array([[.05, .1, .9]], np.float32))

    def test_disjoint_instances_are_not_divided_by_k(self):
        masks = [np.array([[.9, 0]], np.float32), np.array([[0, .8]], np.float32)]
        np.testing.assert_allclose(mean(masks, (1, 2)), [[.9, .8]])

    def test_permutation_and_threshold_extremes(self):
        masks = np.array([[[.9, .05, .3]], [[.7, .8, .01]], [[.2, .6, .1]]], np.float32)
        for order in itertools.permutations(masks):
            np.testing.assert_allclose(mean(order, (1, 3)), [[.6, .7, .3]], rtol=1e-6)
        np.testing.assert_allclose(mean([np.array([[.8]]), np.zeros((1, 1))], (1, 1), tau=0), [[.8]])
        self.assertFalse(mean(masks, (1, 3), tau=1).any())

    def test_bit_exact_frozen_pilot_reference(self):
        from ablation.mask_consolidation import thresholded_pixel_mean
        rng = np.random.default_rng(20260913)
        for count in (0, 1, 2, 8):
            masks = rng.random((count, 16, 17), dtype=np.float32)
            for tau in (0, .1, .2, 1):
                np.testing.assert_array_equal(mean(masks, (16, 17), tau),
                                              thresholded_pixel_mean(masks, (16, 17), tau))

    def test_invalid_inputs(self):
        for shape in ((0, 2), (2,), (1.5, 2), (True, 2)):
            with self.assertRaises(ValueError):
                mean([], shape)
        for tau in (-1, 2, np.nan, np.inf):
            with self.assertRaises(ValueError):
                mean([], (1, 1), tau)
        for value in (-.1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                mean([np.array([[value]])], (1, 1))
        with self.assertRaises(ValueError):
            mean([np.zeros((2, 2))], (1, 1))


class SlotIntegrationTests(unittest.TestCase):
    def test_role_view_ownership_and_absent_slots(self):
        # Interleaved owners and deliberately nonzero dummy predictions expose
        # bugs that accidentally include padded slots or mix roles/views.
        masks = [[np.full((1, 1), v, np.float32) for v in row]
                 for row in ((.9, .05), (.7, 1), (.5, 1))]
        points = [[[[0, 0]], [[0, 0]]], [[[0, 0]], []], [[[0, 0]], []]]
        out = stage2.merge_sam_subqueries(masks, points, [0, 1, 0], 3, [(1, 1)] * 2)
        np.testing.assert_allclose(np.asarray(out).reshape(3, 2), [[.7, 0], [.7, 0], [0, 0]])
        out[2][0][:] = 1
        self.assertFalse(out[2][1].any())
        self.assertFalse(out[0][1].any())
        self.assertEqual(float(masks[2][1][0, 0]), 1)

    def test_legacy_mode_keeps_grouping(self):
        masks = [[np.full((1, 5), .9, np.float32)],
                 [np.array([[.9, .9, .9, .9, .05]], np.float32)]]
        pts = [[[[0, 0]]], [[[0, 0]]]]
        out = stage2.merge_sam_subqueries(masks, pts, [0, 0], 1, [(1, 5)], method="iou_group")
        self.assertAlmostEqual(float(out[0][0][0, -1]), .475, places=6)
        out[0][0][:] = 0
        self.assertAlmostEqual(float(masks[0][0][0, -1]), .9, places=6)

    def test_invalid_slot_layout(self):
        for heat, pts, owners in (([[]], [], [0]), ([[]], [[]], [1]), ([[]], [[]], [0])):
            with self.assertRaises(ValueError):
                stage2.merge_sam_subqueries(heat, pts, owners, 1, [(1, 1)])
        with self.assertRaises(ValueError):
            stage2.merge_sam_subqueries([], [], [], 0, [], method="fixed_k")


class ConfigurationTests(unittest.TestCase):
    def test_cli_defaults_and_overrides(self):
        args = stage2.parse_args([])
        self.assertEqual(args.mask_consolidation, "conditional_mean")
        self.assertEqual(args.mask_tau, .1)
        args = stage2.parse_args(["--mask_tau", ".2", "--mask_consolidation", "iou_group"])
        self.assertEqual(stage2.consolidation_settings(args.mask_consolidation, args.mask_tau),
                         dict(consolidation="iou_group", consolidation_threshold=None))
        for tau in ("nan", "inf", "-0.1", "1.1"):
            with self.subTest(tau=tau), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                stage2.parse_args(["--mask_tau", tau])

    def test_resume_only_matching_recipe_and_complete_files(self):
        with tempfile.TemporaryDirectory(prefix="stage2-resume-test-") as d:
            query = {"task": "hold object"}
            queries = [(0, query)]
            qdir = Path(d) / "q0_hold_object"
            qdir.mkdir()
            settings = stage2.consolidation_settings("conditional_mean", .1)
            self.assertFalse(stage2.existing_object_is_complete(d, queries, settings))
            (qdir / "meta.json").write_text(json.dumps({"engine": settings}))
            self.assertFalse(stage2.existing_object_is_complete(d, queries, settings))
            np.savez(qdir / "scores.npz", scoreA=np.zeros(1))
            self.assertTrue(stage2.existing_object_is_complete(d, queries, settings))
            with self.assertRaisesRegex(ValueError, "Existing consolidation differs"):
                stage2.existing_object_is_complete(d, queries, stage2.consolidation_settings("conditional_mean", .2))
            (qdir / "meta.json").write_text(json.dumps({"engine": {}}))
            with self.assertRaises(ValueError):
                stage2.existing_object_is_complete(d, queries, settings)
            self.assertTrue(stage2.existing_object_is_complete(d, queries, stage2.consolidation_settings("iou_group", .1)))
            # Missing first query must not bypass validation of later queries.
            with self.assertRaises(ValueError):
                stage2.existing_object_is_complete(d, [(1, {"task": "missing"}), *queries], settings)
            (qdir / "meta.json").write_text("invalid json")
            with self.assertRaisesRegex(ValueError, "Cannot verify"):
                stage2.existing_object_is_complete(d, queries, settings)

    def test_resume_mismatch_fails_before_loading_models(self):
        with tempfile.TemporaryDirectory(prefix="stage2-preflight-test-") as d:
            qdir = Path(d) / "obj/q0_hold"
            qdir.mkdir(parents=True)
            (qdir / "meta.json").write_text(json.dumps({"engine": {}}))
            source = Path(d) / "stage1.json"
            source.write_text(json.dumps([dict(object_id="obj", queries=[dict(task="hold", roles=[{}, {}], molmo_queries=["A", "B"])])]))
            args = stage2.parse_args(["--stage1", str(source), "--output_dir", d, "--skip_existing"])
            cwd = os.getcwd()
            try:
                with mock.patch.object(stage2, "parse_args", return_value=args), mock.patch.dict(os.environ), self.assertRaisesRegex(ValueError, "Existing consolidation differs"):
                    # The production engine is not installed in the test path;
                    # reaching its import would fail instead of this preflight.
                    stage2.main()
            finally:
                os.chdir(cwd)


class PipelineSmokeTests(unittest.TestCase):
    def test_main_routes_default_merger_and_saves_recipe(self):
        """Real main/merger/save with model and geometry boundaries mocked."""
        with tempfile.TemporaryDirectory(prefix="stage2-main-test-") as d:
            shape, views = (64, 64), 5
            roles = [dict(id=r, role="hold", target="body", contact_region="side") for r in "AB"]
            query = dict(task="hold object", roles=roles, molmo_queries=["point A", "point B"], category="intra")
            source = Path(d) / "stage1.json"
            source.write_text(json.dumps([dict(object_id="obj", object_name="Object", queries=[query])]))
            output = Path(d) / "output"
            args = stage2.parse_args(["--stage1", str(source), "--output_dir", str(output),
                                      "--num_views", str(views), "--sam_chunk", "2", "--overlap2d", "0"])
            points = [[[[0, 0], [10, 10]] for _ in range(views)], [[[50, 50]] for _ in range(views)]]
            scene = SimpleNamespace(view_images=[None] * views, view_images_np=[np.zeros((*shape, 3), np.uint8)] * views,
                                    n_views=views, cameras=[], K_list=[], depth_maps=[])
            canvas = SimpleNamespace(xyz=np.zeros((4, 3), np.float32), xyz_vote=np.zeros((4, 3), np.float32))
            engine = SimpleNamespace(load_engine=mock.Mock(return_value=(None, None)),
                                     ground_queries=mock.Mock(return_value=(points, [[1.] * views] * 2)))
            package = ModuleType("single_region.molmo2_vllm")
            package.engine = engine
            def sam(images, *args, **kwargs):
                return [[np.full(shape, v, np.float32) for _ in images] for v in (.9, .05, .7)]
            def vote(proj, hms, **kwargs):
                return np.full(4, np.mean([h[0, 0] for h in hms]), np.float32), np.ones(4, np.int32)
            patches = dict(load_sam2_model=mock.Mock(return_value=None), build_aff_map=mock.Mock(return_value={}),
                           resolve_object=mock.Mock(return_value=(d, d)), count_available_views=mock.Mock(return_value=views),
                           load_canvas=mock.Mock(return_value=canvas), load_scene=mock.Mock(return_value=scene),
                           run_sam2_object_queries=mock.Mock(side_effect=sam), precompute_projection=mock.Mock(return_value=[]),
                           knn_indices=mock.Mock(return_value=np.zeros((4, 1), int)), sample_heatmaps_projected=mock.Mock(side_effect=vote),
                           refine_scores=mock.Mock(side_effect=lambda s, idx: s.copy()),
                           partition_two_roles=mock.Mock(side_effect=lambda a, b, *args: (a.copy(), np.zeros_like(b))),
                           prune_partitioned=mock.Mock(side_effect=lambda s, ref, idx: s.copy()))
            cwd = os.getcwd()
            try:
                with mock.patch.dict(sys.modules, {"single_region.molmo2_vllm": package}), mock.patch.dict(os.environ), \
                     mock.patch.object(stage2, "parse_args", return_value=args), mock.patch.multiple(sra, **patches), \
                     contextlib.redirect_stdout(io.StringIO()):
                    stage2.main()
            finally:
                os.chdir(cwd)
            self.assertEqual(patches["run_sam2_object_queries"].call_count, 3)
            qdir = output / "obj/q0_hold_object"
            meta = json.loads((qdir / "meta.json").read_text())
            self.assertEqual(meta["engine"]["consolidation"], "conditional_mean")
            self.assertEqual(meta["engine"]["consolidation_threshold"], .1)
            with np.load(qdir / "scores.npz") as z:
                np.testing.assert_allclose(z["scoreA_raw"], .9)
                np.testing.assert_allclose(z["scoreB_raw"], .7)


if __name__ == "__main__":
    unittest.main()
