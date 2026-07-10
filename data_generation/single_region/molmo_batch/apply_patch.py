#!/usr/bin/env python
"""Apply the MolmoPoint-8B 6-hunk batch fix to the HF modules cache.

Standalone CLI (ships next to molmo_point_batch.py, which is the single
source of truth for the hunks). Run this after any model re-download or on
a fresh machine, BEFORE the first batched (B>1) model load:

    python apply_patch.py                       # locate, back up, patch, verify
    python apply_patch.py --populate            # also fetch model CODE (no
                                                #   weights) if the modules
                                                #   cache is empty
    python apply_patch.py --path /path/to/modeling_molmo_point.py
    python apply_patch.py --skip-import-check   # skip the (heavy) torch/
                                                #   transformers import step

What it does per target file:
  1. backs the pristine file up to <file>.orig (never overwrites an
     existing .orig),
  2. applies the 6 hunks as exact old-block -> new-block replacements
     (idempotent: pre-applied hunks are skipped; unrecognized code fails
     loudly instead of guessing),
  3. purges stale __pycache__ bytecode and syntax-checks the result,
  4. verifies the sentinel comment "FIX (batch>1)" appears exactly 6 times,
  5. (unless --skip-import-check) imports the patched module through the
     transformers dynamic-module machinery — CPU-only, no weights, no GPU —
     and re-checks the sentinel in the imported module's source.

Exit code 0 = every located copy is patched and verified.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import molmo_point_batch as mpb
except ImportError as e:  # pragma: no cover - shipping error
    sys.exit(f"apply_patch.py must sit next to molmo_point_batch.py "
             f"(the hunk definitions live there): {e}")


def verify_import(path: str) -> str:
    """Import ``path`` via transformers' dynamic-module package and re-check
    the sentinel on the imported module. CPU-only; downloads nothing."""
    import transformers.dynamic_module_utils as dmu
    from transformers.utils import HF_MODULES_CACHE

    dmu.init_hf_modules()  # puts HF_MODULES_CACHE on sys.path (+ __init__.py)
    rel = os.path.relpath(os.path.abspath(path), HF_MODULES_CACHE)
    if rel.startswith(".."):
        return (f"SKIPPED (file is outside the transformers modules cache "
                f"{HF_MODULES_CACHE}; import check only supports cache copies)")
    dotted = rel[:-len(".py")].replace(os.sep, ".")
    mod = importlib.import_module(dotted)
    src = inspect.getsource(mod)
    n = src.count(mpb.PATCH_SENTINEL)
    if n != mpb.N_HUNKS:
        raise RuntimeError(
            f"imported {dotted} but found {n}/{mpb.N_HUNKS} sentinels in its "
            f"source — a stale copy was imported?")
    if not hasattr(mod, "extract_image_points"):
        raise RuntimeError(
            f"imported {dotted} lacks extract_image_points — wrong module?")
    return f"import OK ({dotted}, {n}/{mpb.N_HUNKS} sentinels)"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Apply the MolmoPoint-8B batch>1 fix (6 hunks) to the "
                    "HF modules cache.")
    ap.add_argument("--model-id", default=mpb.MODEL_ID,
                    help=f"HF repo id (default: {mpb.MODEL_ID})")
    ap.add_argument("--path", default=None,
                    help="explicit modeling_molmo_point.py to patch instead "
                         "of auto-locating the modules cache")
    ap.add_argument("--populate", action="store_true",
                    help="if the modules cache is empty, fetch the model "
                         "CODE (a few .py files, no weights) via "
                         "transformers.get_cached_module_file")
    ap.add_argument("--skip-import-check", action="store_true",
                    help="skip the final import verification (avoids "
                         "importing torch/transformers)")
    args = ap.parse_args()

    if args.path:
        targets = [args.path]
    else:
        targets = mpb.find_module_files(args.model_id)
        if not targets and args.populate:
            mpb.ensure_batch_patch(args.model_id, populate=True)
            targets = mpb.find_module_files(args.model_id)
        if not targets:
            print(f"ERROR: no {mpb.MODULE_BASENAME} in the modules cache for "
                  f"{args.model_id}.\nEither rerun with --populate, or load "
                  f"the processor/config once with trust_remote_code=True "
                  f"and rerun. Do NOT run batch>1 before patching.")
            return 1

    rc = 0
    for path in targets:
        try:
            rep = mpb.patch_file(path)  # prints its own status line
            if rep["backed_up"]:
                print(f"  backup written: {path}.orig")
        except (RuntimeError, OSError) as e:
            print(f"FAILED: {path}: {e}")
            rc = 1
            continue
        if not args.skip_import_check:
            try:
                print(f"  {verify_import(path)}")
            except Exception as e:  # noqa: BLE001 - report and fail the run
                print(f"FAILED import check: {path}: {type(e).__name__}: {e}")
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
