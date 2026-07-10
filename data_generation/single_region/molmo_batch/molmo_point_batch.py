"""Batched multi-view pointing with allenai/MolmoPoint-8B.

One self-contained reference module consolidating the 2026-07-10 batch-bug
verification suite (step0_exoneration / step2_parity / step3_speed) and the
k-chunk x B-batch stacking test (k4xb_test). Intended to be lifted directly
into the affogato repo ("backport") — it has no imports from the scratchpad
scripts and no repo-specific loaders: callers pass plain PIL images.

Background
----------
Upstream ``modeling_molmo_point.py`` (snapshot 188130f9...) is B=1-only: six
distinct bugs make ``generate()`` with batch size > 1 either crash with shape
errors or silently corrupt scores across samples (see PATCH_README.md for a
hunk-by-hunk explanation). This module manages a 6-hunk patch to the HF
modules-cache copy of that file and provides a verified batched inference
path on top of it.

The scheme: "k-chunk x B-batch"
-------------------------------
* Split the T views of an object into chunks of ``k`` images; each chunk
  becomes ONE multi-image sample ("Point to X" + k images in one prompt).
* Stack ``B`` chunk-samples into one batched ``generate()`` call
  (this is what needs the patch when B > 1).
* Decode each sample's text against that sample's OWN per-sample pointing
  metadata (obtained via a B=1 processor call — the batched processor call
  returns one flat metadata blob that cannot be split per sample).

Measured numbers (GPU3 = 95-GiB RTX PRO 6000 Blackwell, mm env,
transformers 4.57.1, bf16, greedy; affogato mug renders)
----------------------------------------------------------------
k=1 (one image per sample, 8 views, max_new_tokens=200):
  8 x B=1 sequential : 11.03 s  (1.38 s/view)
  1 x B=8  batched   :  4.19 s  (0.52 s/view, 2.64x)   peak alloc 39.3 GiB
  1 x B=16 batched   :  7.99 s  (0.50 s/view, plateau) peak alloc 46.2 GiB
k=4 (four images per sample, 40 views = 10 chunk-prompts, max_new_tokens=512):
  B=1 : 0.527 s/view  peak 50.7 GiB
  B=2 : 0.339 s/view  peak 51.7 GiB
  B=4 : 0.264 s/view  peak 55.6 GiB   <- best verified working combination
  B=8 : OOM while sharing the card with a ~30-GiB neighbour process
        -> B=8 at k=4 needs the GPU exclusively.

Parity (patched code, greedy):
  * B=1 pre-patch vs post-patch: byte-identical text and points, 8/8 views.
  * k=1, B=8 vs B=1: 8/8 views exact (both canonical input forms).
  * k=4, B=2/B=4 vs B=1: hit-sets identical; 30/32 resp. 29/32 views exact.
    The 2-3 differing views are rare bf16 tie-flips (batch padding changes
    the reduction order, so near-tied patch argmaxes can flip); they are
    legitimate alternative answers, not cross-sample corruption.
  * "hit" semantics at k>1: with multi-image chunk prompts the model may
    return no point for views where the target is not visible (32/40 on the
    mug). Serial and batched agree on WHICH views get points.

Known-unfixed (do not use through this module):
  * the video pointing path (``video_token_pooling`` branch) — untouched;
  * ``num_return_sequences > 1`` / beam search — the point-embedding cache
    and logit processor assume exactly one sequence per sample.

Env contract (inherited from the test suite): set CUDA_VISIBLE_DEVICES /
CUDA_DEVICE_ORDER before importing torch. This module deliberately does NOT
mutate the environment.

Typical use::

    import molmo_point_batch as mpb
    mpb.ensure_batch_patch()                    # idempotent, do this first
    model, processor = mpb.load_official()
    pts = mpb.point_batch(views, "Point to the handle of the mug.",
                          k=4, batch=4)         # -> [(x, y) | None] per view
    report = mpb.parity_check(views, question, k=4, batch=4,
                              save_path="parity.json")
"""
from __future__ import annotations

import glob
import inspect
import json
import os
import py_compile
import shutil
import sys
import time

MODEL_ID = "allenai/MolmoPoint-8B"
MODULE_BASENAME = "modeling_molmo_point.py"

# Every hunk inserts a comment containing this string; a fully patched file
# contains it exactly N_HUNKS times. This is the idempotency sentinel.
PATCH_SENTINEL = "FIX (batch>1)"
N_HUNKS = 6

# --------------------------------------------------------------------------
# The 6 hunks (single source of truth; byte-equivalent to
# all_changes_vs_upstream.diff, applied as exact old-block -> new-block
# replacements so partial states are handled and drift fails loudly).
# --------------------------------------------------------------------------
HUNKS = [
    (
        "1_rotate_by_broadcast",
        """\
                rotate_by = image_pos_ids[batch_idx, last_predicted_patch_id]
                rotate_by = torch.where(last_predicted_patch_id >= 0, rotate_by, 0)
                rotate_by = rotate_by.squeeze(-1)
""",
        """\
                # FIX (batch>1): last_predicted_patch_id is [B, 1]; indexing with
                # ([B]-batch_idx, [B,1]) broadcasts to [B, B]. Flatten to [B] first
                # (same pattern as `last_predicted_patch_id.view(batch_size)` in
                # MolmoPointModel.forward). -1 sentinel semantics preserved: index
                # -1 wraps, then torch.where replaces those rows with 0, exactly as
                # before. Result is [B], matching the old post-squeeze(-1) shape.
                last_patch_id_flat = last_predicted_patch_id.view(batch_size)
                rotate_by = image_pos_ids[batch_idx, last_patch_id_flat]
                rotate_by = torch.where(last_patch_id_flat >= 0, rotate_by, 0)
""",
    ),
    (
        "2_should_embed_tensor_and",
        """\
            should_embed = (input_patch_ids >= 0) and (input_patch_ids < (bounds.patch_end-1))
            input_patch_ids_flat = (input_patch_ids + image_token_offset).view(-1)[should_embed.view(-1)]
""",
        """\
            # FIX (batch>1):
            #   1) `and` was Python bool-AND between two tensors → only worked at B=1
            #   2) image_token_offset has shape [B]; adding to input_patch_ids [B,T] broadcasts
            #      to [B,B] (wrong) instead of [B,T]. Need [:, None] for seq-dim broadcast.
            should_embed = (input_patch_ids >= 0) & (input_patch_ids < (bounds.patch_end - 1))
            input_patch_ids_flat = (input_patch_ids + image_token_offset[:, None]).view(-1)[should_embed.view(-1)]
""",
    ),
    (
        "3_subpatch_feature_pairing",
        """\
                for_patches = (last_predicted_patch_id.view(batch_size) + image_token_offset)[input_subpatch_ids.view(batch_size) >= 0]
                vit_features_to_embed = vit_features_flat[for_patches, input_subpatch_ids]
""",
        """\
                # FIX (batch>1): indexing [K]-shaped `for_patches` with [B,1]-shaped
                # `input_subpatch_ids` broadcasts to [B,K] / [B,B]. Pair the K selected
                # rows with their own subpatch ids ([K] with [K]) instead. B=1 (K=1)
                # values unchanged (old [1,1,D] vs new [1,D] assign identically).
                sub_sel = input_subpatch_ids.view(batch_size)
                for_patches = (last_predicted_patch_id.view(batch_size) + image_token_offset)[sub_sel >= 0]
                vit_features_to_embed = vit_features_flat[for_patches, sub_sel[sub_sel >= 0]]
""",
    ),
    (
        "4_logit_processor_flat_pooling",
        """\
    def build_logit_processor_from_inputs(self, inputs) -> LogitsProcessorList:
        if inputs.get("image_token_pooling") is not None:
            pooling = inputs["image_token_pooling"]
        elif inputs.get("video_token_pooling") is not None:
""",
        """\
    def build_logit_processor_from_inputs(self, inputs) -> LogitsProcessorList:
        if inputs.get("image_token_pooling") is not None:
            pooling = inputs["image_token_pooling"]
            # FIX (batch>1): the processor emits a FLAT pooling [sum_patches_over_batch, C],
            # but the model's extended token-id space (see build_token_bounds use at
            # decode time) is laid out with the per-example MAX-PADDED pooling
            # [B, P_max, C] from build_batched_images. At B=1 sum == P_max so both
            # agree; at B>1 the flat pooling shifts every bound, so the logit
            # processor silently masks the wrong score ranges. Rebuild the batched
            # pooling here (pure indexing, no model weights involved).
            if inputs.get("input_ids") is not None and inputs.get("pixel_values") is not None:
                _, pooling = self.model.build_batched_images(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    image_token_pooling=pooling,
                    image_grids=inputs["image_grids"],
                    image_num_crops=inputs["image_num_crops"],
                )
        elif inputs.get("video_token_pooling") is not None:
""",
    ),
    (
        "5_prefill_pooling_source",
        """\
        bs, seq, _ = logits.shape
        if image_data is not None:
            token_pooling = image_data.token_pooling
        else:
            token_pooling = video_token_pooling if video_token_pooling is not None else image_token_pooling
""",
        """\
        bs, seq, _ = logits.shape
        if image_data is not None:
            token_pooling = image_data.token_pooling
        elif outputs.image_data is not None:
            # FIX (batch>1): on the prefill step (no image_data kwarg yet) the flat
            # processor pooling [sum_patches_over_batch, C] gives n_patches summed
            # over the batch, while every decode step uses the max-padded
            # image_data.token_pooling [B, P_max, C]. Use the cache built by this
            # very forward pass so prefill and decode agree on the extended-vocab
            # layout. At B=1 the shapes coincide, so behavior is unchanged.
            token_pooling = outputs.image_data.token_pooling
        else:
            token_pooling = video_token_pooling if video_token_pooling is not None else image_token_pooling
""",
    ),
    (
        "6_argmax_patch_scatter",
        """\
            argmax_patch_logits[batch_idx.view(-1, 1, 1), seq_ix.view(1, -1, 1), selected_patches] = patch_token_logits
""",
        """\
            # FIX (batch>1): `selected_patches` is [B, S]; right-aligned it broadcasts
            # against ([B,1,1], [1,S,1]) to [B, B, 1] at S=1, writing every sample's
            # score into every other sample's argmax column. Make it an explicit
            # per-sample [B, S, 1] scatter. B=1, S=1: identical single write.
            argmax_patch_logits[batch_idx.view(-1, 1, 1), seq_ix.view(1, -1, 1), selected_patches[:, :, None]] = patch_token_logits[:, :, None]
""",
    ),
]


# ==========================================================================
# Patch management
# ==========================================================================
def _hf_modules_root() -> str:
    """The HF dynamic-modules cache root (mirrors transformers' defaults)."""
    if os.environ.get("HF_MODULES_CACHE"):
        return os.environ["HF_MODULES_CACHE"]
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "huggingface",
    )
    return os.path.join(hf_home, "modules")


def find_module_files(model_id: str = MODEL_ID) -> list:
    """All modules-cache copies of modeling_molmo_point.py for ``model_id``.

    trust_remote_code copies the model's .py files out of the hub snapshot
    into ``<HF modules cache>/transformers_modules/<org>/<name>/<rev>/``;
    THAT copy is what gets imported, so that is what must carry the patch.
    Newer transformers sanitize repo names ("MolmoPoint-8B" ->
    "MolmoPoint_hyphen_8B"), so we glob over both spellings and over all
    cached revisions. Returns [] if the cache has never been populated.
    """
    org, name = model_id.split("/", 1)
    root = os.path.join(_hf_modules_root(), "transformers_modules", org)
    hits = []
    for variant in {name, name.replace("-", "_hyphen_")}:
        hits += glob.glob(os.path.join(root, variant, "*", MODULE_BASENAME))
        hits += glob.glob(os.path.join(root, variant, MODULE_BASENAME))
    return sorted(set(hits))


def apply_hunks_to_text(text: str):
    """Apply the 6 hunks to ``text``; return (new_text, applied, skipped).

    Per hunk: if the NEW block is already present -> skip (idempotent); else
    the OLD block must occur exactly once -> replace; anything else raises
    RuntimeError (i.e. a different model revision: re-derive the patch from
    PATCH_README.md / all_changes_vs_upstream.diff, do NOT batch unpatched).
    """
    applied, skipped = [], []
    for name, old, new in HUNKS:
        if new in text:
            skipped.append(name)
            continue
        n = text.count(old)
        if n == 1:
            text = text.replace(old, new)
            applied.append(name)
        elif n == 0:
            raise RuntimeError(
                f"patch hunk '{name}': neither the patched nor the upstream "
                f"code block was found. The model code has probably changed "
                f"(new snapshot/revision?). Refusing to guess — re-derive the "
                f"patch (see PATCH_README.md) before running batch > 1."
            )
        else:
            raise RuntimeError(
                f"patch hunk '{name}': upstream block found {n} times, "
                f"expected exactly 1. Refusing to patch ambiguously."
            )
    return text, applied, skipped


def _purge_pycache(py_file: str) -> None:
    """Drop stale bytecode next to a just-patched source file."""
    stem = os.path.splitext(os.path.basename(py_file))[0]
    cache_dir = os.path.join(os.path.dirname(py_file), "__pycache__")
    for pyc in glob.glob(os.path.join(cache_dir, stem + "*.pyc")):
        try:
            os.remove(pyc)
        except OSError:
            pass


def _already_imported_module_names() -> list:
    return [
        n for n in sys.modules
        if n.startswith("transformers_modules") and "molmo_point" in n.lower()
    ]


def patch_file(path: str, verbose: bool = True) -> dict:
    """Patch one modules-cache file in place (idempotent).

    Creates ``<path>.orig`` on first modification (never overwrites an
    existing .orig), purges stale __pycache__ bytecode, and syntax-checks the
    result. Raises if the source was already imported into this process (the
    in-memory class would silently stay unpatched — restart instead).
    Returns {"path", "applied", "skipped", "sentinels", "backed_up"}.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    new_text, applied, skipped = apply_hunks_to_text(text)

    backed_up = False
    if applied:  # something actually changes on disk
        imported = _already_imported_module_names()
        if imported:
            raise RuntimeError(
                f"{path} needs patching but its module is already imported "
                f"in this process ({imported}); the loaded class would remain "
                f"unpatched. Restart the process and call ensure_batch_patch() "
                f"BEFORE loading the model."
            )
        orig = path + ".orig"
        if not os.path.exists(orig):
            shutil.copy2(path, orig)
            backed_up = True
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_text)
        _purge_pycache(path)
        py_compile.compile(path, doraise=True)  # syntax gate

    sentinels = new_text.count(PATCH_SENTINEL)
    if sentinels != N_HUNKS:
        raise RuntimeError(
            f"{path}: expected {N_HUNKS} '{PATCH_SENTINEL}' sentinels after "
            f"patching, found {sentinels}."
        )
    if verbose:
        state = "already patched" if not applied else (
            f"applied {len(applied)} hunk(s) ({', '.join(applied)})"
            + (f", {len(skipped)} pre-existing" if skipped else "")
        )
        print(f"[molmo_point_batch] {path}: {state} "
              f"[{sentinels}/{N_HUNKS} sentinels]")
    return {"path": path, "applied": applied, "skipped": skipped,
            "sentinels": sentinels, "backed_up": backed_up}


def ensure_batch_patch(model_id: str = MODEL_ID, populate: bool = True,
                       verbose: bool = True) -> list:
    """Idempotently ensure the 6-hunk batch fix is present. Call this FIRST.

    Checks every modules-cache copy of modeling_molmo_point.py for the
    ``PATCH_SENTINEL`` comment (x6 = fully patched); applies missing hunks
    programmatically otherwise. Handles the historical half-patched state
    too (a May backup carrying only hunk 2 = 1 sentinel).

    If the modules cache has never been populated and ``populate`` is True,
    the model CODE (not weights) is fetched via transformers'
    ``get_cached_module_file`` — a small CPU-only download — and then
    patched. With ``populate=False`` it raises with instructions instead.

    NEVER lets B>1 run unpatched silently: every failure mode raises.
    Returns the list of per-file reports from :func:`patch_file`.
    """
    paths = find_module_files(model_id)
    if not paths and populate:
        if verbose:
            print(f"[molmo_point_batch] modules cache empty for {model_id}; "
                  f"fetching model code (no weights) ...")
        try:
            from transformers.dynamic_module_utils import get_cached_module_file
            get_cached_module_file(model_id, MODULE_BASENAME)
        except Exception as e:  # noqa: BLE001 - loud, actionable failure
            raise RuntimeError(
                f"could not populate the HF modules cache for {model_id} "
                f"({type(e).__name__}: {e}). Populate it manually — e.g. "
                f"AutoConfig.from_pretrained('{model_id}', "
                f"trust_remote_code=True) — then rerun ensure_batch_patch() "
                f"in a FRESH process before any model load."
            ) from e
        paths = find_module_files(model_id)
    if not paths:
        raise RuntimeError(
            f"no {MODULE_BASENAME} found under "
            f"{os.path.join(_hf_modules_root(), 'transformers_modules')} for "
            f"{model_id}. Populate the modules cache (load the config/processor "
            f"once with trust_remote_code=True, or rerun with populate=True), "
            f"then call ensure_batch_patch() again. Do NOT run batch>1 until "
            f"this succeeds."
        )
    return [patch_file(p, verbose=verbose) for p in paths]


def assert_model_patched(model) -> None:
    """Raise unless the source file the loaded class came from is patched.

    This closes the remaining gap: even if the on-disk cache is patched, a
    stale import (or a transformers re-copy of the pristine snapshot during
    ``from_pretrained``) could leave an unpatched class in memory. We check
    the file the class was ACTUALLY imported from.
    """
    src = inspect.getfile(type(model))
    with open(src, encoding="utf-8") as f:
        n = f.read().count(PATCH_SENTINEL)
    if n != N_HUNKS:
        raise RuntimeError(
            f"model class at {src} carries {n}/{N_HUNKS} batch-fix sentinels "
            f"— batched generation (B>1) would crash or silently corrupt "
            f"samples. Run ensure_batch_patch() in a fresh process before "
            f"loading the model."
        )


# ==========================================================================
# Loading
# ==========================================================================
_MODEL_CACHE: dict = {}


def load_official(model_id: str = MODEL_ID, require_patch: bool = True,
                  cache: bool = True):
    """Load model + processor exactly per the official model card.

    (AutoProcessor with padding_side='left' — required so batched decode
    stays right-aligned — and AutoModelForImageTextToText with dtype='auto',
    device_map='auto'; verified on transformers 4.57.1.)

    With ``require_patch`` (default) this first runs :func:`ensure_batch_patch`
    and afterwards verifies via :func:`assert_model_patched` that the class
    that was really imported is the patched one — so a B>1 run can never
    silently use upstream code. Post-patch B=1 output is byte-identical to
    unpatched B=1 (verified over 8 views: same texts, same points).

    Returns (model, processor); memoized per model_id when ``cache``.
    """
    key = (model_id, require_patch)
    if cache and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if require_patch:
        ensure_batch_patch(model_id)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, padding_side="left"
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, trust_remote_code=True, dtype="auto", device_map="auto"
    )
    if require_patch:
        assert_model_patched(model)
    if cache:
        _MODEL_CACHE[key] = (model, processor)
    return model, processor


def _get_extract_fn(model):
    """Module-level extract_image_points from the actually-imported module."""
    mod = __import__(type(model).__module__, fromlist=["extract_image_points"])
    return mod.extract_image_points


# ==========================================================================
# Inference: k-chunk x B-batch pointing
# ==========================================================================
def _conversation(question, images):
    return [{
        "role": "user",
        "content": [{"type": "text", "text": question},
                    *[{"type": "image", "image": im} for im in images]],
    }]


def _chunk_metadata(processor, question, images):
    """Per-sample pointing metadata for one chunk via a B=1 processor call.

    The batched processor call returns a single FLAT metadata blob that
    cannot be split per sample, so extraction needs this per-chunk call
    (CPU-only tokenize/crop; cheap next to GPU decode time).
    """
    inputs = processor.apply_chat_template(
        _conversation(question, images), tokenize=True,
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
        padding=True, return_pointing_metadata=True,
    )
    return inputs.pop("metadata")


def _extract_chunk_points(extract_fn, model, text, metadata, base_index):
    """Top-1 point per view for one chunk: {global_view_idx: (x, y)}.

    ``extract_image_points`` yields points in decode (confidence) order, so
    the FIRST point emitted for an image is its top-1.
    """
    pts = extract_fn(
        output_text=text,
        pooling=metadata["token_pooling"],
        mappings=metadata["subpatch_mapping"],
        no_more_points_class=model.config.no_more_points_class,
        location=model.config.patch_location is not None,
        image_sizes=metadata["image_sizes"],
    )
    out = {}
    for (_cls, image_ix, x, y) in pts:
        gi = base_index + int(image_ix)
        if gi not in out:
            out[gi] = (float(x), float(y))
    return out


def _group_chunks(chunks, batch):
    """Group chunks for batching, never mixing chunk lengths in one batch.

    Only the tail chunk can be short (len(views) % k != 0); batching a short
    sample with full ones would break per-sample extraction against the
    max-padded batch layout, so it always runs in its own (B=1) group.
    """
    groups, cur = [], []
    for ch in chunks:
        if cur and (len(cur) == batch or len(ch[1]) != len(cur[-1][1])):
            groups.append(cur)
            cur = []
        cur.append(ch)
    if cur:
        groups.append(cur)
    return groups


def point_batch(views, question, k: int = 4, batch: int = 4,
                max_new_tokens: int = 512, model_id: str = MODEL_ID,
                model=None, processor=None, return_details: bool = False,
                checkpoint_path=None):
    """Point at ``question``'s target in every view; k-chunk x B-batch scheme.

    Args:
        views: list of same-size PIL images (equal sizes are REQUIRED —
            per-sample extraction of a padded batched decode is only valid
            when every sample has the same patch-pooling layout; asserted).
        question: e.g. "Point to the handle of the mug."
        k: images per chunk-prompt. k=1 reproduces plain per-view pointing.
        batch: chunk-prompts per generate() call. batch>1 REQUIRES the
            patched model; this is verified and never bypassed silently.
        max_new_tokens: decode budget. 512 for k=4 (multi-image answers list
            points for several views); the k=1 suite used 200.
        model_id / model / processor: pass an existing (model, processor)
            pair or let the module load-and-cache :func:`load_official`.
        return_details: also return per-chunk texts/timing.
        checkpoint_path: if set, resolved points are dumped to this JSON
            after EVERY completed group, so an OOM at a risky batch size can
            never lose earlier results (the k4xb prototype saved only at the
            very end and lost its whole sweep to a B=8 OOM; fixed here).

    Returns:
        ``points``: list aligned with ``views``; entry = (x, y) in original
        image pixels (top-1 = first decoded point for that view) or None if
        the model emitted no point for that view (legitimate at k>1 when the
        target is not visible in a view; 32/40 hits on the mug benchmark,
        hit-set identical between serial and batched).
        With ``return_details``: (points, details_dict).

    Throughput/VRAM (95-GiB card, measured): k=4xB=4 = 0.264 s/view at
    55.6 GiB peak (works next to a ~30-GiB neighbour); k=4xB=8 OOMs unless
    the GPU is exclusive; k=1xB=8 = 0.52 s/view at 39.3 GiB peak.

    On CUDA OOM mid-run, raises RuntimeError with ``.partial_points`` holding
    every view already resolved (a group is never lost retroactively).
    """
    if not views:
        return ([], {"hits": 0}) if return_details else []
    size0 = views[0].size
    assert all(im.size == size0 for im in views), (
        "point_batch requires equal-size views: per-sample metadata "
        f"extraction under batch padding is only valid then (got sizes "
        f"{sorted({im.size for im in views})})."
    )
    import torch

    if model is None or processor is None:
        model, processor = load_official(model_id)
    if batch > 1:
        assert_model_patched(model)  # NEVER silently run unpatched for B>1
    extract_fn = _get_extract_fn(model)

    chunks = [(s, views[s:s + k]) for s in range(0, len(views), k)]
    groups = _group_chunks(chunks, batch)

    points: dict = {}
    texts_by_chunk: dict = {}
    t0 = time.perf_counter()
    for gi, group in enumerate(groups):
        convs = [_conversation(question, imgs) for _, imgs in group]
        # apply_chat_template distinguishes one-conversation vs a list of
        # conversations by nesting; unwrap the singleton.
        inputs = processor.apply_chat_template(
            convs if len(convs) > 1 else convs[0], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
            return_dict=True, padding=True, return_pointing_metadata=True,
        )
        # The batched "metadata" is flat/unsplittable — drop it; extraction
        # uses per-chunk metadata below. But keep image_token_pooling etc.:
        # build_logit_processor_from_inputs needs them.
        inputs.pop("metadata", None)
        inputs = {kk: (v.to(model.device) if hasattr(v, "to") else v)
                  for kk, v in inputs.items()}
        try:
            lp = model.build_logit_processor_from_inputs(inputs)
            with torch.inference_mode():
                out = model.generate(
                    **inputs, logits_processor=lp,
                    max_new_tokens=max_new_tokens, do_sample=False,
                )
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            err = RuntimeError(
                f"CUDA OOM in group {gi + 1}/{len(groups)} "
                f"(B={len(group)}, k={k}); {len(points)} view(s) already "
                f"resolved (see .partial_points). Retry with a smaller "
                f"batch — measured: k=4xB=4 peaks at 55.6 GiB (fits beside "
                f"a ~30-GiB neighbour on a 95-GiB card); k=4xB=8 needs the "
                f"GPU exclusive."
            )
            err.partial_points = dict(points)
            raise err from e
        new_ids = out[:, inputs["input_ids"].shape[1]:]
        texts = processor.tokenizer.batch_decode(
            new_ids, skip_special_tokens=False)
        del inputs, out
        for (start, imgs), txt in zip(group, texts):
            md = _chunk_metadata(processor, question, imgs)
            points.update(_extract_chunk_points(extract_fn, model, txt, md, start))
            texts_by_chunk[start // k] = txt
        if checkpoint_path:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"question": question, "k": k, "batch": batch,
                           "groups_done": gi + 1, "groups_total": len(groups),
                           "points": {str(i): list(p)
                                      for i, p in sorted(points.items())}},
                          f, indent=1)
    dt = time.perf_counter() - t0

    result = [points.get(i) for i in range(len(views))]
    if not return_details:
        return result
    details = {
        "hits": len(points),
        "n_views": len(views),
        "k": k, "batch": batch, "max_new_tokens": max_new_tokens,
        "seconds": dt, "s_per_view": dt / len(views),
        "texts_by_chunk": texts_by_chunk,
    }
    return result, details


# ==========================================================================
# Parity gate
# ==========================================================================
def parity_check(views, question, k: int = 4, batch: int = 4,
                 max_new_tokens: int = 512, tol: float = 1e-3,
                 max_flip_fraction: float = 0.15, model_id: str = MODEL_ID,
                 model=None, processor=None, save_path=None,
                 verbose: bool = True) -> dict:
    """Serial (B=1) vs batched (B=``batch``) parity at chunk size ``k``.

    Runs the SAFE config first (serial reference) and — if ``save_path`` is
    given — persists the report to disk after every phase, so an OOM in the
    risky batched phase can never lose the baseline (the k4xb prototype lost
    its whole sweep to a B=8 OOM at the final json.dump; fixed here).

    Pass criterion (matches the measured behaviour of the patch):
      * batched phase completed, AND
      * hit-sets identical (the SAME views get points in both runs), AND
      * fraction of common views whose top-1 moved by more than ``tol``
        pixels is <= ``max_flip_fraction``.
    Greedy bf16 batching legitimately tie-flips a few near-tied argmaxes
    when padding changes reduction order — measured 2/32 (B=2) and 3/32
    (B=4) flipped views on the mug at k=4, hit-sets identical, everything
    else bit-exact. The 0.15 default is ~2x that measured rate; 0 flips is
    the norm at k=1 (8/8 exact at B=8).

    Returns a machine-readable dict:
      {"config": {...},
       "serial":  {"ok", "seconds", "s_per_view", "hits", "points"},
       "batched": {"ok", "seconds", "s_per_view", "hits", "points",
                   "peak_vram_gib" | "error"},
       "parity":  {"hit_sets_identical", "n_common", "n_exact",
                   "n_within_tol", "n_flips", "flip_fraction", "flips"},
       "pass": bool}
    """
    import torch

    if model is None or processor is None:
        model, processor = load_official(model_id)

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _save(rep):
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=1)

    def _phase(B):
        _sync()
        t0 = time.perf_counter()
        pts, det = point_batch(
            views, question, k=k, batch=B, max_new_tokens=max_new_tokens,
            model=model, processor=processor, return_details=True)
        _sync()
        dt = time.perf_counter() - t0
        return {
            "ok": True, "seconds": dt, "s_per_view": dt / len(views),
            "hits": det["hits"],
            "points": {str(i): list(p) for i, p in enumerate(pts)
                       if p is not None},
        }

    report = {
        "config": {"model_id": model_id, "k": k, "batch": batch,
                   "n_views": len(views), "max_new_tokens": max_new_tokens,
                   "tol": tol, "max_flip_fraction": max_flip_fraction},
        "serial": None, "batched": None, "parity": None, "pass": False,
    }

    # -- phase 1: safe serial reference; persisted before anything risky ----
    report["serial"] = _phase(1)
    _save(report)
    if verbose:
        s = report["serial"]
        print(f"[parity_check] serial  B=1: hits {s['hits']}/{len(views)}  "
              f"{s['s_per_view']:.3f} s/view")

    # -- phase 2: risky batched config --------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        report["batched"] = _phase(batch)
        if torch.cuda.is_available():
            report["batched"]["peak_vram_gib"] = (
                torch.cuda.max_memory_allocated() / 2**30)
    except RuntimeError as e:  # includes the OOM wrapper from point_batch
        report["batched"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        _save(report)
        if verbose:
            print(f"[parity_check] batched B={batch} FAILED: {e}")
        return report
    _save(report)
    if verbose:
        b = report["batched"]
        print(f"[parity_check] batched B={batch}: hits {b['hits']}/{len(views)}  "
              f"{b['s_per_view']:.3f} s/view  "
              f"peak {b.get('peak_vram_gib', float('nan')):.1f} GiB")

    # -- compare -------------------------------------------------------------
    sp, bp = report["serial"]["points"], report["batched"]["points"]
    common = sorted(set(sp) & set(bp), key=int)
    flips, n_exact, n_within = [], 0, 0
    for v in common:
        d = max(abs(sp[v][0] - bp[v][0]), abs(sp[v][1] - bp[v][1]))
        if d == 0.0:
            n_exact += 1
        if d <= tol:
            n_within += 1
        else:
            flips.append({"view": int(v), "serial": sp[v], "batched": bp[v],
                          "dist": d})
    hit_sets_identical = set(sp) == set(bp)
    flip_fraction = (len(flips) / len(common)) if common else 0.0
    report["parity"] = {
        "hit_sets_identical": hit_sets_identical,
        "n_common": len(common), "n_exact": n_exact,
        "n_within_tol": n_within, "n_flips": len(flips),
        "flip_fraction": flip_fraction, "flips": flips,
    }
    report["pass"] = bool(
        hit_sets_identical and flip_fraction <= max_flip_fraction)
    _save(report)
    if verbose:
        p = report["parity"]
        print(f"[parity_check] parity: hit_sets_identical={hit_sets_identical} "
              f"exact {n_exact}/{len(common)} flips {len(flips)} "
              f"(<= {max_flip_fraction:.0%} allowed) -> "
              f"{'PASS' if report['pass'] else 'FAIL'}")
    return report


# ==========================================================================
# Status CLI (read-only; use apply_patch.py to write)
# ==========================================================================
if __name__ == "__main__":
    mid = sys.argv[1] if len(sys.argv) > 1 else MODEL_ID
    files = find_module_files(mid)
    if not files:
        print(f"no modules-cache copy of {MODULE_BASENAME} for {mid} under "
              f"{_hf_modules_root()} (cache not populated yet)")
        sys.exit(1)
    ok = True
    for p in files:
        with open(p, encoding="utf-8") as f:
            n = f.read().count(PATCH_SENTINEL)
        state = "PATCHED" if n == N_HUNKS else ("PARTIAL" if n else "UNPATCHED")
        ok &= n == N_HUNKS
        print(f"{state:9s} [{n}/{N_HUNKS} sentinels] {p}")
    sys.exit(0 if ok else 1)
