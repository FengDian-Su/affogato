# MolmoPoint-8B batch>1 patch — what it fixes and how to keep it applied

Status: **frozen 2026-07-10**, verified on snapshot `188130f961c8e0888a34e11121a1423c461a01ba`
of `allenai/MolmoPoint-8B`, transformers 4.57.1, torch/bf16, `mm` conda env,
95-GiB RTX PRO 6000 (Blackwell).

## The problem

Upstream `modeling_molmo_point.py` only works at batch size 1. With `B>1`,
`generate()` either crashes with shape/indexing errors or — worse — runs but
**silently writes one sample's patch scores into other samples' rows**.
Step 0 of the verification suite ("exoneration test") confirmed this is a
genuine upstream bug: both canonical input forms from the model card
(`apply_chat_template(convs, tokenize=True, ...)` and
`processor(text=[p]*B, images=[...])`) fail at B=2 with no local code involved.

## The 6 hunks

The full diff is in `all_changes_vs_upstream.diff` (this directory). Each hunk
inserts a comment containing the sentinel string **`FIX (batch>1)`**, so a
fully patched file contains it exactly **6** times — that is the idempotency /
verification check used everywhere. All hunks are provably identity at B=1
(verified: pre-patch vs post-patch B=1 outputs are byte-identical over 8 views).

Three bug families:

**A. `[B,1]` vs `[B]` broadcast bugs in the decode-time pointing machinery**

1. `1_rotate_by_broadcast` (~line 449, vision pointer head):
   `image_pos_ids[batch_idx, last_predicted_patch_id]` indexes a `[B]` tensor
   with a `[B,1]` tensor → broadcasts to `[B,B]`. Fixed by flattening
   `last_predicted_patch_id` to `[B]` first (the `-1` sentinel semantics are
   preserved: wrap-around index, then `torch.where(...>=0, ..., 0)`).
2. `3_subpatch_feature_pairing` (~line 1499, subpatch embedding):
   `vit_features_flat[for_patches, input_subpatch_ids]` pairs `[K]` rows with a
   `[B,1]` index → `[B,K]` garbage. Fixed by pairing each selected row with its
   own subpatch id (`[K]` with `[K]`).
3. `6_argmax_patch_scatter` (~line 1788, logits assembly): the scatter
   `argmax_patch_logits[bidx.view(-1,1,1), six.view(1,-1,1), selected_patches]`
   right-aligns `[B,S]` `selected_patches` to `[B,B,1]` at S=1, writing **every
   sample's score into every other sample's argmax column** (the silent
   cross-sample corruption). Fixed with an explicit per-sample `[B,S,1]` scatter.

**B. Tensor boolean bug**

4. `2_should_embed_tensor_and` (~line 1489, patch-token embedding):
   `(t1) and (t2)` is Python bool-AND between tensors (only defined for
   1-element tensors → only worked at B=1); plus `image_token_offset` `[B]`
   added to `[B,T]` without `[:, None]`. Fixed with `&` and an explicit
   seq-dim broadcast.

**C. Flat vs max-padded pooling disagreement (prefill vs decode vocab layout)**

The processor emits a **flat** pooling `[sum_patches_over_batch, C]`, but the
model's extended point-token id space is laid out per the **max-padded**
pooling `[B, P_max, C]` from `build_batched_images`. At B=1 the two coincide;
at B>1 every token bound shifts, so the logit processor masks the wrong score
ranges.

5. `4_logit_processor_flat_pooling` (~line 1636,
   `build_logit_processor_from_inputs`): rebuild the batched pooling from the
   inputs (pure indexing, no weights) before constructing token bounds.
6. `5_prefill_pooling_source` (~line 1765, `forward`): on the prefill step
   (no `image_data` kwarg yet) take `token_pooling` from
   `outputs.image_data` — the cache built by that same forward pass — so
   prefill and decode agree on the extended-vocab layout.

## Where the patch lives

`trust_remote_code=True` copies the model's code out of the hub snapshot into
the **dynamic-modules cache**, and *that copy* is what Python imports:

```
patched :  ~/.cache/huggingface/modules/transformers_modules/allenai/
           MolmoPoint_hyphen_8B/188130f961c8e0888a34e11121a1423c461a01ba/
           modeling_molmo_point.py            <- 6/6 sentinels
backup  :  same path + ".orig"                <- pre-patch state
pristine:  ~/.cache/huggingface/hub/models--allenai--MolmoPoint-8B/
           snapshots/188130.../modeling_molmo_point.py   <- untouched
```

(Newer transformers sanitize the repo name: `MolmoPoint-8B` →
`MolmoPoint_hyphen_8B`. The tooling here globs both spellings.)

Historical note: the current `.orig` is a May 2026 state that already carried
hunk 2 — it is *not* pristine upstream. The programmatic patcher handles any
partial state (pre-applied hunks are skipped, unknown code fails loudly).

## Re-applying after a re-download / on a fresh machine

A model re-download (or `HF_HOME` change, or a transformers upgrade that
re-copies the snapshot into the modules cache) silently restores the
**unpatched** file. Two equivalent remedies, run in a **fresh process before
the first model load**:

```bash
python apply_patch.py                # locate + backup + patch + import-verify
python apply_patch.py --populate    # fresh machine: also fetch model CODE
                                     # (a few .py files, no weights, CPU-only)
```

or from Python:

```python
import molmo_point_batch as mpb
mpb.ensure_batch_patch()             # idempotent; raises rather than guessing
```

`molmo_point_batch.load_official()` runs `ensure_batch_patch()` automatically
**and** re-checks the sentinel on the file the loaded class was actually
imported from (`assert_model_patched`), so a batched run can never silently
execute upstream code — every `point_batch(..., batch>1)` call re-asserts this.
If the hunks no longer match (a *different* model revision), the patcher
refuses to guess and points you back at this README + the diff.

Quick status check (read-only): `python molmo_point_batch.py` →
`PATCHED [6/6 sentinels] <path>`, exit 0.

## What is verified — and what is NOT fixed

Verified (see docstrings in `molmo_point_batch.py` for the full numbers):

* B=1 pre/post patch: byte-identical texts and points (8/8 mug views).
* k=1, B=8 vs B=1: 8/8 views exact, both canonical input forms; B=16 halves
  identical to each other and to B=8.
* k=4 chunks, B=2/B=4 vs B=1: hit-sets identical; 30/32 resp. 29/32 exact.
  The residual 2–3/32 differing views are **rare bf16 tie-flips** — batch
  padding changes reduction order, so near-tied patch argmaxes can flip to an
  equally valid point. This is expected noise, not corruption; the parity
  criterion used for these measurements was therefore "identical hit-sets +
  flip fraction ≤15% (~2x the measured rate)" rather than 100% bitwise
  equality. (The one-off `parity_check()` helper that encoded this gate was
  removed after the verification campaign; this section is its record.)

**Not fixed — do not use with B>1:**

* the **video pointing path** (`video_token_pooling` branch): untouched by all
  six hunks; it still hands the flat pooling straight to the logit processor.
* **`num_return_sequences > 1` / beam search**: the point-embedding cache and
  logit processor assume exactly one sequence per sample; expanding sequences
  per sample re-introduces the same index misalignments.

## Speed / VRAM guidance (measured, greedy, bf16)

| config              | s/view | peak alloc | note                              |
|---------------------|-------:|-----------:|-----------------------------------|
| k=1, 8 x B=1        |  1.38  |     —      | sequential baseline               |
| k=1, B=8            |  0.52  |  39.3 GiB  | 2.64x                             |
| k=1, B=16           |  0.50  |  46.2 GiB  | throughput plateau                |
| k=4, B=1            |  0.527 |  50.7 GiB  | chunking alone ≈ k=1 B=8          |
| k=4, B=2            |  0.339 |  51.7 GiB  |                                   |
| **k=4, B=4**        | **0.264** | **55.6 GiB** | **best verified combination** |
| k=4, B=8            |  OOM   |     —      | with a ~30-GiB neighbour process  |

Rule of thumb on a 95-GiB card: **B=4 (k=4) is safe on a shared GPU** (fits
beside a ~30-GiB neighbour); **B=8 (k=4) needs the GPU exclusively**.
`max_new_tokens`: 512 for k=4 (a multi-image answer lists points for several
views); 200 sufficed for k=1.

Other constraints baked into `point_batch()`:

* all views must be the **same size** (per-sample extraction against a padded
  batched decode is only valid when every sample shares the pooling layout);
* a short tail chunk (`len(views) % k != 0`) is never batched with full
  chunks — it runs in its own group;
* per-sample pointing metadata comes from per-chunk B=1 processor calls (the
  batched call returns one flat, unsplittable metadata blob).

## Files in this directory

| file                          | role                                                        |
|-------------------------------|-------------------------------------------------------------|
| `molmo_point_batch.py`        | the reference module: patch manager (`ensure_batch_patch`, `patch_file`, `assert_model_patched`), `load_official`, `point_batch` (k-chunk x B-batch); hunk definitions live here (single source of truth) |
| `apply_patch.py`              | standalone CLI: locate/populate cache, backup to `.orig`, patch, import-verify sentinel |
| `all_changes_vs_upstream.diff`| the raw 6-hunk diff (documentation copy; the module applies the same change programmatically) |
| `PATCH_README.md`             | this file                                                   |
