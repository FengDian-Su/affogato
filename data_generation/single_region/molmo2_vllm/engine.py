"""Molmo2-8B pointing on vLLM — the stage2 (pipeline/stage2_v2.py) pointing engine.

Validated 2026-07-11 on the 10-object human-GT benchmark (scratchpad
vllm_m2_point.py / vllm_m2_score.py): parity with the transformers reference
(0.847 vs 0.844 meanAUC); 96-113 ms/view = ~15x transformers serial.

Everything official:
- engine: vLLM 0.16 native Molmo2 support (`molmo` conda env, torch 2.9.1+cu128).
  Two required settings on THIS box (shared server, driver 570 = CUDA 12.8, no
  upgrade possible): VLLM_WORKER_MULTIPROC_METHOD=spawn (CUDA can't fork; same as
  bimanual_annotation/gemma.py) and mm_encoder_attn_backend="TORCH_SDPA" (vLLM's
  bundled flash-attn kernels raise cudaErrorUnsupportedPtxVersion on sm_120 with
  this driver — verified on 0.16/cu128 AND 0.22-nightly/cu129).
- prompts: the stage1 molmo_query verbatim, one image per request
  (multi-image chunking retired 07-19, commit 3c500b3).
- parsing: point_formatter_official.py (verbatim copy of AI2 molmo2 repo
  olmo/preprocessing/point_formatter.py) — coords="idx x y" triplets, /1000
  scale, 1-based image indices. We call UnifiedPointFormatter.extract_points
  directly rather than the module-level extract_points wrapper: the wrapper
  falls back to LegacyPointFormatting, whose looser regexes match stray digits
  in prose, and Molmo2 only ever emits the unified format.
"""
import os
import sys
import types
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_ID = "allenai/Molmo2-8B"
DEDUP_PX = 3.0   # two points this close in one view are the same instance


def load_official_extractor():
    """Import the vendored AI2 point_formatter with its two olmo deps stubbed
    (parse_timestamp is only used for video timestamps; PointTrack is a
    TypedDict). Runs at import time and the stubs are process-global: nothing
    else in a stage2 process may import the real `olmo`."""
    for name, attrs in [("olmo", {}), ("olmo.util", {"parse_timestamp": lambda s: float(s)}),
                        ("olmo.data", {}), ("olmo.data.academic_video_track_datasets", {"PointTrack": dict})]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m
    spec = importlib.util.spec_from_file_location(
        "molmo2_point_formatter", os.path.join(HERE, "point_formatter_official.py"))
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)
    return pf


_pf = load_official_extractor()
_UPF = _pf.UnifiedPointFormatter()


def load_engine(gpu_memory_utilization=0.5, max_model_len=8192):
    """Start the vLLM engine + processor. Call ONCE per process, before SAM2."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    llm = LLM(model=MODEL_ID, trust_remote_code=True,
              gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len,
              limit_mm_per_prompt={"image": 1, "video": 0},
              mm_encoder_attn_backend="TORCH_SDPA")
    return llm, proc


def render_prompt(proc, question):
    msgs = [{"role": "user", "content": [dict(type="text", text=question), dict(type="image")]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def p_exist_from_logprobs(logprobs0):
    """P(target exists) from the position-0 alternatives of a pointing call.

    Under this prompt Molmo2 answers in exactly two shapes: '<points ...'
    (target present) or 'There are none.' (absent) — the first generated token
    IS the model's existence decision. Subset softmax between the two camps.
    Measured on 500 objects / 7968 requests: 0 non-standard openers, AUROC
    0.912 vs hallucinated points, monotonically calibrated (bucket [0.99,1] ->
    94% real), while coordinate-token logprobs carry no signal (AUROC 0.45).

    Returns None (= "no opinion", the caller then keeps the view) unless the
    ARGMAX token is itself in a camp. Without that guard the subset softmax
    would happily report 0.999 off a 1e-3-vs-1e-9 tail ratio while 98% of the
    real mass sat on some third opener — a fabricated high confidence in
    precisely the bucket the exist gate trusts most. The formatter also has
    count-first templates ('There are N <points...>') that token 0 cannot
    separate from 'There are none.'; none occurred in the 7968 measured
    requests, and the argmax guard does not catch them — if the template ever
    changes, re-measure rather than re-tune the threshold.
    """
    from math import exp

    def camp(lp):
        # normalize BPE piece markers so camp matching survives tokenizer variants
        t = lp.decoded_token.replace("Ġ", " ").replace("▁", " ").strip().lower()
        if t.startswith("<") and not t.startswith("<|"):
            return "pts"
        return "none" if t.startswith("there") else None

    top = max(logprobs0.values(), key=lambda lp: lp.logprob)
    if camp(top) is None:
        return None
    p_pts = p_none = 0.0
    for lp in logprobs0.values():
        c = camp(lp)
        if c == "pts":
            p_pts += exp(lp.logprob)
        elif c == "none":
            p_none += exp(lp.logprob)
    tot = p_pts + p_none
    return p_pts / tot if tot > 1e-6 else None


def ground_queries(llm, proc, view_images, molmo_queries, max_tokens=512):
    """Point every query over all views in ONE llm.generate (continuous batching).

    Single-image, view-by-view by design (07-19): every (query, view) is its
    own request — all of them submitted together, so vLLM batches the whole
    object concurrently. Per view Molmo either points (one point per visible
    instance, adaptive count, near-duplicates <3px dropped) or declines
    ("There are none."), and the first generated token carries its existence
    confidence. Multi-image chunking was removed: it forced points ("in all
    images": 64.6% vs 45.0% hallucinated-target point rate), correlated
    errors across views (defeating the 40-view vote), and had no per-view
    decision token.

    view_images: list[PIL.Image]
    molmo_queries: flat list[str] of stage1-style "Point to ..." queries (the
        caller passes every query x both roles of one object, in order)
    Returns: (per_query, p_exist) — per query: per-view LIST of [x, y]
    (possibly empty), and per-view P(exist) float | None. A generation that
    hits max_tokens parses to zero points; detected via finish_reason and
    warned about loudly.
    """
    from vllm import SamplingParams
    T = len(view_images)
    reqs = []
    for q in molmo_queries:
        prompt = render_prompt(proc, q)
        for vi in range(T):
            reqs.append({"prompt": prompt, "multi_modal_data": {"image": [view_images[vi]]}})
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=20),
                        use_tqdm=False)
    # vLLM returns outputs in request order; divmod below silently shifts EVERY
    # later (query, view) into the wrong query if that ever stops holding
    assert len(outs) == len(reqs), f"vLLM returned {len(outs)} of {len(reqs)} outputs"
    per_query = [[[] for _ in range(T)] for _ in molmo_queries]
    p_exist = [[None] * T for _ in molmo_queries]
    n_trunc = 0
    for i, out in enumerate(outs):
        qi, vi = divmod(i, T)
        o = out.outputs[0]
        if o.finish_reason == "length":
            n_trunc += 1
        if o.logprobs:
            p_exist[qi][vi] = p_exist_from_logprobs(o.logprobs[0])
        lst = per_query[qi][vi]
        W, H = view_images[vi].size   # the formatter scales normalized coords by THIS view's size
        for x, y in _UPF.extract_points(o.text, W, H):
            if not any((x - px) ** 2 + (y - py) ** 2 < DEDUP_PX ** 2 for px, py in lst):
                lst.append([float(x), float(y)])
    if n_trunc:
        print(f"WARNING molmo2_vllm: {n_trunc}/{len(reqs)} generations hit "
              f"max_tokens={max_tokens} and were truncated (zero points parsed)", flush=True)
    return per_query, p_exist
