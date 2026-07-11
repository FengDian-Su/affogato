"""Molmo2-8B pointing on vLLM — the stage2 (pipeline/stage2_v2.py) pointing engine.

Validated 2026-07-11 on the 10-object human-GT benchmark (scratchpad
vllm_m2_point.py / vllm_m2_score.py): k=1 parity with transformers 0.847-vs-0.844
meanAUC; k=4 multi-image 0.827/0.842; 96-113 ms/view = ~15x transformers serial.

Everything official:
- engine: vLLM 0.16 native Molmo2 support (`molmo` conda env, torch 2.9.1+cu128).
  Two required settings on THIS box (shared server, driver 570 = CUDA 12.8, no
  upgrade possible): VLLM_WORKER_MULTIPROC_METHOD=spawn (CUDA can't fork; same as
  bimanual_annotation/gemma.py) and mm_encoder_attn_backend="TORCH_SDPA" (vLLM's
  bundled flash-attn kernels raise cudaErrorUnsupportedPtxVersion on sm_120 with
  this driver — verified on 0.16/cu128 AND 0.22-nightly/cu129).
- prompts: multi-image question from the official training template
  GENERAL_PROMPTS_V1["multi_image_pointing"] ("Point to {label} in all images.");
  single-image = the stage1 molmo_query verbatim.
- parsing: point_formatter_official.py (verbatim copy of AI2 molmo2 repo
  olmo/preprocessing/point_formatter.py) — coords="idx x y" triplets, /1000
  scale, 1-based image indices.
"""
import os
import re
import sys
import types
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_ID = "allenai/Molmo2-8B"


def load_official_extractor():
    """Import the vendored AI2 point_formatter with its two olmo deps stubbed
    (parse_timestamp is only used for video timestamps; PointTrack is a TypedDict)."""
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


def load_engine(gpu_memory_utilization=0.5, max_images=4, max_model_len=8192):
    """Start the vLLM engine + processor. Call ONCE per process, before SAM2."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    llm = LLM(model=MODEL_ID, trust_remote_code=True,
              gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len,
              limit_mm_per_prompt={"image": max_images, "video": 0},
              mm_encoder_attn_backend="TORCH_SDPA")
    return llm, proc


def render_prompt(proc, question, n_images):
    msgs = [{"role": "user", "content": [dict(type="text", text=question),
                                         *[dict(type="image") for _ in range(n_images)]]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def multi_image_question(molmo_query):
    """Official multi-image template around a stage1 'Point to ...' query."""
    label = re.sub(r"^Point to\s+", "", molmo_query.strip()).rstrip(". ")
    return f"Point to {label} in all images."


def ground_queries(llm, proc, view_images, molmo_queries, k=4, max_tokens=512):
    """Point every query over all views in ONE llm.generate (continuous batching).

    view_images: list[PIL.Image] (equal sizes)
    molmo_queries: flat list[str] of stage1-style "Point to ..." queries (the
        caller passes every query x both roles of one object, in order)
    Returns: list (per query) of per-view [x, y] | None, plus n_oob (points
    whose 1-based image index fell outside the chunk). A generation that hits
    max_tokens is truncated mid-coords and parses to ZERO points for that
    chunk — that silent failure mode is detected via finish_reason and warned
    about loudly instead.
    """
    from vllm import SamplingParams
    T = len(view_images)
    W, H = view_images[0].size
    reqs, keys = [], []
    for qi, q in enumerate(molmo_queries):
        if k == 1:
            prompt = render_prompt(proc, q, 1)
            for vi in range(T):
                reqs.append({"prompt": prompt, "multi_modal_data": {"image": [view_images[vi]]}})
                keys.append((qi, vi, 1))
        else:
            question = multi_image_question(q)
            for s in range(0, T, k):
                chunk = view_images[s:s + k]
                reqs.append({"prompt": render_prompt(proc, question, len(chunk)),
                             "multi_modal_data": {"image": chunk}})
                keys.append((qi, s, len(chunk)))
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=max_tokens),
                        use_tqdm=False)
    per_query = [[None] * T for _ in molmo_queries]
    n_oob = n_trunc = 0
    for (qi, s, klen), out in zip(keys, outs):
        if out.outputs[0].finish_reason == "length":
            n_trunc += 1
        txt = out.outputs[0].text
        if klen == 1:
            pts = _UPF.extract_points(txt, W, H)
            if pts:
                per_query[qi][s] = [float(pts[0][0]), float(pts[0][1])]
        else:
            for ix, x, y in _UPF.extract_multi_image_points(txt, W, H):
                ix = int(ix)
                if 1 <= ix <= klen:               # official 1-based image index
                    gi = s + ix - 1
                    if per_query[qi][gi] is None:
                        per_query[qi][gi] = [float(x), float(y)]
                else:
                    n_oob += 1
    if n_trunc:
        print(f"WARNING molmo2_vllm: {n_trunc}/{len(reqs)} generations hit "
              f"max_tokens={max_tokens} and were truncated — their chunks "
              f"likely lost all points (unparseable coords tail)", flush=True)
    return per_query, n_oob
