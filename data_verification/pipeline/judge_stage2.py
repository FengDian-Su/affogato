"""Verification Stage 2 -- Point-cloud Spatial Judge (README section 6).

Answers only: do the final heatmaps actually sit on the 3D regions the metadata
describes, and can the two together support the task. It never re-judges whether
the task itself is sensible (that is Stage 1's job) and is given no Stage 1
result, no Stage 0 warning and no raw score.

Six SEPARATE labelled combined-heatmap views per sample, rendered on the fly at
the frozen protocol (tau_vis = 0.20, shared by A and B, intensity = original
score) and deleted afterwards. No geometry sheet -- README L1.

Two modes:

  probe  Phase 2 prerequisite: can the model separate red from blue at all in a
         point-cloud render? Forced-choice "is red above or below blue", scored
         against the true A/B centroid heights, on samples whose separation is
         unambiguous. If this fails, every grounding verdict below is noise.

  judge  the real Stage 2 verdict.

    python judge_stage2.py probe --limit 20
    python judge_stage2.py judge --limit 20
"""
import argparse
import json
import os
import random
import shutil
import tempfile

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
# paddlex registers a vLLM plugin whose paddle import segfaults at engine init
os.environ.setdefault("VLLM_PLUGINS", "")

import numpy as np                                       # noqa: E402
from PIL import Image                                    # noqa: E402
from vllm import LLM, SamplingParams                     # noqa: E402
from vllm.sampling_params import GuidedDecodingParams    # noqa: E402
from transformers import AutoProcessor                   # noqa: E402

import common as C                                       # noqa: E402
import render as R                                       # noqa: E402

JUDGE_VERSION = "qwen3vl-spatial-v1.0"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
UP_AXIS = 1        # render.py permutes xyz[:,[0,2,1]], so ORIGINAL y is screen-up

HAND_TAGS = ["wrong_region", "over_expanded", "under_localized"]
DUAL_TAGS = ["ab_collapse", "only_A_valid", "only_B_valid", "only_one_correct",
             "spatially_incompatible"]

_sc = {"type": "integer", "enum": [0, 1, 2]}


def _block(tags):
    return {"type": "object", "additionalProperties": False,
            "properties": {"score": _sc,
                           "error_tags": {"type": "array",
                                          "items": {"type": "string", "enum": tags}},
                           "reason": {"type": "string"}},
            "required": ["score", "error_tags", "reason"]}


JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"hand_A_grounding": _block(HAND_TAGS),
                   "hand_B_grounding": _block(HAND_TAGS),
                   "dual_coordination": _block(DUAL_TAGS),
                   "evidence_views": {"type": "array", "items": {"type": "string"}},
                   "self_reported_confidence": {"type": "number"}},
    "required": ["hand_A_grounding", "hand_B_grounding", "dual_coordination",
                 "evidence_views", "self_reported_confidence"],
}

PROBE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"relation": {"type": "string",
                                "enum": ["red_above_blue", "blue_above_red", "same_height"]},
                   "reason": {"type": "string"}},
    "required": ["relation", "reason"],
}

PROBE_SYSTEM = """You are shown six rendered views of the same 3D point cloud, each labelled
with its viewpoint. All six use the same object orientation: the vertical direction in the
image is the object's own up direction.

Some points are coloured red, some blue, some purple (both), and the rest are grey.

Answer one question: taken over the whole object, is the red region higher up, lower down, or
at about the same height as the blue region? Use every view; ignore left/right and
front/back. Answer "same_height" only if you genuinely cannot separate them vertically."""

# Reasoning principles only -- no object- or part-specific examples, which would make the
# judge overfit to them (README section 6.4.1).
JUDGE_SYSTEM = """You verify where two hands' contact regions have been marked on a 3D object.

You see six rendered views of one point cloud, each labelled with its viewpoint. All six share
the same object, orientation, scale and colour scheme:
- RED   = the region marked for Hand A
- BLUE  = the region marked for Hand B
- PURPLE = points marked for both hands
- GREY  = not marked
Colour intensity follows the underlying score, so stronger colour means a stronger mark.

You are given the task and the metadata that describes what each hand is supposed to touch.
Judge whether the coloured regions match that description.

How to reason:
- Do not re-judge whether the task itself makes sense. Assume the task and the metadata are
  given; your question is only whether the marked regions are in the right places.
- Look at all six views before concluding. A region can look wrong from one viewpoint and
  correct from another, and parts of the object are hidden in any single view.
- For each hand, compare the coloured region against that hand's own contact_region and
  function: is it on the structure the metadata names, and is it somewhere that hand could
  actually apply what its function describes?
- Region SIZE is judged against how specific the metadata is, never against how many points
  are coloured. If the contact_region names a broad or general surface, a large marked area
  is appropriate. If the contact_region names a particular side, height, or local feature,
  then a region that spreads over unrelated surfaces is wrong even though each individual
  point might be touchable.
- under_localized is the opposite failure: the metadata describes a usable contact area but
  the marking has collapsed to a few scattered points that could not support the function.
- For the two hands together, ask whether these two specific regions could be held at the
  same time and would jointly produce what the task requires, and whether their arrangement
  matches the stated relation between the hands.
- If both hands are marked on essentially the same points, that is a collapse: the two hands
  have not actually been separated.

Scoring, for each hand: 2 = the region is on the right part and its extent fits the
description; 1 = partly right, or the boundary is imprecise, or the views are not sufficient
to confirm; 0 = clearly on the wrong part, or unable to support the metadata at all.
Dual coordination: 2 = the two regions jointly support the task; 1 = workable but awkward,
or one hand is questionable; 0 = they cannot jointly support the task, only one is usable, or
the two have collapsed onto each other.

Cite in evidence_views the viewpoint labels you actually relied on. self_reported_confidence
is recorded for analysis only and gates nothing."""


def meta_block(meta):
    a, b = meta["roles"][0], meta["roles"][1]

    def hand(h):
        return ("  role: %s\n  target: %s\n  contact_region: %s\n  function: %s"
                % (h.get("role"), h.get("target"), h.get("contact_region"), h.get("function")))
    return "\n".join([
        "OBJECT: %s" % meta.get("object_name"),
        "TASK: %s" % meta.get("task"),
        "QUERY: %s" % meta.get("query"),
        "HAND A (red)", hand(a), "HAND B (blue)", hand(b),
        "coordination: %s" % meta.get("coordination"),
        "relation between the hands: %s" % meta.get("relation"),
        "symmetric: %s" % meta.get("symmetric"),
        "hand_agnostic: %s" % meta.get("hand_agnostic"),
        "", "Judge the marked regions and return the required JSON.",
    ])


def build_prompt(processor, system, imgs, text):
    msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": ([{"type": "image"} for _ in imgs]
                                         + [{"type": "text", "text": text}])}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def pick_samples(dataset, n, seed, min_sep=0.0):
    """Present samples with usable A/B heatmaps; min_sep filters on vertical separation
    relative to object height, used by the probe to keep ground truth unambiguous."""
    samples, _ = C.expected_samples(C.load_stage1(dataset), C.STAGE2_ROOT.format(ds=dataset))
    present = [s for s in samples if s["state"] == "present"]
    random.Random(seed).shuffle(present)
    out = []
    for sm in present:
        if len(out) >= n:
            break
        try:
            with np.load(os.path.join(sm["qdir"], "scores.npz"), allow_pickle=False) as z:
                xyz, a, b = z["xyz"], z["scoreA"], z["scoreB"]
                ma, mb = a >= R.TAU_VIS, b >= R.TAU_VIS
                if not ma.any() or not mb.any():
                    continue
                ya = float(xyz[ma, UP_AXIS].mean())
                yb = float(xyz[mb, UP_AXIS].mean())
                h = float(xyz[:, UP_AXIS].max() - xyz[:, UP_AXIS].min())
                sep = (ya - yb) / (h + C.EPS)
        except Exception:
            continue
        if abs(sep) < min_sep:
            continue
        sm["sep"] = sep
        sm["truth"] = "red_above_blue" if sep > 0 else "blue_above_red"
        out.append(sm)
    return out


def load_llm(a, n_images):
    processor = AutoProcessor.from_pretrained(a.model)
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=a.max_seqs, max_num_batched_tokens=a.max_len, enforce_eager=True,
              limit_mm_per_prompt={"image": n_images, "video": 0},
              enable_prefix_caching=True, trust_remote_code=True)
    return processor, llm


def render_sample(sm, tmp):
    d = os.path.join(tmp, sm["sample_id"].replace("/", "_"))
    paths, fa, fb = R.render_views(sm["qdir"], d, R.TAU_VIS, "v", "linear")
    imgs = [Image.open(paths[v]).convert("RGB") for v in R.VIEW_ORDER]
    return imgs, fa, fb


def manifest_work(path):
    """CLEAN::<id> and CORRUPT::<id> per entry, both rendered and judged identically."""
    rows = [json.loads(l) for l in open(path)]
    rej = os.path.join(os.path.dirname(path), "rejected.txt")
    if os.path.exists(rej):
        bad = {l.strip() for l in open(rej) if l.strip() and not l.startswith("#")}
        rows = [r for r in rows if r["corruption_id"] not in bad]
        print("[stage2] %d corruptions excluded by human review" % len(bad))
    out = []
    for r in rows:
        for tag, qdir in (("CLEAN", r["clean_qdir"]), ("CORRUPT", r["corrupt_qdir"])):
            out.append({"sample_id": "%s::%s" % (tag, r["corruption_id"]), "qdir": qdir})
    print("[stage2] manifest: %d entries -> %d judgements" % (len(rows), len(out)))
    return out


def run(a):
    probe = a.mode == "probe"
    if a.manifest:
        picks = manifest_work(a.manifest)
        out_dir = os.path.dirname(a.manifest)
    else:
        picks = pick_samples(a.dataset, a.limit, a.seed, min_sep=a.min_sep if probe else 0.0)
        out_dir = a.out or os.path.join(C.REPO,
                                        "data_verification/quality_evaluation/stage2", a.dataset)
    print("[stage2] %s: %d samples" % (a.mode, len(picks)))
    os.makedirs(out_dir, exist_ok=True)
    out_p = os.path.join(out_dir, "judged.jsonl" if a.manifest else "%s.jsonl" % a.mode)

    processor, llm = load_llm(a, len(R.VIEW_ORDER))
    schema = PROBE_SCHEMA if probe else JUDGE_SCHEMA
    system = PROBE_SYSTEM if probe else JUDGE_SYSTEM
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens,
                        guided_decoding=GuidedDecodingParams(json=schema))

    tmp = tempfile.mkdtemp(prefix="s2render_")
    fout = open(out_p, "w")
    n_ok = n_fail = 0
    try:
        for b0 in range(0, len(picks), a.batch):
            batch = picks[b0:b0 + a.batch]
            prompts, keep = [], []
            for sm in batch:
                try:
                    imgs, fa, fb = render_sample(sm, tmp)
                except Exception as e:
                    print("  render failed %s: %s" % (sm["sample_id"], e))
                    continue
                if probe:
                    text = ("The six views are, in order: %s.\n\nWhich region is higher?"
                            % ", ".join(R.VIEW_ORDER))
                else:
                    meta = C.load_meta(os.path.join(sm["qdir"], "meta.json"))
                    text = ("The six views are, in order: %s.\n\n%s"
                            % (", ".join(R.VIEW_ORDER), meta_block(meta)))
                    sm["_meta"] = meta
                prompts.append({"prompt": build_prompt(processor, system, imgs, text),
                                "multi_modal_data": {"image": imgs}})
                sm["_active"] = (fa, fb)
                keep.append(sm)
            if not prompts:
                continue
            for o, sm in zip(llm.generate(prompts, sp), keep):
                txt = o.outputs[0].text
                rec = {"judge_version": JUDGE_VERSION, "model": a.model,
                       "sample_id": sm["sample_id"], "tau_vis": R.TAU_VIS,
                       "active_A": sm["_active"][0], "active_B": sm["_active"][1], "raw": txt}
                if probe:
                    rec.update(truth=sm["truth"], sep=sm["sep"])
                else:
                    m = sm.get("_meta", {})
                    rec["roles"] = m.get("roles")
                try:
                    rec.update(json.loads(txt))
                    n_ok += 1
                except Exception:
                    rec["judge_failure"] = True
                    n_fail += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            shutil.rmtree(tmp, ignore_errors=True)     # render on the fly, keep nothing
            os.makedirs(tmp, exist_ok=True)
            print("[stage2] %d/%d | ok %d | fail %d" % (min(b0 + a.batch, len(picks)),
                                                        len(picks), n_ok, n_fail))
    finally:
        fout.close()
        shutil.rmtree(tmp, ignore_errors=True)
    print("[stage2] wrote %s" % out_p)
    if probe:
        report_probe(out_p)


def report_probe(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ok = [r for r in rows if not r.get("judge_failure")]
    hit = [r for r in ok if r["relation"] == r["truth"]]
    same = [r for r in ok if r["relation"] == "same_height"]
    flip = [r for r in ok if r["relation"] != r["truth"] and r["relation"] != "same_height"]
    print("\n=== A/B discrimination probe ===")
    print("n=%d  correct=%d (%.2f)  said same_height=%d  flipped=%d"
          % (len(ok), len(hit), len(hit) / max(1, len(ok)), len(same), len(flip)))
    if flip:
        print("flipped examples (model said the opposite of the true geometry):")
        for r in flip[:5]:
            print("  %-58s truth=%s sep=%+.2f" % (r["sample_id"][:58], r["truth"], r["sep"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "judge"])
    ap.add_argument("--dataset", default="daily_used")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--min_sep", type=float, default=0.25,
                    help="probe only: minimum |A-B| centroid height gap / object height")
    ap.add_argument("--manifest", default=None,
                    help="spatial corruption manifest; judges CLEAN and CORRUPT variants "
                         "through this exact same render + prompt path")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu_mem", type=float, default=0.92)
    ap.add_argument("--max_seqs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--out", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
