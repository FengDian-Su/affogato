#!/usr/bin/env python3
"""Qwen3.8-27B as a task / role / heatmap rater, measured against the human raters of the quality review.

Each step gets the evidence a human rater has at that step: the task step sees the eight RGB views
(original resolution), the object name and the task; the role step also sees both hands' roles; the
heatmap step sees the three review sheets (object, Hand A, Hand B) at full resolution, 512 px per view
(render_hires.py; the UI shows them at 256 px but lets raters zoom in). The
question is short and general, leaning on the model's everyday knowledge. One request per pair and
criterion, greedy, thinking off (unless --think), JSON constrained to the step fields, then score and reason. The generated score is
the model's label as is; the probabilities of the score token are only logged.

  run       python -m data_verification.quality_review.qwen_judge run --shard 0/2 [--axes heatmap] --out DIR
            (one GPU per shard: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0, and the gemma4
            env needs LD_LIBRARY_PATH=$CONDA/envs/gemma4/lib)
  evaluate  python -m data_verification.quality_review.qwen_judge evaluate --out DIR
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from data_verification.quality_review.app import (
    DEFAULT_DB, DEFAULT_ROOT, axis_ratings, connect, krippendorff_alpha, score,
)

MODEL, REVISION = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
VERSION = "qwen38-27b-v23"
# v3: a short, general question per criterion that leans on the model's everyday knowledge, asked on
# the eight original 512 px views (v1/v2 used the 256 px-per-view sheet and the full rater rubric).
# v4: the middle grade means "unclear", not "awkward". v5: the name can be wrong and the views are a 3D
# model whose moving parts are often fused. Task keeps the v5 prompt: a parts check before the score
# helps (dropping it: false alarms 33 -> 83 on dev), while step-by-step "checker" versions (v10, v11)
# made it stricter. Role: the model first writes how a person would normally do the task, then compares
# the roles with that, instead of auditing the 3D model ("this lid is fixed, it cannot be pushed"):
# role score 91.2 -> 93.9 on dev (role-v12). v15 adds what a role is: an affordance annotation, where
# the touched region need not move or deform and the object stays whole, and an action that is the
# closest of eight verbs, judged as such (93.9 -> 96.9; either idea alone: 93.4 / 94.4). Spelling out
# verb meanings and fault cases added nothing (v10, v14a). Thinking mode did not help (v5-think).
# v18 task: the 0 grade is more lenient - a part that is not visible or not modelled separately is not
# missing, and size is no reason against a task (the hands may be any person or robot; picking up and
# moving are asked of every object): all 1000 pairs 92.5 -> 93.3, universal-task flags 6 -> 0.
# Telling the model to accept a task even when its parts are not shown (v20) matched the raters' score
# (96.2) but by leniency: it caught half as many of the pairs the raters flagged. v18 stays grounded on
# what the views show.
# v21 role: says what the most used verbs cover (lift also holds the object up or steady, rotate turns or
# tilts the whole object through the part held, hold keeps it still while the other hand works), and to
# rely on the name and everyday knowledge when the views are hard to see: all 1000 pairs 94.8 -> 96.5
# (dev 96.9 -> 97.5, test 90.2 -> 94.0).
# v22: each hand is listed as separate fields (touched part and where, action, purpose) instead of one
# imperative "rotate the handle", which read as the handle itself turning: of the 63 pairs that tilt or
# flip an object by a handle or rim, flagged 7 -> 1; Good pairs flagged 33 -> 27.
# v23 adds the heatmap criterion; task and role are unchanged from v22. Heatmap prompt history (dev, two
# raters): text-only wording moved along one trade-off curve (Good pairs flagged vs pairs both raters
# flagged that the model also flags); what moved the curve was seeing the sheets at full resolution and
# grading on the raters' own scale - clean / flawed but usable / unusable - instead of "usable" alone,
# which let visible flaws pass. Pipeline-specific explanations (straight-cut halves) and the role's
# "where" text are left out: the former reads as excusing our own output, the latter made the model
# match regions against the exact spot named (4x the flags). The size sentence covers every hand whose
# possible contacts span the object (holding, supporting or moving the whole object).
AXES = ("task", "role", "heatmap")
PROMPTS = {
    "task": """You will see one object in eight views and a task someone proposes to do with it using two hands.
Using what you see and your everyday knowledge of objects like this, judge whether the task makes sense
for this object and is a meaningful thing to do with it.
The object name can be wrong, so go by what the views show. The views show a 3D model, where parts that
move on the real object, such as lids, doors or knobs, are often modelled as one solid piece; judge the
object as the real thing it shows.
2 = it makes sense and is meaningful for this object.
1 = unclear: you cannot tell whether it makes sense, or whether it has a real point.
0 = it clearly does not make sense or has no real point for this object, for example it needs a part
that objects of this kind do not have. A part that is small, hidden or not modelled separately is not
missing, and size or weight is no reason either: the hands may belong to a person or a robot of any size,
and picking up and moving the object are asked of every object. When a real object of this kind would
have the part, judge the task as if the part were there. If you are not sure, give 1.
First say whether the parts the task needs are visible in the views (yes, no or unsure), then give
the score and one short sentence explaining it.""",
    "role": """You check affordance annotations for 3D object models. An annotation tells a person or robot with two
hands how to act on the object to do a task: for each hand, an action, the part it touches, where on that
part, and what it is for. The touched part and region are where the hand makes contact and applies its
action; they do not have to move, bend or change shape themselves. The object stays whole: the hands act
on it as it is, often moving the whole object through the part they hold.
The action is chosen from only eight verbs (hold, lift, push, pull, press, slide, rotate, squeeze).
Many tasks need a motion none of them names exactly, so the closest verb is used; check whether each
hand's verb is a reasonable closest choice among the eight, not whether it names the motion exactly.
So lift also stands for holding the object up or steady against gravity, rotate for turning, tilting or
flipping the whole object through the part a hand holds, and hold for keeping the object or its base still
while the other hand works.
You get eight views of the object (they show only the object, not the hands), its name, the task, and the
two roles. If the views are dark or hard to see, rely on the name and everyday knowledge. Answer in these steps:
1. plan: in one sentence, how a person would normally do this task with two hands.
2. parts: are the parts the hands touch on this object? (yes, no or unsure) Count parts a real object like
this has, even if the 3D model shows them fused in place.
3. score:
2 = the roles are a reasonable way to do the task with these eight verbs, even if not exactly how
you would do it.
1 = you cannot tell.
0 = clearly not a way to do the task.
4. reason: one short sentence.
Judge the roles for the task as given, even if the task itself is odd.""",
    "heatmap": """You check contact heatmaps in affordance annotations for 3D object models. An annotation tells a person
or robot with two hands how to act on an object to do a task. For each hand there is a role (the part it touches, the action and its purpose) and a heatmap.
What a heatmap shows: all the possible places on the object where that hand could make contact to do its
role. The object is drawn as dark points and the region is colored, orange for Hand A and teal for Hand B;
brighter means a more likely contact. A usable region is one clear
area, or a few, on places where the hand could do its job; specks scattered over unrelated parts, or a
trace too faint to see, give the hand no clear place to act.
The size of a region does not by itself make it good or bad: a hand that holds, supports or moves the whole
object can touch it in many places, so a large region is right for it, while a hand that works one
particular part can only touch that part.
You get three images of the object from the same eight cameras: the object itself, Hand A's heatmap and
Hand B's heatmap. The views show only the object, not the hands. The role text says roughly where each hand
acts; judge the regions you see against the task.
Answer in these steps:
1. hand_a_region: look at the orange points and say where they are on the object and how they look: one
clear area, a few separate areas, scattered specks, a faint trace, or spread over most of the object.
2. hand_a_quality: is the orange region clean (on places where Hand A can do its action, with nothing
notable elsewhere), flawed (usable, but part of it lies elsewhere, it is broken into small bits, or it is
very small) or unusable (missing, too faint, or not where Hand A can do its action)?
3. hand_b_region: the same as step 1 for the teal points.
4. hand_b_quality: the same as step 2 for Hand B.
5. score:
2 = both regions are clean.
1 = a region is flawed, or you cannot tell.
0 = a region is unusable.
6. reason: one short sentence.""",
}
YES_NO = {"type": "string", "enum": ["yes", "no", "unsure"]}
SCORE = {"type": "integer", "enum": [0, 1, 2]}
TEXT = {"type": "string"}
QUALITY = {"type": "string", "enum": ["clean", "flawed", "unusable"]}
SCHEMAS = {
    "task": {"type": "object", "additionalProperties": False, "required": ["parts_visible", "score", "reason"],
             "properties": {"parts_visible": YES_NO, "score": SCORE, "reason": TEXT}},
    "role": {"type": "object", "additionalProperties": False, "required": ["plan", "parts", "score", "reason"],
             "properties": {"plan": TEXT, "parts": YES_NO, "score": SCORE, "reason": TEXT}},
    "heatmap": {"type": "object", "additionalProperties": False,
                "required": ["hand_a_region", "hand_a_quality", "hand_b_region", "hand_b_quality", "score", "reason"],
                "properties": {"hand_a_region": TEXT, "hand_a_quality": QUALITY, "hand_b_region": TEXT,
                               "hand_b_quality": QUALITY, "score": SCORE, "reason": TEXT}},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def case_text(sample: Any, axis_id: str) -> str:
    lines = [f"Object: {sample['object_name']}", f"Task: {sample['task']}"]
    for role in json.loads(sample["roles_json"]) if axis_id != "task" else []:
        where = f" at {role['contact_region']}" if axis_id == "role" else ""
        lines.append(f"Hand {role['id']}: touches {role['target']}{where}; "
                     f"action: {role['role']}; purpose: {role['function']}")
    return "\n".join(lines)


def engine_kwargs() -> dict[str, Any]:
    """What this box needs for Qwen3.8 on the RTX PRO 6000 (sm_120) under a CUDA 12.8 driver: the
    bundled flash-attn ships cu129 PTX this driver cannot load, so the decoder and the vision
    encoder use other backends, and flashinfer JIT gets a real nvcc and an explicit arch."""
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12:
        return {}
    cuda = "/usr/local/cuda-12.8"
    if os.path.isdir(cuda):
        os.environ.setdefault("CUDA_HOME", cuda)
        os.environ["PATH"] = f"{cuda}/bin:{os.path.dirname(sys.executable)}:" + os.environ.get("PATH", "")
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0a")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return {"attention_backend": "TRITON_ATTN", "mm_encoder_attn_backend": "TORCH_SDPA"}


def evidence(sample: Any, axis_id: str) -> tuple[list[str], list[str]]:
    """The images a rater sees at this step, each with the caption placed before it."""
    if axis_id == "heatmap":
        return (["Object views:", "Hand A heatmap (orange):", "Hand B heatmap (teal):"],
                [sample[key].replace("/release1000/renders/", "/release1000/renders_2x/")
                 for key in ("rgb_path", "heat_a_path", "heat_b_path")])
    return ([f"View: {label}" for label in json.loads(sample["view_labels_json"])],
            json.loads(sample["manifest_json"])["views"])


def chat_prompt(processor: Any, system: str, captions: list[str], user_text: str, think: bool = False) -> str:
    """The question, then each image under its caption, then the case text."""
    content: list[dict[str, Any]] = []
    for caption in captions:
        content += [{"type": "text", "text": caption}, {"type": "image"}]
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [*content, {"type": "text", "text": user_text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=think)


def score_probs(completion: Any, think: bool = False) -> dict[str, float] | None:
    """p(0), p(1), p(2) at the generated score digit (the first digit after the answer's "score" key),
    renormalised over the three digits."""
    answer, answering = "", not think
    for token_id, position in zip(completion.token_ids, completion.logprobs or []):
        sampled = position.get(token_id)
        text = (sampled.decoded_token or "") if sampled else ""
        if not answering:
            answering = "</think>" in text
            continue
        answer += text
        if '"score":' not in answer or text.strip() not in ("0", "1", "2"):
            continue
        probs = {"0": 0.0, "1": 0.0, "2": 0.0}
        for candidate in position.values():
            text = (candidate.decoded_token or "").strip()
            if text in probs:
                probs[text] += math.exp(candidate.logprob)
        total = sum(probs.values())
        return {key: value / total for key, value in probs.items()}
    return None


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("VLLM_PLUGINS", "")
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    connection = connect(args.db)
    try:
        samples = connection.execute("SELECT * FROM samples ORDER BY sample_index").fetchall()
    finally:
        connection.close()
    index, count = (int(part) for part in args.shard.split("/"))
    samples = [sample for sample in samples if args.split in ("all", sample["split"])]
    samples = samples[index::count][: args.limit or None]
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"shard{index}.jsonl"
    done = {(row["sample_id"], row["axis"]) for row in read_jsonl(out)} if out.exists() else set()
    axes = [axis for axis in AXES if axis in args.axes.split(",")]
    jobs = [(sample, axis) for sample in samples for axis in axes if (sample["sample_id"], axis) not in done]
    prompts = {axis: PROMPTS[axis] for axis in axes}
    (args.out / "config.json").write_text(json.dumps({
        "version": VERSION, "model": MODEL, "revision": REVISION,
        "images": {"task": "8 original views, 512 px", "role": "8 original views, 512 px",
                   "heatmap": "object / Hand A / Hand B sheets, 512 px per view (renders_2x)"},
        "decoding": {"temperature": 0, "thinking": args.think, "logprobs": 10}, "system_prompts": prompts,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"shard {index}/{count}: {len(jobs)} requests to run ({len(done)} already done)", flush=True)
    if not jobs:
        return

    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    extra = engine_kwargs()
    llm = LLM(model=MODEL, revision=REVISION, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              max_num_seqs=args.batch, enforce_eager=not extra, limit_mm_per_prompt={"image": 8, "video": 0},
              trust_remote_code=True, disable_log_stats=True,
              structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True,
                                         "reasoning_parser": "qwen3" if args.think else ""}, **extra)
    params = {axis: SamplingParams(temperature=0.0, max_tokens=8192 if args.think else 1024, logprobs=10,
                                   structured_outputs=StructuredOutputsParams(json=SCHEMAS[axis])) for axis in axes}

    started = time.time()
    for start in range(0, len(jobs), args.batch * 4):
        chunk = jobs[start:start + args.batch * 4]
        requests = []
        for sample, axis in chunk:
            captions, paths = evidence(sample, axis)
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))   # transparent background -> black, as in the UI
            requests.append({"prompt": chat_prompt(processor, prompts[axis], captions,
                                                   case_text(sample, axis), args.think),
                             "multi_modal_data": {"image": images}})
        results = llm.generate(requests, [params[axis] for _, axis in chunk], use_tqdm=False)
        with out.open("a", encoding="utf-8") as stream:
            for (sample, axis), result in zip(chunk, results):
                completion = result.outputs[0]
                try:
                    answer = json.loads(completion.text.split("</think>")[-1])
                    label = int(answer["score"])
                except (ValueError, KeyError, TypeError):
                    answer, label = None, None
                stream.write(json.dumps({
                    "sample_id": sample["sample_id"], "axis": axis, "score": label,
                    "answer": answer, "p": score_probs(completion, args.think),
                    "raw": completion.text, "version": VERSION,
                }, ensure_ascii=False) + "\n")
        finished = start + len(chunk)
        rate = (time.time() - started) / finished
        print(f"[{finished}/{len(jobs)}] {rate:.2f} s/request, eta {rate * (len(jobs) - finished) / 60:.1f} min",
              flush=True)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    connection = connect(args.db)
    try:
        everything = axis_ratings(connection)
        splits = {row["sample_id"]: row["split"] for row in connection.execute("SELECT sample_id, split FROM samples")}
    finally:
        connection.close()
    model = {(row["sample_id"], row["axis"]): row for path in sorted(args.out.glob("shard*.jsonl"))
             for row in read_jsonl(path) if row["score"] is not None}
    report: dict[str, Any] = {"version": VERSION, "run": str(args.out)}
    for split in ("all", "dev", "test"):
        report[split] = {}
        for axis in AXES:
            pairs = {sid: per[axis] for sid, per in everything.items()
                     if per[axis] and (sid, axis) in model and split in ("all", splits[sid])}
            raters = sorted({username for ratings in pairs.values() for username in ratings})
            machine = {sid: model[(sid, axis)]["score"] for sid in pairs}
            per_rater = {}
            for username in raters:
                mine = {sid: ratings[username]["label"] for sid, ratings in pairs.items() if username in ratings}
                both = [(machine[sid], label) for sid, label in mine.items()]
                confusion = Counter(both)
                per_rater[username] = {
                    "n": len(both),
                    "exact": sum(m == h for m, h in both) / len(both),
                    "alpha_ordinal": krippendorff_alpha([[m, h] for m, h in both]),
                    "ac1": gwet_ac1([m for m, _ in both], [h for _, h in both]),
                    "human_score": score([h for _, h in both]),
                    "model_score": score([m for m, _ in both]),
                    "non_good_recall": _ratio(sum(m < 2 and h < 2 for m, h in both), sum(h < 2 for _, h in both)),
                    "non_good_precision": _ratio(sum(m < 2 and h < 2 for m, h in both), sum(m < 2 for m, _ in both)),
                    "confusion_human_by_model": [[confusion[(m, h)] for m in (0, 1, 2)] for h in (0, 1, 2)],
                }
            humans_only = [[r["label"] for r in ratings.values()] for ratings in pairs.values() if len(ratings) >= 2]
            report[split][axis] = {
                "pairs": len(pairs),
                "model_score": score(list(machine.values())) if machine else None,
                "model_distribution": [list(machine.values()).count(v) for v in (0, 1, 2)],
                "human_alpha_ceiling": krippendorff_alpha(humans_only),
                "per_rater": per_rater,
                "table": table_row(pairs, machine) if pairs else None,
            }
        ids = [sid for sid in everything if split in ("all", splits[sid])
               and all((sid, axis) in model for axis in AXES)]
        report[split]["all_three_good"] = all_three_good(everything, model, ids) if ids else None
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for split in ("all", "dev", "test"):
        for axis in AXES:
            entry = report[split][axis]
            for username, stats in entry["per_rater"].items():
                print(f"{split:4s} {axis:4s} vs {username:12s} n={stats['n']:4d}  human {stats['human_score']:5.1f}"
                      f"  model {stats['model_score']:5.1f}  exact {stats['exact']:.1%}  alpha "
                      f"{_fmt(stats['alpha_ordinal'])}  AC1 {stats['ac1']:5.2f}  non-Good recall "
                      f"{_fmt(stats['non_good_recall'])} precision {_fmt(stats['non_good_precision'])}")
    for split in ("all", "dev", "test"):
        for axis in AXES:
            row = report[split][axis]["table"]
            if row and row["experts"]:
                print(f"{split:4s} {axis:7s} experts {row['expert_score']:5.1f} +- {_fmt(row['expert_score_sd'])}"
                      f" ({row['expert_good_pct']:.1f}% Good)  judge {row['judge_score']:5.1f}"
                      f" ({row['judge_good_pct']:.1f}% Good)  AC1 judge-expert {row['ac1_judge_expert']:.3f}"
                      f"  expert-expert {_fmt(row['ac1_expert_expert'])}  ({len(row['experts'])} experts)")
        row = report[split]["all_three_good"]
        if row and row["experts"]:
            print(f"{split:4s} all three Good: experts {row['expert_pct']:.1f}%  judge {row['judge_pct']:.1f}%")
    return report


def gwet_ac1(a: list[int], b: list[int], categories: tuple = (0, 1, 2)) -> float:
    """Gwet's AC1 between two raters. Unlike kappa or alpha it does not collapse when one grade
    dominates, as Good does here."""
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    shares = [(a.count(k) + b.count(k)) / (2 * len(a)) for k in categories]
    chance = sum(p * (1 - p) for p in shares) / (len(categories) - 1)
    return (observed - chance) / (1 - chance)


def good_pct(labels: list[int]) -> float:
    return 100 * labels.count(2) / len(labels)


def table_row(pairs: dict[str, dict], machine: dict[str, int]) -> dict[str, Any]:
    """One criterion of the paper's table. Experts are the raters who rated every pair; their score is
    the mean of each expert's own score with the SD across experts (as in the review app), and AC1 is
    averaged over judge-expert and over expert-expert pairs."""
    experts = sorted(u for u in {u for ratings in pairs.values() for u in ratings}
                     if all(u in ratings for ratings in pairs.values()))
    judge = [machine[sid] for sid in pairs]
    human = [[pairs[sid][u]["label"] for sid in pairs] for u in experts]
    return {
        "experts": experts,
        "expert_score": statistics.mean(map(score, human)) if human else None,
        "expert_score_sd": statistics.stdev(map(score, human)) if len(human) > 1 else None,
        "expert_good_pct": statistics.mean(map(good_pct, human)) if human else None,
        "judge_score": score(judge),
        "judge_good_pct": good_pct(judge),
        "ac1_judge_expert": statistics.mean(gwet_ac1(judge, h) for h in human) if human else None,
        "ac1_expert_expert": (statistics.mean(gwet_ac1(a, b) for a, b in combinations(human, 2))
                              if len(human) > 1 else None),
    }


def all_three_good(everything: dict, model: dict, ids: list[str]) -> dict[str, Any]:
    """Share of pairs graded Good on every criterion, by the judge and by each expert (averaged)."""
    experts = sorted(u for u in {u for sid in ids for axis in AXES for u in everything[sid][axis]}
                     if all(u in everything[sid][axis] for sid in ids for axis in AXES))
    return {
        "experts": experts,
        "expert_pct": statistics.mean(100 * statistics.mean(
            all(everything[sid][axis][u]["label"] == 2 for axis in AXES) for sid in ids) for u in experts)
            if experts else None,
        "judge_pct": 100 * statistics.mean(all(model[(sid, axis)]["score"] == 2 for axis in AXES) for sid in ids),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _fmt(value: float | None) -> str:
    return "  -- " if value is None else f"{value:5.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--db", type=Path, default=DEFAULT_DB)
        command.add_argument("--out", type=Path, default=DEFAULT_ROOT / "qwen38_27b" / VERSION)
    sub.choices["run"].add_argument("--shard", default="0/1")
    sub.choices["run"].add_argument("--limit", type=int, default=0)
    sub.choices["run"].add_argument("--split", choices=("dev", "test", "all"), default="dev",
                                    help="tune on dev; score test once, at the end")
    sub.choices["run"].add_argument("--batch", type=int, default=32)
    sub.choices["run"].add_argument("--think", action="store_true", help="let the model think before it answers")
    sub.choices["run"].add_argument("--max-model-len", type=int, default=16384, help="context length")
    sub.choices["run"].add_argument("--axes", default=",".join(AXES), help="comma-separated subset of the criteria")
    sub.choices["run"].add_argument("--gpu-mem", type=float, default=0.88)
    args = parser.parse_args()
    run(args) if args.command == "run" else evaluate(args)


if __name__ == "__main__":
    main()
