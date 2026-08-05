"""Verification Stage 1 -- RGB-conditioned Visual-Semantic Judge (README section 5).

Answers only: is this object/task/two-hand metadata sensible, given the same
G-Objaverse views the generator itself saw? It never sees scoreA/scoreB, hitA/hitB,
coverage, Stage 0 warnings, Molmo/SAM2 confidence or any Stage 2 result, so its
verdict cannot be contaminated by downstream grounding.

Model: Qwen3-VL-8B-Instruct (pilot; swaps to 30B-A3B when the big GPUs free up --
8B numbers are a LOWER BOUND and do not go in the paper's main tables).

The 8 views of an object are loaded once and placed at the FRONT of the prompt,
with the per-query metadata last, so vLLM prefix caching reuses the vision
encoding across every query of that object.

    CUDA_DEVICE_ORDER=PCI_BUS_ID python judge_stage1.py --dataset daily_used --limit 20
"""
import argparse
import json
import os
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")   # GPU2 is the only free card
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
# vLLM autoloads plugins from installed entry points at engine init. In the `mm` env
# paddlex registers one, and importing paddle SEGFAULTS inside its own
# monkey_patch_variable -> the engine dies before printing a single log line.
# We use no vLLM plugins, so switch discovery off entirely.
os.environ.setdefault("VLLM_PLUGINS", "")

from PIL import Image                                       # noqa: E402
from vllm import LLM, SamplingParams                        # noqa: E402
from vllm.sampling_params import GuidedDecodingParams       # noqa: E402
from vllm.config import StructuredOutputsConfig               # noqa: E402
from transformers import AutoProcessor                      # noqa: E402

import common as C                                          # noqa: E402

# v2.0 is the FORMAL BASELINE prompt. v2.1 (direct-physical-effect wording + 5 worked
# examples) is kept and selectable via --judge_version but is NOT the default: on 8B it only
# moved role_function target-strict 0.12 -> 0.20, and a prompt carrying worked examples
# measures the examples as much as it measures the model. Schema and derive_error_tags() are
# identical across both, so their records are structurally comparable.
JUDGE_VERSION = "qwen3vl-visual-semantic-v2.0"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
# The judge no longer emits tags. It reports structured verdicts only and the tags below are
# DERIVED from them by derive_error_tags(); the 8B run showed free-form tagging is not
# discriminative (role_task_conflict fired 15x on bimanual samples, 0x on real role conflicts),
# so the tag vocabulary is now a deterministic function of the fields, not a second opinion.
ERROR_TAGS = ["implausible_task", "contact_part_absent", "bimanual_conflict",
              "role_function_conflict"]

_score = {"type": "integer", "enum": [0, 1, 2]}
_hand = {"type": "object", "additionalProperties": False,
         "properties": {"score": _score,
                        "contact_region_evidence": {"type": "string",
                                                    "enum": ["supported", "uncertain", "absent"]},
                        "role_function_consistency": {"type": "string",
                                                      "enum": ["consistent", "uncertain",
                                                               "conflict"]},
                        "reason": {"type": "string"}},
         "required": ["score", "contact_region_evidence", "role_function_consistency", "reason"]}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "task_plausibility": {"type": "object", "additionalProperties": False,
                              "properties": {"score": _score, "reason": {"type": "string"}},
                              "required": ["score", "reason"]},
        "bimanual_validity": {"type": "object", "additionalProperties": False,
                              "properties": {"label": {"type": "string",
                                                       "enum": ["valid", "acceptable", "invalid"]},
                                             "reason": {"type": "string"}},
                              "required": ["label", "reason"]},
        "hand_A_visual_semantic_consistency": _hand,
        "hand_B_visual_semantic_consistency": _hand,
        "evidence_views": {"type": "array", "items": {"type": "string"}},
        "self_reported_confidence": {"type": "number"},
    },
    "required": ["task_plausibility", "bimanual_validity",
                 "hand_A_visual_semantic_consistency", "hand_B_visual_semantic_consistency",
                 "evidence_views", "self_reported_confidence"],
}

# Reasoning principles only -- deliberately no object- or part-specific examples,
# which would make the judge overfit to them (README section 6.4.1 note).
_SYSTEM_TMPL = """You audit metadata for a two-handed (bimanual) manipulation dataset.

You are shown several rendered views of ONE 3D object, in a fixed order, each labelled
with its view id. You are then given a proposed task and the metadata describing what
each of the two hands does. Judge that metadata against what the images actually show.

How to reason:
- Use only the images and the given text. You are given no grounding, no heatmaps and no
  confidence scores; do not speculate about them.
- Separate two different situations. A part that does not exist on this object is a
  metadata error. A part that plausibly exists but is hidden, occluded or too small to
  resolve in these views is uncertainty, not an error. Check every view before concluding
  a part is absent.
- A role names a physical primitive. Ask whether the force or motion that primitive
  implies can actually be produced at the stated contact region, given the object's
  visible geometry, scale and articulation.
- A function must be the consequence of that role acting at that contact region, not a
  restatement of the task.
- The two hands must be able to act at the same time: on material that exists, at places
  they can both reach, without occupying the same region, and their two contributions
  together must accomplish the task.
- Judge the object in front of you, not the object category in general. Scale and
  construction visible in the renders govern what is possible.

Scoring:
- task_plausibility: 2 = plausible and the images support it; 1 = possibly plausible but
  the images are insufficient, the relevant part is occluded, or the manner of operation
  is uncertain; 0 = clearly implausible, the object cannot support the task, or the task
  belongs to a different object.
- bimanual_validity: "valid" = two hands are natural, necessary, or carry a clear division
  of labour; "acceptable" = two hands work but one hand would plausibly suffice;
  "invalid" = the two-handed framing is forced, or the two roles cannot form a coherent
  operation.
- hand_A / hand_B consistency, judged as one integrated verdict over
  role + target + contact_region + function: 2 = metadata agrees with the visual evidence;
  1 = broadly reasonable but the part cannot be fully confirmed or the description is too
  vague to check; 0 = the role and function conflict, the part does not exist, or the
  contact region is clearly implausible for the stated function.
- contact_region_evidence: "supported" = the described part is visible and the region is a
  sensible place on it; "uncertain" = cannot be confirmed from these views; "absent" = the
  described part is not present on this object.
%(role_function)s
Cite in evidence_views the view ids you actually relied on. self_reported_confidence is
recorded for analysis only and gates nothing; report your genuine uncertainty rather than a
default high value.%(examples)s"""

# v2.0: the plain definition. This is the FORMAL BASELINE -- the 30B comparison runs on it,
# because a prompt carrying worked examples measures the examples as much as the model.
_RF_V20 = """- role_function_consistency, judged from the role and function TEXT ALONE -- do not use the
  images and do not let the contact region or the task decide it: "consistent" = the primitive
  named by role can plausibly produce the described function; "uncertain" = the wording is
  too vague or too general to decide either way; "conflict" = the two contradict each other
  in motion, in the direction of applied force, or in what the action is for.
"""

# v2.1: direct-physical-effect wording. Kept and runnable, but no longer the default -- on 8B
# it moved role_function target-strict 0.12 -> 0.20 (within noise at n=25).
_RF_V21 = """- role_function_consistency, judged from that hand's role and function TEXT ALONE. Do not use
  the images, the task, the target or the contact region. Ask one question only: does the
  DIRECT physical effect of this primitive -- the motion it makes and the force it applies --
  produce the stated function? A function that merely helps the overall task succeed is not
  enough; the role has to be what produces it.
  "consistent" = the primitive's direct physical effect can plausibly produce the function;
  "uncertain" = the function is too broad, too indirect, or too vague to decide;
  "conflict" = the two contradict each other in the motion performed, in the direction of the
  applied force, in whether the object ends up moving or held still, or in what the action
  is for.
"""

_EXAMPLES_V21 = """

Worked examples for role_function_consistency. Compare direct physical effects; do not learn
these particular words. The same primitive can be consistent or conflicting depending on what
the function actually claims.

Example 1 - consistent
role: hold
function: keeps the object stationary during the operation
judgment: consistent
reason: Holding directly restrains motion and provides stability.

Example 2 - conflict
role: slide
function: keeps the object stationary throughout the operation
judgment: conflict
reason: Sliding produces translational motion, which conflicts with remaining stationary.

Example 3 - consistent
role: press
function: presses the object against the table to prevent movement
judgment: consistent
reason: Pressing against a support surface can directly stabilize the object.

Example 4 - conflict
role: press
function: pulls the component outward toward the user
judgment: conflict
reason: Pressing cannot directly produce an outward pulling action.

Example 5 - uncertain
role: slide
function: helps control the object during the task
judgment: uncertain
reason: The function is too broad to determine whether sliding directly provides the stated
effect."""

# Prompt variants are selected by --judge_version, and the chosen version string is stamped
# on every record, so a judged.jsonl always says which prompt produced it.
PROMPTS = {
    "qwen3vl-visual-semantic-v2.0": _SYSTEM_TMPL % {"role_function": _RF_V20, "examples": ""},
    "qwen3vl-visual-semantic-v2.1": _SYSTEM_TMPL % {"role_function": _RF_V21,
                                                    "examples": _EXAMPLES_V21},
}

HANDS = ("hand_A_visual_semantic_consistency", "hand_B_visual_semantic_consistency")


def derive_error_tags(parsed):
    """Tags are a deterministic function of the structured verdict (README section 5.4).

    "uncertain" on either hand deliberately produces NO tag -- it is a Human Review signal
    for Stage 3, not an error -- so Stage 3 must read role_function_consistency /
    contact_region_evidence themselves rather than infer uncertainty from an empty tag list.
    Emitted in ERROR_TAGS order so the list is stable across runs.
    """
    hands = [parsed.get(k) or {} for k in HANDS]
    fired = set()
    if (parsed.get("task_plausibility") or {}).get("score") == 0:
        fired.add("implausible_task")
    if any(h.get("contact_region_evidence") == "absent" for h in hands):
        fired.add("contact_part_absent")
    if (parsed.get("bimanual_validity") or {}).get("label") == "invalid":
        fired.add("bimanual_conflict")
    if any(h.get("role_function_consistency") == "conflict" for h in hands):
        fired.add("role_function_conflict")
    return [t for t in ERROR_TAGS if t in fired]


def view_id(path):
    return os.path.basename(path).split(".")[0]


def object_block(rec, ids, missing):
    comps = rec.get("components") or []
    lines = ["OBJECT", "name: %s" % rec.get("object_name"),
             "views shown, in order: %s" % ", ".join(ids)]
    if missing:
        lines.append("views listed but unreadable (not shown): %s" % ", ".join(missing))
    if comps:
        lines.append("components reported by the generator:")
        lines += ["  - %s: %s" % (c.get("name"), c.get("interaction")) for c in comps]
    return "\n".join(lines)


def query_block(q):
    def hand(h):
        return ("  role: %s\n  target: %s\n  contact_region: %s\n  function: %s"
                % (h.get("role"), h.get("target"), h.get("contact_region"), h.get("function")))
    a, b = q["roles"][0], q["roles"][1]
    op = q.get("operation")
    return "\n".join([
        "PROPOSED TASK", "task: %s" % q.get("task"), "query: %s" % q.get("query"),
        "goal: %s" % q.get("goal"), "why two hands: %s" % q.get("why_bimanual"),
        "category: %s" % q.get("category"), "mechanism: %s" % q.get("mechanism"),
        "operation: %s" % (json.dumps(op, ensure_ascii=False) if isinstance(op, (dict, list)) else op),
        "coordination: %s" % q.get("coordination"),
        "HAND A", hand(a), "HAND B", hand(b),
        "relation between the hands: %s" % q.get("relation"),
        "symmetric: %s" % q.get("symmetric"), "hand_agnostic: %s" % q.get("hand_agnostic"),
        "", "Judge this metadata and return the required JSON.",
    ])


def build(processor, rec, ids, missing, q, system):
    msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user",
             "content": ([{"type": "image"} for _ in ids]
                         + [{"type": "text", "text": object_block(rec, ids, missing)
                             + "\n\n" + query_block(q)}])}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="daily_used")
    ap.add_argument("--limit", type=int, default=0,
                    help="max OBJECTS (not samples); every query of a kept object is judged")
    ap.add_argument("--out_name", default=None, help="output filename inside --out")
    ap.add_argument("--meta_name", default=None, help="run_meta filename inside --out")
    ap.add_argument("--judge_version", default=JUDGE_VERSION, choices=sorted(PROMPTS),
                    help="selects the SYSTEM prompt AND is stamped on every record")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--manifest", default=None,
                    help="corruption manifest; judges CLEAN and CORRUPT variants of each entry "
                         "through this exact same path")
    ap.add_argument("--obj_batch", type=int, default=4)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu_mem", type=float, default=0.92)
    ap.add_argument("--max_len", type=int, default=8192)
    # 3090 headroom: ~17.6 GB of bf16 weights on a 24 GB card leaves ~6 GB for the vision
    # tower, sampler buffers and KV cache. vLLM's startup profiling runs a dummy sampler
    # over max_num_seqs x vocab (152k) logits and OOMs at the default; keep it small and
    # skip CUDA-graph capture, which needs another chunk of memory it cannot spare.
    ap.add_argument("--max_seqs", type=int, default=8)
    ap.add_argument("--eager", action="store_true", default=True)
    # 640 truncated four long reasons mid-JSON; guided decoding cannot rescue a cut-off
    # generation, so it surfaced as a judge_failure rather than as invalid JSON.
    ap.add_argument("--max_tokens", type=int, default=1024)
    # A JSON grammar permits unbounded whitespace, so a model can fall into a tab/newline
    # loop that burns max_tokens and surfaces as judge_failure (30B did this 1/8 on smoke;
    # 8B did it 0/200). Opt-in, because it changes the constraint the tokens are sampled
    # under and the existing 8B baselines were produced without it.
    ap.add_argument("--disable_ws", action="store_true",
                    help="forbid whitespace inside the generated JSON (xgrammar)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ver, system = a.judge_version, PROMPTS[a.judge_version]

    recs = C.load_stage1(a.dataset)[a.start:]
    if a.limit and not a.manifest:
        recs = recs[:a.limit]
    if a.manifest:
        recs, work = _manifest_work(a.manifest, recs)
    # manifest runs default to sitting next to their manifest, but an explicit --out lets a
    # re-run of the SAME manifest under a new prompt version land in its own file instead of
    # being swallowed by the resume check against the previous run's judged.jsonl
    out_dir = a.out or (os.path.dirname(a.manifest) if a.manifest else
                        os.path.join(C.REPO, "data_verification/quality_evaluation/stage1", a.dataset))
    os.makedirs(out_dir, exist_ok=True)
    out_p = os.path.join(out_dir, a.out_name or
                         ("judged.jsonl" if a.manifest else "judge_%06d.jsonl" % a.start))
    done = set()
    if os.path.exists(out_p):
        with open(out_p) as f:
            done = {json.loads(l)["sample_id"] for l in f if l.strip()}
        print("[stage1] resuming, %d already judged" % len(done))

    t_boot = time.time()
    processor = AutoProcessor.from_pretrained(a.model)
    # vLLM 0.11: whitespace suppression is an ENGINE-level setting and only this form takes
    # effect. Both of the obvious alternatives fail SILENTLY -- passing
    # disable_any_whitespace on SamplingParams.guided_decoding, and the deprecated
    # guided_decoding_disable_any_whitespace= kwarg -- leaving the engine at
    # StructuredOutputsConfig(backend='auto', disable_any_whitespace=False). backend must be
    # named explicitly too: the flag is rejected under backend='auto'.
    so = (StructuredOutputsConfig(backend="xgrammar", disable_any_whitespace=True)
          if a.disable_ws else None)
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=a.max_seqs, max_num_batched_tokens=a.max_len,
              enforce_eager=a.eager, limit_mm_per_prompt={"image": 8, "video": 0},
              enable_prefix_caching=True, trust_remote_code=True,
              **({"structured_outputs_config": so} if so else {}))
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens,
                        guided_decoding=GuidedDecodingParams(json=SCHEMA))

    fout = open(out_p, "a")
    # load time is excluded from the per-sample figure on purpose: it is a fixed ~2 min
    # engine boot, so folding it in makes a 20-object test look far slower than a real run
    t_load = time.time() - t_boot
    t0 = time.time()
    n_ok = n_fail = 0
    for b0 in range(0, len(recs), a.obj_batch):
        prompts, metas = [], []
        for rec in recs[b0:b0 + a.obj_batch]:
            paths, imgs, ids, missing = rec.get("views_used") or [], [], [], []
            for p in paths:                       # fixed order; never substitute another view
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    ids.append(view_id(p))
                except Exception:
                    missing.append(view_id(p))
            if not imgs:
                continue
            if a.manifest:
                items = work.get(rec["object_id"], [])
            else:
                items = [("%s/q%d_%s" % (rec["object_id"], qi, C.slugify(q.get("task", ""))), q)
                         for qi, q in enumerate(rec.get("queries", []))]
            for sid, q in items:
                if not (isinstance(q.get("roles"), list) and len(q["roles"]) == 2):
                    continue
                if sid in done:
                    continue
                prompts.append({"prompt": build(processor, rec, ids, missing, q, system),
                                "multi_modal_data": {"image": imgs}})
                metas.append({"sample_id": sid, "object_id": rec["object_id"],
                              "views_shown": ids, "views_missing": missing})
        if not prompts:
            continue

        outs = llm.generate(prompts, sp)
        retry_i = []
        for i, (o, m) in enumerate(zip(outs, metas)):
            txt = o.outputs[0].text
            try:
                parsed = json.loads(txt)
            except Exception:
                retry_i.append(i)
                continue
            _write(fout, m, parsed, txt, a.model, ver)
            n_ok += 1
        if retry_i:                                # README section 5.5: max retry = 1
            outs2 = llm.generate([prompts[i] for i in retry_i],
                                 SamplingParams(temperature=0.2, max_tokens=a.max_tokens,
                                                guided_decoding=GuidedDecodingParams(json=SCHEMA)))
            for i, o in zip(retry_i, outs2):
                txt = o.outputs[0].text
                try:
                    _write(fout, metas[i], json.loads(txt), txt, a.model, ver)
                    n_ok += 1
                except Exception:
                    _write(fout, metas[i], None, txt, a.model, ver)
                    n_fail += 1
        fout.flush()
        print("[stage1] objects %d/%d | ok %d | judge_failure %d"
              % (min(b0 + a.obj_batch, len(recs)), len(recs), n_ok, n_fail))
    fout.close()
    elapsed = time.time() - t0
    meta = {"judge_version": ver, "model": a.model, "dataset": a.dataset,
            "start": a.start, "limit_objects": a.limit, "manifest": a.manifest,
            "n_objects_in_scope": len(recs), "n_judged_this_run": n_ok + n_fail,
            "n_ok": n_ok, "n_judge_failure": n_fail, "n_resumed_skipped": len(done),
            "load_seconds": round(t_load, 1), "judge_seconds": round(elapsed, 1),
            "seconds_per_sample": round(elapsed / max(n_ok + n_fail, 1), 2),
            "output": out_p, "disable_any_whitespace": a.disable_ws}
    with open(os.path.join(out_dir, a.meta_name or "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("[stage1] wrote %s | %d judged in %.1fs (%.2f s/sample, load %.1fs)"
          % (out_p, n_ok + n_fail, elapsed, meta["seconds_per_sample"], t_load))


def _manifest_work(path, recs):
    """CLEAN::<id> and CORRUPT::<id> for every manifest entry, keyed by object.

    Both variants go through the identical prompt builder and sampling params, so a
    clean-vs-corrupted difference can only come from the metadata that was edited.
    Human-rejected corruptions (rejected.txt) are dropped before any GPU time is spent.
    """
    rows = [json.loads(l) for l in open(path)]
    rej_p = os.path.join(os.path.dirname(path), "rejected.txt")
    if os.path.exists(rej_p):
        bad = {l.strip() for l in open(rej_p) if l.strip() and not l.startswith("#")}
        rows = [r for r in rows if r["corruption_id"] not in bad]
        print("[stage1] %d corruptions excluded by human review" % len(bad))
    work = {}
    for r in rows:
        work.setdefault(r["object_id"], []).extend(
            [("CLEAN::" + r["corruption_id"], r["original"]),
             ("CORRUPT::" + r["corruption_id"], r["corrupted"])])
    keep = [rec for rec in recs if rec["object_id"] in work]
    print("[stage1] manifest: %d entries over %d objects -> %d judgements"
          % (len(rows), len(keep), 2 * len(rows)))
    return keep, work


def _write(f, meta, parsed, raw, model, version):
    rec = {"judge_version": version, "model": model, "sample_id": meta["sample_id"],
           "object_id": meta["object_id"], "views_shown": meta["views_shown"],
           "views_missing": meta["views_missing"], "raw": raw}
    if parsed is None:
        rec["judge_failure"] = True
    else:
        rec.update(parsed)
        rec["error_tags"] = derive_error_tags(parsed)   # rule-based; the model emits none
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
