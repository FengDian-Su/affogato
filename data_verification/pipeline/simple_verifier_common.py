"""Shared utilities for the deliberately small Stage 1/2 verifiers.

This module has no vLLM/transformers imports at module load time.  In particular,
``--dry-run`` can validate an entire benchmark without touching CUDA or loading a
model.  The existing full judges remain separate baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DIRECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "integer", "enum": [0, 1, 2]},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
}
BAD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "clear_bad": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["clear_bad", "reason"],
}
GOOD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "fully_good": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["fully_good", "reason"],
}


def mcq_answers(findings: list[str]) -> list[str]:
    return list(findings) + [MCQ_OK, MCQ_GOOD]


def mcq_schema(findings: list[str]) -> dict[str, Any]:
    """One single-select question whose answer determines the score AND the error type. Single-select
    matters: a multi-label tag lets the model fill every slot, which is how earlier taggers emitted a
    fixed number of labels regardless of the sample."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "finding": {"type": "string", "enum": mcq_answers(findings)},
            "reason": {"type": "string"},
        },
        "required": ["finding", "reason"],
    }


@dataclass(frozen=True)
class WorkItem:
    sample_id: str
    object_id: str
    image_path: str
    user_text: str


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def stable_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def common_parser(axis: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Simple {axis} verifier")
    p.add_argument("--manifest", required=True)
    p.add_argument("--split-file", required=True)
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--variant", choices=["direct", "cascade", "mcq", "multi", "gated", "per_axis"], required=True)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--winner-file", default=None,
                   help="required for locked test; must name this axis/variant")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--gpu-mem", type=float, default=0.92)
    p.add_argument("--eager", dest="eager", action="store_true", default=None,
                   help="force eager (default: eager on Ampere, CUDA graphs on Blackwell)")
    p.add_argument("--no-eager", dest="eager", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    return p


def select_manifest_rows(manifest_path: str, split_path: str, split: str,
                         limit: int = 0) -> list[dict[str, Any]]:
    rows = read_jsonl(manifest_path)
    ids = [r.get("sample_id") for r in rows]
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError("manifest sample_id values must be present and unique")
    split_rows = read_jsonl(split_path)
    split_by_id = {r["sample_id"]: r["split"] for r in split_rows}
    if len(split_by_id) != len(split_rows):
        raise ValueError("split file contains duplicate sample_id values")
    missing = set(ids) - set(split_by_id)
    extra = set(split_by_id) - set(ids)
    if missing or extra:
        raise ValueError(f"manifest/split mismatch: missing={len(missing)} extra={len(extra)}")
    if split != "all":
        rows = [r for r in rows if split_by_id[r["sample_id"]] == split]
    rows.sort(key=lambda r: r["sample_index"])
    return rows[:limit] if limit else rows


def validate_locked_test(axis: str, variant: str, split: str,
                         winner_file: str | None) -> None:
    if split != "test":
        return
    if not winner_file:
        raise ValueError("--winner-file is required for the locked test split")
    with open(winner_file, encoding="utf-8") as f:
        winners = json.load(f)
    selected = winners.get(axis)
    if not selected or selected.get("variant") != variant:
        raise ValueError(
            f"locked test requires selected {axis} variant "
            f"{(selected or {}).get('variant')!r}, not {variant!r}"
        )


VARIANT_KEYS = {"direct": ["direct"], "cascade": ["bad", "good"], "mcq": ["mcq"],
                "multi": ["multi"], "gated": ["gate", "tag"],
                # one isolated call per aspect: the same four items as `multi`, but asked
                # separately so the model cannot stamp one gestalt verdict across all of them
                "per_axis": ["axis_task", "axis_bimanual", "axis_hand_A", "axis_hand_B"]}

# `gated` reverses the cascade. Call 1 is the isolated soundness question and it ALONE decides
# whether the sample is auto-accepted, so nothing pollutes the strong signal. Call 2 only runs on
# what call 1 already rejected, and only assigns an error type - it can turn a reject into a 0 or a
# 1, never into an accept, so it cannot lower the catch rate.

# `multi` scores each part separately in ONE call and aggregates with min(), the way
# judge_stage1.py's rubric and the human RUBRIC.md do. Separate sub-scores force separate
# examinations instead of one gestalt verdict, and min() means any single fatal part decides.
# The axis that hit the minimum doubles as the error type.
MULTI_AXES: tuple[str, ...] = ()          # set per judge module via verifier_main(axes=...)

# `mcq` is ONE call. The answer space carries the score and the error type together: any failure id
# means 0 and is itself the tag, plus two non-failure outcomes for 1 and 2.
MCQ_OK = "acceptable_imprecise"      # -> 1
MCQ_GOOD = "fully_sound"             # -> 2


def prompt_hash(axis: str, variant: str, prompts: dict[str, str]) -> str:
    return stable_hash(axis, variant, *(prompts[k] for k in VARIANT_KEYS[variant]))


def _failure_record(axis: str, variant: str, version: str, model: str,
                    sample_id: str, reason: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "axis": axis,
        "variant": variant,
        "prompt_version": version,
        "model": model,
        "score": None,
        "reason": reason,
        "judge_failure": True,
    }


def prepare_items(rows: Iterable[dict[str, Any]], item_builder: Callable[[dict[str, Any]], WorkItem],
                  axis: str, variant: str, version: str, model: str
                  ) -> tuple[list[WorkItem], list[dict[str, Any]]]:
    items, failures = [], []
    for row in rows:
        sid = row.get("sample_id", "<missing>")
        try:
            item = item_builder(row)
            with Image.open(item.image_path) as im:
                im.verify()
            items.append(item)
        except Exception as exc:
            failures.append(_failure_record(axis, variant, version, model, sid,
                                            f"input_error: {exc}"))
    return items, failures


def write_dry_run(out_dir: str, axis: str, variant: str, version: str, model: str,
                  prompts: dict[str, str], rows: list[dict[str, Any]], items: list[WorkItem],
                  failures: list[dict[str, Any]]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p_hash = prompt_hash(axis, variant, prompts)
    preview_path = out / "prompt_previews.jsonl"
    with open(preview_path, "w", encoding="utf-8") as f:
        for item in items[:10]:
            rec = {
                "sample_id": item.sample_id,
                "image_path": item.image_path,
                "user_text": item.user_text,
                "system_prompts": [prompts[k] for k in VARIANT_KEYS[variant]],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "dry_run": True,
        "axis": axis,
        "variant": variant,
        "prompt_version": version,
        "prompt_sha256": p_hash,
        "model": model,
        "rows_in_scope": len(rows),
        "valid_inputs": len(items),
        "input_failures": len(failures),
        "failure_records": failures[:20],
        "preview": str(preview_path),
        "model_loaded": False,
    }
    write_json(out / "dry_run_summary.json", summary)
    if failures:
        raise RuntimeError(f"dry-run found {len(failures)} invalid inputs")


def build_chat_prompt(processor: Any, system: str, item: WorkItem) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": item.user_text},
        ]},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _parse(text: str, schema_kind: str, findings: list[str] | None = None) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("reason"), str):
        raise ValueError("missing object/reason")
    if schema_kind == "direct":
        if type(value.get("score")) is not int or value["score"] not in (0, 1, 2):
            raise ValueError("invalid score")
    elif schema_kind == "mcq":
        if value.get("finding") not in mcq_answers(findings or []):
            raise ValueError(f"invalid finding {value.get('finding')!r}")
    elif schema_kind == "multi":
        for a in (findings or []):          # `findings` carries the axis list for this variant
            if type(value.get(a)) is not int or value[a] not in (0, 1, 2):
                raise ValueError(f"invalid sub-score {a}={value.get(a)!r}")
    else:
        key = "clear_bad" if schema_kind == "bad" else "fully_good"
        if type(value.get(key)) is not bool:
            raise ValueError(f"invalid {key}")
    return value


def cascade_score(pass1: dict[str, Any], pass2: dict[str, Any] | None) -> tuple[int, str]:
    if pass1["clear_bad"]:
        return 0, pass1["reason"]
    if pass2 is None:
        raise ValueError("non-bad cascade verdict requires pass 2")
    return (2 if pass2["fully_good"] else 1), pass2["reason"]


def multi_schema(axes: list[str]) -> dict[str, Any]:
    props: dict[str, Any] = {a: {"type": "integer", "enum": [0, 1, 2]} for a in axes}
    props["reason"] = {"type": "string"}
    return {"type": "object", "additionalProperties": False, "properties": props,
            "required": list(axes) + ["reason"]}


def per_axis_score(sub: dict[str, dict[str, Any]], axes: list[str]) -> tuple[int, str, str]:
    """min() over four INDEPENDENT calls. Because each axis was asked alone, the weakest axis is a
    real attribution rather than a tie-break artefact."""
    lo = min(sub[a]["score"] for a in axes)
    worst = next(a for a in axes if sub[a]["score"] == lo)
    return lo, sub[worst]["reason"], ("none" if lo == 2 else worst)


def gated_score(gate: dict[str, Any], tag: dict[str, Any] | None) -> tuple[int, str, str]:
    if gate["fully_good"]:
        return 2, gate["reason"], "none"
    if tag is None:
        return 1, gate["reason"], "unclassified"     # tagging failed; still not auto-accepted
    if tag["finding"] == MCQ_OK:
        return 1, tag["reason"], "none"
    return 0, tag["reason"], tag["finding"]


def multi_score(answer: dict[str, Any], axes: list[str]) -> tuple[int, str, str]:
    """min() over the sub-scores, and the weakest axis is the error type. Ties resolve in the
    declared axis order so the tag is deterministic."""
    lo = min(answer[a] for a in axes)
    worst = next(a for a in axes if answer[a] == lo)
    return lo, answer["reason"], ("none" if lo == 2 else worst)


def mcq_score(answer: dict[str, Any]) -> tuple[int, str, str]:
    """One answer -> score and error type. Naming a concrete failure condition IS the 0; the model
    never rates quality on an abstract scale."""
    f = answer["finding"]
    if f == MCQ_GOOD:
        return 2, answer["reason"], "none"
    if f == MCQ_OK:
        return 1, answer["reason"], "none"
    return 0, answer["reason"], f


def _structured_params(schema: dict[str, Any]) -> Any:
    """vLLM renamed GuidedDecodingParams -> StructuredOutputsParams (and the SamplingParams field
    guided_decoding -> structured_outputs) between 0.11 (mm env) and the 0.22 nightly (gemma4 env).
    Returns (kwarg_name, value) so the same runner works in both."""
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return "structured_outputs", StructuredOutputsParams(json=schema)
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return "guided_decoding", GuidedDecodingParams(json=schema)


def _blackwell_engine_kwargs() -> dict[str, Any]:
    """Extra engine args needed on sm_120 (RTX PRO 6000) with a CUDA 12.8 driver; empty elsewhere.
    Same set the qwen_vl.py wrapper needed: the bundled flash-attn ships cu129 PTX this driver
    cannot load, so route the decoder and the ViT encoder off it."""
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12:
        return {}
    cuda = "/usr/local/cuda-12.8"
    if os.path.isdir(cuda):
        os.environ.setdefault("CUDA_HOME", cuda)
        os.environ["PATH"] = f"{cuda}/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0a")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return {"attention_backend": "TRITON_ATTN", "mm_encoder_attn_backend": "TORCH_SDPA"}


def run_inference(args: argparse.Namespace, axis: str, version: str, prompts: dict[str, str],
                  items: list[WorkItem], input_failures: list[dict[str, Any]],
                  findings: list[str] | None = None) -> None:
    # Heavy imports are intentionally local: dry-run never reaches this block.
    os.environ.setdefault("VLLM_PLUGINS", "")        # paddle's plugin segfaults vLLM at init
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "judged.jsonl"
    done: set[str] = set()
    if out_path.exists():
        done = {r["sample_id"] for r in read_jsonl(out_path)}
    pending = [item for item in items if item.sample_id not in done]

    processor = AutoProcessor.from_pretrained(args.model)
    extra = _blackwell_engine_kwargs()
    # eager is required on the memory-tight 3090; on a 97 GB card let vLLM capture CUDA graphs
    eager = args.eager if args.eager is not None else not extra
    llm = LLM(
        model=args.model,
        max_model_len=args.max_len,
        gpu_memory_utilization=args.gpu_mem,
        max_num_seqs=args.batch,
        max_num_batched_tokens=max(args.max_len, args.batch * args.max_len // 4),
        enforce_eager=eager,
        limit_mm_per_prompt={"image": 1, "video": 0},
        enable_prefix_caching=False,
        trust_remote_code=True,
        # xgrammar's unbounded-whitespace loop must be killed at ENGINE level (README 1091-1109);
        # the per-request ways silently do nothing
        structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True},
        **extra,
    )

    def generate(batch: list[WorkItem], system: str, schema: dict[str, Any], kind: str
                 ) -> list[tuple[dict[str, Any] | None, str]]:
        reqs = []
        opened = []
        try:
            for item in batch:
                im = Image.open(item.image_path).convert("RGB")
                opened.append(im)
                reqs.append({
                    "prompt": build_chat_prompt(processor, system, item),
                    "multi_modal_data": {"image": im},
                })
            so_key, so_val = _structured_params(schema)
            params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                    **{so_key: so_val})
            outputs = llm.generate(reqs, params)
            parsed = []
            for output in outputs:
                raw = output.outputs[0].text
                try:
                    parsed.append((_parse(raw, kind, findings), raw))
                except Exception:
                    parsed.append((None, raw))
            return parsed
        finally:
            for im in opened:
                im.close()

    started = time.time()
    with open(out_path, "a", encoding="utf-8") as fout:
        for failure in input_failures:
            if failure["sample_id"] not in done:
                fout.write(json.dumps(failure, ensure_ascii=False) + "\n")
        for b0 in range(0, len(pending), args.batch):
            batch = pending[b0:b0 + args.batch]
            if args.variant == "direct":
                outputs = generate(batch, prompts["direct"], DIRECT_SCHEMA, "direct")
                for item, (parsed, raw) in zip(batch, outputs):
                    if parsed is None:
                        rec = _failure_record(axis, args.variant, version, args.model,
                                              item.sample_id, "invalid model JSON")
                        rec["raw"] = raw
                    else:
                        rec = {
                            "sample_id": item.sample_id, "axis": axis,
                            "variant": args.variant, "prompt_version": version,
                            "model": args.model, "score": parsed["score"],
                            "reason": parsed["reason"], "judge_failure": False,
                            "raw": raw,
                        }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            elif args.variant in ("mcq", "multi"):
                # ONE call either way. mcq: the chosen answer carries score + error type.
                # multi: per-part sub-scores, aggregated with min().
                is_multi = args.variant == "multi"
                schema = (multi_schema(findings or []) if is_multi
                          else mcq_schema(findings or []))
                outputs = generate(batch, prompts[args.variant], schema, args.variant)
                for item, (parsed, raw) in zip(batch, outputs):
                    if parsed is None:
                        rec = _failure_record(axis, args.variant, version, args.model,
                                              item.sample_id, "invalid model JSON")
                        rec["raw"] = raw
                    else:
                        score, reason, fault = (multi_score(parsed, findings or []) if is_multi
                                                else mcq_score(parsed))
                        rec = {
                            "sample_id": item.sample_id, "axis": axis,
                            "variant": args.variant, "prompt_version": version,
                            "model": args.model, "score": score, "reason": reason,
                            "fault": fault, "judge_failure": False, "raw": raw,
                            **({"sub_scores": {a: parsed[a] for a in (findings or [])}}
                               if is_multi else {}),
                        }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            elif args.variant == "per_axis":
                axes = findings or []
                per = {a: generate(batch, prompts[VARIANT_KEYS["per_axis"][i]], DIRECT_SCHEMA,
                                   "direct") for i, a in enumerate(axes)}
                for j, item in enumerate(batch):
                    sub = {a: per[a][j][0] for a in axes}
                    if any(v is None for v in sub.values()):
                        rec = _failure_record(axis, args.variant, version, args.model,
                                              item.sample_id, "invalid JSON on an axis call")
                        rec["raw"] = {a: per[a][j][1] for a in axes}
                    else:
                        score, reason, fault = per_axis_score(sub, axes)
                        rec = {"sample_id": item.sample_id, "axis": axis,
                               "variant": args.variant, "prompt_version": version,
                               "model": args.model, "score": score, "reason": reason,
                               "fault": fault, "judge_failure": False,
                               "sub_scores": {a: sub[a]["score"] for a in axes},
                               "sub_reasons": {a: sub[a]["reason"] for a in axes}}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            elif args.variant == "gated":
                gate = generate(batch, prompts["gate"], GOOD_SCHEMA, "good")
                rejected = [item for item, (p, _) in zip(batch, gate)
                            if p is not None and not p["fully_good"]]
                tags = generate(rejected, prompts["tag"], mcq_schema(findings or []), "mcq")
                tag_by_id = dict(zip((x.sample_id for x in rejected), tags))
                for item, (g, raw1) in zip(batch, gate):
                    if g is None:
                        rec = _failure_record(axis, args.variant, version, args.model,
                                              item.sample_id, "invalid gate JSON")
                        rec["raw"] = raw1
                    else:
                        t, raw2 = tag_by_id.get(item.sample_id, (None, None))
                        score, reason, fault = gated_score(g, t)
                        rec = {"sample_id": item.sample_id, "axis": axis,
                               "variant": args.variant, "prompt_version": version,
                               "model": args.model, "score": score, "reason": reason,
                               "fault": fault, "judge_failure": False,
                               "gate": g, "tag": t, "raw": [raw1, raw2]}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            elif args.variant == "cascade":
                first = generate(batch, prompts["bad"], BAD_SCHEMA, "bad")
                second_items = [item for item, (parsed, _) in zip(batch, first)
                                if parsed is not None and not parsed["clear_bad"]]
                second_outputs = generate(second_items, prompts["good"], GOOD_SCHEMA, "good")
                second_by_id = dict(zip((x.sample_id for x in second_items), second_outputs))
                for item, (pass1, raw1) in zip(batch, first):
                    if pass1 is None:
                        rec = _failure_record(axis, args.variant, version, args.model,
                                              item.sample_id, f"invalid {args.variant} pass 1 JSON")
                        rec["cascade"] = {"pass1_raw": raw1}
                    else:
                        pass2, raw2 = second_by_id.get(item.sample_id, (None, None))
                        try:
                            score, reason = cascade_score(pass1, pass2)
                            rec = {
                                "sample_id": item.sample_id, "axis": axis,
                                "variant": args.variant, "prompt_version": version,
                                "model": args.model, "score": score, "reason": reason,
                                "judge_failure": False,
                                "cascade": {"pass1": pass1, "pass1_raw": raw1,
                                            "pass2": pass2, "pass2_raw": raw2},
                            }
                        except Exception as exc:
                            rec = _failure_record(axis, args.variant, version, args.model,
                                                  item.sample_id, f"{args.variant} failure: {exc}")
                            rec["cascade"] = {"pass1": pass1, "pass1_raw": raw1,
                                              "pass2": pass2, "pass2_raw": raw2}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                raise ValueError(f"unknown variant {args.variant!r}")
            fout.flush()

    judged = read_jsonl(out_path)
    failed = sum(1 for r in judged if r.get("judge_failure"))
    rate = failed / max(1, len(judged))
    if rate > 0.01:
        print(f"!! {axis}/{args.variant}: {failed}/{len(judged)} ({rate:.0%}) judge failures - "
              f"if the raw output ends mid-JSON the generation hit --max-tokens ({args.max_tokens}); "
              f"this run is NOT usable", flush=True)
    write_json(out_dir / "run_meta.json", {
        "judge_failures": failed, "judge_failure_rate": round(rate, 4),
        "axis": axis, "variant": args.variant, "prompt_version": version,
        "prompt_sha256": prompt_hash(axis, args.variant, prompts), "model": args.model,
        "split": args.split, "items_in_scope": len(items) + len(input_failures),
        "resumed": len(done), "attempted": len(pending),
        "elapsed_seconds": round(time.time() - started, 2), "output": str(out_path),
    })


def verifier_main(axis: str, version: str, prompts: dict[str, str],
                  item_builder: Callable[[dict[str, Any]], WorkItem],
                  findings: list[str] | None = None,
                  axes: list[str] | None = None) -> None:
    parser = common_parser(axis)
    args = parser.parse_args()
    missing = [k for k in VARIANT_KEYS[args.variant] if k not in prompts]
    if missing:
        raise SystemExit(f"{axis}: --variant {args.variant} needs prompt(s) {missing}, which this "
                         f"judge does not define (has: {sorted(prompts)})")
    if args.variant in ("multi", "per_axis"):
        findings = axes          # for `multi` the same slot carries the sub-score axis list
    if args.variant in ("mcq", "multi", "gated", "per_axis") and not findings:
        raise SystemExit(f"{axis}: --variant {args.variant} needs the answer/axis list from the "
                         f"judge module")
    validate_locked_test(axis, args.variant, args.split, args.winner_file)
    rows = select_manifest_rows(args.manifest, args.split_file, args.split, args.limit)
    items, failures = prepare_items(rows, item_builder, axis, args.variant, version, args.model)
    if args.dry_run:
        write_dry_run(args.out, axis, args.variant, version, args.model,
                      prompts, rows, items, failures)
        print(f"[{axis}] dry-run: {len(items)}/{len(rows)} valid; model not loaded")
        return
    run_inference(args, axis, version, prompts, items, failures, findings)

