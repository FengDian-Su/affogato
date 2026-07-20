#!/usr/bin/env python
"""
Stage 0 QC JUDGE — full-coverage semantic audit of stage0_full KEPT records.

For EVERY kept object (61,399 across 13 parts): feed its 8 views + accepted components to Gemma
and verify the acceptance holds on the actual geometry. Flags the junk taxonomy that a 275-object
image-verified audit confirmed (2026-07-20): multi-object scenes, corrupted/degenerate meshes,
flat slivers/graphic panels, human figures/statues, one-hand tiny items, ungrounded components.

Output: outputs/stage0_full/judge_part{N}.json — one record per kept object:
  {object_id, object_name, verdict: ok|junk, junk_types: [...], actual_object,
   components_grounded, ungrounded_components, notes}
Resume: re-running skips object_ids already judged (error stubs re-run). One model load for the
whole run (iterates parts internally). Windowed batching identical to stage0 (W=8 @ 40960).

  /home/michaellee/miniconda3/envs/gemma4/bin/python \
      data_generation/pipeline/stage0_judge.py --gpu 0
"""
import os
import sys
import json
import time
import shutil
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))            # data_generation/verify/
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline"))
from stage0_filter_components import (REPO_ROOT, DATA_GEN, BIMANUAL_DIR, parse_json,
                                      safe_prepare_views, view_label)

VIEW_DIR = os.path.join("/tmp", f"tmp_views_judge_{os.getpid()}")

JUDGE_PROMPT = {
    "system": (
        "You are a strict quality auditor for a 3D bimanual-manipulation object dataset. Each item "
        "was ALREADY accepted by an earlier filter; your job is to verify that acceptance against "
        "the actual rendered geometry. WORK IN THIS ORDER: first describe what the GEOMETRY shows "
        "as if you had never seen the category name (fill actual_object from shape alone); only "
        "then compare it to the claim. The name is the single most misleading signal in this "
        "dataset - corrupted blobs, flat sprites, and random scans routinely carry functional-"
        "sounding names, and a texture PAINTED with keypads, buttons, text, or pixel art is not "
        "geometry: a feature counts only if it exists as actual 3D shape (edges, recesses, "
        "protrusions). Do not rationalize - if the claimed function's defining shape is not "
        "visibly there, say so. Low-poly but clearly-shaped objects are fine; an undifferentiated "
        "primitive or blob that only the NAME makes functional is not."
    ),
    "user": (
        "Claimed object: {object_name}\n"
        "Accepted manipulable components:\n{components}\n"
        "The images are several views of the item (oblique ring, then top-down, then underside).\n\n"
        "Verify the acceptance with these checks:\n"
        "1. SINGLE_OBJECT: the item is ONE object - not a multi-object scene, diorama, or a set of "
        "separate items arranged on a base/ground plane (an axe embedded in a stump on a grass "
        "mound is a scene). One object on a small pedestal/display stand still passes.\n"
        "2. GEOMETRY_OK: a recognizable coherent object emerges from the SHAPE ITSELF - it fails "
        "for an amorphous/melted blob from every angle, disconnected fragments floating apart, an "
        "object fused into a large ground-plane slab (scan artifact), or a bare primitive "
        "(sphere/cylinder/cube with no distinguishing structure) whose claimed identity depends "
        "entirely on the name or on painted texture. A pot needs a visible opening or rim "
        "geometry; an ATM needs actual recessed/protruding panels, not a printed front. "
        "CARVE-OUT: thin PARTS are normal (a knife blade, hat brim, page, table top are supposed "
        "to be thin); flat_or_sliver applies when the ENTIRE object is one paper-thin sheet or an "
        "extruded flat sprite with no volumetric structure.\n"
        "3. IDENTITY: the geometry shows a real FUNCTIONAL object. It fails for a full human "
        "figure or clothed mannequin (head/limbs visible), a statue/figurine/decorative sculpture/"
        "relief, a flat card or panel whose face merely carries a graphic, or an abstract shape "
        "that matches NO functional object. CARVE-OUTS: if the claimed name is wrong but the "
        "geometry is still clearly SOME usable object (a box, a container, furniture), it passes - "
        "note the mismatch in actual_object instead. Ignore printed text / texture content "
        "entirely.\n"
        "4. MANIPULABLE: two hands would realistically manipulate it in everyday use - it fails "
        "for a tiny item one hand fully handles (a card, coin, marble, crumpled paper wad); judge "
        "size from the object's real-world identity.\n"
        "5. COMPONENTS: each accepted component above is actually VISIBLE on the geometry - a "
        "component invented from the category name (e.g. a 'lid' on a seamless solid, an 'inner "
        "rim' with no visible opening) is ungrounded.\n\n"
        "verdict is \"ok\" only if checks 1-4 ALL pass (components affect components_grounded, not "
        "the verdict).\n\n"
        "Return JSON only:\n"
        "{{\n"
        "  \"verdict\": \"ok|junk\",\n"
        "  \"junk_types\": [\"multi_object_scene|corrupted_mesh|flat_or_sliver|"
        "human_figure_or_statue|flat_graphic_panel|one_hand_tiny|not_functional_object|"
        "fixed_installation|other\"],\n"
        "  \"actual_object\": \"<what the geometry really shows, a few words>\",\n"
        "  \"components_grounded\": true/false,\n"
        "  \"ungrounded_components\": [\"<names from the accepted list that are not visible>\"],\n"
        "  \"notes\": \"<one concise sentence>\"\n"
        "}}"
    ),
}


def components_to_text(comps):
    return "\n".join(f"- {c.get('name')}: {c.get('interaction', '')}" for c in comps) or "- (none)"


def judge_batch(model, window):
    """ONE text_images_batch over the window; returns judge records aligned to window."""
    outs = model.text_images_batch([{
        "user_text": JUDGE_PROMPT["user"].format(
            object_name=w["name"], components=components_to_text(w["comps"])),
        "image_urls": w["clean"], "system_text": JUDGE_PROMPT["system"],
        "labels": w["labels"]} for w in window])
    recs = []
    for w, o in zip(window, outs):
        j = parse_json(o)
        recs.append({"object_id": w["oid"], "object_name": w["name"],
                     "verdict": j.get("verdict"), "junk_types": j.get("junk_types") or [],
                     "actual_object": j.get("actual_object"),
                     "components_grounded": j.get("components_grounded"),
                     "ungrounded_components": j.get("ungrounded_components") or [],
                     "notes": j.get("notes"),
                     **({"parse_error": j["error"]} if "error" in j else {})})
    return recs


def run_part(model, part, in_path, out_path, batch_size):
    kept = [r for r in json.load(open(in_path)) if r.get("keep")]
    results, done = [], set()
    if os.path.exists(out_path):
        try:
            results = [r for r in json.load(open(out_path)) if "error" not in r]
            done = {r["object_id"] for r in results}
        except Exception:
            results, done = [], set()
    todo = [r for r in kept if r["object_id"] not in done]
    print(f"judge part{part}: {len(kept)} kept, {len(done)} done, {len(todo)} to judge", flush=True)

    def write():
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)
        os.replace(tmp, out_path)

    window = []

    def flush():
        nonlocal window
        if not window:
            return
        t0 = time.time()
        try:
            recs = judge_batch(model, window)
        except Exception as e:
            print(f"  window of {len(window)} failed ({e}); retrying one by one", flush=True)
            recs = []
            for w in window:
                try:
                    recs.extend(judge_batch(model, [w]))
                except Exception as e1:
                    recs.append({"object_id": w["oid"], "object_name": w["name"],
                                 "error": str(e1) or type(e1).__name__})
        results.extend(recs)
        write()
        junk = sum(1 for r in recs if r.get("verdict") == "junk")
        print(f"  [part{part} {len(results)}/{len(kept)}] window {len(window)} objs "
              f"{time.time()-t0:.0f}s  junk+{junk}", flush=True)
        for w in window:
            shutil.rmtree(w["odir"], ignore_errors=True)
        window = []

    for r in todo:
        views = r.get("views_used") or []
        odir = os.path.join(VIEW_DIR, str(r["object_id"]))
        pairs = safe_prepare_views(views, odir)
        if len(pairs) < 5:
            results.append({"object_id": r["object_id"], "object_name": r.get("object_name"),
                            "error": "views unreadable"})
            shutil.rmtree(odir, ignore_errors=True)
            continue
        window.append({"oid": r["object_id"], "name": r.get("object_name"),
                       "comps": r.get("components") or [], "odir": odir,
                       "clean": [p for _, p in pairs],
                       "labels": [view_label(views[idx]) for idx, _ in pairs]})
        if len(window) >= batch_size:
            flush()
    flush()
    write()
    junk_total = sum(1 for r in results if r.get("verdict") == "junk")
    print(f"judge part{part} done: {len(results)} judged, {junk_total} junk -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="outputs/stage0_full")
    ap.add_argument("--parts", default="0-12", help="e.g. 0-12 or 3")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_num_batched_tokens", type=int, default=40960)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--model_id", default=os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-26B-A4B-it"))
    ap.add_argument("--judge_model", default="qwen", choices=["gemma", "qwen"],
                    help="judge family. Default qwen (Qwen3.5-35B-A3B): CROSS-FAMILY QC - the "
                         "gemma filter must never be verified by its own family (gemma option kept "
                         "for ablation only). Outputs: judge_qwen_part{N}.json / judge_part{N}.json")
    ap.add_argument("--limit", type=int, default=None, help="judge at most N objects per part (smoke)")
    args = ap.parse_args()

    d = args.dir if os.path.isabs(args.dir) else os.path.join(DATA_GEN, args.dir)
    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    os.chdir(REPO_ROOT)
    sys.path.insert(0, BIMANUAL_DIR)
    if args.judge_model == "qwen":
        from qwen_vl import QwenVL
        model = QwenVL(model_id="Qwen/Qwen3.5-35B-A3B", max_new_tokens=args.max_new_tokens,
                       gpu_memory_utilization=args.gpu_mem,
                       max_num_batched_tokens=args.max_num_batched_tokens)
    else:
        os.environ["GEMMA_MODEL_ID"] = args.model_id
        from gemma import Gemma
        model = Gemma(max_new_tokens=args.max_new_tokens, backend="vllm",
                      gpu_memory_utilization=args.gpu_mem,
                      max_num_batched_tokens=args.max_num_batched_tokens)

    prefix = "judge_qwen" if args.judge_model == "qwen" else "judge"
    lo, _, hi = args.parts.partition("-")
    parts = list(range(int(lo), int(hi or lo) + 1))
    for p in parts:
        in_path = os.path.join(d, f"stage0_part{p}.json")
        out_path = os.path.join(d, f"{prefix}_part{p}.json")
        if args.limit is not None:
            # smoke mode: temporary shallow run — judge first N kept only, separate output
            kept = [r for r in json.load(open(in_path)) if r.get("keep")][:args.limit]
            tmp_in = os.path.join("/tmp", f"judge_smoke_in_{os.getpid()}.json")
            json.dump(kept, open(tmp_in, "w"))
            run_part(model, p, tmp_in, os.path.join(d, f"{prefix}_smoke_part{p}.json"), args.batch_size)
        else:
            run_part(model, p, in_path, out_path, args.batch_size)


if __name__ == "__main__":
    main()
