#!/usr/bin/env python
"""
Stage 0 — object filtering + component extraction (vLLM-batched).

For each object in the gObjaverse->affogato mapping:
  load name -> sample N views -> Gemma FILTER (keep/drop) ->
  if kept: Gemma COMPONENT extraction -> emit one record with
  {object_id, object_name, views_used, keep, filter, components}.

Output feeds stage1 (stage1_v2.py).

BATCHED like stage1_v2: objects are processed in windows of --batch_size; per window the
FILTER phase runs as ONE llm.generate over the whole window (vLLM continuous batching),
then the COMPONENT phase runs as ONE llm.generate over the kept subset. The kept objects'
views were just encoded in the FILTER phase, so with the encoder-cache budget
(--max_num_batched_tokens) they are NOT re-encoded in phase 2 (encode-once locality,
same mechanism stage1 tuned: window 8 x 8 views = 64 images fit a 40960 budget).
Backend defaults to vLLM (GEMMA_BACKEND / --backend override); on transformers the same
code runs serially (text_images_batch falls back to a loop).

Reuses bimanual_annotation/{gemma.py, get_component.py}. Run with the **gemma4** env
(transformers 5.x + vllm nightly cu129). Paths default to data_generation/-relative.

  /home/michaellee/miniconda3/envs/gemma4/bin/python \
      data_generation/pipeline/stage0_filter_components.py \
      --mapping dataset/daily_used_to_affogato.json \
      --out outputs/stage0_filtered.json --start 0 --end 50 --gpu 3 --batch_size 8
"""
import os
import sys
import math
import json
import time
import shutil
import argparse
import subprocess

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate corrupt/truncated render PNGs (libpng CRC errors)

HERE = os.path.dirname(os.path.abspath(__file__))            # data_generation/pipeline/


def _find_repo_root(d):
    """Walk up until we find the repo (the dir containing bimanual_annotation/)."""
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "bimanual_annotation")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.dirname(HERE))


REPO_ROOT = _find_repo_root(HERE)
DATA_GEN = os.path.join(REPO_ROOT, "data_generation")
BIMANUAL_DIR = os.path.join(REPO_ROOT, "bimanual_annotation")
# per-process temp dir for cleaned views (concurrent shards must NOT share one); under /tmp so it
# never clutters the repo. Each object gets its own subdir: a window holds several objects' clean
# views ALIVE at once, so they cannot share one flat dir.
VIEW_DIR = os.path.join("/tmp", f"tmp_views_stage0_{os.getpid()}")


def parse_json(text):
    """Lenient: grab the outermost {...} and json.load it."""
    try:
        s = text.find("{"); e = text.rfind("}") + 1
        return json.loads(text[s:e])
    except Exception as ex:
        return {"error": f"parse failed: {ex}", "raw": text}


# Re-save each view as a clean PNG in an ISOLATED subprocess: a corrupt/truncated render decodes
# with a libpng CRC error -> hard crash (SIGBUS), uncatchable in-process. Isolating it (with a
# timeout for hung NFS reads) drops the bad view (or skips the object) instead of killing the batch.
_PREP_WORKER = (
    "import sys, os\n"
    "from PIL import Image, ImageFile\n"
    "ImageFile.LOAD_TRUNCATED_IMAGES = True\n"
    "out_dir = sys.argv[1]; views = sys.argv[2:]\n"
    "for i, p in enumerate(views):\n"
    "    try:\n"
    "        op = os.path.join(out_dir, f'v{i:02d}.png')\n"
    "        Image.open(p).convert('RGB').save(op)\n"
    "        print(f'{i}\\t{op}', flush=True)\n"
    "    except Exception:\n"
    "        pass\n"
)


# gObjaverse fixed 40-view rig (verified across objects): 0-24 upper orbit, 25 = top-down (+90),
# 26 = underside (-90), 27-39 = eye-level orbit. The old 4-view 2x2 grid missed the top-down AND the
# underside AND downscaled each view. We now feed ~8 SEPARATE full-res views spanning elevations.
def _gobja_indices(n):
    """n views: (n-2) spread around the OBLIQUE ring (elevated orbit 0-24, ~23.6deg - the rig's only
    angled elevation, so each view shows top + sides together) + the top-down (25) and underside (26).
    NOTE: this rig has NO oblique top/bottom, so 25/26 are necessarily straight (+/-90)."""
    n_ring = max(0, n - 2)

    def spread(lo, hi, k):           # k azimuths evenly around the cyclic orbit [lo..hi]
        if k <= 0:
            return []
        span = hi - lo + 1
        return [lo + round(span * j / k) for j in range(k)]

    return spread(0, 24, n_ring) + [25, 26]


def sample_views_gobjaverse(src_path, n_views=8):
    """Elevation-spanning SEPARATE views (incl. top-down + underside); fall back to even spacing."""
    dirs = sorted(d for d in os.listdir(src_path)
                  if d.isdigit() and os.path.isdir(os.path.join(src_path, d)))
    have = {int(d): d for d in dirs}
    pth = lambda d: os.path.join(src_path, d, f"{d}.png")
    if len(have) >= 27 and 25 in have and 26 in have:
        out = [pth(have[i]) for i in _gobja_indices(n_views) if i in have and os.path.exists(pth(have[i]))]
        if len(out) >= 5:
            return out
    if not dirs:
        return []
    step = max(len(dirs) // n_views, 1)
    return [pth(d) for d in dirs[::step][:n_views] if os.path.exists(pth(d))]


def safe_prepare_views(views, out_dir, timeout=120):
    """Re-save views as clean PNGs in an isolated subprocess; return [(orig_index, clean_path), ...]
    for the views that survived (a corrupt/hung render is dropped)."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        try:
            os.remove(os.path.join(out_dir, f))
        except OSError:
            pass
    out = []
    try:
        r = subprocess.run([sys.executable, "-c", _PREP_WORKER, out_dir, *views],
                           capture_output=True, text=True, timeout=timeout)
        for ln in r.stdout.splitlines():
            parts = ln.strip().split("\t")
            if len(parts) == 2 and parts[1].endswith(".png") and os.path.exists(parts[1]):
                out.append((int(parts[0]), parts[1]))
    except subprocess.TimeoutExpired:
        return []
    return out


def view_label(view_path):
    """Human-readable viewpoint label for a gObjaverse view, from its camera json."""
    d = os.path.dirname(view_path)
    idx = os.path.basename(d)
    try:
        o = json.load(open(os.path.join(d, f"{idx}.json"))).get("origin")
        r = math.sqrt(sum(t * t for t in o)) or 1.0
        elev = math.degrees(math.asin(o[2] / r))
        az = math.degrees(math.atan2(o[1], o[0])) % 360
    except Exception:
        return "A view of the object."
    if elev > 60:
        return ("TOP-DOWN view: camera looking straight DOWN on the TOP of the object "
                "(use it to see lids, openings, controls, or anything on the top face).")
    if elev < -60:
        return "UNDERSIDE view: camera looking straight UP at the BOTTOM / base of the object."
    return (f"Oblique view from ~{round(elev)} deg above, rotated to azimuth ~{round(az)} deg around "
            f"the object (shows its top edge plus the side facing the camera).")


# Object filter: KEEP if (everyday object) AND (a two-handed task exists) — tests
# EXISTENCE of a bimanual task, not "designed for two hands" (so a box is kept).
FILTER_PROMPT = {
    # "system": "You screen objects for a bimanual (two-arm) robot manipulation dataset.",
    "system": (
        "You are a data quality filter for a bimanual (two-arm) robot manipulation dataset. "
        "Your job is to decide whether a 3D object should be included in the dataset.\n\n"
        "Core filtering principle:\n"
        "KEEP only objects that are (1) real, everyday functional objects people genuinely "
        "handle with their hands, AND (2) have at least one realistic task requiring both hands "
        "to coordinate at the same time.\n\n"
        "When in doubt, REJECT. It is better to exclude a borderline object than to let "
        "noise into the dataset. A false negative (missing a good object) is far less harmful "
        "than a false positive (including a decorative plaque or figurine).\n\n"
        "Judge from the rendered 3D geometry, not the category name. A functional-sounding "
        "name may belong to a flat decorative panel — trust what you see."
    ),
    "user": (
        "Object category: {object_name}\n"
        "The images are several views of ONE object from different angles (including a top-down "
        "view and an underside view).\n\n"
        "Decide whether to KEEP this object. KEEP only if BOTH hold:\n"
        "1. EVERYDAY FUNCTIONAL OBJECT: it must be a REAL, common, functional physical object that "
        "people genuinely pick up and USE/manipulate with their hands in everyday life - e.g. "
        "kitchenware, tableware, tools, appliances/electronics you operate, bottles/jars/containers, "
        "boxes/packaging, bags, movable furniture, household items. REJECT anything that is NOT such "
        "an object, INCLUDING: badges / emblems / medallions / coins / pins / plaques / trophies / "
        "awards; signs / signage / boards / posters / banners / nameplates / labels / logos / text or "
        "graphic panels; decorative sculptures / ornaments / figurines / reliefs / wall art; fantasy / "
        "game / cartoon characters; body parts; abstract or artistic shapes / 3D scans of surfaces; "
        "vehicles / buildings / large fixed structures. If it is mainly decorative, a flat graphic or "
        "sign, an emblem/badge, or not something a person routinely handles in order to USE it, REJECT.\n"
        "JUDGE FROM THE ACTUAL RENDERED GEOMETRY, NOT THE CATEGORY NAME: a name like 'Sunglasses', "
        "'Cocktail Shaker', or 'Machine' can be attached to a flat decorative plaque. Use the TOP-DOWN "
        "and UNDERSIDE views - if the object is essentially FLAT / 2D / a single thin slab or panel "
        "(it appears as just a thin strip from above and below) whose front face merely carries a "
        "graphic, emblem, logo, or text, it is a sign/plaque: REJECT it even though the name implies a "
        "functional 3D object.\n"
        "2. BIMANUAL TASK EXISTS: there is at least ONE realistic task in which two hands "
        "would coordinate to operate or move it. Judge whether such a task EXISTS, not "
        "whether the object was 'designed' for two hands. Carrying / lifting / moving a "
        "large or heavy object counts (two hands needed to carry it), as do opening, "
        "folding, twisting, pulling apart, stabilize-and-actuate, etc.\n"
        "REJECT only if even with two hands there is no meaningful coordinated task "
        "(e.g. a tiny object one hand fully handles, or something you never manipulate).\n\n"
        "Return JSON only:\n"
        "{{\n"
        "  \"everyday_object\": true/false,\n"
        "  \"bimanual_task_exists\": true/false,\n"
        "  \"example_task\": \"one realistic two-handed task, or null\",\n"
        "  \"keep\": true/false\n"
        "}}"
    ),
}


# ---------------------------------------------------------------- batched pipeline
def run_batched(model, parse_prompt, load_name, batch, args, out, done, results):
    """Batched path: objects are prepared serially (cheap CPU/NFS work) into windows of
    args.batch_size; per window, FILTER runs as ONE model.text_images_batch over the whole window,
    then COMPONENT extraction as ONE batch over the kept subset (vLLM continuous-batches internally;
    no manual chunking). Writes the file + frees temp view dirs per window. A window that raises is
    recorded as error stubs; error stubs are re-run on the next resume."""
    window = []   # [{obj, gi, name, odir, views, clean, labels}]

    def write():
        tmp = out + ".tmp"                                     # atomic: a crash mid-dump can't truncate
        with open(tmp, "w") as f:                              # the accumulated output file
            json.dump(results, f, indent=2, ensure_ascii=False)
        os.replace(tmp, out)

    def run_window(objs):
        """FILTER + COMPONENT the given objects (2 batched calls); return their records (no side
        effects until return, so a raise appends nothing and the caller can retry per-object)."""
        filts = [parse_json(o) for o in model.text_images_batch([{
            "user_text": FILTER_PROMPT["user"].format(object_name=w["name"]),
            "image_urls": w["clean"], "system_text": FILTER_PROMPT["system"],
            "labels": w["labels"]} for w in objs])]
        kept_idx = [i for i, f in enumerate(filts) if bool(f.get("keep"))]
        comp_out = model.text_images_batch([{
            "user_text": parse_prompt["user"].format(object_name=objs[i]["name"]),
            "image_urls": objs[i]["clean"], "system_text": parse_prompt["system"],
            "labels": objs[i]["labels"]} for i in kept_idx])
        comps = {i: (parse_json(o).get("canonical_components") or [])
                 for i, o in zip(kept_idx, comp_out)}
        recs = []
        for i, w in enumerate(objs):
            rec = {"object_id": w["obj"]["object_id"], "object_name": w["name"],
                   "views_used": w["views"], "keep": i in comps and bool(comps[i]),
                   "filter": filts[i], "components": comps.get(i, [])}
            if i in comps and not comps[i]:
                # passed the filter but no groundable parts -> unusable, treat as fail
                rec["skip_reason"] = "no components extracted"
            recs.append(rec)
        return recs

    def flush():
        if not window:
            return
        t0 = time.time()
        try:
            recs = run_window(window)
        except Exception as e:                                 # a bad window shouldn't kill the run:
            print(f"  window of {len(window)} failed ({e}); retrying objects one by one")
            recs = []                                          # fall back to per-object isolation so ONE
            for w in window:                                   # deterministically-bad object can't stub
                try:                                           # (and livelock) its window-mates
                    recs.extend(run_window([w]))
                except Exception as e1:
                    recs.append({"object_id": w["obj"]["object_id"], "keep": False,
                                 "error": str(e1) or type(e1).__name__})
                    print(f"  [{w['gi']}] ERROR {w['obj']['object_id']}: {e1}")
        for rec, w in zip(recs, window):
            results.append(rec)
            if "error" not in rec:
                print(f"  [{w['gi']}] {w['name'][:28]:28} keep={rec['keep']}  "
                      f"comps={len(rec['components'])}  ex={rec['filter'].get('example_task')}")
        write()
        print(f"  [window {len(window)} objs] {time.time() - t0:.0f}s")
        for w in window:
            shutil.rmtree(w["odir"], ignore_errors=True)       # free temp views (bounded /tmp)
        window.clear()

    for i, ex in enumerate(batch):
        gi = args.start + i
        object_id = ex["object_id"]
        if object_id in done:
            continue                                           # its record is already in `results`
        try:
            name = load_name(ex["dst"])
            views = sample_views_gobjaverse(ex["src"], n_views=args.views)
            if name == "unknown object" or len(views) < 5:
                results.append({"object_id": object_id, "object_name": name,
                                "views_used": views, "keep": False,
                                "skip_reason": f"name/views (got {len(views)})"})
                continue
            odir = os.path.join(VIEW_DIR, str(object_id))
            pairs = safe_prepare_views(views, odir)            # [(orig_idx, clean_path), ...]
            if len(pairs) < 5:
                results.append({"object_id": object_id, "object_name": name, "views_used": views,
                                "keep": False, "skip_reason": "corrupt/unreadable renders"})
                print(f"  [{gi}] {name[:28]:28} SKIP (corrupt renders)")
                shutil.rmtree(odir, ignore_errors=True)        # dir was created above -> don't leak it
                continue
            window.append({"obj": ex, "gi": gi, "name": name, "odir": odir, "views": views,
                           "clean": [p for _, p in pairs],
                           "labels": [view_label(views[idx]) for idx, _ in pairs]})
        except Exception as e:
            results.append({"object_id": object_id, "keep": False,
                            "error": str(e) or type(e).__name__})
            print(f"  [{gi}] ERROR {object_id}: {e}")
        if len(window) >= args.batch_size:
            flush()
    flush()
    write()                                                    # persist trailing skip/error records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", default="dataset/daily_used_to_affogato.json",
                    help="gobjaverse->affogato mapping json (data_generation/-relative)")
    ap.add_argument("--out", default="outputs/stage0_filtered.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None, help="exclusive; default = all")
    ap.add_argument("--gpu", type=int, default=3, help="CUDA device index (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--views", type=int, default=8,
                    help="# separate views to feed (gObjaverse: top-down + underside + orbit azimuths)")
    ap.add_argument("--model_id", default=os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-26B-A4B-it"))
    ap.add_argument("--fresh", action="store_true", help="ignore existing output; restart from scratch")
    ap.add_argument("--backend", default=os.environ.get("GEMMA_BACKEND", "vllm"),
                    choices=["vllm", "transformers"],
                    help="vllm = batched fast path (default); transformers = old eager serial")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="objects per window. All of a window's views must fit vLLM's encoder cache "
                         "(max_num_batched_tokens/560 imgs) so the kept objects' 8 views encode ONCE "
                         "in the FILTER phase and are reused by the COMPONENT phase. 8 objs x 8 views "
                         "= 64 imgs fits the 40960 default. Measured 07-15 (26B-A4B, GPU0): W=8@40960 "
                         "1.6s/obj; W=16@40960 2.1 (cache overflow re-encodes); W=16@81920 1.7; "
                         "W=32@163840+gpu_mem 0.92 1.6 - throughput PLATEAUS once the window fits, "
                         "and a bigger step budget only squeezes KV (W=32@0.85 fails startup: KV<min), "
                         "so W=8 is the optimum, not just a safe default.")
    ap.add_argument("--gpu_mem", type=float, default=0.85,
                    help="vLLM gpu_memory_utilization (bigger -> more KV cache -> larger concurrent batch)")
    ap.add_argument("--max_num_seqs", type=int, default=None,
                    help="vLLM max concurrent requests (vLLM default 128); lower if OOM/preemption")
    ap.add_argument("--max_num_batched_tokens", type=int, default=40960,
                    help="vLLM per-step token budget (= encoder-cache budget); 40960 fits a full "
                         "8-obj window's 64 images (64*560=35.8k soft tokens)")
    ap.add_argument("--max_new_tokens", type=int, default=1024,
                    help="output token cap. Old serial path used 512 (occasionally truncating a long "
                         "components JSON mid-list); greedy stops at EOS so headroom is ~free")
    args = ap.parse_args()

    # resolve paths relative to data_generation/ BEFORE we chdir
    mapping = args.mapping if os.path.isabs(args.mapping) else os.path.join(DATA_GEN, args.mapping)
    out = args.out if os.path.isabs(args.out) else os.path.join(DATA_GEN, args.out)
    kept_out = os.path.splitext(out)[0] + ".kept.json"   # qualified-only (keep == true)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # GPU + model selection must be set before torch / gemma import
    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["GEMMA_MODEL_ID"] = args.model_id

    os.chdir(REPO_ROOT)                  # so prompt/get_component_prompt.json resolves
    sys.path.insert(0, BIMANUAL_DIR)     # so `from gemma import Gemma` resolves
    from gemma import Gemma
    from get_component import load_object_name, load_prompts, PROMPT_PATH

    parse_prompt = load_prompts(PROMPT_PATH)["parse"]    # COMPONENT-extraction prompt (unchanged)

    data = [d for d in json.load(open(mapping)) if d.get("dst")]
    end = args.end if args.end is not None else len(data)
    batch = data[args.start:end]
    print(f"stage0: {len(batch)} objects [{args.start}:{end}] of {len(data)} -> {out}")

    # resume: keep ALL prior records (even outside [start:end) - shards may share one file);
    # only error stubs INSIDE the current slice are dropped so their objects re-run.
    results, done = [], set()
    if not args.fresh and os.path.exists(out):
        try:
            prior = json.load(open(out))
            batch_ids = {ex["object_id"] for ex in batch}
            results = [r for r in prior
                       if not ("error" in r and r.get("object_id") in batch_ids)]
            done = {r.get("object_id") for r in results}
            n_err = len(prior) - len(results)
            print(f"resume: {len(results)} records kept from {out}"
                  + (f"; {n_err} error stubs re-run" if n_err else ""))
        except Exception:
            results, done = [], set()

    model = Gemma(max_new_tokens=args.max_new_tokens, backend=args.backend,
                  gpu_memory_utilization=args.gpu_mem, max_num_seqs=args.max_num_seqs,
                  max_num_batched_tokens=args.max_num_batched_tokens,
                  max_images=max(8, args.views))
    if getattr(model, "backend", None) != "vllm":     # text_images_batch only truly batches on vLLM
        print("  WARNING: backend != vLLM -> batching INACTIVE (serial speed)", flush=True)
    print(f"  filter+components batched, window={args.batch_size}")

    run_batched(model, parse_prompt, load_object_name, batch, args, out, done, results)

    # qualified-only file: keep == true now means (passed filter) AND (>=1 component)
    kept_records = [r for r in results if r.get("keep")]
    with open(kept_out, "w") as f:
        json.dump(kept_records, f, indent=2, ensure_ascii=False)

    print(f"\nstage0 done: {len(kept_records)}/{len(batch)} kept (filter + components)")
    print(f"  full log   -> {out}")
    print(f"  qualified  -> {kept_out}")


if __name__ == "__main__":
    main()
