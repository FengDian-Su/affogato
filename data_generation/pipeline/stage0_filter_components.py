#!/usr/bin/env python
"""
Stage 0 — object filtering + component extraction.

For each object in the gObjaverse->affogato mapping:
  load name -> sample N views -> 2x2 grid -> Gemma FILTER (keep/drop) ->
  if kept: Gemma COMPONENT extraction -> emit one record with
  {object_id, object_name, views_used, keep, filter, components}.

Output feeds stage1_task_role_assemble.py.

Reuses bimanual_annotation/{gemma.py, get_component.py}. Run with the **gemma4** env
(transformers 5.x + gemma-4-12B-it). Paths default to data_generation/-relative.

  CUDA_VISIBLE_DEVICES=3 \
  /home/michaellee/miniconda3/envs/gemma4/bin/python \
      data_generation/stage0_filter_components.py \
      --mapping dataset/daily_used_to_affogato.json \
      --out outputs/stage0_filtered.json --start 0 --end 50
"""
import os
import sys
import math
import json
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
# never clutters the repo
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
    "system": "You screen objects for a bimanual (two-arm) robot manipulation dataset.",
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
    ap.add_argument("--model_id", default=os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-12B-it"))
    ap.add_argument("--fresh", action="store_true", help="ignore existing output; restart from scratch")
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
    from get_component import load_object_name, sample_views, make_grid, ComponentExtractor

    ce = ComponentExtractor()            # loads Gemma (uses GEMMA_MODEL_ID) + parse prompt
    model = ce.model

    data = [d for d in json.load(open(mapping)) if d.get("dst")]
    end = args.end if args.end is not None else len(data)
    batch = data[args.start:end]
    print(f"stage0: {len(batch)} objects [{args.start}:{end}] of {len(data)} -> {out}")

    results, done = [], set()
    if not args.fresh and os.path.exists(out):
        try:
            results = json.load(open(out))
            done = {r.get("object_id") for r in results}
            print(f"resume: {len(results)} already processed in {out}; skipping those")
        except Exception:
            results, done = [], set()

    for i, ex in enumerate(batch):
        object_id = ex["object_id"]
        if object_id in done:
            continue
        try:
            name = load_object_name(ex["dst"])
            views = sample_views_gobjaverse(ex["src"], n_views=args.views)
            if name == "unknown object" or len(views) < 5:
                results.append({"object_id": object_id, "object_name": name,
                                "views_used": views, "keep": False,
                                "skip_reason": f"name/views (got {len(views)})"})
                continue

            pairs = safe_prepare_views(views, VIEW_DIR)   # [(orig_idx, clean_path), ...]
            if len(pairs) < 5:
                results.append({"object_id": object_id, "object_name": name, "views_used": views,
                                "keep": False, "skip_reason": "corrupt/unreadable renders"})
                print(f"  [{args.start+i}] {name[:28]:28} SKIP (corrupt renders)")
                continue
            clean = [p for _, p in pairs]
            labels = [view_label(views[idx]) for idx, _ in pairs]   # viewpoint label per image
            filt = parse_json(model.text_images(
                FILTER_PROMPT["user"].format(object_name=name), clean, FILTER_PROMPT["system"], labels=labels))
            keep = bool(filt.get("keep"))
            rec = {"object_id": object_id, "object_name": name, "views_used": views,
                   "keep": keep, "filter": filt, "components": []}
            if keep:
                pp = ce.prompts["parse"]
                rec["components"] = parse_json(model.text_images(
                    pp["user"].format(object_name=name), clean, pp["system"], labels=labels)).get("canonical_components", [])
                if not rec["components"]:
                    # passed the filter but no groundable parts -> unusable, treat as fail
                    rec["keep"] = False
                    rec["skip_reason"] = "no components extracted"
            results.append(rec)
            print(f"  [{args.start+i}] {name[:28]:28} keep={rec['keep']}  "
                  f"comps={len(rec['components'])}  ex={filt.get('example_task')}")
        except Exception as e:
            results.append({"object_id": object_id, "keep": False, "error": str(e)})
            print(f"  [{args.start+i}] ERROR {object_id}: {e}")

        with open(out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # qualified-only file: keep == true now means (passed filter) AND (>=1 component)
    kept_records = [r for r in results if r.get("keep")]
    with open(kept_out, "w") as f:
        json.dump(kept_records, f, indent=2, ensure_ascii=False)

    print(f"\nstage0 done: {len(kept_records)}/{len(batch)} kept (filter + components)")
    print(f"  full log   -> {out}")
    print(f"  qualified  -> {kept_out}")


if __name__ == "__main__":
    main()
