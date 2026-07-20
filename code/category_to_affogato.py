"""
Build a gObjaverse-category -> affogato mapping (generalises dailyused_to_affogato.py to any
category: Electronics / Furnitures / Daily-Used).

Chain: category manifest "group/id" -> gobjaverse_280k_index_to_objaverse.json -> objaverse UID
       -> affogato/<uid> (holds queries.json class_name + xyzc.npy GT).

Reads the MANIFEST (no per-object NFS listdir, so a partially-synced render tree cannot silently
truncate the mapping — the failure mode of the original electronics mapping, which was built by
scanning a half-downloaded directory against a half-downloaded affogato) and writes ABSOLUTE
src/dst plus object_id, the record shape stage0 consumes.

  python code/category_to_affogato.py --category electronics
  python code/category_to_affogato.py --category furnitures --src_root /nfs_drive/gobjaverse/furnitures
"""
import os
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_GEN = os.path.join(PROJECT_ROOT, "data_generation")
GOBJ_DIR = os.path.join(DATA_GEN, "dataset", "gobjaverse")
INDEX_PATH = os.path.join(GOBJ_DIR, "gobjaverse_280k_index_to_objaverse.json")
AFFOGATO_ROOT = os.path.join(PROJECT_ROOT, "dataset", "affogato")        # full 16-part affogato

CATEGORIES = {                      # category -> (manifest file, default render root)
    "electronics": ("gobjaverse_280k_Electronics.json", "/nfs_drive/gobjaverse/electronics"),
    "furnitures": ("gobjaverse_280k_Furnitures.json", "/nfs_drive/gobjaverse/furnitures"),
    "daily-used": ("gobjaverse_280k_Daily-Used.json", "/nfs_drive/gobjaverse/daily-used"),
}


def build_affogato_uid_index(root):
    uid2path = {}
    for part in sorted(os.listdir(root)):
        pdir = os.path.join(root, part)
        if not os.path.isdir(pdir):
            continue
        for uid in os.listdir(pdir):
            uid2path[uid] = os.path.abspath(os.path.join(pdir, uid))
    return uid2path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", required=True, choices=sorted(CATEGORIES))
    ap.add_argument("--src_root", default=None, help="render tree root (default per category)")
    ap.add_argument("--out", default=None,
                    help="output json (default data_generation/dataset/<category>_to_affogato.json)")
    args = ap.parse_args()

    manifest_file, default_root = CATEGORIES[args.category]
    src_root = args.src_root or default_root
    out = args.out or os.path.join(DATA_GEN, "dataset", f"{args.category}_to_affogato.json")

    print(f"affogato index from {AFFOGATO_ROOT} ...")
    uid2path = build_affogato_uid_index(AFFOGATO_ROOT)
    print(f"  affogato objects: {len(uid2path)}")

    index_map = json.load(open(INDEX_PATH))
    manifest = json.load(open(os.path.join(GOBJ_DIR, manifest_file)))
    print(f"  index entries: {len(index_map)}   {args.category} objects: {len(manifest)}")

    results, miss_index, miss_aff = [], 0, 0
    for key in manifest:
        glb = index_map.get(key)
        if not glb:
            miss_index += 1
            dst = oid = ""
        else:
            uid = os.path.splitext(os.path.basename(glb))[0]
            dst = uid2path.get(uid, "")
            oid = uid if dst else ""
            if not dst:
                miss_aff += 1
        results.append({"src": os.path.join(src_root, key), "dst": dst, "object_id": oid})

    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    with_dst = sum(1 for r in results if r["dst"])
    print(f"\nDone -> {out}")
    print(f"  total: {len(results)}")
    print(f"  with affogato dst: {with_dst} ({100 * with_dst / len(results):.1f}%)")
    print(f"  missing index: {miss_index}   missing affogato: {miss_aff}")


if __name__ == "__main__":
    main()
