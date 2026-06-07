"""
Build a daily-used -> affogato mapping (analogue of gobjaverse_to_affogato.py, but for the
gObjaverse "Daily-Used" subset on /nfs_drive).

Chain: daily-used "group/id"  ->  gobjaverse_280k_index_to_objaverse.json  ->  objaverse UID
       ->  affogato/<uid>  (which holds queries.json class_name + xyzc.npy GT).

Uses the manifest list directly (no per-object NFS listdir) and writes ABSOLUTE paths so the
generator can consume src (on /nfs_drive) and dst (top-level dataset/affogato) without any prefix.
"""
import os
import json
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DAILY_ROOT    = "/nfs_drive/gobjaverse/daily-used"
MANIFEST_PATH = "/nfs_drive/gobjaverse/gobjaverse_280k_Daily-Used.json"
INDEX_PATH    = os.path.join(PROJECT_ROOT, "data_generation/dataset/gobjaverse/gobjaverse_280k_index_to_objaverse.json")
AFFOGATO_ROOT = os.path.join(PROJECT_ROOT, "dataset/affogato")          # full 16-part affogato
OUTPUT_PATH   = os.path.join(PROJECT_ROOT, "data_generation/dataset/daily_used_to_affogato.json")


def build_affogato_uid_index(root):
    uid2path = {}
    for part in os.listdir(root):
        pdir = os.path.join(root, part)
        if not os.path.isdir(pdir):
            continue
        for uid in os.listdir(pdir):
            uid2path[uid] = os.path.abspath(os.path.join(pdir, uid))
    return uid2path


def main():
    print("Building affogato UID index from", AFFOGATO_ROOT, "...")
    uid2path = build_affogato_uid_index(AFFOGATO_ROOT)
    print(f"  affogato objects: {len(uid2path)}")

    index_map = json.load(open(INDEX_PATH))
    manifest = json.load(open(MANIFEST_PATH))
    print(f"  index entries: {len(index_map)}   daily-used objects: {len(manifest)}")

    results = []
    miss_index = miss_aff = 0
    for key in tqdm(manifest, desc="mapping"):
        src = os.path.join(DAILY_ROOT, key)                  # /nfs_drive/.../daily-used/<group>/<id>
        glb = index_map.get(key)
        if not glb:
            miss_index += 1
            dst, oid = "", ""
        else:
            uid = os.path.splitext(os.path.basename(glb))[0]
            dst = uid2path.get(uid, "")
            oid = uid if dst else ""
            if not dst:
                miss_aff += 1
        results.append({"src": src, "dst": dst, "object_id": oid})

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    with_dst = sum(1 for r in results if r["dst"])
    print("\nDone ->", OUTPUT_PATH)
    print(f"  total: {len(results)}")
    print(f"  with affogato dst (name + GT): {with_dst} ({100*with_dst/len(results):.1f}%)")
    print(f"  missing index: {miss_index}   missing affogato: {miss_aff}")


if __name__ == "__main__":
    main()
