import os
import json

# ====== 路徑設定 ======
BASE_ROOT = "/mnt/home/sufengdian/affogato/dataset/gobjaverse/electronics"
INDEX_JSON_PATH = "/mnt/home/sufengdian/affogato/dataset/gobjaverse/gobjaverse_280k_index_to_objaverse.json"
AFFOGATO_ROOT = "/mnt/home/sufengdian/affogato/dataset/affogato"
OUTPUT_JSON_PATH = "/mnt/home/sufengdian/affogato/dataset/objaverse_to_affogato.json"


# ====== 載入 index JSON ======
with open(INDEX_JSON_PATH, "r") as f:
    index_map = json.load(f)


# ====== 搜尋 function ======
def find_affogato_path(root_dir, target_id):
    for root, dirs, files in os.walk(root_dir):
        if target_id in dirs:
            return os.path.join(root, target_id)
    return ""


# ====== 主流程 ======
results = []

for i in range(0, 160):  # 👈 包含 0 ~ 159

    gobj_root = os.path.join(BASE_ROOT, str(i))

    if not os.path.exists(gobj_root):
        continue

    for folder_name in os.listdir(gobj_root):

        src_path = os.path.join(gobj_root, folder_name)

        if not os.path.isdir(src_path):
            continue

        key = f"{i}/{folder_name}"

        # ===== 查 index =====
        if key not in index_map:
            dst_path = ""
        else:
            glb_path = index_map[key]
            objaverse_id = os.path.basename(glb_path).replace(".glb", "")
            dst_path = find_affogato_path(AFFOGATO_ROOT, objaverse_id)

        results.append({
            "src": src_path,
            "dst": dst_path
        })


# ====== 一次寫入 JSON ======
with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(results, f, indent=2)