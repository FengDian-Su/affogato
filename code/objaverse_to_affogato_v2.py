import os
import json
from tqdm import tqdm

# ====== 可調參數 ======
START = 120
END = 159  # inclusive

# ====== 路徑設定 ======
BASE_ROOT = "/mnt/home/sufengdian/affogato/dataset/gobjaverse/electronics"
INDEX_JSON_PATH = "/mnt/home/sufengdian/affogato/dataset/gobjaverse/gobjaverse_280k_index_to_objaverse.json"
AFFOGATO_ROOT = "/mnt/home/sufengdian/affogato/dataset/affogato"
OUTPUT_JSON_PATH = "/mnt/home/sufengdian/affogato/dataset/objaverse_to_affogato_v2.json"


# ====== 載入 index JSON ======
with open(INDEX_JSON_PATH, "r") as f:
    index_map = json.load(f)


# ====== 搜尋 function ======
def find_affogato_path(root_dir, target_id):
    for root, dirs, files in os.walk(root_dir):
        if target_id in dirs:
            return os.path.join(root, target_id)
    return ""


# ====== 收集 tasks ======
tasks = []

for i in range(START, END + 1):

    gobj_root = os.path.join(BASE_ROOT, str(i))

    if not os.path.exists(gobj_root):
        continue

    for folder_name in os.listdir(gobj_root):
        src_path = os.path.join(gobj_root, folder_name)

        if os.path.isdir(src_path):
            tasks.append((i, folder_name, src_path))


# ====== 主流程 ======
for i, folder_name, src_path in tqdm(tasks, desc="Processing"):

    key = f"{i}/{folder_name}"

    # ===== 查 index =====
    if key not in index_map:
        dst_path = ""
    else:
        glb_path = index_map[key]
        objaverse_id = os.path.basename(glb_path).replace(".glb", "")
        dst_path = find_affogato_path(AFFOGATO_ROOT, objaverse_id)

    new_entry = {
        "src": src_path,
        "dst": dst_path
    }

    # ===== 讀取既有 JSON =====
    if os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, "r") as f:
            try:
                results = json.load(f)
            except:
                results = []
    else:
        results = []

    # ===== 檢查是否已存在（避免重複）=====
    existing_srcs = set([x["src"] for x in results])

    if src_path in existing_srcs:
        continue

    # ===== append =====
    results.append(new_entry)

    # ===== 寫回 =====
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)