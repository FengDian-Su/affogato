import os
import json
from tqdm import tqdm

# =========================================================
# 🔧 Project Root
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data_generation/"
print(f"Project Root: {PROJECT_ROOT}")

# =========================================================
# 🔧 Paths
# =========================================================
BASE_ROOT = os.path.join(PROJECT_ROOT, "dataset/gobjaverse/electronics")
INDEX_JSON_PATH = os.path.join(PROJECT_ROOT, "dataset/gobjaverse/gobjaverse_280k_index_to_objaverse.json")
AFFOGATO_ROOT = os.path.join(PROJECT_ROOT, "dataset/affogato")

OUTPUT_JSON_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset/gobjaverse_to_affogato.json"
)

# =========================================================
# 🔧 Step 1: 建立 affogato index（in-memory）
# =========================================================
def build_affogato_index(root):
    mapping = {}
    for sub in os.listdir(root):
        sub_path = os.path.join(root, sub)

        if not os.path.isdir(sub_path):
            continue

        for obj_id in os.listdir(sub_path):
            full_path = os.path.join(sub_path, obj_id)

            if os.path.isdir(full_path):
                mapping[obj_id] = full_path

    return mapping


print("🔨 Building affogato index...")
affogato_index = build_affogato_index(AFFOGATO_ROOT)
print(f"Total affogato objects: {len(affogato_index)}")

# =========================================================
# 🔧 Step 2: 載入 index_map
# =========================================================
with open(INDEX_JSON_PATH, "r") as f:
    index_map = json.load(f)

# =========================================================
# 🔧 Step 3: 收集 tasks（全部）
# =========================================================
tasks = []

for i in sorted(os.listdir(BASE_ROOT)):
    gobj_root = os.path.join(BASE_ROOT, i)

    if not os.path.isdir(gobj_root):
        continue

    for folder_name in os.listdir(gobj_root):
        src_path = os.path.join(gobj_root, folder_name)

        if os.path.isdir(src_path):
            tasks.append((i, folder_name, src_path))

print(f"📊 Total tasks: {len(tasks)}")

# =========================================================
# 🔧 Step 4: mapping
# =========================================================
results = []

missing_index = 0
missing_affogato = 0

for i, folder_name, src_path in tqdm(tasks, desc="Processing"):

    key = f"{i}/{folder_name}"

    dst_path = ""

    if key in index_map:
        glb_path = index_map[key]
        objaverse_id = os.path.splitext(os.path.basename(glb_path))[0]

        dst_path = affogato_index.get(objaverse_id, "")

        if dst_path == "":
            missing_affogato += 1
    else:
        missing_index += 1

    # 轉相對路徑
    rel_src = os.path.relpath(src_path, PROJECT_ROOT)
    rel_dst = os.path.relpath(dst_path, PROJECT_ROOT) if dst_path else ""

    results.append({
        "src": rel_src,
        "dst": rel_dst
    })

# =========================================================
# 🔧 Step 5: 寫入
# =========================================================
os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Done!")
print(f"Saved to: {OUTPUT_JSON_PATH}")
print(f"Missing index_map: {missing_index}")
print(f"Missing affogato: {missing_affogato}")
print(f"Total output: {len(results)}")