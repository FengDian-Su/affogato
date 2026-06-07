import os
import json
import shutil
from tqdm import tqdm

# ====== 路徑設定 ======
AFFOGATO_ROOT = "dataset/affogato"
GOBJAVERSE_ROOT = "dataset/gobjaverse"
MAPPING_JSON = "dataset/gobjaverse_to_affogato.json"
OUTPUT_ROOT = "dataset/bimanual_dataset"

# ====== 行為設定 ======
SKIP_IF_EXISTS = True   # 已存在就跳過
VERBOSE = False         # 印詳細 log

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ====== 讀 JSON ======
with open(MAPPING_JSON, "r") as f:
    mappings = json.load(f)

print(f"Total mappings: {len(mappings)}")

# ====== 主流程 ======
num_success = 0
num_skip = 0
num_fail = 0

for item in tqdm(mappings):
    try:
        src_rel = item["src"]
        dst_rel = item["dst"]

        # ====== 絕對路徑 ======
        src_path = os.path.join("/mnt/home/sufengdian/affogato", src_rel)
        dst_path = os.path.join("/mnt/home/sufengdian/affogato", dst_rel)

        # ====== object_id ======
        object_id = os.path.basename(dst_path)

        target_dir = os.path.join(OUTPUT_ROOT, object_id)
        target_mvi = os.path.join(target_dir, "mvi")
        target_npy = os.path.join(target_dir, "xyzc.npy")

        # ====== skip if exists ======
        if SKIP_IF_EXISTS and os.path.exists(target_dir):
            num_skip += 1
            continue

        # ====== 檢查來源 ======
        xyzc_src = os.path.join(dst_path, "xyzc.npy")

        if not os.path.exists(xyzc_src):
            num_fail += 1
            if VERBOSE:
                print(f"[FAIL] missing xyzc: {xyzc_src}")
            continue

        if not os.path.exists(src_path):
            num_fail += 1
            if VERBOSE:
                print(f"[FAIL] missing mvi src: {src_path}")
            continue

        # ====== 建立資料夾 ======
        os.makedirs(target_dir, exist_ok=True)

        # ====== copy xyzc.npy ======
        shutil.copy2(xyzc_src, target_npy)

        # ====== copy mvi (整個資料夾內容) ======
        # 注意：copy "contents"，不是包一層
        shutil.copytree(src_path, target_mvi)

        num_success += 1

    except Exception as e:
        num_fail += 1
        if VERBOSE:
            print(f"[ERROR] {e}")
        continue

# ====== summary ======
print("\n===== DONE =====")
print(f"Success: {num_success}")
print(f"Skipped: {num_skip}")
print(f"Failed: {num_fail}")