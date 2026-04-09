import json
import argparse

from read_npy import visualize_xyzc


# ====== JSON 路徑 ======
JSON_PATH = "dataset/gobjaverse_to_affogato.json"


def load_mapping(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def find_item(data, object_id):
    for item in data:
        if object_id in item["dst"]:
            return item
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_id", type=str, required=True)
    args = parser.parse_args()

    object_id = args.object_id

    # ====== 1. load json ======
    data = load_mapping(JSON_PATH)

    # ====== 2. find item ======
    item = find_item(data, object_id)

    if item is None:
        raise ValueError(f"❌ 找不到 object_id: {object_id}")

    dst_path = item["dst"]
    src_path = item["src"]

    print(f"✅ src path: {src_path}/00000/00000.png")

    # ====== 3. call read_npy function ======
    visualize_xyzc(dst_path)


if __name__ == "__main__":
    main()