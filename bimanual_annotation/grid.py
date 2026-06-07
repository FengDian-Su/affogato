import os
import json
from PIL import Image


# =========================
# Config
# =========================
INPUT_PATH = "output_components.json"
OUTPUT_DIR = "grids"


# =========================
# Utility: sample views
# =========================
def sample_views(src_path: str, num_views=4):
    views = sorted(os.listdir(src_path))

    total = len(views)
    if total == 0:
        return []

    step = max(total // num_views, 1)

    selected = [views[i] for i in range(0, total, step)]
    selected = selected[:num_views]

    image_paths = [
        os.path.join(src_path, v, f"{v}.png")
        for v in selected
    ]

    return image_paths


# =========================
# Utility: make 2x2 grid
# =========================
def make_grid(image_paths: list, output_path: str):
    images = [Image.open(p).convert("RGB") for p in image_paths]

    w, h = images[0].size
    grid = Image.new("RGB", (w * 2, h * 2))

    positions = [(0, 0), (w, 0), (0, h), (w, h)]
    for img, pos in zip(images, positions):
        grid.paste(img, pos)

    grid.save(output_path)
    return output_path


# =========================
# Main
# =========================
if __name__ == "__main__":

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for sample in data[1:3]:
        object_id = sample.get("object_id")
        image_paths = sample.get("views_used", [])

        if not image_paths:
            print(f"Skipping {object_id}: no views found")
            continue

        output_path = os.path.join(OUTPUT_DIR, f"{object_id}.jpg")
        make_grid(image_paths, output_path)
        print(f"Saved: {output_path}")