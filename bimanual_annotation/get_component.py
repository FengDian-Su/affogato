import os
import json
from PIL import Image
from gemma import Gemma


# =========================
# Config
# =========================
NUM_VIEWS = 4
PROMPT_PATH = "prompt/get_component_prompt.json"
GRID_PATH = "tmp_grid.png"


# =========================
# Utility: load prompts
# =========================
def load_prompts(prompt_path: str) -> dict:
    with open(prompt_path, "r") as f:
        return json.load(f)


# =========================
# Utility: load object name
# =========================
def load_object_name(dst_path: str) -> str:
    query_path = os.path.join(dst_path, "queries.json")

    if not os.path.exists(query_path):
        return "unknown object"

    try:
        with open(query_path, "r") as f:
            data = json.load(f)
        return data[0].get("class_name", "unknown object")
    except:
        return "unknown object"


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
# Main Class
# =========================
class ComponentExtractor:
    def __init__(self):
        self.model = Gemma(max_new_tokens=512)
        self.prompts = load_prompts(PROMPT_PATH)

    # =========================
    # Step 1: filter
    # =========================
    def is_operable(self, grid_path, object_name):
        prompt = self.prompts["filter"]

        user_text = prompt["user"].format(object_name=object_name)

        raw = self.model.text_image(
            user_text=user_text,
            image_url=grid_path,
            system_text=prompt["system"]
        )

        return raw.strip().lower() == "yes"

    # =========================
    # Step 2: parse grid image
    # =========================
    def parse_grid(self, grid_path, object_name):
        prompt = self.prompts["parse"]

        user_text = prompt["user"].format(object_name=object_name)

        raw = self.model.text_image(
            user_text=user_text,
            image_url=grid_path,
            system_text=prompt["system"]
        )

        parsed = self.parse_json(raw)

        # list[dict]: [{"name": "...", "interaction": "..."}, ...]
        return parsed.get("canonical_components", [])

    # =========================
    # Full pipeline
    # =========================
    def predict(self, src_path, dst_path):

        object_name = load_object_name(dst_path)

        # ===== Step 0: sample views =====
        image_paths = sample_views(src_path, NUM_VIEWS)

        # ===== Step 1: make grid =====
        grid_path = make_grid(image_paths, GRID_PATH)

        # ===== Step 2: filter =====
        if not self.is_operable(grid_path, object_name):
            return {
                "object_name": object_name,
                "views_used": image_paths,
                "operable": False,
                "final_components": [],
            }

        # ===== Step 3: parse grid =====
        final_components = self.parse_grid(grid_path, object_name)

        return {
            "object_name": object_name,
            "views_used": image_paths,
            "operable": True,
            "final_components": final_components,
        }

    # =========================
    # JSON parsing
    # =========================
    def parse_json(self, text):
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {"error": "parse failed", "raw": text}


# =========================
# Main
# =========================
if __name__ == "__main__":

    extractor = ComponentExtractor()

    json_path = "data_generation/dataset/gobjaverse_to_affogato_nj.json"
    output_path = "data_generation/output_components_test.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    # load existing results if any
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            content = f.read().strip()
            results = json.loads(content) if content else []
    else:
        results = []

    # 這裡控制 data 資料筆數
    for sample in data[:10]:
        src_path = f"data_generation/{sample["src"]}"
        dst_path = f"data_generation/{sample["dst"]}"
        object_id = os.path.basename(dst_path)

        result = extractor.predict(
            src_path=src_path,
            dst_path=dst_path,
        )

        result["object_id"] = object_id

        print(json.dumps(result, indent=2))

        results.append(result)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)