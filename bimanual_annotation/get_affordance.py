import os
import json
from PIL import Image
from gemma import Gemma


# =========================
# Config
# =========================
PROMPT_PATH = "prompt/get_component_prompt.json"
INPUT_PATH = "data_generation/output_components_test.json"
GRID_PATH = "tmp_grid.png"


# =========================
# Utility: load prompts
# =========================
def load_prompts(prompt_path: str) -> dict:
    with open(prompt_path, "r") as f:
        return json.load(f)


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
class AffordanceExtractor:
    def __init__(self):
        self.model = Gemma(max_new_tokens=512)
        self.prompts = load_prompts(PROMPT_PATH)

    # =========================
    # Affordance reasoning
    # =========================
    def reason_affordance(self, object_name, components):
        prompt = self.prompts["affordance"]

        user_text = prompt["user"].format(
            object_name=object_name,
            components=json.dumps(components, indent=2)
        )

        raw = self.model.text(
            user_text=user_text,
            system_text=prompt["system"]
        )

        return self.parse_json(raw)

    # =========================
    # Full pipeline
    # =========================
    def predict(self, sample):

        object_name = sample["object_name"]
        components = sample["final_components"]
        image_paths = sample["views_used"]

        # ===== Step 0: make grid =====
        make_grid(image_paths, GRID_PATH)

        # ===== Step 1: affordance reasoning =====
        affordance = self.reason_affordance(object_name, components)

        return {
            "object_id": sample["object_id"],
            "object_name": object_name,
            "final_components": components,
            "affordance": affordance,
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
 
    extractor = AffordanceExtractor()
 
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)
 
    # build lookup by object_id for efficient update
    data_map = {sample["object_id"]: sample for sample in data}
 
    # 這裡控制 data 資料筆數
    for sample in data[:10]:
        if not sample.get("operable", False):
            print(f"Skipping {sample.get('object_id')} ({sample.get('object_name')}): not operable")
            continue
 
        result = extractor.predict(sample)
 
        print(json.dumps(result, indent=2))
 
        # update affordance in-place by object_id
        data_map[result["object_id"]]["affordance"] = result["affordance"]
 
        with open(INPUT_PATH, "w") as f:
            json.dump(list(data_map.values()), f, indent=2)