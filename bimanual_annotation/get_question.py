import os
import json
from gemma import Gemma


# =========================
# Config
# =========================
PROMPT_PATH = "prompt/get_component_prompt.json"
INPUT_PATH = "data_generation/output_components_test.json"


# =========================
# Utility: load prompts
# =========================
def load_prompts(prompt_path: str) -> dict:
    with open(prompt_path, "r") as f:
        return json.load(f)


# =========================
# Main Class
# =========================
class QuestionGenerator:
    def __init__(self):
        self.model = Gemma(max_new_tokens=256)
        self.prompts = load_prompts(PROMPT_PATH)

    # =========================
    # Generate question
    # =========================
    def generate_question(self, object_name, reason, affordance_parts):
        prompt = self.prompts["question"]

        user_text = prompt["user"].format(
            object_name=object_name,
            reason=reason,
            affordance_parts=affordance_parts
        )

        raw = self.model.text(
            user_text=user_text,
            system_text=prompt["system"]
        )

        return raw.strip()

    # =========================
    # Full pipeline
    # =========================
    def predict(self, sample):

        object_name = sample["object_name"]
        affordance = sample["affordance"]
        reason = affordance["reason"]
        affordance_parts = affordance["affordance_parts"]

        question = self.generate_question(object_name, reason, affordance_parts)

        return {
            "object_id": sample["object_id"],
            "object_name": object_name,
            "question": question,
            "answer": affordance_parts,
        }


# =========================
# Main
# =========================
if __name__ == "__main__":

    generator = QuestionGenerator()

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    # build lookup by object_id for efficient update
    data_map = {sample["object_id"]: sample for sample in data}

    # 這裡控制 data 資料筆數
    for sample in data[:10]:
        if not sample.get("operable", False):
            print(f"Skipping {sample.get('object_id')} ({sample.get('object_name')}): not operable")
            continue

        if "affordance" not in sample:
            print(f"Skipping {sample.get('object_id')} ({sample.get('object_name')}): no affordance")
            continue

        result = generator.predict(sample)

        print(json.dumps(result, indent=2))

        # update question and answer in-place by object_id
        data_map[result["object_id"]]["question"] = result["question"]
        data_map[result["object_id"]]["answer"] = result["answer"]

        with open(INPUT_PATH, "w") as f:
            json.dump(list(data_map.values()), f, indent=2)