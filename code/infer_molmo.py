"""
Molmo 7B-D 推論腳本
需先執行 download_molmo.py 完成下載
"""

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import torch

MODEL_ID = "allenai/Molmo-7B-D-0924"

# ── 從本機快取載入（不會重新下載）────────────────────────────────────────────
print("從本機載入模型中...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # 從 "auto"(float32) 改為 bfloat16，省約一半 VRAM
    device_map="auto",           # 自動分配到兩張卡
    local_files_only=True,
)

print("模型載入完成！\n")


# ── 純文字對話 ────────────────────────────────────────────────────────────────
def chat(user_text: str) -> str:
    inputs = processor.process(
        images=None,
        text=user_text,
    )
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )

    generated_tokens = output[0, inputs["input_ids"].size(1):]
    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


# ── 圖片 + 文字對話 ───────────────────────────────────────────────────────────
def chat_with_image(user_text: str, image_url: str) -> str:
    image = Image.open(requests.get(image_url, stream=True).raw)

    # 確保圖片為 RGB 格式
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor.process(
        images=[image],
        text=user_text,
    )
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )

    generated_tokens = output[0, inputs["input_ids"].size(1):]
    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


# ── 執行單次對話 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 【純文字】先註解起來，要用時取消註解
    # response = chat("請用繁體中文介紹一下你自己")
    # print(response)

    # 【圖片 + 文字】
    response = chat_with_image(
        user_text="Describe this image.",
        image_url="https://picsum.photos/id/237/536/354",
    )
    print(response)