"""
Gemma 3 4B 推論腳本
需先執行 download_model.py 完成下載
之後每次執行都從本機快取載入，不會重新下載
"""

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

MODEL_ID = "google/gemma-3-4b-it"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512

# ── 從本機快取載入（不會重新下載）────────────────────────────────────────────
print("從本機載入模型中...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=DTYPE,
    local_files_only=True,   # 強制只從本機讀取，不連網
).eval()

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    local_files_only=True,
)
print("模型載入完成！\n")


# ── 純文字對話 ────────────────────────────────────────────────────────────────
def chat(user_text: str, system_text: str = "You are a helpful assistant.") -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=DTYPE)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    return processor.decode(generation[0][input_len:], skip_special_tokens=True)


# ── 圖片 + 文字對話 ───────────────────────────────────────────────────────────
def chat_with_image(user_text: str, image_url: str, system_text: str = "You are a helpful assistant.") -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=DTYPE)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            top_p=None,
            top_k=None,
        )

    return processor.decode(generation[0][input_len:], skip_special_tokens=True)


# ── 執行單次對話 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 【純文字】先註解起來，要用時取消註解
    response = chat("你好，請用繁體中文介紹一下你自己")
    print(response)

    # 【圖片 + 文字】
    # response = chat_with_image(
    #     user_text="請描述這張圖片的內容",
    #     image_url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    # )
    print(response)