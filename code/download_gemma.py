"""
Gemma 3 4B 模型下載腳本
只需執行一次，將模型快取到本機 ~/.cache/huggingface/
"""

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

MODEL_ID = "google/gemma-3-4b-it"

print(f"開始下載模型：{MODEL_ID}")
print("首次下載約需 10–20 分鐘，視網速而定...\n")

Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)

AutoProcessor.from_pretrained(MODEL_ID)

print("\n✅ 模型下載完成，已快取至 ~/.cache/huggingface/")
print("之後執行 gemma3_inference.py 將直接從本機載入，不需重新下載。")