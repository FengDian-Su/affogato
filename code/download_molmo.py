"""
Molmo 7B-D 模型下載腳本
授權：Apache 2.0，不需要申請，可直接下載
只需執行一次，將模型快取到本機 (~/.cache/huggingface/)
"""

from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "allenai/Molmo-7B-D-0924"

print(f"開始下載模型：{MODEL_ID}")
print("首次下載約需 10–20 分鐘，請耐心等候...\n")

# 下載 processor
print("下載 Processor 中...")
AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
print("✓ Processor 下載完成")

# 下載模型權重
print("下載模型權重中...")
AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
print("✓ 模型權重下載完成")

print("\n全部下載完成！之後執行 infer_molmo.py 會直接從本機載入。")