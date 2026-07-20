# qwen_vl.py — minimal vLLM wrapper for Qwen3.5 / Qwen3-VL multimodal models exposing the SAME
# text_images_batch(items) API as gemma.py's Gemma, so stage0_judge.py can swap judge families
# (cross-family QC: never let a model family verify its own filtering decisions alone).
import os


def _load_image(u):
    from PIL import Image
    if isinstance(u, Image.Image):
        return u.convert("RGB")
    return Image.open(u).convert("RGB")


class QwenVL:
    def __init__(self, model_id="Qwen/Qwen3.5-35B-A3B", max_new_tokens=512,
                 max_model_len=32768, gpu_memory_utilization=0.85,
                 max_images=8, max_num_batched_tokens=None, max_num_seqs=64):
        # max_num_seqs: the hybrid GDN/Mamba layers allocate one Mamba cache block per decode
        # seq; the nightly default (1024) exceeds the blocks available at 0.85 gpu_mem (385).
        # Judge windows are 8 concurrent requests, so 64 is ample.
        # Blackwell sm_120 on driver CUDA 12.8 (per vllm-blackwell guide + this box's constraints):
        # - flashinfer JIT needs a FULL CUDA toolkit on PATH (PyPI fragments can't compile sm_12x);
        #   /usr/local/cuda-12.8's nvcc supports sm_120
        # - spawn: CUDA can't fork
        import sys as _sys
        cuda = "/usr/local/cuda-12.8"
        env_bin = os.path.dirname(_sys.executable)          # ninja lives here (pip-installed)
        if os.path.isdir(cuda):
            os.environ.setdefault("CUDA_HOME", cuda)
            os.environ["PATH"] = f"{cuda}/bin:{env_bin}:" + os.environ.get("PATH", "")
        # torch on this driver (12.8) can't read sm_120 device capability ("SM 12.x requires
        # CUDA >= 12.9"), leaving flashinfer's TARGET_CUDA_ARCHS empty -> "No supported CUDA
        # architectures". Its documented override is FLASHINFER_CUDA_ARCH_LIST; a SUFFIXED arch
        # ("12.0a") is respected as-is, skipping the normalizer that demands CUDA >= 12.9 for the
        # "f" family suffix - and nvcc 12.8 does compile sm_120a.
        os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0a")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # greedy-equivalent native sampler (gemma-proven on this box); the flashinfer sampler
        # probe imports/JITs more sm_12x kernels than we need
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        from transformers import AutoProcessor
        self.max_new_tokens = max_new_tokens
        self.backend = "vllm"
        self.processor = AutoProcessor.from_pretrained(model_id)
        sched = {}
        if max_num_seqs is not None:
            sched["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            sched["max_num_batched_tokens"] = max_num_batched_tokens
        print(f"Loading QwenVL (vLLM)... ({model_id})")
        kw = dict(model=model_id, max_model_len=max_model_len, trust_remote_code=True,
                  limit_mm_per_prompt={"image": max_images, "video": 0},
                  gpu_memory_utilization=gpu_memory_utilization,
                  # the bundled vllm_flash_attn ships cu129 PTX this box's 12.8 driver can't load
                  # (cudaErrorUnsupportedPtxVersion) -> route BOTH the decoder (TRITON_ATTN, the
                  # backend gemma-4 uses here; this nightly takes it as an engine arg, the old
                  # VLLM_ATTENTION_BACKEND env is gone) and the ViT encoder (TORCH_SDPA) off it.
                  attention_backend="TRITON_ATTN",
                  mm_encoder_attn_backend="TORCH_SDPA", **sched)
        if os.environ.get("QWEN_MOE_TRITON", "").lower() in ("1", "true"):
            kw["kernel_config"] = {"moe_backend": "triton"}   # fallback if flashinfer JIT fails
        self.llm = LLM(**kw)
        print("QwenVL loaded!")

    def text_images_batch(self, items):
        """items: [{user_text, image_urls, system_text, labels}] -> list of output strings."""
        if not items:
            return []
        from vllm import SamplingParams
        reqs = []
        for it in items:
            content, images = [], []
            labels = it.get("labels")
            for i, u in enumerate(it["image_urls"]):
                if labels and i < len(labels) and labels[i]:
                    content.append({"type": "text", "text": labels[i]})
                content.append({"type": "image"})
                images.append(_load_image(u))
            content.append({"type": "text", "text": it["user_text"]})
            messages = [
                {"role": "system", "content": [{"type": "text",
                                                "text": it.get("system_text", "You are a helpful assistant.")}]},
                {"role": "user", "content": content},
            ]
            try:
                # Qwen3.5 thinks by default (official serve uses --reasoning-parser); the judge
                # needs plain JSON, and thinking would eat the whole max_tokens budget
                prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                            add_generation_prompt=True,
                                                            enable_thinking=False)
            except TypeError:
                prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                            add_generation_prompt=True)
            req = {"prompt": prompt}
            if images:
                req["multi_modal_data"] = {"image": images}
            reqs.append(req)
        sp = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens, skip_special_tokens=True)
        outs = self.llm.generate(reqs, sp, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]
