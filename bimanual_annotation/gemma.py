# gemma.py

import os
import re
import torch
import transformers as _tf
from transformers import AutoProcessor

# transformers>=5 (Gemma 4) wants `dtype=`; 4.x (Gemma 3) wants `torch_dtype=`.
_DTYPE_KW = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"

# Env-overridable so the SAME code runs gemma-3-4b-it (env `mm`) or
# gemma-4-12B-it (env `gemma4`, with GEMMA_MODEL_ID set). AutoModelForImageTextToText
# dispatches to the right Gemma3/Gemma4 class from each model's config.
DEFAULT_MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
# "transformers" (HF eager) or "vllm" (fast paged-attn). vLLM path = gemma-4 on Blackwell:
# TRITON_ATTN (heterogeneous head_dim) + native sampler (VLLM_USE_FLASHINFER_SAMPLER=0, greedy-equiv).
DEFAULT_BACKEND = os.environ.get("GEMMA_BACKEND", "transformers")


def _load_image(u):
    """Path or http(s) URL -> RGB PIL.Image (vLLM wants the decoded image, not a url)."""
    from PIL import Image
    if isinstance(u, Image.Image):
        return u.convert("RGB")
    if str(u).startswith(("http://", "https://")):
        import io, requests
        return Image.open(io.BytesIO(requests.get(u, timeout=30).content)).convert("RGB")
    return Image.open(u).convert("RGB")


def _parse_thinking(raw):
    """Gemma-4 thinking emits '<|channel>thought ... <channel|> <answer>'. Split on the LAST
    close-of-thought so the trace's own braces don't confuse JSON parsing, then drop stray tags."""
    ans = raw.rsplit("<channel|>", 1)[-1] if "<channel|>" in raw else raw
    return re.sub(r"<[^<>]*>", "", ans).strip()


class Gemma:
    def __init__(
        self,
        model_id=None,
        dtype=torch.bfloat16,
        max_new_tokens=64,
        think_max_new_tokens=4096,   # generous cap so a thinking trace is never truncated (soft-limited in the prompt)
        backend=None,
        # vLLM-only knobs:
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        max_num_seqs=None,             # concurrent requests (vLLM default 128); throughput knob for batching
        max_num_batched_tokens=None,   # per-step token budget = the encoder-cache budget too; raise (>8192)
        max_images=8,
        vision_soft_tokens=1120,     # per-image capacity (hf_overrides); per-request budget = max_soft_tokens
        max_soft_tokens=560,         # image detail (official values: 70/140/280/560/1120)
        enforce_eager=False,         # False -> compile cudagraphs (steady-state speed); True only for quick tests
    ):
        model_id = model_id or DEFAULT_MODEL_ID
        self.model_id = model_id
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.think_max_new_tokens = think_max_new_tokens
        self.backend = (backend or DEFAULT_BACKEND).lower()
        self.max_soft_tokens = max_soft_tokens
        # GEMMA_NO_THINKING=1 forces every call to no-thinking (e.g. to A/B a model w/o the reasoning channel)
        self.no_thinking = os.environ.get("GEMMA_NO_THINKING", "").lower() in ("1", "true", "yes")

        if self.backend == "vllm":
            self._init_vllm(model_id, max_model_len, gpu_memory_utilization,
                            max_images, vision_soft_tokens, enforce_eager,
                            max_num_seqs, max_num_batched_tokens)
        else:
            self._init_transformers(model_id, dtype)

    # ── backends ────────────────────────────────────────────────
    def _init_transformers(self, model_id, dtype):
        from transformers import AutoModelForImageTextToText
        print(f"Loading Gemma (transformers)... ({model_id})")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map="auto", local_files_only=True, **{_DTYPE_KW: dtype},
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        print("Gemma loaded!")

    def _init_vllm(self, model_id, max_model_len, gpu_memory_utilization,
                   max_images, vision_soft_tokens, enforce_eager,
                   max_num_seqs=None, max_num_batched_tokens=None):
        # must be set before importing vllm: spawn (CUDA can't fork) + native sampler (no flashinfer JIT)
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        # Official throughput knobs (docs.vllm.ai/configuration/optimization): max_num_seqs = concurrent
        # requests; max_num_batched_tokens = per-step token budget (ALSO the multi-image encoder-cache
        # budget). Pass only when set so vLLM's own defaults hold otherwise. Batching itself = one
        # llm.generate(list) with continuous batching (see text_images_batch); NO manual chunking.
        sched = {}
        if max_num_seqs is not None:
            sched["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            sched["max_num_batched_tokens"] = max_num_batched_tokens
        print(f"Loading Gemma (vLLM)... ({model_id})  max_model_len={max_model_len} "
              f"soft_tokens={self.max_soft_tokens}/{vision_soft_tokens} enforce_eager={enforce_eager} "
              f"sched={sched or 'defaults'}")
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        self.llm = LLM(
            model=model_id, max_model_len=max_model_len, trust_remote_code=True,
            limit_mm_per_prompt={"image": max_images, "video": 0, "audio": 0},
            hf_overrides={"vision_config": {"default_output_length": vision_soft_tokens},
                          "vision_soft_tokens_per_image": vision_soft_tokens},
            mm_processor_kwargs={"max_soft_tokens": self.max_soft_tokens},
            gpu_memory_utilization=gpu_memory_utilization, enforce_eager=enforce_eager,
            **sched,
        )
        print("Gemma loaded!")

    def _vllm_generate(self, messages, enable_thinking, images):
        from vllm import SamplingParams
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        max_tok = self.think_max_new_tokens if enable_thinking else self.max_new_tokens
        # thinking: keep special tokens so we can split on the channel marker; else clean decode
        sp = SamplingParams(temperature=0.0, max_tokens=max_tok,
                            skip_special_tokens=not enable_thinking)
        req = {"prompt": prompt}
        if images:
            req["multi_modal_data"] = {"image": images}
            req["mm_processor_kwargs"] = {"max_soft_tokens": self.max_soft_tokens}
        raw = self.llm.generate(req, sp, use_tqdm=False)[0].outputs[0].text
        return _parse_thinking(raw) if enable_thinking else raw.strip()

    # ── 純文字 ─────────────────────────
    def text(self, user_text, system_text="You are a helpful assistant."):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
        if self.backend == "vllm":
            return self._vllm_generate(messages, enable_thinking=False, images=None)

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device, dtype=self.dtype)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()

    # ── 圖片 + 文字 ─────────────────────
    def text_image(self, user_text, image_url, system_text="You are a helpful assistant."):
        return self.text_images(user_text, [image_url], system_text)

    # ── 多張圖片 + 文字 (separate full-res views, optional per-image labels) ──
    # enable_thinking=True turns on Gemma-4's reasoning channel; the model emits a thinking trace then
    # the answer. We split on the LAST '<channel|>' (close-of-thought) so the thinking's own inline
    # braces don't confuse JSON parsing, and use a generous token cap (soft-limit via the prompt).
    def text_images(self, user_text, image_urls, system_text="You are a helpful assistant.",
                    labels=None, enable_thinking=False):
        if self.no_thinking:
            enable_thinking = False
        content, images = [], []
        for i, u in enumerate(image_urls):
            if labels and i < len(labels) and labels[i]:
                content.append({"type": "text", "text": labels[i]})   # viewpoint label before its image
            if self.backend == "vllm":
                content.append({"type": "image"})        # placeholder; image passed via multi_modal_data
                images.append(_load_image(u))
            else:
                content.append({"type": "image", "url": u})
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": content},
        ]

        if self.backend == "vllm":
            return self._vllm_generate(messages, enable_thinking, images)

        tmpl_kw = {"enable_thinking": True} if enable_thinking else {}
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", **tmpl_kw,
        ).to(self.model.device, dtype=self.dtype)
        input_len = inputs["input_ids"].shape[-1]
        max_new = self.think_max_new_tokens if enable_thinking else self.max_new_tokens
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        gen = generation[0][input_len:]
        if enable_thinking:
            raw = self.processor.decode(gen, skip_special_tokens=False)
            return _parse_thinking(raw)
        return self.processor.decode(gen, skip_special_tokens=True).strip()

    # ── batched (thinking OFF) ──────────
    def text_images_batch(self, items):
        """Batched generation for a LIST of items, each a dict {user_text, image_urls, system_text,
        labels}. THINKING OFF (one SamplingParams for the whole batch). vLLM: ONE llm.generate over all
        requests (continuous batching -> big throughput). transformers: falls back to a serial loop (no
        real batching). Returns a list of output strings aligned to `items`."""
        if not items:
            return []
        if self.backend != "vllm":
            return [self.text_images(it["user_text"], it["image_urls"],
                                     it.get("system_text", "You are a helpful assistant."),
                                     labels=it.get("labels"), enable_thinking=False) for it in items]
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
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            req = {"prompt": prompt}
            if images:
                req["multi_modal_data"] = {"image": images}
                req["mm_processor_kwargs"] = {"max_soft_tokens": self.max_soft_tokens}
            reqs.append(req)
        sp = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens, skip_special_tokens=True)
        outs = self.llm.generate(reqs, sp, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]
