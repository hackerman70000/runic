"""End-to-end text generation with the watermark logits processor injected."""

from __future__ import annotations

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from runic.core.config import WatermarkConfig
from runic.generation.processor import WatermarkLogitsProcessor


class WatermarkGenerator:
    def __init__(
        self,
        model_name: str,
        config: WatermarkConfig | None = None,
        device: str = "auto",
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model_name = model_name
        self.config = config or WatermarkConfig()
        self.device = self._resolve_device(device)
        self.dtype = dtype or self._default_dtype(self.device)

        logger.info(f"Loading {model_name} on {self.device} ({self.dtype})")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype
        )
        model.to(torch.device(self.device))  # type: ignore[arg-type]
        model.eval()  # type: ignore[no-untyped-call]
        self.model = model

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.processor = WatermarkLogitsProcessor(self.config, vocab_size=len(self.tokenizer))

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _default_dtype(device: str) -> torch.dtype:
        return torch.float16 if device in {"cuda", "mps"} else torch.float32

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 200,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        watermark: bool = True,
        seed: int | None = None,
    ) -> str:
        if seed is not None:
            torch.manual_seed(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        processors = LogitsProcessorList([self.processor]) if watermark else None

        output = self.model.generate(  # type: ignore[operator]
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            logits_processor=processors,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output[0, prompt_len:]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return decoded if isinstance(decoded, str) else " ".join(decoded)
