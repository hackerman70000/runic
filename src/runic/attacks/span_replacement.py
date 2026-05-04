"""T5 span-replacement attack (Kirchenbauer §7.1).

Iteratively masks single words in the watermarked text and asks T5 to
infill them. The attacker stops once a fraction ``budget`` of tokens has
been replaced. Increasing ``budget`` raises perplexity but degrades
watermark detection.
"""

from __future__ import annotations

import random

import torch
from loguru import logger
from transformers import T5ForConditionalGeneration, T5Tokenizer


class SpanReplacementAttack:
    def __init__(
        self,
        replacement_model: str = "t5-large",
        device: str = "auto",
    ) -> None:
        self.device = self._resolve_device(device)
        logger.info(f"Loading replacement LM '{replacement_model}' on {self.device}")
        self.tokenizer = T5Tokenizer.from_pretrained(replacement_model)
        model = T5ForConditionalGeneration.from_pretrained(replacement_model)
        model.to(torch.device(self.device))  # type: ignore[arg-type]
        model.eval()  # type: ignore[no-untyped-call]
        self.model = model

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @torch.no_grad()
    def attack(
        self,
        text: str,
        *,
        budget: float = 0.1,
        candidates_per_span: int = 20,
        max_iterations: int | None = None,
        seed: int | None = None,
    ) -> str:
        if not 0 <= budget <= 1:
            raise ValueError(f"budget must be in [0, 1], got {budget}")
        rng = random.Random(seed)

        words = text.split()
        if len(words) < 2:
            return text

        target_replacements = int(len(words) * budget)
        if max_iterations is None:
            max_iterations = max(1, target_replacements * 5)

        replaced = 0
        attempts = 0
        while replaced < target_replacements and attempts < max_iterations:
            attempts += 1
            idx = rng.randrange(len(words))
            original_word = words[idx]
            words[idx] = "<extra_id_0>"
            masked_text = " ".join(words)

            input_ids = self.tokenizer(masked_text, return_tensors="pt").input_ids.to(self.device)
            outputs = self.model.generate(  # type: ignore[misc]
                input_ids,
                num_beams=candidates_per_span,
                num_return_sequences=candidates_per_span,
                max_new_tokens=10,
                early_stopping=True,
            )

            new_word: str | None = None
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=False)
                fill = self._extract_first_span(decoded)
                if fill and fill.strip() and fill.strip() != original_word:
                    new_word = fill.strip()
                    break

            if new_word:
                words[idx] = new_word
                replaced += 1
            else:
                words[idx] = original_word

        logger.info(f"Replaced {replaced}/{target_replacements} words in {attempts} attempts.")
        return " ".join(words)

    @staticmethod
    def _extract_first_span(decoded: str) -> str | None:
        marker = "<extra_id_0>"
        if marker not in decoded:
            return None
        start = decoded.index(marker) + len(marker)
        tail = decoded[start:]
        for end_marker in ("<extra_id_1>", "</s>", "<pad>"):
            if end_marker in tail:
                tail = tail[: tail.index(end_marker)]
        return tail
