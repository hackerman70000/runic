"""HuggingFace `LogitsProcessor` that injects the green-list bias.

Plug into any `model.generate()` call via `LogitsProcessorList([...])`.
Soft mode (default): adds `delta` to green-list logits.
Hard mode (`delta == inf`): masks red-list logits to `-inf`.
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor

from runic.core.config import WatermarkConfig
from runic.core.vocab_partition import green_mask_for_token


class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, config: WatermarkConfig, vocab_size: int) -> None:
        self.config = config
        self.vocab_size = vocab_size

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if input_ids.shape[1] == 0:
            return scores

        biased = scores.clone()
        for batch_idx in range(input_ids.shape[0]):
            prev_token = int(input_ids[batch_idx, -1].item())
            mask = green_mask_for_token(
                prev_token,
                vocab_size=self.vocab_size,
                gamma=self.config.gamma,
                hash_key=self.config.hash_key,
                device=scores.device,
            )
            if self.config.is_hard:
                biased[batch_idx, ~mask] = float("-inf")
            else:
                biased[batch_idx, mask] = biased[batch_idx, mask] + self.config.delta
        return biased  # type: ignore[return-value]
