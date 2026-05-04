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

        score_dim = scores.shape[-1]
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
            mask = self._align_mask_to_scores(mask, score_dim)
            if self.config.is_hard:
                biased[batch_idx, ~mask] = float("-inf")
            else:
                biased[batch_idx, mask] = biased[batch_idx, mask] + self.config.delta
        return biased  # type: ignore[return-value]

    @staticmethod
    def _align_mask_to_scores(mask: torch.Tensor, score_dim: int) -> torch.Tensor:
        """Reconcile model output dim with tokenizer vocab.

        OPT-style models pad the model's output projection to a multiple
        of e.g. 8 for performance, leaving `score_dim > tokenizer vocab`.
        We zero-pad the green mask so the extra positions act as red.
        """
        if mask.shape[0] == score_dim:
            return mask
        if mask.shape[0] < score_dim:
            extra = score_dim - mask.shape[0]
            padding = torch.zeros(extra, dtype=torch.bool, device=mask.device)
            return torch.cat([mask, padding])
        return mask[:score_dim]
