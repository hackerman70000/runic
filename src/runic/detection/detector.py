"""Watermark detection — count green-list hits and compute z / p-value.

Detection is *model-free*: only the tokenizer and the watermark config
are required. Anyone who knows ``hash_key`` can verify provenance.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from runic.core.config import WatermarkConfig
from runic.core.statistics import DetectionStats, p_value_from_z, z_statistic
from runic.core.vocab_partition import green_mask_for_token


class WatermarkDetector:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: WatermarkConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or WatermarkConfig()
        self.vocab_size = len(tokenizer)

    def detect(self, text: str) -> DetectionStats:
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return self.detect_token_ids(list(token_ids))

    def detect_token_ids(self, token_ids: list[int]) -> DetectionStats:
        if len(token_ids) < 2:
            return DetectionStats(
                green_count=0,
                total_tokens=0,
                z_score=0.0,
                p_value=1.0,
                threshold=self.config.z_threshold,
            )

        green_count = 0
        total = len(token_ids) - 1
        for i in range(1, len(token_ids)):
            mask = green_mask_for_token(
                token_ids[i - 1],
                vocab_size=self.vocab_size,
                gamma=self.config.gamma,
                hash_key=self.config.hash_key,
            )
            if bool(mask[token_ids[i]].item()):
                green_count += 1

        z = z_statistic(green_count, total, self.config.gamma)
        return DetectionStats(
            green_count=green_count,
            total_tokens=total,
            z_score=z,
            p_value=p_value_from_z(z),
            threshold=self.config.z_threshold,
        )

    def detect_tensor(self, token_ids: torch.LongTensor) -> DetectionStats:
        return self.detect_token_ids([int(t) for t in token_ids.flatten().tolist()])
