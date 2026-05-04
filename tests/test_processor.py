from __future__ import annotations

import torch

from runic.core.config import WatermarkConfig
from runic.core.vocab_partition import green_mask_for_token
from runic.generation.processor import WatermarkLogitsProcessor


def test_soft_processor_adds_delta_to_green_logits():
    cfg = WatermarkConfig(gamma=0.5, delta=2.0, hash_key=7)
    vocab_size = 100
    proc = WatermarkLogitsProcessor(cfg, vocab_size=vocab_size)

    input_ids = torch.tensor([[42]])
    scores = torch.zeros(1, vocab_size)
    biased = proc(input_ids, scores)

    mask = green_mask_for_token(42, vocab_size=vocab_size, gamma=0.5, hash_key=7)
    expected_green = (biased[0] == 2.0).sum().item()
    expected_red = (biased[0] == 0.0).sum().item()
    assert expected_green == int(mask.sum().item())
    assert expected_green + expected_red == vocab_size


def test_hard_processor_masks_red_logits_to_neg_inf():
    cfg = WatermarkConfig(gamma=0.4, delta=float("inf"), hash_key=11)
    vocab_size = 50
    proc = WatermarkLogitsProcessor(cfg, vocab_size=vocab_size)

    input_ids = torch.tensor([[3]])
    scores = torch.zeros(1, vocab_size)
    biased = proc(input_ids, scores)

    mask = green_mask_for_token(3, vocab_size=vocab_size, gamma=0.4, hash_key=11)
    assert torch.all(biased[0, mask] == 0.0)
    assert torch.all(torch.isinf(biased[0, ~mask]))


def test_processor_handles_batched_input():
    cfg = WatermarkConfig(gamma=0.3, delta=1.5, hash_key=99)
    vocab_size = 30
    proc = WatermarkLogitsProcessor(cfg, vocab_size=vocab_size)

    input_ids = torch.tensor([[10], [20]])
    scores = torch.zeros(2, vocab_size)
    biased = proc(input_ids, scores)

    assert biased.shape == (2, vocab_size)
    assert not torch.equal(biased[0], biased[1])


def test_processor_with_empty_input_returns_unchanged():
    cfg = WatermarkConfig()
    proc = WatermarkLogitsProcessor(cfg, vocab_size=10)
    input_ids = torch.zeros(1, 0, dtype=torch.long)
    scores = torch.randn(1, 10)
    out = proc(input_ids, scores)
    assert torch.equal(out, scores)
