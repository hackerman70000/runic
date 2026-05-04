from __future__ import annotations

from typing import Any

from runic.core.config import WatermarkConfig
from runic.core.vocab_partition import green_list_for_token
from runic.detection.detector import WatermarkDetector


class StubTokenizer:
    """Minimal duck-typed tokenizer good enough for the detector unit tests."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        ids = [int(tok) for tok in text.split()]
        return {"input_ids": ids}


def _build_watermarked_token_ids(
    cfg: WatermarkConfig,
    vocab_size: int,
    length: int,
    prefix: int = 5,
) -> list[int]:
    """Construct a token sequence where every step picks a green-list token."""
    ids = [prefix]
    for _ in range(length - 1):
        green = green_list_for_token(
            ids[-1],
            vocab_size=vocab_size,
            gamma=cfg.gamma,
            hash_key=cfg.hash_key,
        )
        ids.append(int(green[0].item()))
    return ids


def test_detector_flags_watermarked_sequence():
    cfg = WatermarkConfig(gamma=0.25, delta=2.0, hash_key=7, z_threshold=4.0)
    vocab_size = 200
    tokens = _build_watermarked_token_ids(cfg, vocab_size, length=50)

    det = WatermarkDetector(tokenizer=StubTokenizer(vocab_size), config=cfg)
    result = det.detect_token_ids(tokens)

    assert result.is_watermarked is True
    assert result.green_count == result.total_tokens
    assert result.z_score > 4.0


def test_detector_clears_random_sequence():
    cfg = WatermarkConfig(gamma=0.25, hash_key=7, z_threshold=4.0)
    vocab_size = 1000
    tokens = list(range(50))

    det = WatermarkDetector(tokenizer=StubTokenizer(vocab_size), config=cfg)
    result = det.detect_token_ids(tokens)

    assert result.is_watermarked is False
    assert -3 < result.z_score < 3


def test_detector_handles_too_short_input():
    cfg = WatermarkConfig()
    det = WatermarkDetector(tokenizer=StubTokenizer(100), config=cfg)
    assert det.detect_token_ids([]).total_tokens == 0
    assert det.detect_token_ids([1]).total_tokens == 0


def test_detector_uses_tokenizer_call_path():
    cfg = WatermarkConfig(gamma=0.25, hash_key=7)
    vocab_size = 200
    tokens = _build_watermarked_token_ids(cfg, vocab_size, length=30)
    text = " ".join(str(t) for t in tokens)

    det = WatermarkDetector(tokenizer=StubTokenizer(vocab_size), config=cfg)
    via_text: Any = det.detect(text)
    via_ids: Any = det.detect_token_ids(tokens)
    assert via_text.green_count == via_ids.green_count
    assert via_text.total_tokens == via_ids.total_tokens
