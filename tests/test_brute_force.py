from __future__ import annotations

from runic.core.config import WatermarkConfig
from runic.core.vocab_partition import green_list_for_token
from runic.detection.brute_force import brute_force_detect


class StubTokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        ids = [int(tok) for tok in text.split()]
        return {"input_ids": ids}


def _watermarked_token_ids(cfg: WatermarkConfig, vocab_size: int, length: int) -> list[int]:
    ids = [5]
    for _ in range(length - 1):
        green = green_list_for_token(
            ids[-1], vocab_size=vocab_size, gamma=cfg.gamma, hash_key=cfg.hash_key
        )
        ids.append(int(green[0].item()))
    return ids


def test_brute_force_recovers_known_hash_key():
    """Search over a grid that contains the true hash_key; expect it to win."""
    true_cfg = WatermarkConfig(gamma=0.25, delta=2.0, hash_key=42, z_threshold=4.0)
    vocab_size = 200
    tokens = _watermarked_token_ids(true_cfg, vocab_size, length=80)
    text = " ".join(str(t) for t in tokens)
    tokenizer = StubTokenizer(vocab_size)

    result = brute_force_detect(
        text,
        tokenizer,
        gammas=(0.25, 0.5),
        hash_keys=(7, 42, 1337),
    )

    assert result.best_config.hash_key == 42
    assert result.best_config.gamma == 0.25
    assert result.best_stats.is_watermarked


def test_brute_force_clears_random_text():
    """If no config matches the text, the best z-score stays low."""
    tokenizer = StubTokenizer(500)
    text = " ".join(str(t) for t in range(50))
    result = brute_force_detect(
        text,
        tokenizer,
        gammas=(0.25, 0.5),
        hash_keys=(1, 7, 42),
    )
    assert not result.best_stats.is_watermarked


def test_brute_force_returns_all_attempts():
    tokenizer = StubTokenizer(200)
    text = " ".join(str(t) for t in range(40))
    result = brute_force_detect(
        text,
        tokenizer,
        gammas=(0.25, 0.5),
        hash_keys=(1, 7),
    )
    assert len(result.all_attempts) == 4
    for cfg, _ in result.all_attempts:
        assert cfg.gamma in {0.25, 0.5}
        assert cfg.hash_key in {1, 7}
