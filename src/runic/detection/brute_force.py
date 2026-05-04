"""Watermark detection when the parameters aren't known.

If you receive watermarked text without `gamma`, `delta`, or `hash_key`,
sweep a small grid of plausible values and pick the configuration that
produces the strongest z-score. The watermark embeds at sampling time
with one specific configuration; the detector that matches it sees a
big z, the others see noise.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from runic.core.config import WatermarkConfig
from runic.core.statistics import DetectionStats
from runic.detection.detector import WatermarkDetector


@dataclass
class BruteForceResult:
    best_stats: DetectionStats
    best_config: WatermarkConfig
    all_attempts: list[tuple[WatermarkConfig, DetectionStats]]


DEFAULT_GAMMAS: tuple[float, ...] = (0.25, 0.5)
DEFAULT_HASH_KEYS: tuple[int, ...] = (
    1,
    7,
    42,
    1337,
    2024,
    15485863,
)


def brute_force_detect(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    gammas: Iterable[float] = DEFAULT_GAMMAS,
    hash_keys: Iterable[int] = DEFAULT_HASH_KEYS,
    z_threshold: float = 4.0,
) -> BruteForceResult:
    """Grid search over `(gamma, hash_key)` pairs.

    Delta is irrelevant for detection (it only affects sampling-time bias)
    so it's not part of the search. Returns the configuration that produced
    the highest z-score, plus every attempt so callers can plot the grid.
    """
    attempts: list[tuple[WatermarkConfig, DetectionStats]] = []
    for gamma in gammas:
        for hash_key in hash_keys:
            cfg = WatermarkConfig(
                gamma=gamma,
                delta=2.0,
                hash_key=hash_key,
                z_threshold=z_threshold,
            )
            detector = WatermarkDetector(tokenizer=tokenizer, config=cfg)
            stats = detector.detect(text)
            attempts.append((cfg, stats))

    best_cfg, best_stats = max(attempts, key=lambda pair: pair[1].z_score)
    return BruteForceResult(
        best_stats=best_stats,
        best_config=best_cfg,
        all_attempts=attempts,
    )
