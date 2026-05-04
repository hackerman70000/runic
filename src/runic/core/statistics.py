"""Hypothesis test for watermark detection (Kirchenbauer eq. 2 / 3)."""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats


def z_statistic(green_count: int, total_tokens: int, gamma: float) -> float:
    """One-proportion z-statistic for green-list hits.

    H_0: tokens are sampled with no knowledge of the green list, so the
    expected proportion of green hits is ``gamma``. The z-statistic is

        z = (|s|_G - gamma * T) / sqrt(T * gamma * (1 - gamma))

    Larger ``z`` ⇒ stronger evidence the watermark is present.
    """
    if total_tokens <= 0:
        return 0.0
    if not 0 < gamma < 1:
        raise ValueError(f"gamma must be in (0, 1), got {gamma}")
    expected = gamma * total_tokens
    variance = total_tokens * gamma * (1 - gamma)
    return float((green_count - expected) / (variance**0.5))


def p_value_from_z(z: float) -> float:
    """One-sided upper-tail p-value of the standard normal."""
    return float(stats.norm.sf(z))


@dataclass(frozen=True)
class DetectionStats:
    green_count: int
    total_tokens: int
    z_score: float
    p_value: float
    threshold: float

    @property
    def is_watermarked(self) -> bool:
        return self.z_score >= self.threshold

    @property
    def green_fraction(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.green_count / self.total_tokens
