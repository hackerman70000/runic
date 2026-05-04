from __future__ import annotations

import pytest

from runic.core.statistics import DetectionStats, p_value_from_z, z_statistic


def test_z_zero_when_observed_matches_expected():
    z = z_statistic(green_count=25, total_tokens=100, gamma=0.25)
    assert z == pytest.approx(0.0)


def test_z_large_when_green_excess():
    z = z_statistic(green_count=80, total_tokens=100, gamma=0.25)
    assert z > 10


def test_z_negative_when_green_deficit():
    z = z_statistic(green_count=5, total_tokens=100, gamma=0.25)
    assert z < -3


def test_p_value_decreases_with_z():
    p_low = p_value_from_z(0.0)
    p_mid = p_value_from_z(2.0)
    p_high = p_value_from_z(4.0)
    assert p_low > p_mid > p_high
    assert p_low == pytest.approx(0.5)


def test_z_zero_total_returns_zero():
    assert z_statistic(0, 0, 0.5) == 0.0


def test_detection_stats_threshold():
    stats = DetectionStats(
        green_count=50,
        total_tokens=100,
        z_score=5.0,
        p_value=1e-6,
        threshold=4.0,
    )
    assert stats.is_watermarked is True
    assert stats.green_fraction == 0.5

    weak = DetectionStats(
        green_count=27,
        total_tokens=100,
        z_score=0.5,
        p_value=0.3,
        threshold=4.0,
    )
    assert weak.is_watermarked is False
