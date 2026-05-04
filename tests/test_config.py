from __future__ import annotations

import pytest

from runic.core.config import WatermarkConfig


def test_defaults_are_valid():
    cfg = WatermarkConfig()
    assert cfg.gamma == 0.25
    assert cfg.delta == 2.0
    assert cfg.is_hard is False


def test_hard_mode_when_delta_infinite():
    cfg = WatermarkConfig(delta=float("inf"))
    assert cfg.is_hard is True


def test_invalid_gamma_rejected():
    with pytest.raises(ValueError, match="gamma"):
        WatermarkConfig(gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        WatermarkConfig(gamma=1.0)


def test_negative_delta_rejected():
    with pytest.raises(ValueError, match="delta"):
        WatermarkConfig(delta=-0.1)


def test_non_positive_threshold_rejected():
    with pytest.raises(ValueError, match="z_threshold"):
        WatermarkConfig(z_threshold=0.0)
