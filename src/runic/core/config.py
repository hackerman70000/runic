from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WatermarkConfig:
    """Watermark hyperparameters (Kirchenbauer et al. 2023).

    gamma: fraction of vocabulary placed in the green list, in (0, 1).
    delta: additive logit bias for green-list tokens (soft watermark).
        Use ``delta=float('inf')`` for the hard variant (Algorithm 1) which
        forbids red tokens entirely.
    hash_key: secret key mixed with the previous token to seed the PRNG.
        Public detection still works as long as both sides know this key.
    z_threshold: z-statistic threshold for the detector. Paper default 4.0
        gives a one-sided p-value of approximately 3e-5.
    """

    gamma: float = 0.25
    delta: float = 2.0
    hash_key: int = 15485863
    z_threshold: float = 4.0

    def __post_init__(self) -> None:
        if not 0 < self.gamma < 1:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma}")
        if self.delta < 0:
            raise ValueError(f"delta must be non-negative, got {self.delta}")
        if self.z_threshold <= 0:
            raise ValueError(f"z_threshold must be positive, got {self.z_threshold}")

    @property
    def is_hard(self) -> bool:
        return self.delta == float("inf")
