"""Quick plots for watermark detection runs.

Lazy-imports matplotlib so the base install stays light.
Install the optional extra with `uv sync --extra plotting`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _lazy_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("matplotlib not installed. Run `uv sync --extra plotting`.") from exc
    return plt


def plot_z_histogram(
    z_scores: Sequence[float],
    *,
    threshold: float = 4.0,
    title: str = "Watermark z-scores",
    bins: int = 50,
    save_path: Path | str | None = None,
) -> None:
    """Histogram of per-sample z-scores with the detection threshold marked."""
    plt = _lazy_pyplot()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(list(z_scores), bins=bins, color="steelblue", edgecolor="white")
    ax.axvline(threshold, color="crimson", linestyle="--", label=f"z = {threshold}")
    ax.set_xlabel("z-score")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_attack_curve(
    budgets: Sequence[float],
    z_scores: Sequence[float],
    *,
    threshold: float = 4.0,
    title: str = "Watermark robustness vs attack budget",
    save_path: Path | str | None = None,
) -> None:
    """Detection z-score as the attacker's perturbation budget grows."""
    plt = _lazy_pyplot()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(budgets), list(z_scores), marker="o", color="steelblue")
    ax.axhline(threshold, color="crimson", linestyle="--", label=f"z = {threshold}")
    ax.set_xlabel("attack budget (fraction of tokens replaced)")
    ax.set_ylabel("z-score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
