"""Per-token green/red list partitioning of the vocabulary.

The previous token is mixed with a secret key to seed a torch PRNG, which
produces a deterministic but pseudo-random permutation of the vocabulary.
The first ``gamma * |V|`` indices form the green list.
"""

from __future__ import annotations

import torch


def green_list_for_token(
    prev_token_id: int,
    *,
    vocab_size: int,
    gamma: float,
    hash_key: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a 1-D LongTensor of green-list token ids on ``device``.

    The size is ``floor(gamma * vocab_size)`` and the contents depend
    only on ``prev_token_id`` and ``hash_key``.
    """
    if not 0 < gamma < 1:
        raise ValueError(f"gamma must be in (0, 1), got {gamma}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    seed = (hash_key * (prev_token_id + 1)) & 0x7FFFFFFFFFFFFFFF
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    permutation = torch.randperm(vocab_size, generator=rng)
    n_green = max(1, int(gamma * vocab_size))
    return permutation[:n_green].to(device)


def green_mask_for_token(
    prev_token_id: int,
    *,
    vocab_size: int,
    gamma: float,
    hash_key: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Boolean mask of length ``vocab_size``; ``True`` for green-list ids."""
    green = green_list_for_token(
        prev_token_id,
        vocab_size=vocab_size,
        gamma=gamma,
        hash_key=hash_key,
        device=device,
    )
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    mask[green] = True
    return mask
