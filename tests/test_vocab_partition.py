from __future__ import annotations

import pytest
import torch

from runic.core.vocab_partition import green_list_for_token, green_mask_for_token


def test_partition_size_matches_gamma():
    green = green_list_for_token(42, vocab_size=1000, gamma=0.25, hash_key=7)
    assert green.shape[0] == 250


def test_partition_is_deterministic():
    a = green_list_for_token(42, vocab_size=500, gamma=0.3, hash_key=99)
    b = green_list_for_token(42, vocab_size=500, gamma=0.3, hash_key=99)
    assert torch.equal(a, b)


def test_partition_differs_for_different_prev_tokens():
    a = green_list_for_token(1, vocab_size=500, gamma=0.3, hash_key=99)
    b = green_list_for_token(2, vocab_size=500, gamma=0.3, hash_key=99)
    assert not torch.equal(a, b)


def test_partition_differs_for_different_hash_keys():
    a = green_list_for_token(1, vocab_size=500, gamma=0.3, hash_key=99)
    b = green_list_for_token(1, vocab_size=500, gamma=0.3, hash_key=100)
    assert not torch.equal(a, b)


def test_mask_matches_list():
    green = green_list_for_token(7, vocab_size=200, gamma=0.4, hash_key=11)
    mask = green_mask_for_token(7, vocab_size=200, gamma=0.4, hash_key=11)
    assert int(mask.sum().item()) == green.shape[0]
    for token_id in green.tolist():
        assert mask[token_id].item() is True


def test_partition_rejects_invalid_gamma():
    with pytest.raises(ValueError, match="gamma"):
        green_list_for_token(0, vocab_size=100, gamma=0.0, hash_key=1)


def test_partition_rejects_invalid_vocab():
    with pytest.raises(ValueError, match="vocab_size"):
        green_list_for_token(0, vocab_size=0, gamma=0.5, hash_key=1)
