"""L1 unit tests for the augmentation transforms (probe E)."""

from __future__ import annotations

import torch

from polyjepa.augment import augment_view, drop_edges, mask_features


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_drop_edges_p0_returns_input():
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    assert drop_edges(ei, 0.0, _gen()) is ei


def test_drop_edges_p1_drops_all():
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    out = drop_edges(ei, 1.0, _gen())
    assert out.shape[1] == 0


def test_drop_edges_is_deterministic():
    ei = torch.arange(20).reshape(2, 10)
    a = drop_edges(ei, 0.5, _gen(7))
    b = drop_edges(ei, 0.5, _gen(7))
    assert torch.equal(a, b)
    assert a.shape[0] == 2


def test_mask_features_p0_noise0_is_identity():
    x = torch.randn(5, 4)
    out = mask_features(x, 0.0, _gen(), noise_std=0.0)
    assert torch.allclose(out, x)
    assert out is not x  # returns a copy


def test_mask_features_p1_zeros_all_columns():
    x = torch.randn(5, 4)
    out = mask_features(x, 1.0, _gen())
    assert torch.allclose(out, torch.zeros_like(x))


def test_mask_features_noise_changes_values_preserves_shape():
    x = torch.zeros(5, 4)
    out = mask_features(x, 0.0, _gen(1), noise_std=1.0)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


def test_augment_view_returns_features_and_edges():
    x = torch.randn(6, 3)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    x_aug, e_aug = augment_view(x, ei, 0.3, 0.3, _gen(2))
    assert x_aug.shape == x.shape
    assert e_aug.shape[0] == 2
