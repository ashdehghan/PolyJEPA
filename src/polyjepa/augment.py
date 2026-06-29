"""Stochastic graph augmentations for the augmented-view probe (E).

Two standard, label-free augmentations from the BGRL/GRACE family: random edge
dropping and random feature masking (with optional Gaussian noise). All sampling
uses a caller-supplied ``torch.Generator`` for reproducibility.
"""

from __future__ import annotations

import torch


def drop_edges(
    edge_index: torch.Tensor, p: float, generator: torch.Generator
) -> torch.Tensor:
    """Randomly drop a fraction ``p`` of edges. Returns a new ``edge_index``.

    Dropping is per-column. An undirected edge is stored as two columns (one per
    direction), so a given edge may survive in only one direction, making the
    augmented view's message passing mildly asymmetric. This is intentional and
    matches the GRACE/BGRL augmentation family.
    """
    if p <= 0.0 or edge_index.numel() == 0:
        return edge_index
    num_edges = edge_index.shape[1]
    keep = (torch.rand(num_edges, generator=generator) >= p).to(edge_index.device)
    return edge_index[:, keep]


def mask_features(
    x: torch.Tensor,
    p: float,
    generator: torch.Generator,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """Randomly zero a fraction ``p`` of feature columns (shared across nodes),
    then optionally add Gaussian noise. Returns a new feature tensor.
    """
    out = x.clone()
    if p > 0.0:
        num_feat = x.shape[1]
        drop = (torch.rand(num_feat, generator=generator) < p).to(x.device)
        out[:, drop] = 0.0
    if noise_std > 0.0:
        noise = (torch.randn(out.shape, generator=generator) * noise_std).to(x.device)
        out = out + noise
    return out


def augment_view(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    feat_drop: float,
    edge_drop: float,
    generator: torch.Generator,
    noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce one augmented view: feature masking + edge dropping."""
    x_aug = mask_features(x, feat_drop, generator, noise_std=noise_std)
    e_aug = drop_edges(edge_index, edge_drop, generator)
    return x_aug, e_aug
