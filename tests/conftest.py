"""Shared fixtures: small planted synthetic graphs for fast, deterministic tests."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from polyjepa.synthetic import inject_feature_outliers, planted_sbm


def make_deterministic(seed: int = 42) -> None:
    """Seed every RNG a test might touch (torch, numpy, random).

    The engine seeds its own generators internally, so most tests do not need
    this; use it for tests that build inputs with bare ``torch.randn`` and assert
    exact, reproducible values.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture
def sbm_bridges() -> Data:
    """A 3-community SBM with planted bridge nodes."""
    return planted_sbm(
        num_communities=3, nodes_per_comm=20, num_bridges=4, bridge_degree=5, seed=0
    )


@pytest.fixture
def sbm_outliers() -> Data:
    """A single-community graph with planted feature outliers."""
    base = planted_sbm(
        num_communities=2, nodes_per_comm=20, num_bridges=0, p_in=0.4, seed=2
    )
    return inject_feature_outliers(base, frac=0.15, scale=8.0, seed=3)


@pytest.fixture
def clustered_with_isolated() -> Data:
    """A dense cluster plus one isolated node (last index, no edges)."""
    base = planted_sbm(
        num_communities=1, nodes_per_comm=20, p_in=0.5, num_bridges=0, seed=1
    )
    x = torch.cat([base.x, torch.ones(1, base.x.shape[1]) * 5.0], dim=0)
    data = Data(x=x, edge_index=base.edge_index)
    data.num_nodes = base.x.shape[0] + 1
    data.isolated_idx = base.x.shape[0]
    return data


def fast_kwargs(**overrides) -> dict:
    """Small, fast engine settings for tests."""
    kw = dict(epochs=120, hidden_dim=64, latent_dim=64, seed=0, diag_every=10)
    kw.update(overrides)
    return kw
