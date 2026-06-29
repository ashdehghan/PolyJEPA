"""L1 unit tests for the pluggable encoder backbones."""

from __future__ import annotations

import pytest
import torch
from tests.conftest import make_deterministic

from polyjepa.backbones import build_backbone


def _small_graph() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    return x, edge_index


@pytest.mark.parametrize("name", ["gcn", "sage", "gat", "mlp"])
def test_backbone_output_shape(name):
    make_deterministic(0)
    enc = build_backbone(name, in_dim=4, hidden_dim=8, latent_dim=6)
    x, edge_index = _small_graph()
    out = enc(x, edge_index)
    assert out.shape == (6, 6)
    assert torch.isfinite(out).all()


def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="unknown backbone"):
        build_backbone("does-not-exist", 4, 8, 6)


def test_gat_head_divisibility_error():
    with pytest.raises(ValueError, match="divisible"):
        build_backbone("gat", in_dim=4, hidden_dim=9, latent_dim=6, heads=4)


def test_mlp_ignores_edge_index():
    make_deterministic(0)
    enc = build_backbone("mlp", in_dim=4, hidden_dim=8, latent_dim=6)
    enc.eval()
    x = torch.randn(10, 4)
    e1 = torch.tensor([[0, 1], [1, 2]])
    e2 = torch.tensor([[3, 4, 5], [4, 5, 6]])
    with torch.no_grad():
        out1 = enc(x, e1)
        out2 = enc(x, e2)
    assert torch.allclose(out1, out2)
