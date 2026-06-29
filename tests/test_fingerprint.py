"""Fingerprint assembly: shape, cross-probe matrix, descriptors, embedding bank."""

from __future__ import annotations

import torch
from tests.conftest import fast_kwargs

from polyjepa import RecoverFocal, SampledNeighbor, fingerprint


def test_fingerprint_full_shape(sbm_bridges):
    fp = fingerprint(sbm_bridges, **fast_kwargs(epochs=40))
    n = sbm_bridges.num_nodes
    p = len(fp.probe_names)
    assert fp.residuals.shape == (n, p)
    assert fp.cross_probe.shape == (p, p)
    for name in fp.probe_names:
        assert fp.embeddings[name].shape[0] == n
        assert torch.isfinite(fp.embeddings[name]).all()


def test_cross_probe_symmetric_unit_diagonal(sbm_bridges):
    fp = fingerprint(sbm_bridges, **fast_kwargs(epochs=40))
    m = fp.cross_probe
    assert torch.allclose(m.diagonal(), torch.ones(m.shape[0]), atol=1e-4)
    assert torch.allclose(m, m.t(), atol=1e-4, equal_nan=True)


def test_custom_probe_subset(sbm_bridges):
    probes = {"A": RecoverFocal(), "C": SampledNeighbor()}
    fp = fingerprint(sbm_bridges, probes=probes, **fast_kwargs(epochs=40))
    assert fp.probe_names == ["A", "C"]
    assert fp.residuals.shape[1] == 2


def test_descriptors_present(sbm_bridges):
    fp = fingerprint(sbm_bridges, **fast_kwargs(epochs=40))
    d = fp.graph_descriptors["A"]
    for key in ("mean", "std", "skew", "p10", "p50", "p90", "gini", "n_scored"):
        assert key in d
    assert d["pooled_embedding"].shape[0] == fp.embeddings["A"].shape[1]


def test_full_fingerprint_is_deterministic(sbm_bridges):
    """The top-level drift detector: same seed gives an identical fingerprint
    (residuals and embeddings) on the same machine. Any unintended behavioral
    change fails this contract.
    """
    fp1 = fingerprint(sbm_bridges, **fast_kwargs(epochs=40, seed=5))
    fp2 = fingerprint(sbm_bridges, **fast_kwargs(epochs=40, seed=5))
    assert torch.allclose(fp1.residuals, fp2.residuals, equal_nan=True)
    for name in fp1.probe_names:
        assert torch.allclose(fp1.embeddings[name], fp2.embeddings[name])
