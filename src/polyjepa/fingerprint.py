"""The graph fingerprint: the public output of PolyJEPA.

For each probe, training one JEPA yields a per-node embedding (the clean-graph
target representation under that probe's objective) and a per-node residual. The
:class:`Fingerprint` collects these into a two-layer description of the graph:

- an **embedding bank** (one ``(N, d)`` tensor per probe),
- a **residual fingerprint** (an ``(N, P)`` matrix),

plus per-graph descriptor statistics and the cross-probe Spearman matrix that
shows whether the probes are complementary axes or redundant copies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import skew, spearmanr
from torch_geometric.data import Data

from polyjepa.engine import Diagnostics, JEPAEngine
from polyjepa.pair_design import PairDesign
from polyjepa.probes import default_probes


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of non-negative values (a spread/concentration measure)."""
    v = np.sort(values[np.isfinite(values)])
    if v.size == 0 or v.sum() == 0:
        return float("nan")
    v = v - v.min() if v.min() < 0 else v
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _descriptors(residual: torch.Tensor, embedding: torch.Tensor) -> dict:
    arr = residual.numpy()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        stats = {k: float("nan") for k in ("mean", "std", "skew", "p10", "p50", "p90")}
    else:
        stats = {
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "skew": float(skew(finite)) if finite.size > 2 else float("nan"),
            "p10": float(np.percentile(finite, 10)),
            "p50": float(np.percentile(finite, 50)),
            "p90": float(np.percentile(finite, 90)),
        }
    stats["gini"] = _gini(arr)
    stats["n_scored"] = int(np.isfinite(arr).sum())
    stats["pooled_embedding"] = embedding.mean(dim=0)
    return stats


def _cross_probe_spearman(residuals: torch.Tensor) -> torch.Tensor:
    arr = residuals.numpy()
    p = arr.shape[1]
    out = np.full((p, p), np.nan)
    for i in range(p):
        for j in range(p):
            if i == j:
                # A probe is perfectly rank-correlated with itself by definition;
                # spearmanr returns NaN for a constant column, so set it directly.
                out[i, j] = 1.0
                continue
            a, b = arr[:, i], arr[:, j]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 3:
                rho, _ = spearmanr(a[ok], b[ok])
                out[i, j] = rho
    return torch.from_numpy(out).float()


@dataclass
class Fingerprint:
    """Two-layer fingerprint of a graph produced by the PolyJEPA probes."""

    probe_names: list[str]
    residuals: torch.Tensor  # (N, P), may contain NaN for unscored nodes
    embeddings: dict[str, torch.Tensor]  # name -> (N, d)
    graph_descriptors: dict[str, dict]  # name -> stats
    cross_probe: torch.Tensor  # (P, P) Spearman
    diagnostics: dict[str, Diagnostics]

    def residual(self, name: str) -> torch.Tensor:
        return self.residuals[:, self.probe_names.index(name)]

    def healthy(self) -> bool:
        """True if every probe's training diagnostics passed the collapse floor."""
        return all(d.healthy() for d in self.diagnostics.values())


def fingerprint(
    data: Data,
    probes: dict[str, PairDesign] | None = None,
    backbone: str = "gcn",
    seed: int = 0,
    **engine_kwargs,
) -> Fingerprint:
    """Run the probes on ``data`` and assemble its :class:`Fingerprint`.

    ``probes`` maps a name to a :class:`PairDesign`; defaults to all six (A-F).
    Extra keyword arguments are forwarded to :class:`JEPAEngine` (e.g. ``epochs``,
    ``device``, ``hidden_dim``).
    """
    probe_map = probes if probes is not None else default_probes()
    names = list(probe_map.keys())

    residual_cols: list[torch.Tensor] = []
    embeddings: dict[str, torch.Tensor] = {}
    descriptors: dict[str, dict] = {}
    diagnostics: dict[str, Diagnostics] = {}

    for name in names:
        engine = JEPAEngine(
            probe_map[name], backbone=backbone, seed=seed, **engine_kwargs
        )
        engine.fit(data)
        resid = engine.score()
        emb = engine.embeddings()
        residual_cols.append(resid)
        embeddings[name] = emb
        descriptors[name] = _descriptors(resid, emb)
        diagnostics[name] = engine.diagnostics

    residuals = torch.stack(residual_cols, dim=1)
    cross = _cross_probe_spearman(residuals)
    return Fingerprint(
        probe_names=names,
        residuals=residuals,
        embeddings=embeddings,
        graph_descriptors=descriptors,
        cross_probe=cross,
        diagnostics=diagnostics,
    )
