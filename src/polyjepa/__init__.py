"""PolyJEPA: a multi-probe JEPA that fingerprints graphs.

One fixed JEPA template; a family of pair designs (probes A-F). Each probe is a
self-supervised objective whose per-node residual and embedding measure a
different property of a node's role in the graph.

Public API::

    from polyjepa import fingerprint
    fp = fingerprint(data)            # all six probes
    fp.residuals                      # (N, P) residual fingerprint
    fp.embeddings["A"]                # (N, d) embedding under probe A
    fp.cross_probe                    # (P, P) Spearman across probes
"""

from __future__ import annotations

from polyjepa.engine import Diagnostics, JEPAEngine
from polyjepa.fingerprint import Fingerprint, fingerprint
from polyjepa.pair_design import Pair, PairContext, PairDesign
from polyjepa.probes import (
    AugmentedView,
    CommunitySibling,
    PooledNeighborhood,
    RecoverFocal,
    SampledNeighbor,
    TwoHopRing,
    default_probes,
)

__all__ = [
    "fingerprint",
    "Fingerprint",
    "JEPAEngine",
    "Diagnostics",
    "PairDesign",
    "PairContext",
    "Pair",
    "RecoverFocal",
    "PooledNeighborhood",
    "SampledNeighbor",
    "TwoHopRing",
    "AugmentedView",
    "CommunitySibling",
    "default_probes",
]

__version__ = "0.1.0"
