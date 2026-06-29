"""The pair-design abstraction: the only thing a probe overrides.

Every PolyJEPA probe shares one fixed template (encoder, EMA target, predictor,
latent L2 loss). A ``PairDesign`` describes how to build the context/target pair
for a chunk of focal nodes:

- which nodes have their features replaced by the ``[MASK]`` token in the context
  view (or, for the augmented-view probe, which augmentations to apply),
- which target node(s) each focal node must predict, and how to pool them.

``make_pair`` returns a :class:`Pair` that the engine consumes uniformly for both
training and scoring. The general principle for the masked probes: *the context
hides what it must predict*, so the target nodes are masked in the context view.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from polyjepa.graph_index import GraphIndex


@dataclass
class PairContext:
    """Inputs handed to ``PairDesign.make_pair`` for one focal chunk."""

    x: torch.Tensor  # (N, F) clean features, on device
    edge_index: torch.Tensor  # clean edges, on device
    focal_idx: torch.Tensor  # (C,) focal node ids for this chunk, on device
    mask_token: torch.Tensor  # (F,) learnable mask token, on device
    index: GraphIndex
    generator: torch.Generator  # CPU generator for reproducible sampling
    num_nodes: int
    device: torch.device


@dataclass
class Pair:
    """A built context/target pair for one focal chunk.

    ``focal`` are the valid focal nodes (those with at least one target).
    ``tgt_index`` and ``tgt_group`` describe, in flat form, the target nodes to
    mean-pool per focal: target entry ``j`` belongs to focal slot
    ``tgt_group[j]`` and points at node ``tgt_index[j]``.
    """

    x_ctx: torch.Tensor
    edge_index_ctx: torch.Tensor
    x_tgt: torch.Tensor
    edge_index_tgt: torch.Tensor
    focal: torch.Tensor  # (K,)
    tgt_index: torch.Tensor  # (M,)
    tgt_group: torch.Tensor  # (M,) values in [0, K)


def segment_mean(
    values: torch.Tensor, group: torch.Tensor, num_groups: int
) -> torch.Tensor:
    """Mean of ``values`` rows within each group id. Returns ``(num_groups, d)``."""
    out = values.new_zeros(num_groups, values.shape[-1])
    out.index_add_(0, group, values)
    counts = values.new_zeros(num_groups)
    counts.index_add_(0, group, values.new_ones(group.shape[0]))
    return out / counts.clamp(min=1.0).unsqueeze(-1)


def assemble_targets(
    focal_idx_cpu: torch.Tensor, target_lists: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Drop focal nodes with empty targets and build flat (focal, index, group).

    All tensors returned on CPU; the caller moves them to the device.
    """
    valid_focal: list[int] = []
    flats: list[torch.Tensor] = []
    groups: list[torch.Tensor] = []
    k = 0
    for fv, targets in zip(focal_idx_cpu.tolist(), target_lists, strict=True):
        if targets.numel() == 0:
            continue
        valid_focal.append(fv)
        flats.append(targets)
        groups.append(torch.full((targets.numel(),), k, dtype=torch.long))
        k += 1
    if not valid_focal:
        return None
    return (
        torch.tensor(valid_focal, dtype=torch.long),
        torch.cat(flats),
        torch.cat(groups),
    )


def apply_mask(
    x: torch.Tensor, mask_nodes: torch.Tensor, mask_token: torch.Tensor
) -> torch.Tensor:
    """Return a copy of ``x`` with ``mask_nodes`` rows replaced by ``mask_token``."""
    x_ctx = x.clone()
    if mask_nodes.numel() > 0:
        x_ctx[mask_nodes] = mask_token
    return x_ctx


class PairDesign(ABC):
    """Base class for a probe's pair construction.

    Attributes
    ----------
    name : str
        Short identifier (a single letter A-F is the convention).
    static_target : bool
        If True, the target view is the clean static graph, so the engine can
        cache the target-encoder embedding once. False for augmentation-based
        probes whose target view changes every pass.
    needs_communities : bool
        If True, the GraphIndex must build community structure.
    """

    name: str = "?"
    static_target: bool = True
    needs_communities: bool = False

    @abstractmethod
    def make_pair(self, ctx: PairContext) -> Pair | None:
        """Build the pair for one focal chunk, or ``None`` if no focal is valid."""
