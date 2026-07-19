"""The six PolyJEPA probes (pair designs A-F) and a name registry.

Each probe is a :class:`~polyjepa.pair_design.PairDesign`. The masked probes
follow one principle: the context view hides the nodes it must predict, so the
target nodes are replaced by the mask token in the context. The augmented-view
probe (E) uses two stochastic augmentations instead of masking.
"""

from __future__ import annotations

import torch

from polyjepa.augment import augment_view
from polyjepa.pair_design import (
    Pair,
    PairContext,
    PairDesign,
    apply_mask,
    assemble_targets,
)


def _to_device_pair(
    x_ctx: torch.Tensor,
    edge_index_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    edge_index_tgt: torch.Tensor,
    assembled: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Pair:
    focal, tgt_index, tgt_group = assembled
    return Pair(
        x_ctx=x_ctx,
        edge_index_ctx=edge_index_ctx,
        x_tgt=x_tgt,
        edge_index_tgt=edge_index_tgt,
        focal=focal.to(device),
        tgt_index=tgt_index.to(device),
        tgt_group=tgt_group.to(device),
    )


class RecoverFocal(PairDesign):
    """A: mask the focal node (and optionally a fraction of its neighbors);
    predict the focal node's own clean embedding from the remaining context.
    """

    name = "A"

    def __init__(
        self,
        mask_focal: bool = True,
        neighbor_mask_ratio: float = 0.0,
        neighbor_hops: int = 1,
    ) -> None:
        if not mask_focal and neighbor_mask_ratio == 0.0:
            raise ValueError(
                "RecoverFocal needs a mask source: mask_focal=True or "
                "neighbor_mask_ratio > 0."
            )
        if not 0.0 <= neighbor_mask_ratio <= 1.0:
            raise ValueError("neighbor_mask_ratio must lie in [0, 1]")
        if neighbor_hops not in (1, 2):
            raise ValueError("neighbor_hops must be 1 or 2")
        self.mask_focal = mask_focal
        self.neighbor_mask_ratio = neighbor_mask_ratio
        self.neighbor_hops = neighbor_hops

    def make_pair(self, ctx: PairContext) -> Pair | None:
        focal_cpu = ctx.focal_idx.cpu()
        masked: set[int] = set()
        if self.mask_focal:
            masked.update(focal_cpu.tolist())
        if self.neighbor_mask_ratio > 0.0:
            for v in focal_cpu.tolist():
                cand = ctx.index.neighbors(v)
                if self.neighbor_hops == 2:
                    cand = torch.cat([cand, ctx.index.ring2(v)])
                if cand.numel() == 0:
                    continue
                n_take = int(round(self.neighbor_mask_ratio * cand.numel()))
                if n_take == 0:
                    continue
                perm = torch.randperm(cand.numel(), generator=ctx.generator)
                masked.update(cand[perm[:n_take]].tolist())
        target_lists = [torch.tensor([v], dtype=torch.long) for v in focal_cpu.tolist()]
        assembled = assemble_targets(focal_cpu, target_lists)
        if assembled is None:
            return None
        mask_nodes = torch.tensor(sorted(masked), dtype=torch.long).to(ctx.device)
        x_ctx = apply_mask(ctx.x, mask_nodes, ctx.mask_token)
        return _to_device_pair(
            x_ctx, ctx.edge_index, ctx.x, ctx.edge_index, assembled, ctx.device
        )


class _MaskedNeighborProbe(PairDesign):
    """Shared logic for probes whose targets are other nodes, masked in context.

    Subclasses implement ``_targets(ctx, v)`` returning the target node ids for
    focal node ``v``. All such target nodes are masked in the context view; the
    focal node itself stays visible.

    Masking design: the context graph masks the UNION of all focal nodes' target
    sets in a chunk. For probes B and D (full 1-hop / 2-hop neighborhoods), this
    union covers 96–100% of the graph at the default focal_ratio=0.30, making the
    context featureless and the residuals invalid. Setting ``per_focal_scoring =
    True`` instructs the engine to score each node in a singleton focal chunk
    (one node at a time), so the union collapses to just that node's targets.
    Training is unaffected (the encoder still learns a useful representation even
    with the union masking; only the stored fingerprint residuals are corrected).
    """

    per_focal_scoring: bool = True

    def _targets(self, ctx: PairContext, v: int) -> torch.Tensor:
        raise NotImplementedError

    def make_pair(self, ctx: PairContext) -> Pair | None:
        focal_cpu = ctx.focal_idx.cpu()
        target_lists = [self._targets(ctx, v) for v in focal_cpu.tolist()]
        assembled = assemble_targets(focal_cpu, target_lists)
        if assembled is None:
            return None
        masked: set[int] = set()
        for t in target_lists:
            masked.update(t.tolist())
        mask_nodes = torch.tensor(sorted(masked), dtype=torch.long).to(ctx.device)
        x_ctx = apply_mask(ctx.x, mask_nodes, ctx.mask_token)
        return _to_device_pair(
            x_ctx, ctx.edge_index, ctx.x, ctx.edge_index, assembled, ctx.device
        )


class PooledNeighborhood(_MaskedNeighborProbe):
    """B: predict the mean clean embedding of the focal node's 1-hop neighbors."""

    name = "B"

    def _targets(self, ctx: PairContext, v: int) -> torch.Tensor:
        return ctx.index.neighbors(v)


class SampledNeighbor(_MaskedNeighborProbe):
    """C: predict one sampled 1-hop neighbor's clean embedding (per-edge)."""

    name = "C"

    def _targets(self, ctx: PairContext, v: int) -> torch.Tensor:
        nbrs = ctx.index.neighbors(v)
        if nbrs.numel() == 0:
            return nbrs
        idx = int(torch.randint(nbrs.numel(), (1,), generator=ctx.generator))
        return nbrs[idx : idx + 1]


class TwoHopRing(_MaskedNeighborProbe):
    """D: predict the pooled clean embedding of the radius-2 ring."""

    name = "D"

    def _targets(self, ctx: PairContext, v: int) -> torch.Tensor:
        return ctx.index.ring2(v)


class CommunitySibling(_MaskedNeighborProbe):
    """F: predict a non-neighbor sibling in the same detected community."""

    name = "F"
    needs_communities = True

    def _targets(self, ctx: PairContext, v: int) -> torch.Tensor:
        u = ctx.index.sample_sibling(v, ctx.generator)
        if u is None:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor([u], dtype=torch.long)


class AugmentedView(PairDesign):
    """E: predict the focal node's embedding across two stochastic augmentations.

    No masking. The context view is augmentation A; the target view is
    augmentation B. The target view changes every pass, so ``static_target`` is
    False and the engine recomputes the target embedding each time.
    """

    name = "E"
    static_target = False

    def __init__(
        self,
        feat_drop_a: float = 0.2,
        edge_drop_a: float = 0.2,
        feat_drop_b: float = 0.1,
        edge_drop_b: float = 0.3,
    ) -> None:
        self.feat_drop_a = feat_drop_a
        self.edge_drop_a = edge_drop_a
        self.feat_drop_b = feat_drop_b
        self.edge_drop_b = edge_drop_b

    def make_pair(self, ctx: PairContext) -> Pair | None:
        focal_cpu = ctx.focal_idx.cpu()
        target_lists = [torch.tensor([v], dtype=torch.long) for v in focal_cpu.tolist()]
        assembled = assemble_targets(focal_cpu, target_lists)
        if assembled is None:
            return None
        x_ctx, e_ctx = augment_view(
            ctx.x, ctx.edge_index, self.feat_drop_a, self.edge_drop_a, ctx.generator
        )
        x_tgt, e_tgt = augment_view(
            ctx.x, ctx.edge_index, self.feat_drop_b, self.edge_drop_b, ctx.generator
        )
        return _to_device_pair(x_ctx, e_ctx, x_tgt, e_tgt, assembled, ctx.device)


# Default probe registry. A is the canonical recover-focal config (J0).
def default_probes() -> dict[str, PairDesign]:
    return {
        "A": RecoverFocal(),
        "B": PooledNeighborhood(),
        "C": SampledNeighbor(),
        "D": TwoHopRing(),
        "E": AugmentedView(),
        "F": CommunitySibling(),
    }


PROBE_CLASSES: dict[str, type[PairDesign]] = {
    "A": RecoverFocal,
    "B": PooledNeighborhood,
    "C": SampledNeighbor,
    "D": TwoHopRing,
    "E": AugmentedView,
    "F": CommunitySibling,
}

__all__ = [
    "RecoverFocal",
    "PooledNeighborhood",
    "SampledNeighbor",
    "TwoHopRing",
    "AugmentedView",
    "CommunitySibling",
    "default_probes",
    "PROBE_CLASSES",
]
