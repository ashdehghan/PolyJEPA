"""The fixed JEPA engine shared by every probe.

One ``JEPAEngine`` trains a single probe's objective: a context encoder
``f_theta`` consumes the probe's context view, a bottleneck MLP predictor
``g_phi`` maps it toward the target space, and an EMA target encoder ``f_xi``
(stop-gradient) reads the probe's target view. The per-focal loss is the squared
L2 residual between the prediction and the pooled target embedding.

Design choices follow BGRL (Thakoor et al., ICLR 2022) and I-JEPA (Assran et
al., CVPR 2023):

- predictor on the online branch only (the core anti-collapse device),
- EMA momentum on a cosine schedule from ``ema_base`` to 1.0,
- AdamW with linear warmup then cosine LR decay,
- collapse diagnostics (embedding std and mean norm) logged during training.

After training, :meth:`score` returns the per-node residual field (averaged over
several sweeps) and :meth:`embeddings` returns the clean-graph target embedding,
the canonical per-node representation under this probe's objective.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch_geometric.data import Data

from polyjepa.backbones import Encoder, build_backbone
from polyjepa.graph_index import GraphIndex
from polyjepa.pair_design import PairContext, PairDesign, segment_mean


class _Predictor(nn.Module):
    """Bottleneck MLP predictor: ``latent -> hidden -> ... -> latent``."""

    def __init__(self, latent_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("predictor_depth must be >= 1")
        if depth == 1:
            self.net: nn.Module = nn.Linear(latent_dim, latent_dim)
            return
        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.PReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.PReLU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _TargetEMA:
    """EMA copy of the encoder; forward is no-grad, in eval mode."""

    def __init__(self, encoder: nn.Module) -> None:
        self.encoder = copy.deepcopy(encoder)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    @torch.no_grad()
    def update(self, source: nn.Module, momentum: float) -> None:
        for tp, sp in zip(
            self.encoder.parameters(), source.parameters(), strict=True
        ):
            tp.data.mul_(momentum).add_(sp.data, alpha=1.0 - momentum)
        # Track encoder buffers (e.g. BatchNorm running stats) on the target.
        # These are hard-copied, not EMA-blended: the target's normalization
        # statistics equal the online encoder's exactly each step. This is an
        # intentional choice that matches some BGRL implementations; only the
        # parameters follow the EMA momentum.
        for tb, sb in zip(self.encoder.buffers(), source.buffers(), strict=True):
            tb.data.copy_(sb.data)

    def to(self, device: torch.device) -> _TargetEMA:
        self.encoder.to(device)
        return self


def _sweep_chunks(
    num_nodes: int, focal_ratio: float, generator: torch.Generator
) -> list[torch.Tensor]:
    """Disjoint focal-index chunks covering all nodes exactly once."""
    perm = torch.randperm(num_nodes, generator=generator)
    chunk_size = max(1, int(round(focal_ratio * num_nodes)))
    return [perm[i : i + chunk_size] for i in range(0, num_nodes, chunk_size)]


@dataclass
class Diagnostics:
    """Training-time collapse diagnostics (BGRL Figs 7-8)."""

    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    embed_std: list[float] = field(default_factory=list)
    embed_norm: list[float] = field(default_factory=list)

    def healthy(self, std_floor: float = 1e-3) -> bool:
        """True if the final embedding std is above a collapse floor."""
        return bool(self.embed_std) and self.embed_std[-1] > std_floor


class JEPAEngine:
    """Train one probe's JEPA objective and produce its residual + embedding.

    Parameters mirror the BGRL/I-JEPA defaults; see module docstring. ``probe``
    is the :class:`PairDesign` whose objective this engine trains.
    """

    def __init__(
        self,
        probe: PairDesign,
        backbone: str = "gcn",
        hidden_dim: int = 256,
        latent_dim: int = 256,
        predictor_hidden_dim: int = 256,
        predictor_depth: int = 2,
        num_layers: int = 2,
        ema_base: float = 0.99,
        epochs: int = 300,
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 30,
        focal_ratio: float = 0.30,
        score_passes: int = 4,
        diag_every: int = 25,
        seed: int | None = None,
        device: str | None = None,
        backbone_kwargs: dict | None = None,
    ) -> None:
        if not 0.0 < focal_ratio <= 1.0:
            raise ValueError("focal_ratio must lie in (0, 1]")
        self.probe = probe
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        self.predictor_depth = predictor_depth
        self.num_layers = num_layers
        self.ema_base = ema_base
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.focal_ratio = focal_ratio
        self.score_passes = score_passes
        self.diag_every = diag_every
        self.seed = seed
        self.device = device or "cpu"
        self.backbone_kwargs = backbone_kwargs or {}

        self._encoder: Encoder | None = None
        self._predictor: _Predictor | None = None
        self._target: _TargetEMA | None = None
        self._mask_token: nn.Parameter | None = None
        self._index: GraphIndex | None = None
        self._scores: torch.Tensor | None = None
        self._embeddings: torch.Tensor | None = None
        self.diagnostics = Diagnostics()

    # -- schedules -----------------------------------------------------------
    def _lr_factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.epochs - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def _ema_tau(self, step: int) -> float:
        progress = step / max(1, self.epochs)
        return 1.0 - (1.0 - self.ema_base) * (1.0 + math.cos(math.pi * progress)) / 2.0

    # -- fit -----------------------------------------------------------------
    def fit(self, data: Data) -> JEPAEngine:
        if data.x is None:
            raise ValueError("JEPAEngine requires data.x")
        num_nodes = int(data.num_nodes)
        if num_nodes == 0:
            raise ValueError("JEPAEngine requires a non-empty graph")
        in_dim = int(data.x.shape[1])
        device = torch.device(self.device)

        if self.seed is not None:
            torch.manual_seed(self.seed)
        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(self.seed)

        x = data.x.float().to(device)
        edge_index = data.edge_index.to(device)
        self._index = GraphIndex(data.edge_index, num_nodes, seed=self.seed)
        if self.probe.needs_communities:
            self._index.community_of(0)  # force community build up front

        encoder = build_backbone(
            self.backbone,
            in_dim,
            self.hidden_dim,
            self.latent_dim,
            self.num_layers,
            **self.backbone_kwargs,
        ).to(device)
        predictor = _Predictor(
            self.latent_dim, self.predictor_hidden_dim, self.predictor_depth
        ).to(device)
        target = _TargetEMA(encoder).to(device)
        mask_token = nn.Parameter(torch.zeros(in_dim, device=device))
        nn.init.normal_(mask_token, std=0.02)

        params = (
            list(encoder.parameters()) + list(predictor.parameters()) + [mask_token]
        )
        optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )

        n_focal = max(1, int(round(self.focal_ratio * num_nodes)))
        for step in range(self.epochs):
            encoder.train()
            predictor.train()
            for g in optimizer.param_groups:
                g["lr"] = self.lr * self._lr_factor(step)
            optimizer.zero_grad()

            perm = torch.randperm(num_nodes, generator=gen)
            focal_idx = perm[:n_focal].to(device)
            ctx = PairContext(
                x=x,
                edge_index=edge_index,
                focal_idx=focal_idx,
                mask_token=mask_token,
                index=self._index,
                generator=gen,
                num_nodes=num_nodes,
                device=device,
            )
            pair = self.probe.make_pair(ctx)
            if pair is None:
                continue

            z_ctx = encoder(pair.x_ctx, pair.edge_index_ctx)
            z_pred = predictor(z_ctx)
            with torch.no_grad():
                z_tgt = target.forward(pair.x_tgt, pair.edge_index_tgt)
            pooled = segment_mean(
                z_tgt[pair.tgt_index], pair.tgt_group, pair.focal.shape[0]
            )
            diff = z_pred[pair.focal] - pooled
            loss = diff.pow(2).sum(dim=-1).mean()

            loss.backward()
            optimizer.step()
            target.update(encoder, self._ema_tau(step))

            if step % self.diag_every == 0 or step == self.epochs - 1:
                with torch.no_grad():
                    self.diagnostics.steps.append(step)
                    self.diagnostics.losses.append(float(loss.detach().cpu()))
                    self.diagnostics.embed_std.append(
                        float(z_tgt.std(dim=0).mean().cpu())
                    )
                    self.diagnostics.embed_norm.append(
                        float(z_tgt.norm(dim=1).mean().cpu())
                    )

        self._encoder = encoder
        self._predictor = predictor
        self._target = target
        self._mask_token = mask_token

        self._embeddings = target.forward(x, edge_index).detach().cpu()
        self._scores = self._compute_scores(x, edge_index, num_nodes, gen).cpu()
        return self

    # -- scoring -------------------------------------------------------------
    @torch.no_grad()
    def _compute_scores(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        gen: torch.Generator,
    ) -> torch.Tensor:
        assert self._encoder is not None
        assert self._predictor is not None
        assert self._target is not None
        assert self._mask_token is not None
        assert self._index is not None

        self._encoder.eval()
        self._predictor.eval()
        device = x.device

        z_tgt_static = (
            self._target.forward(x, edge_index) if self.probe.static_target else None
        )
        scores = torch.zeros(num_nodes, device=device)
        counts = torch.zeros(num_nodes, device=device)

        for _ in range(self.score_passes):
            for chunk in _sweep_chunks(num_nodes, self.focal_ratio, gen):
                ctx = PairContext(
                    x=x,
                    edge_index=edge_index,
                    focal_idx=chunk.to(device),
                    mask_token=self._mask_token,
                    index=self._index,
                    generator=gen,
                    num_nodes=num_nodes,
                    device=device,
                )
                pair = self.probe.make_pair(ctx)
                if pair is None:
                    continue
                z_ctx = self._encoder(pair.x_ctx, pair.edge_index_ctx)
                z_pred = self._predictor(z_ctx)
                if z_tgt_static is not None:
                    z_tgt = z_tgt_static
                else:
                    z_tgt = self._target.forward(pair.x_tgt, pair.edge_index_tgt)
                pooled = segment_mean(
                    z_tgt[pair.tgt_index], pair.tgt_group, pair.focal.shape[0]
                )
                resid = (z_pred[pair.focal] - pooled).pow(2).sum(dim=-1)
                scores.index_add_(0, pair.focal, resid)
                counts.index_add_(0, pair.focal, torch.ones_like(resid))

        # Nodes never valid as focal (e.g. isolated nodes under probe B) are
        # undefined for this probe; mark them NaN rather than a misleading 0.
        out = scores / counts.clamp(min=1.0)
        out[counts == 0] = float("nan")
        return out

    def score(self, data: Data | None = None) -> torch.Tensor:
        if self._scores is None:
            assert data is not None, "call fit(data) first or pass data"
            self.fit(data)
        assert self._scores is not None
        return self._scores

    def embeddings(self) -> torch.Tensor:
        assert self._embeddings is not None, "call fit(data) first"
        return self._embeddings
