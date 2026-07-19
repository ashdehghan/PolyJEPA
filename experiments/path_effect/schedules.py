"""Schedules: a ``(T, N)`` matrix ``W`` of per-node loss weights over epochs.

The experiment isolates the *path* from the *marginal*, so every schedule satisfies
two exact constraints:

- **column sums** ``sum_t W[t, i] == T`` for every node ``i`` — each node receives the
  same total gradient mass as under flat training. This is the time-marginal, held
  fixed. Without it we would be measuring reweighting, which Wu et al. (ICLR 2021)
  already showed is where curriculum's benefit actually comes from.
- **row sums** ``sum_i W[t, i] == N`` for every epoch ``t`` — each step carries the same
  total gradient mass, so schedules cannot differ merely by taking bigger steps.

A matrix satisfying both is doubly stochastic up to scale, so we build any raw affinity
and project it onto that set with Sinkhorn iterations. Flat training is ``W == 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.path_effect.coords import COORD_NAMES


@dataclass(frozen=True)
class Schedule:
    name: str
    family: str  # flat | random | sweep | curriculum
    weights: np.ndarray  # (T, N)


def sinkhorn(raw: np.ndarray, iters: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Project a positive matrix onto {row sums == N, col sums == T}."""
    t, n = raw.shape
    w = np.clip(raw.astype(np.float64), 1e-12, None)
    for _ in range(iters):
        w *= n / w.sum(axis=1, keepdims=True)  # rows -> N
        w *= t / w.sum(axis=0, keepdims=True)  # cols -> T
        if (
            np.abs(w.sum(axis=1) - n).max() < tol
            and np.abs(w.sum(axis=0) - t).max() < tol
        ):
            break
    # One last row pass leaves columns exact to ~1e-12; re-run cols to finish on the
    # constraint that matters most (the marginal), then assert both in the test.
    w *= n / w.sum(axis=1, keepdims=True)
    w *= t / w.sum(axis=0, keepdims=True)
    return w


def flat(t: int, n: int) -> Schedule:
    return Schedule("flat", "flat", np.ones((t, n), dtype=np.float64))


def _spotlight(order: np.ndarray, t: int, sigma: float) -> np.ndarray:
    """A moving Gaussian spotlight sweeping along ``order`` (a node ranking in [0,1])."""
    centres = np.linspace(0.0, 1.0, t)[:, None]  # (T, 1)
    return np.exp(-((order[None, :] - centres) ** 2) / (2.0 * sigma**2))


def _pacing(order: np.ndarray, t: int, floor: float) -> np.ndarray:
    """Classic cumulative curriculum: the visible set grows along ``order``."""
    centres = np.linspace(0.15, 1.0, t)[:, None]
    return np.where(order[None, :] <= centres, 1.0, floor)


def build_schedules(
    coords: np.ndarray,
    t: int,
    n_random: int,
    seed: int,
    sigma: float = 0.15,
    floor: float = 0.05,
) -> list[Schedule]:
    """Flat + coordinate sweeps (both directions) + cumulative curricula + random paths."""
    n = coords.shape[0]
    rng = np.random.default_rng(seed)
    out: list[Schedule] = [flat(t, n)]

    # Structured extremes: what a person would actually design.
    for c, name in enumerate(COORD_NAMES):
        rank = coords[:, c].argsort().argsort() / max(1, n - 1)
        for direction, tag in ((rank, "asc"), (1.0 - rank, "desc")):
            out.append(
                Schedule(
                    f"sweep:{name}:{tag}", "sweep", sinkhorn(_spotlight(direction, t, sigma))
                )
            )
            out.append(
                Schedule(
                    f"curriculum:{name}:{tag}",
                    "curriculum",
                    sinkhorn(_pacing(direction, t, floor)),
                )
            )

    # Random paths: sweeps along random directions in coordinate space, plus
    # unstructured smooth noise. Together these sample the space rather than
    # three points in it.
    for k in range(n_random):
        if k % 2 == 0:
            w = rng.normal(size=coords.shape[1])
            proj = coords @ (w / np.linalg.norm(w))
            rank = proj.argsort().argsort() / max(1, n - 1)
            sig = float(rng.uniform(0.08, 0.40))
            raw = _spotlight(rank, t, sig)
            fam = "random"
        else:
            base = rng.normal(size=(max(2, t // 8), n))
            raw = np.exp(
                np.repeat(base, int(np.ceil(t / base.shape[0])), axis=0)[:t]
                * float(rng.uniform(0.5, 2.0))
            )
            fam = "random"
        out.append(Schedule(f"random:{k}", fam, sinkhorn(raw)))

    return out


def arrival_time(w: np.ndarray) -> np.ndarray:
    """Mean arrival epoch per node, in [0, 1]: *when* a node's mass lands."""
    t = w.shape[0]
    tt = np.linspace(0.0, 1.0, t)[:, None]
    return (w * tt).sum(axis=0) / w.sum(axis=0)


def descriptors(w: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Describe a schedule by HOW IT COUPLES ARRIVAL TIME TO NODE COORDINATES.

    This is the only place node identity enters the descriptor, which is what makes
    the scrambled-coordinate control meaningful: scrambling the coordinates destroys
    the coupling while leaving the schedule itself untouched.
    """
    tau_raw = arrival_time(w)
    spread = float(tau_raw.std())  # how spread out arrivals are (coordinate-free)
    tau = (tau_raw - tau_raw.mean()) / (tau_raw.std() + 1e-12)
    feats = [spread]
    for c in range(coords.shape[1]):
        col = coords[:, c]
        feats.append(float(np.corrcoef(col, tau)[0, 1]))  # early-vs-late coupling
        # concentration: does this coordinate sit at the extremes of the sweep or its middle?
        feats.append(float(np.corrcoef(col, np.abs(tau))[0, 1]))
    return np.nan_to_num(np.array(feats, dtype=np.float64))
