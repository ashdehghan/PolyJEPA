"""The guard on the whole experiment.

If the time-marginal is not exactly equal across arms, the experiment measures
reweighting rather than sequencing -- which is the confound that sank the curriculum
literature (Wu et al., ICLR 2021: with pacing fixed, random order matches easy-to-hard).
These assertions are the difference between a result and an artifact.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.path_effect.schedules import arrival_time, build_schedules, descriptors

T, N = 60, 140


@pytest.fixture(scope="module")
def coords() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(N, 5))


@pytest.fixture(scope="module")
def scheds(coords):
    return build_schedules(coords, t=T, n_random=12, seed=0)


def test_every_schedule_has_the_flat_marginal(scheds):
    """Column sums == T: every node gets exactly the gradient mass it gets under flat."""
    for s in scheds:
        col = s.weights.sum(axis=0)
        assert np.allclose(col, T, rtol=0, atol=1e-6), (
            f"{s.name}: per-node marginal deviates by "
            f"{np.abs(col - T).max():.2e} -- this arm measures reweighting, not path"
        )


def test_every_epoch_carries_the_same_mass(scheds):
    """Row sums == N: no schedule can win merely by taking bigger steps."""
    for s in scheds:
        row = s.weights.sum(axis=1)
        assert np.allclose(row, N, rtol=0, atol=1e-6), (
            f"{s.name}: per-epoch mass deviates by {np.abs(row - N).max():.2e}"
        )


def test_schedules_are_actually_different_paths(scheds):
    """Guard against Sinkhorn flattening everything into the baseline."""
    flat = next(s for s in scheds if s.name == "flat")
    others = [s for s in scheds if s.name != "flat"]
    assert others, "no non-flat schedules were built"
    for s in others:
        spread = arrival_time(s.weights).std()
        assert spread > 1e-3, f"{s.name} collapsed to a flat path (arrival std {spread:.2e})"
    assert np.allclose(arrival_time(flat.weights).std(), 0.0, atol=1e-9)


def test_scrambling_coordinates_destroys_the_descriptor(coords, scheds):
    """The control must actually control: scrambling coords must change descriptors."""
    rng = np.random.default_rng(1)
    perm = rng.permutation(coords.shape[0])
    sweep = next(s for s in scheds if s.name.startswith("sweep:"))
    real = descriptors(sweep.weights, coords)
    scrambled = descriptors(sweep.weights, coords[perm])
    assert not np.allclose(real, scrambled, atol=1e-3)
    # A real sweep couples arrival time to its coordinate almost perfectly; a scrambled
    # one cannot. This is the signal the predictability regression is looking for.
    assert np.abs(real).max() > 0.5
    assert np.abs(scrambled).max() < np.abs(real).max()
