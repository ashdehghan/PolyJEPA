"""L1 unit tests for pair-design helpers (pure functions)."""

from __future__ import annotations

import torch
from hypothesis import given
from hypothesis import strategies as st

from polyjepa.pair_design import apply_mask, assemble_targets, segment_mean


def test_segment_mean_matches_manual():
    values = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [7.0, 7.0]])
    group = torch.tensor([0, 0, 1, 2])
    out = segment_mean(values, group, 3)
    assert torch.allclose(out, torch.tensor([[2.0, 2.0], [5.0, 5.0], [7.0, 7.0]]))


def test_segment_mean_empty_group_is_zero():
    values = torch.tensor([[2.0, 0.0]])
    group = torch.tensor([0])
    out = segment_mean(values, group, 3)
    assert torch.allclose(out[1], torch.zeros(2))
    assert torch.allclose(out[2], torch.zeros(2))


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=4),
            st.floats(
                min_value=-1e3,
                max_value=1e3,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_segment_mean_property(pairs):
    k = 5
    group = torch.tensor([g for g, _ in pairs])
    values = torch.tensor([[v] for _, v in pairs], dtype=torch.float64)
    out = segment_mean(values, group, k)
    for c in range(k):
        sel = [v for g, v in pairs if g == c]
        expected = sum(sel) / len(sel) if sel else 0.0
        assert abs(out[c, 0].item() - expected) < 1e-4


def test_assemble_targets_drops_empty_focal():
    focal = torch.tensor([10, 11, 12])
    target_lists = [
        torch.tensor([1, 2]),
        torch.empty(0, dtype=torch.long),
        torch.tensor([3]),
    ]
    out = assemble_targets(focal, target_lists)
    assert out is not None
    focal_out, idx, group = out
    assert focal_out.tolist() == [10, 12]
    assert idx.tolist() == [1, 2, 3]
    assert group.tolist() == [0, 0, 1]


def test_assemble_targets_all_empty_returns_none():
    focal = torch.tensor([0, 1])
    target_lists = [torch.empty(0, dtype=torch.long) for _ in range(2)]
    assert assemble_targets(focal, target_lists) is None


def test_apply_mask_replaces_only_listed_rows():
    x = torch.arange(12.0).reshape(4, 3)
    token = torch.tensor([9.0, 9.0, 9.0])
    out = apply_mask(x, torch.tensor([1, 3]), token)
    assert torch.allclose(out[1], token)
    assert torch.allclose(out[3], token)
    assert torch.allclose(out[0], x[0])
    assert torch.allclose(out[2], x[2])


def test_apply_mask_empty_is_noop():
    x = torch.arange(6.0).reshape(2, 3)
    out = apply_mask(x, torch.empty(0, dtype=torch.long), torch.zeros(3))
    assert torch.allclose(out, x)
