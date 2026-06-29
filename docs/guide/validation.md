# Validation methodology

How do we know a probe measures the property it claims to? We test it on **planted
synthetic graphs** where the ground truth is known, then check that the residual
lines up with the planted structure.

## Planted generators

[`polyjepa.synthetic`](../reference/synthetic.md) builds graphs with known structure:

- `planted_sbm` — a stochastic block model with planted cross-community
  **bridge** nodes (`bridge_mask`);
- `inject_feature_outliers` — nodes whose features are off-distribution
  (`outlier_mask`).

```python
from polyjepa.synthetic import planted_sbm, inject_feature_outliers

data = planted_sbm(num_communities=3, nodes_per_comm=26, num_bridges=6, seed=0)
data = inject_feature_outliers(data, frac=0.08, seed=1)
data.bridge_mask, data.outlier_mask     # the ground truth
```

## What the contracts assert

The test suite encodes these as behavioral contracts (paraphrased):

- the **isolated node tops probe A**;
- planted **feature outliers** score higher on **A** and **B**;
- planted **bridges** score higher on **C**;
- **D** ranks differently from **C** (a distinct axis).

These are visible in the residual-field figures on each probe page: the planted
markers (squares for bridges, diamonds for outliers) sit on the high-residual nodes
of the matching probe.

## Why behavioral contracts, not exact snapshots

Exact floating-point results are not guaranteed across PyTorch versions or
platforms, so the suite asserts stable behavioral properties (rankings, signs,
separations, cross-probe distinctness) plus within-run determinism (same seed gives
an identical fingerprint), rather than pinning exact numbers. The seed reaches
every source of randomness, including Louvain community detection, so probe F's
partition is fixed per seed and varies across seeds. See the
[development guide](../development.md).

## Honest scope

These guarantees hold on planted graphs where the ground truth is known. On real
graphs without labels, the instrument is mechanically healthy and the probes are
distinct, but mapping the rankings onto interpretable real-graph properties is an
ongoing line of work.
