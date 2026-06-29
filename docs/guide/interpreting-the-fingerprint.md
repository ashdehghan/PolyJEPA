# Interpreting the fingerprint

## Within a probe: rank, do not compare magnitudes

A probe's residual orders nodes from predictable to surprising. The absolute value
is arbitrary (it depends on that probe's latent scale), so work with ranks or
percentiles, and never compare a raw residual on one probe to another.

```python
import torch

resid_C = fp.residual("C")                 # (N,)
finite = torch.isfinite(resid_C)
most_surprising = torch.argsort(resid_C[finite], descending=True)[:10]
```

To see *what* a probe ranks by, correlate its residual against interpretable node
statistics (degree, betweenness, clustering, k-core, homophily). That is exactly the
[probe-by-property signature](../probes/index.md#what-each-probe-tracks).

## Across probes: the cross-probe matrix

The $P\times P$ Spearman matrix is the real product. Off-diagonal entries near zero
mean two probes rank nodes independently (complementary axes); entries near one mean
they are redundant.

```python
fp.cross_probe        # (6, 6) tensor, rows/cols follow fp.probe_names
```

A useful summary is the mean absolute off-diagonal correlation: low values confirm
the fingerprint has distinct axes rather than six copies of one signal.

## Per-graph descriptors

Each probe is reduced to graph-level statistics, handy for comparing whole graphs:

```python
fp.graph_descriptors["A"]
# {'mean', 'std', 'skew', 'p10', 'p50', 'p90', 'gini', 'n_scored',
#  'pooled_embedding'}
```

## Unscored nodes

Probes B, C, D, and F can leave some nodes unscored (no neighbor, ring, or community
sibling); those residuals are `NaN` by design, not zero. Always filter with
`torch.isfinite` before ranking or computing statistics.
