<!--
  PyPI renders this README as Markdown with no scripts/styles, and does not
  render SVG. The rich visuals (diagrams, per-probe figures) live on the docs
  site linked below. Raster hero images can be added here, via absolute URLs,
  once the project has a public home.
-->

# PolyJEPA

**A multi-probe Joint-Embedding Predictive Architecture that fingerprints graphs.**

A graph goes in; a family of self-supervised probes comes out as a *fingerprint*.
One fixed JEPA template (context encoder, EMA target encoder, predictor, latent L2
loss) is held constant; the only thing that varies between probes is how the
context/target pair is built. Each probe is therefore a different objective that
measures a different property of every node's role in the graph.

📖 **[Full documentation](https://polyjepa.readthedocs.io)** — concepts, a page per
probe with diagrams and real results, the API reference, and the validation
methodology. *(Published when the project goes public.)*

Self-contained: depends only on PyTorch, PyTorch Geometric, NumPy, SciPy, and
NetworkX. No dependency on any other project; the package can be lifted out as its
own repository.

## What you get

For each probe, training one JEPA yields, per node, a learned **embedding** and a
prediction **residual**. Across the six probes, a graph's fingerprint is:

- an **embedding bank**: one `(N, d)` representation per probe,
- a **residual fingerprint**: an `(N, P)` matrix, one residual column per probe,
- **per-graph descriptors** and a **cross-probe correlation matrix**.

Residuals rank nodes by surprise (anomaly, difficulty, curriculum); embeddings serve
similarity and link prediction; the cross-probe matrix shows whether the probes are
complementary axes or redundant copies.

## The six probes

| Probe | Pair design | High residual means |
|-------|-------------|---------------------|
| A `recover-focal`       | mask the focal node; predict its own embedding             | isolated, boundary, feature/topology clash |
| B `pooled-neighborhood` | mask the 1-hop neighbors; predict their mean embedding     | node atypical for its neighborhood |
| C `sampled-neighbor`    | mask one neighbor; predict it (per edge)                   | surprising edges, bridges, rare roles |
| D `two-hop-ring`        | mask the radius-2 ring; predict it                         | periphery, weak clustering, heterophily |
| E `augmented-view`      | no mask; predict the focal across two augmentations        | fragile, low-redundancy nodes |
| F `community-sibling`   | mask a same-community non-neighbor; predict it             | community outliers, cluster mismatch |

## Install

```bash
pip install polyjepa            # once published
# from a checkout:
pip install -e .
```

## Quickstart

```python
from torch_geometric.datasets import Planetoid
from polyjepa import fingerprint

data = Planetoid(root="/tmp/Cora", name="Cora")[0]
fp = fingerprint(data, seed=0)          # trains all six probes

fp.residuals          # (N, P) per-node residual fingerprint (NaN where unscored)
fp.embeddings["A"]    # (N, d) embedding under probe A
fp.cross_probe        # (P, P) Spearman correlation across probes
fp.healthy()          # True if no probe collapsed
```

For a single property, use the engine directly:

```python
from polyjepa import JEPAEngine, SampledNeighbor

scores = JEPAEngine(SampledNeighbor(), seed=0).fit(data).score()
```

## Architecture (literature-grounded)

The template follows BGRL (Thakoor et al., ICLR 2022) with I-JEPA-style masked
prediction (Assran et al., CVPR 2023):

- 2-layer GCN encoder (PReLU + BatchNorm); pluggable backbone (GCN / GraphSAGE / GAT
  / MLP control);
- shallow bottleneck predictor on the online branch only (the core anti-collapse
  device);
- EMA target with stop-gradient, momentum scheduled 0.99 → 1.0;
- AdamW with linear warmup then cosine decay; collapse diagnostics logged.

## Validation

Probe semantics are validated on planted synthetic graphs where the ground truth is
known (isolated node tops A; feature outliers rank high on A and B; bridges rank high
on C). The package ships a comprehensive test suite (unit, engine-mechanics,
probe-semantics, integration) with behavioral-contract regression tests.

## License

MIT
