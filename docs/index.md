# PolyJEPA

**A multi-probe Joint-Embedding Predictive Architecture that fingerprints graphs.**

A graph goes in; a family of self-supervised probes comes out as a *fingerprint*.
Every probe shares one fixed architecture and differs only in how it builds its
context/target pair, so each probe is a distinct objective that measures a
different property of every node's role in the graph.

<figure markdown="span">
  ![PolyJEPA architecture](assets/architecture.svg){ width="760" }
  <figcaption>One fixed JEPA template. Only the construction of the context/target pair changes between probes.</figcaption>
</figure>

## What you get

For each probe, training one JEPA yields, per node, a learned **embedding** and a
prediction **residual**. Across the six probes, a graph's fingerprint is:

- an **embedding bank**: one `(N, d)` representation per probe,
- a **residual fingerprint**: an `(N, P)` matrix, one residual column per probe,
- **per-graph descriptors** and a **cross-probe correlation matrix**.

Residuals rank nodes by surprise (anomaly, difficulty, curriculum); embeddings
serve similarity and link prediction; the cross-probe matrix shows whether the
probes are complementary axes or redundant copies.

## Quickstart

```python
from torch_geometric.datasets import Planetoid
from polyjepa import fingerprint

data = Planetoid(root="/tmp/Cora", name="Cora")[0]
fp = fingerprint(data, seed=0)        # trains all six probes

fp.residuals          # (N, P) per-node residual fingerprint
fp.embeddings["A"]    # (N, d) embedding under probe A
fp.cross_probe        # (P, P) Spearman across probes
```

## Where to go next

<div class="grid cards" markdown>

-   :material-cog: **[The architecture](concepts/architecture.md)**

    The fixed JEPA template, the loss, and why it does not collapse.

-   :material-fingerprint: **[The fingerprint](concepts/fingerprint.md)**

    The two-layer output and how to read it.

-   :material-grid: **[The probes](probes/index.md)**

    All six pair designs, each with a schematic and real results.

-   :material-book-open-variant: **[Quickstart & guide](guide/quickstart.md)**

    Fingerprint a graph and interpret the result.

</div>

!!! note "Status"
    PolyJEPA is a standalone, self-contained package (PyTorch + PyTorch Geometric
    only). It ships with a comprehensive test suite and reproducible figure
    pipeline. Probe semantics are validated on planted synthetic graphs; see
    [Validation methodology](guide/validation.md).
