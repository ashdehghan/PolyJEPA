# Quickstart

## Install

PolyJEPA depends only on PyTorch and PyTorch Geometric.

```bash
pip install polyjepa            # once published
# or, from a checkout:
pip install -e .
```

## Fingerprint a graph

`fingerprint` takes any PyTorch Geometric `Data` object and runs all six probes.

```python
from torch_geometric.datasets import Planetoid
from polyjepa import fingerprint

data = Planetoid(root="/tmp/Cora", name="Cora")[0]
fp = fingerprint(data, seed=0)

fp.probe_names        # ['A', 'B', 'C', 'D', 'E', 'F']
fp.residuals.shape    # (N, 6)
fp.embeddings["C"]    # (N, d) embedding under probe C
fp.cross_probe        # (6, 6) Spearman matrix
fp.healthy()          # True if no probe collapsed
```

Extra keyword arguments are forwarded to the engine, for example `epochs`,
`hidden_dim`, `latent_dim`, `backbone`, `device`.

## Run a single probe

For one property, use [`JEPAEngine`](../reference/engine.md) directly:

```python
from polyjepa import JEPAEngine, SampledNeighbor

eng = JEPAEngine(SampledNeighbor(), seed=0, epochs=200).fit(data)
residuals = eng.score()        # (N,) higher = less predictable, NaN where unscored
embedding = eng.embeddings()   # (N, d) clean-graph target embedding
```

## Choose a backbone

```python
fingerprint(data, backbone="mlp")   # structure-free control
# also: "gcn" (default), "sage", "gat"
```

Next: [interpreting the fingerprint](interpreting-the-fingerprint.md).
