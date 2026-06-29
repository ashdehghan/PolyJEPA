# Development

PolyJEPA is a published library, so the priority is **confidence and comprehensive
coverage over speed**: every change must prove the system still behaves as expected
and has not silently drifted. Testing is a habit, run as part of every change.

## The test layers

- **L1 pure-function units** (`test_pair_design.py`, `test_graph_index.py`,
  `test_augment.py`, `test_synthetic.py`, `test_backbones.py`): the math and helpers,
  with Hypothesis where invariants are crisp.
- **L2 engine mechanics** (`test_engine.py`): shape and finiteness, determinism,
  non-collapse, EMA-update correctness, gradient flow to all parameters, a strong
  loss decrease (bootstrap losses plateau, not zero), and `NaN`-marking.
- **L3 probe semantics** (`test_probes.py`, `test_probes_unit.py`): on planted
  graphs, each probe measures its property; plus construction validation and the
  "no valid focal" branch.
- **L4 integration** (`test_fingerprint.py`): full fingerprint shape, the cross-probe
  matrix, descriptors, the embedding bank, and full-fingerprint determinism.

## How drift is caught

Behavioral contracts (rankings, signs, separations, cross-probe distinctness) plus
within-run determinism (same seed gives an identical fingerprint). No bit-exact
golden snapshots, since PyTorch does not guarantee identical floats across versions
or platforms.

## Commands

```bash
pip install -e ".[dev]"

python -m pytest                                   # full default suite
python -m pytest --cov=polyjepa --cov-branch \
    --cov-report=term-missing --cov-fail-under=90  # definition-of-done
python -m ruff check src tests                     # lint
```

## Definition of done

A change is done only when: any new pure function has L1 tests; any new probe has an
L3 property contract plus L2 determinism/shape coverage plus `NaN`-marking if it can
leave nodes unscored; the full suite passes; coverage holds at 90% (line+branch);
and `ruff` is clean.

## Adding a probe

1. Subclass `PairDesign` (or `_MaskedNeighborProbe` and implement `_targets`) in
   `probes/__init__.py`; set `name`, `static_target`, `needs_communities`.
2. Register it in `default_probes()` and `PROBE_CLASSES`.
3. Add an L3 property contract on a planted graph (extend `synthetic.py` if needed),
   plus an L2 determinism/shape check and a `NaN`-marking check if applicable.
4. Add a docs page under `docs/probes/` and a figure in `generate_figures.py`.
5. Run the definition-of-done commands. All green before you call it done.

## Regenerating the documentation figures

```bash
PYTHONPATH=src python docs/scripts/generate_figures.py
mkdocs serve            # preview at http://127.0.0.1:8000
```
