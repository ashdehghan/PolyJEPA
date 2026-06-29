# Contributing to PolyJEPA

Thanks for your interest in PolyJEPA. This is a small, focused library, and the
priority is confidence and comprehensive coverage over speed: every change should
prove the system still behaves as expected and has not silently drifted.

## Development setup

PolyJEPA uses a `src/` layout and depends only on PyTorch, PyTorch Geometric,
NumPy, SciPy, and NetworkX. Set up a project-local virtual environment and install
the package with its dev (and, for docs work, docs) extras:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

Run everything from the repository root.

## Definition of done

A change is complete only when all of the following hold:

1. Any new pure function has unit tests.
2. Any new probe has a property contract proving it measures its property, plus
   determinism/shape coverage, plus NaN-marking coverage if it can leave nodes
   unscored.
3. Any new behavior has a contract test in the matching layer, written before or
   alongside the implementation.
4. The full suite passes: `python -m pytest`.
5. Coverage holds (line + branch): `python -m pytest --cov=polyjepa --cov-branch
   --cov-fail-under=90`.
6. Lint is clean: `python -m ruff check src tests`.
7. Documentation is in sync: docstrings updated for any new or changed public
   symbol, the affected concept/guide/probe page updated, and
   `mkdocs build --strict` is clean.

## Testing philosophy

Exact floating-point results are not guaranteed across torch versions or
platforms, so we do not pin exact numbers. Drift is caught by asserting stable
behavioral properties (rankings, signs, separations, cross-probe distinctness,
healthy diagnostics) plus within-run determinism (same seed gives identical
output). Never add a bit-exact golden-value test.

## Conventions

- Writing style (code comments, docstrings, markdown): no em dashes; use commas,
  parentheses, or colons. Keep a direct, technical tone.
- The package must stay import-clean: it depends only on its declared third-party
  dependencies, never on any sibling project. Real datasets come from
  `torch_geometric.datasets`.
- The engine is fixed. New measurement ideas are new `PairDesign` probes, not
  forks of the engine.

See `CLAUDE.md` for the full architecture overview and the recipe for adding a
probe.
