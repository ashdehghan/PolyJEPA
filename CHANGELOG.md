# Changelog

All notable changes to PolyJEPA are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Community detection is now seed-aware: the engine/fingerprint seed is threaded
  into Louvain so probe F's partition varies across seeds (still deterministic per
  seed), instead of a fixed `seed=0` partition for every run.
- Packaging metadata points at the canonical repository
  (`github.com/ashdehghan/PolyJEPA`); the package version is single-sourced from
  `polyjepa.__version__`.

### Fixed
- The pooled-target probes (B `PooledNeighborhood`, D `TwoHopRing`) no longer
  diverge during training. The EMA target encoder ran BatchNorm in eval mode using
  running statistics hard-copied from the online encoder, which gathers them on the
  heavily-masked context view; applied to the clean target view this exploded the
  embeddings (loss and embedding std diverged by orders of magnitude). The target
  now normalizes with batch statistics, consistent with the online branch.
- `Diagnostics.healthy()` now catches divergence, not only collapse: it requires the
  final embedding std to stay below a ceiling and the loss not to blow up, in
  addition to the existing collapse floor.
- The cross-probe Spearman matrix now always has a unit diagonal, even when a
  probe's residual column is constant.
- `JEPAEngine.fit` guards against empty graphs instead of indexing out of bounds.

### Removed
- Dropped the unused `scikit-learn` dependency from the `docs` extra.

## [0.1.0]

- Initial standalone release: the fixed JEPA engine, the six pair-design probes
  (A-F), pluggable backbones (GCN / GraphSAGE / GAT / MLP), planted-ground-truth
  synthetic generators, and the public `fingerprint(...)` entry point.

[Unreleased]: https://github.com/ashdehghan/PolyJEPA/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ashdehghan/PolyJEPA/releases/tag/v0.1.0
