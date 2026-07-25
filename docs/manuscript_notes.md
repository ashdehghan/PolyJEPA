# Manuscript working notes (research version)

## Title shortlist (decide with the abstract; Ash chose "what/when + transfer" direction)

1. **What to Train on, and When: Separating the Marginal from the Path in GNN Training**
   — the working candidate; names the decomposition, covers both halves.
2. **The Marginal and the Path: What Transfers in Graph Neural Network Training Schedules**
   — decomposition + the transfer punchline in one breath.
3. **What, When, and What Carries Over: Schedule Decomposition and Transfer in GNN Training**
   — explicit three-beat; slightly busy.
4. **Separating What from When: Path Dependence, Selection, and the Limits of Transfer in GNN Training**
   — most complete; longest.
5. **Order Matters, Recipes Don't Transfer: The Marginal and the Path in GNN Training**
   — punchiest; risks overclaiming "order matters" as ours post-LeDoux.

Recommendation pending abstract: 1 or 2.

## Abstract skeleton (results-first)

1. Contradiction: path dependence real (Frankle/Achille/LeDoux) yet curricula don't beat random (Wu).
   Resolution: the field conflated *what* (marginal) with *when* (path); formalize schedules as
   measures, decompose, test each half.
2. Timing: at exactly equal per-node budget (Sinkhorn), order alone moves final accuracy on
   3 citation graphs (existence, graded, multi-seed — the natural-benchmark complement to
   LeDoux's synthetic demonstration); signal is early-training; BUT the per-node timing signal
   is dataset-local and not a function of any static coordinate system (nonparametric nulls) —
   answering LeDoux's open question about ordering design in analytically unknown domains.
3. Selection: at low label budgets, representativeness transfers (ProbCover density centers,
   label-free; class-stratified easy-first, label-aware) across [3–5] graphs incl. non-citation
   [pending Amazon]; confirms cold-start phase transition + coverage/coreset duality in the
   transductive setting; probe fingerprint holds real but non-transferable signal.
4. Verdict: what to train on is learnable and transfers via representativeness; when to train
   is real but dataset-local; the search for portable timing recipes is the wrong search.

## Post-LeDoux framing rules (from claim audit rows 1–2)

- Never claim "first order-only experiment"; claim: first on standard natural benchmarks with
  continuous Sinkhorn-exact schedules + graded multi-seed effects.
- Frame LeDoux as concurrent work, prominently, §1 + §2: mechanism (their gradient-level
  entanglement) complements our design-space formalism (schedule-level measure).
- Use their p.26 open question as the bridge INTO our predictability program.
