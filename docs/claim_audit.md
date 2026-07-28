# Claim audit — PolyJEPA research manuscript

Every novelty-bearing claim in the paper, mapped to our evidence and the prior art that could
contest it. Verdicts: **ours** | **ours-narrowed** | **confirmation** | **needs-attribution** |
**retracted** | **PENDING** (reading incomplete). Feeds Appendix F of the manuscript.

Last updated: 2026-07-25 (session: manuscript rewrite, phase 1–2).

| # | Claim | Where stated | Our evidence | Closest prior art | Verdict |
|---|---|---|---|---|---|
| 1 | First marginal-exact, semantically structured ordering experiment measuring final quality | §2 "The gap" | replicate.py: 8k Sinkhorn schedules, 3 graphs | **LeDoux 2026 (2603.25047), deep-read 2026-07-25**: satisfies every clause of our gap sentence — per-epoch marginal exactly uniform ("The only variable is the ordering of training examples within each epoch," p. 12), STRIDE/TARGET are task-semantic orderings, outcome is final test accuracy (99.5% vs 0.30% at a shared compute cap). Synthetic modular arithmetic only; effects essentially binary (generalize/memorize/fail); RANDOM+TARGET single-seed; early-stop makes "equal compute" = "equal cap, losers used more" | **RETRACTED as stated → ours-narrowed**: rewrite §2 gap as: LeDoux demonstrates the order-only channel decisively on a synthetic task; what does not exist is such an experiment on *standard natural benchmarks*, with *graded* effects, *multi-seed* statistics, and *continuous Sinkhorn-exact per-example weight schedules* (strictly tighter marginal control than per-epoch permutation + early stop). Credit prominently as concurrent work. TEXT FIX PENDING (restructure pass) |
| 2 | Schedule-as-measure formalism: w: D×[0,T]→R≥0, decomposed into time-marginal (what) vs path (when); marginal null H1 | §3 | formalism itself | LeDoux's decomposition is a *gradient-space* object (entanglement term −ηH_B·∇L_A, Eq. 1; counterfactual projection Eqs. 5–6) — mechanistic, per-step; nothing in the paper decomposes a *schedule* into marginal + path. Their sharpest prose statement (p. 26): "varies the sequential relationships between examples while holding the per-epoch example set constant" | **ours-narrowed (final)**: the measure-theoretic decomposition + doubly-stochastic operationalization is ours; credit LeDoux for articulating the fixed-marginal/varying-order idea informally and supplying the gradient-level mechanism (Hessian-gradient entanglement) as the complement — they explain *how* order enters the gradient; we formalize *what the design space is* |
| 3 | Arrival-compass instrument (ridge from arrival profiles → val acc; directional, self-calibrating) | §6 (Results I) | replicate.py, 3 datasets, fwd-vs-rev +4.0/+1.6/+4.5 | Nothing similar known; LeDoux's orderings are hand-designed, not searched/fit | **ours** (PENDING tier-1 sweep) |
| 4 | β is gradient-dynamic and dataset-local; even direct dynamic measurement (GIS) fails to transfer | §6 | probe_compass + gis.py sign flip | Dataset cartography/GraNd measure training dynamics but for *selection*, not timing; no transferability claims to contest | **ours** (PENDING) |
| 5 | Timing null is airtight: β, |β|, sign(β) are not functions of probe/hand-crafted/SGC space (nonparametric, permutation-tested) | §6 anatomy | probe_anatomy.py: 18 LOO-kNN tests ≤ null | none known | **ours** (PENDING) |
| 6 | Low-budget selection: representativeness transfers (ProbCover q25 + stratified easy-first) | §7 | selection.py 3 graphs + Amazon Photo (Computers excluded: fails flat-baseline sanity gate, 0.368 vs ~0.80) | **Hacohen 2022** phase transition; **Yehuda 2022** coverage/coreset duality; read end-to-end 2026-07-24 | **confirmation, REFINED 2026-07-25 by the domain jump**: ProbCover transfers to Photo at every radius/budget (r* is BEST there, +27.4 at 5%, 10/10 — the purity rule works on co-purchase features where it failed on citation SGC); stratified easy-first does NOT transfer cleanly (Photo gain ≈ the stratification component; CLNode difficulty catastrophic −23 to −45). Sharpened principle for the paper: label-free representative COVERAGE transfers across domains; DIFFICULTY signals of every flavor are domain-local |
| 7 | Raw easy/hard flip is a class-coverage artifact; stratified easy-first transfers; consistent with Sorscher keep-easy-when-scarce | §7 | run2/run3 controls, random_strat decomposition | Sorscher 2022 (PENDING end-to-end read); CLNode read 2026-07-24 (no such control in their paper) | **ours** (the control + decomposition), consistent-with Sorscher. PENDING |
| 8 | Probe fingerprint holds real but non-transferable selection signal (CiteSeer wins, Cora catastrophic) | §7 | selection.py | AutoSSL/RCL scalarize residuals; nobody tests residual-space coverage for selection (PENDING reads) | **ours** (PENDING jin2022autossl, zhang2023rcl reads) |
| 9 | "Curriculum searched one dimension" — difficulty scalar admits only sort+sweep schedules; curves in probe space generalize | §3/§4 | formalism | Wu 2021 (empirical), Dataset Cartography (2-D map, no schedules) | **ours** as formal statement (PENDING) |
| 10 | 48-pt easy-vs-hard gap attributed to CLNode | (former §4) | — | CLNode Table 2: real gains +1 to +5.7 | **retracted 2026-07-24** — was our own pilot; fixed in §4 + App D Phase 5 |

## Reading log (tier 1)

| Paper | Status | Key takeaways for us |
|---|---|---|
| wei2023clnode | ✅ read 2026-07-24 | difficulty = pseudo-label neighborhood entropy + prototype sim, α=1; paced cumulative curriculum; gains +1–5.7; no 48-pt gap |
| hacohen2022typiclust | ✅ read 2026-07-24 | phase transition: typical@low budget; TypiClust = density per cluster |
| yehuda2022probcover | ✅ read 2026-07-24 | max-coverage↔coreset duality (§2.4); δ* purity rule (§3.2) = our r* |
| ledoux2026order | ✅ read 2026-07-25 (pp.1–4 direct + agent deep-read 5–51) | Ordering = information channel via Hessian-gradient entanglement (−ηH_B·∇L_A); epoch-level counterfactual decomposition → ordering component ≈85% of epoch-mean gradient energy in ALL strategies incl. IID (bandwidth, not productive signal — coherence differentiates); modular arithmetic p=9973, 2-layer transformer, per-epoch permutations (marginal uniform/epoch), 4 strategies (STRIDE/FIXED-RANDOM/RANDOM/TARGET) → 99.5/99.5/0.30/0.01%; Fourier fundamental = dual of stride (F=⌊p/s⌋, derivable!); FIXED-RANDOM works too (cyclic-group accident, flagged domain-specific). NO natural data, NO predictability/transferability tests, NO selection ("orthogonal axis," p. 6). Their open question (p. 26: "In domains where the task is not known analytically, deliberate ordering design is not straightforward") is EXACTLY what our compass + dataset-locality results answer — frame our timing results as the empirical answer in natural domains. Instrumented runs single-seed (199); RANDOM/TARGET never multi-seeded. |
| frankle2020lmc | ✅ read 2026-07-25 (agent, cover-to-cover) | Instability analysis accurate BUT: (1) their SGD noise = data order AND augmentation jointly — order never isolated; write "SGD noise (data order + augmentation)"; (2) it's an ERROR barrier, not loss barrier; (3) at-init barriers ≈ random guessing (~90% err CIFAR / ~99.9% ImageNet), instability ~80% at k=0 for ResNet-20/VGG-16 — but LeNet is STABLE at init, don't say "all"; barriers are graphical (Figs. 2/3, App. D/E), no table; (4) stability threshold <2%; stabilization: ResNet-20 iter 2000 (~3% of training), VGG-16 iter 1000 (~1.5%), ResNet-50 epoch 18 (~20%), Inception epoch 28 (~16%) — "early" literal only for CIFAR; (5) say "LINEARLY disconnected" (nonlinear paths still connect) |
| achille2019critical | ✅ read 2026-07-25 (agent, cover-to-cover) | Deficit = cataract-like blur (8×8 downsample); permanent if not removed in first ~40–60 epochs; damage peaks at onset ~epoch 30; headline: up to 3× test-error increase; Fisher trace rise/fall = information-plasticity loss. **ERROR CAUGHT: "survives even in deep linear networks" is NOT in this paper** (they test All-CNN, ReLU FC net, ResNet-18) — the deep-linear result is a different (later, Kleinman/Achille/Soatto) paper; re-source or drop |
| wu2021curricula | ✅ read 2026-07-25 (agent) | Characterizations ACCURATE (>25k models, 540 runs/dataset, pacing held fixed). STRENGTHENER: vs their fair baseline even *pacing* is n.s. in the standard regime — "pacing, not ordering" can be stated harder. Our marginal-confound critique technically accurate with one honest qualifier: in expectation over random permutations marginals DO equalize; the confound is per-realization. MUST represent their positive regimes: curricula help at short time budgets (352/1760 steps) and under label noise (20–80%, large margins) |
| sorscher2022beyond | ✅ read 2026-07-25 (agent) | Prototype-pruning characterization ACCURATE (k-means in SWaV space, cosine to centroid, label-free scalar). Keep-easy-when-scarce: their theory (perceptron) + CIFAR-10/EL2N empirics; NOT demonstrated with the prototype metric nor on ImageNet — pairing "scarce→easy" with the SSL metric is OUR extrapolation, say so. Operative variable = data per parameter (α_tot) jointly with kept fraction, not absolute size. Image-domain only; class balancing essential; example interactions unmodeled — our transductive analogy is beyond their evidence, state as such |
| lu2022grab / rajput2022permutation | ✅ read 2026-07-25 (agent) | Marginal-equalization + herding-not-difficulty + Rajput's exponential-to-nonexistent range all ACCURATE. **QUALIFIER REQUIRED: GraB explicitly claims + shows generalization gains** (val acc MNIST ~92–93 vs ~89, WikiText-2 val ppl ~200 vs ~210; "lets the model generalize better," p. 2) — our "measures convergence rate, not generalization" must be scoped to their *guarantees* |
| jin2022autossl / zhang2023rcl | ✅ read 2026-07-25 (agent) | **AutoSSL sentence WRONG on mechanism**: they learn per-TASK scalar loss weights (Eq. 1) searched via pseudo-homophily (CMA-ES / meta-gradient), not per-node weights, no gating network. The negative half survives (no residual-as-coordinates, schedules nothing). RCL characterization ACCURATE (per-edge reconstruction residual, easiest edges first, paced by age λ); nuance: their "self-supervised" probe is trained jointly with the supervised loss |

## Learned-schedules appendix (E5) — reading round 2026-07-28

| # | Claim | Verdict |
|---|---|---|
| 11 | Learning the coupling (time-indexed T×N weights, both marginals pinned via differentiable Sinkhorn, full-run unroll) is new | **ours-narrowed, verified by 6 full reads**: Franceschi 2017 §5.1 (data hyper-cleaning) is REAL precedent — gradient-learned per-example weights with box + ℓ1 budget through full unroll, but constant in time (= learned marginal, knob 1). Maclaurin 2015 learned per-iteration LR schedules (= knob 2) + training-set pixels; never example×time weights. Ren/Shu = reactive per-step meta-gradient weights, batch-simplex only. Graves = closed-loop bandit over tasks. Tay = Sinkhorn inside attention on activations. Knob 3 (the coupling) unlearned in all six. Appendix states this map explicitly |
| 12 | E5 pre-registration | Locked before run: primary = held-out gain > compass +1.9 (Cora); secondary = > best val-selected random schedule; else reported as timing-channel ceiling |

Reading log additions (all ✅ read 2026-07-28, notes in library): ren2018reweight,
shu2019metaweightnet, graves2017automated, maclaurin2015hypergrad, franceschi2017forward,
tay2020sinkhornattn.

## Strong-claim ledger (rule 12 decisions, abstract + §1–2, 2026-07-25)

| Claim | Decision |
|---|---|
| "large applied literature, thin evidential one" | REPLACED with facts+cites: applied literature is large (soviany survey); most systematic test of ordering found no benefit (wu) |
| "Order was never isolated." | SCOPED: "None of these designs isolates order." (about the designs just described) |
| "null hypothesis the curriculum literature never stated" | HEDGED: "to our knowledge ... has not stated" |
| "the field searched one axis" (thesis) | SCOPED: "the curricula that have been tried all search along a single axis" |
| Abstract "the two effects have never been measured separately" | DELETED the clause; LeDoux sentence in §2 covers the near-exception |
| Abstract "Exploiting timing therefore requires..." | SCOPED: "In our experiments, exploiting timing required a new search on every dataset." |
| Abstract "rules ... are specific to each dataset" | SCOPED: "the rules we found are specific to each dataset" |
| "equalizes marginals exactly by construction" (perm-SGD) | KEPT: mathematical fact |
| Frankle "barriers are enormous/large" | REPLACED with the fact: "interpolated error rises to near random guessing ... except LeNet" |
| Saglietti "same verdict" | KEPT with the consolidation-loss parenthetical scoping it |

## Text-fix queue (apply in the restructure pass)

1. §2 gap sentence + footnote → LeDoux re-attribution (audit row 1). 
2. §1 Frankle sentences: "data orders" → "SGD noise (data order and augmentation)"; "loss barrier"
   → "error barrier"; add "linearly" before "disconnected"; qualify "all but LeNet"; replace the
   unchecked-numbers footnote with real ones (≈random-guessing barriers at init; stability by
   1.5–3% of training on CIFAR, 16–20% on ImageNet).
3. §1 Achille sentence: DROP "and this survives even in deep linear networks" or re-source to the
   later deep-linear critical-periods paper (Kleinman/Achille/Soatto — verify exact ref before
   citing); add the 3× test-error headline number and the blur-deficit description.
4. Abstract: "identical initializations trained with different data orders land in linearly
   disconnected basins" → credit SGD noise jointly, keep "linearly."
5. §1 Wu paragraph: add per-realization qualifier to our marginal critique; state the
   strengthener (even pacing n.s. vs fair baseline in the standard regime); add their positive
   regimes (short budgets, label noise) for fairness.
6. §7/§8 Sorscher engagement: phrase as "consistent with the scarce-regime prediction of their
   perceptron theory (validated by them on CIFAR-10/EL2N); the extension to prototype-style
   metrics and transductive graphs is ours."
7. §2 permutation-SGD sentence: scope "measures convergence rate, not generalization" to their
   *guarantees* — GraB reports validation-quality gains (Fig. 2) and claims better
   generalization in its abstract.
8. §4 AutoSSL sentence: rewrite to "searches per-task scalar loss weights via a
   pseudo-homophily objective (CMA-ES / meta-gradient)" — currently says per-node weights +
   gating, which is wrong; keep the negative half (no residual-as-coordinates, schedules
   nothing).
9. §1 Saglietti: scope the "decisive negative" — their curriculum-aware consolidation loss
   yields a large positive; the null is for standard training objectives.
10. §1 Dodge: move the ANOVA parenthetical to the best-vs-worst-orderings finding; "2,100
    fine-tuning trials."
11. Formalism/App: cite Sinkhorn & Knopp (1967) for the projection algorithm, Cuturi (2013)
    for efficiency; "nearest" only in the KL sense (add bib entry sinkhorn1967).
12. §1 Jastrzebski: "present evidence that" + note the lever is LR/batch size.
13. §4 PT4AL: add "then samples within batches by main-task uncertainty."
14. §4 CCS: add the high-pruning-rate regime qualifier (favorable — matches our low budgets).
15. §1 Bengio quote: restore exact two-sentence wording with ellipsis or drop quotation marks;
    rename cite key swayamditta2020cartography → swayamdipta2020cartography.

## Tier-2 verification sweep

✅ complete 2026-07-25 (agent, all remaining cite keys vs sources). **No WRONG citations.**
Qualifiers needed (→ text-fix queue 9–15): saglietti (their consolidation half yields a
positive result — soften "decisive negative" to standard-objective scope), dodge (ANOVA
parenthetical attached to the wrong finding; 2,100 trials not "over a hundred"), cuturi
(nearest-in-KL only; algorithm is Sinkhorn–Knopp 1967 — cite both), jastrzebski ("fixes" →
"present evidence that"; lever is LR/batch not data order), yi2022pt4al (batch selection uses
uncertainty sampler — add clause), zheng2023ccs (coverage-beats-top-k holds at HIGH pruning
rates — favorable regime qualifier, matches our low-budget setting), bengio2009 (quoted
definition is a stitched condensation inside quote marks — restore exact wording or drop
quotes; verify vs PDF pre-submission). Mechanical: cite key `swayamditta2020cartography`
misspells Swayamdipta — rename. Verified accurate: meng2017spl, li2022sgdnoise, entezari,
ainsworth, swayamdipta content, you2020whendoes, **zhu2024dyfss per-node is CORRECT** (unlike
AutoSSL), vakil (26 indices), cai2017age (linear combination Eq. 7), thakoor/assran/kipf.
Cannot-verify (no PDF, high-confidence from knowledge): ding2019dominant α-sum — spot-check
pre-submission. Dead bib entries (cited nowhere): garipov2018mode, frankle2020early,
toneva2019forgetting, paul2021grand, sener2018coreset, skenderi2024graphjepa,
weinshall2020theory — keep or cut deliberately in the restructure (several are natural cites
for the expanded related work).

## E5 outcome (2026-07-28)

- **Run**: `schedule_learn.py`, Cora, 150 outer steps x 2 restarts, ~2.8h. Artifact:
  `schedule_learn_cora.json` (+ `.npz`, untracked).
- **Result**: held-out (seeds 200-209, standard pipeline) flat 0.7894, learned 0.8169 =
  **+2.75 pts, 10/10 wins**. Primary criterion (beat compass +1.9): PASS. Secondary
  (beat best val-selected random, +1.64): PASS.
- **Diagnostics**: search val_acc at selected step 86.9% (vs 78.7% flat start) -> ~8 pts
  search-side, 2.75 survives held-out (winner's curse visible, guard held).
  corr(arrival(W_learned), beta) = +0.13 -> not the compass direction.
  mean|W-1| = 1.24, max entry 48. Heatmap (fig_learned): staged/serialized structure,
  arrival windows tile the run.
- **Claim discipline**: appendix reports one dataset, no transfer claim, winner's-curse
  gap stated in the same breath as the headline.

## E0 seed stability (2026-07-28)

- **Gap found**: S3 claimed "we test for it" but no stability run existed. Fixed by running it.
- **Run**: `seed_stability.py`, Cora, fingerprints at seeds 1 and 2 vs cached seed 0
  (200 epochs/probe, ~7.2h). Artifact: `seed_stability_cora.json`.
- **Result**: per-probe Spearman across seed pairs — A .92-.94, B .88-.90, D .92-.93 (stable);
  C .69-.78, E .63-.74 (moderate); F .26-.35 (near noise). Mean 0.75. 10-NN overlap in the
  6-D z-scored space 0.24-0.27 vs chance 0.072.
- **Verdict**: gate holds in the weak sense. Geometry is mostly graph, not accident, but seed
  variance is real and probe F's column is close to noise. Anatomy negatives don't hinge on it
  (surface-stat space has zero seed variance and fails identically); the caveat is attenuation
  of probe-column signal. S3 text now states the measured result; E0 bullet annotated.
