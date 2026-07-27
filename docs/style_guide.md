# Manuscript style guide

The register of the approved abstract (pdf rev 18), condensed into rules. Every section of
the manuscript follows this. When editing, fix register only; content and citations are
governed by `claim_audit.md`.

## Rules

1. **Plain words.** The everyday word over the writerly one. "Mixed together," not
   "conflated." "Does not carry across datasets," not "fails to exhibit cross-dataset
   generalization."
2. **No em dashes.** If the dash introduces an explanation, use a colon or start a new
   sentence. If it wraps an aside, promote the aside to its own sentence or cut it;
   parentheses only for citations and short genuine asides. If it is a dramatic pause,
   delete the pause.
3. **Rhythm: short for pivots and claims, long for evidence.** Pivot sentences land short:
   "We argue these results are consistent." "Here the outcome is different." Evidence and
   mechanism sentences may run long and should often end on the numbers.
4. **Active first person.** "We find," "we could not." Never "it can be observed that."
5. **No intensity decoration.** Delete: cleanly, decisively, strikingly, remarkably,
   spectacularly, airtight, enormous (when rhetorical), "most dangerously," "sits
   uncomfortably." If a result is remarkable, the sentence carrying it needs no decoration.
6. **Concrete before abstract.** Show the phenomenon first, then name the concept.
7. **Hedges stated plainly.** "We could not verify X." "This is our extrapolation, not their
   result."
8. **Scope inside the sentence.** "on three citation graphs," not a trailing pile of
   defensive clauses.
9. **Emphasis is rationed.** `\emph` for definitions and true contrast only. Bold signposts
   sparingly (the abstract's "The path:" / "The marginal:" pattern).
10. **No rhetorical questions in prose.**

## Examples (before → after, from our own draft)

| Before | After |
|---|---|
| Two established results sit uncomfortably together. | Neural network training depends on the order in which examples are presented. |
| curriculum learning --- the field built to exploit ordering --- does not... | curriculum learning, which orders examples from easy to hard, does not... |
| falsifies the null cleanly: at exactly equal per-node budget... | rules out the null: with per-node budgets held exactly equal, timing alone changes the model. |
| Both probe-based arms collapse entirely. | Both probe-based arms fail. All six ridge coefficients are numerically zero. |
| Most dangerously, CCS already shows that covering a score range beats taking its top-$k$ | CCS already shows that covering a score range beats taking its top-$k$ at high pruning rates. This is the closest existing result to ours. |
| the compass works --- but it costs thousands of runs | The compass works, but it costs thousands of runs. |
| $R$ requires no labels --- it is a lens on the data | $R$ requires no labels: it is a lens on the data. |

## Exceptions

- Appendix G (research journey) keeps its lab-notebook voice on purpose. It is a journal.
- Math display conventions are untouched by this guide.
