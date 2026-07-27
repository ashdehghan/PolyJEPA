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
3. **Rhythm: short for pivots and claims, long for evidence — and never staccato.** Pivot
   sentences land short: "We argue these results are consistent." Evidence and mechanism
   sentences may run long and should often end on the numbers. The short-sentence license is
   for pivots ONLY: an explanation must flow as connected prose with explicit logical links
   (because, so, which means, however). Chopping an explanation into fragments ("They fixed
   pacing and randomized order. But a growing subset means...") delivers no information per
   sentence and is banned. When explaining, prefer one full sentence that carries the whole
   step of reasoning over three fragments that each carry a shard of it.
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
11. **No term of art may carry the argument.** If removing a technical term leaves the
    sentence contentless, the sentence is wrong. Explain what happens in plain words first;
    name the term only if the name is needed later. The reader is a researcher, not a
    subfield insider: convey the idea and the intuition, never make them decode.
12a. **Numbers need context (abstract and introduction).** A number appears only where the
    sentence itself gives it meaning. One summarizing magnitude with a clear referent ("up to
    $4.5$ accuracy points," "up to $22$ points over random selection") beats a per-dataset
    list ("$4.0$, $1.6$, and $4.5$") that the reader cannot yet attach to anything.
    Dataset-by-dataset numbers live in the results, where the datasets have names.
12c. **No aphoristic compression.** Do not pack an idea into a clever short construction
    that the reader must unpack. Banned patterns: question words as bare predicates
    ("Everything the marginal discards is *when*"); metadiscourse flourishes ("Two notes
    ground this in practice," "Here is what X buys," "worth stating because"). Write the
    full sentence with an explicit subject, verb, and object: "The second part is the
    information the marginal throws away: the timing." A scientific manuscript is not
    poetry; short sentences are for clarity, never for compression.
12b. **Tense scheme.** Cited work: past tense for reporting verbs ("Wu et al. compared,"
    "Frankle et al. showed"); a content clause after a past reporting verb may stay present
    when it states a general truth ("showed that sampling noise biases the solution"). Our
    own work in this paper: present tense ("we run," "we find," "Table 2 shows"). Mathematics,
    definitions, and general facts: present tense.
12. **Strong-claim discipline.** For every definitive, universal, or evaluative statement:
    (a) if it is verifiably true, attach the evidence in the same breath (a citation or a
    number); (b) if it is our judgment, mark it as ours; (c) if it cannot be defended at full
    strength, scope it ("within the curriculum literature," "in our experiments," "to our
    knowledge") or delete it. State the facts and let them convict; do not pronounce
    verdicts.

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
| (Rule 11) prove self-paced learning is majorization--minimization on an implicit objective with a non-convex robust regularizer: a robust loss, not a curriculum | Self-paced learning looks like a curriculum: the model trains on its easiest examples first and adds harder ones as it improves. Meng et al. proved this procedure is exactly equivalent to ordinary training on a fixed objective in which each example's loss is capped, so that hard examples simply count less. Whatever it gains, it gains from suppressing hard examples, not from order. |
| (Rule 12) a large applied literature and a thin evidential one | The applied literature is large [survey]. Direct evidence that the ordering itself helps is harder to find, and the most systematic test of that question found none [Wu]. |

13. **Headings are descriptive noun phrases.** Section and paragraph titles name their
    content in standard scientific form ("Related work," "Directional consistency,"
    "Cross-domain evaluation"). Never editorial one-liners ("Related work, and what is
    actually ours"), never claims-as-titles ("The probe bank cannot predict the compass"),
    never staging ("The negative results, first").
14. **Write clearly, in natural connected prose. Full stop.** One complete thought per
    sentence, stated plainly, joined with explicit logic (because, so, however) when the
    reasoning continues. Short sentences are fine when they are clear and complete. Long
    sentences are fine when the thought needs them. Never shorten to be punchy, never
    lengthen by chaining clauses, never compress an idea into a construction the reader
    must decode. The final check is always a human read of the full text, not a script.

## Exceptions

- Appendix G (research journey) keeps its lab-notebook voice on purpose. It is a journal.
- Math display conventions are untouched by this guide.
