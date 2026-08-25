# Does the NLI faithfulness metric notice a falsified value?

**Run date:** 2026-08-25 · **Cost:** $0 (NLI is local) · **Scope:** `checkpoints/n1000_v3`
**Design:** `src/perturbation_check.py`, 200 paired cases × 16 (dataset × generator × embedder)

---

## 1. The question

A grounded answer says "$1,000" because $1,000 is in the retrieved document.
If the answer instead said "$1,500", would `nli_max` — the paper's faithfulness
metric — go down?

Paired, four conditions per answer, scored against the *same* chunks:

| condition | what it is |
|---|---|
| `orig` | untouched answer |
| `number` | one **grounded** value replaced with one **not** in the context |
| `entity` | one grounded entity replaced — comparison substitution |
| `random` | `orig` scored against another query's chunks — floor of the scale |

Eligibility is two-sided and load-bearing: the original value must appear in
the retrieved context (or it was never grounded), and the replacement must not
(or the "wrong" value is accidentally supported).

## 2. Result: it depends almost entirely on the generator

| generator | mean `orig` | after falsifying | Δ | caught at gate 0.5 |
|---|---|---|---|---|
| **Claude Haiku 4.5** | 0.906 | 0.873 | **+0.033 … +0.094** | **2 – 15 %** |
| **GPT-4o-mini** | 0.896 | 0.416 | **+0.357 … +0.480** | **55 – 76 %** |

Same NLI model, same chunks, same perturbation code. The only thing that
changed is which generator wrote the answer.

On Claude's answers the falsified version still scores **0.86–0.89**, far above
the random-context floor of **0.50**. A wrong figure remains "faithful" by any
usable threshold. On GPT-4o-mini's answers the falsified version lands at
**0.42**, essentially *at* that generator's floor of 0.43 — correctly rejected.

**For a document-QA product using a verbose generator, an NLI grounding gate
catches roughly one falsified figure in twenty.**

## 3. Numbers are harder than entities, by about 2×

Claude, per condition: number Δ +0.033/+0.038/+0.051/+0.030 against entity Δ
+0.081/+0.066/+0.023/+0.064. On HotpotQA the split is wider — number +0.059 to
+0.094, entity +0.127 to +0.179.

So there is a numeric-specific weakness, but it sits on top of a much larger
effect that applies to every substitution. On GPT-4o-mini's terser answers the
two are comparable (+0.425 number, +0.415 entity) — numbers are caught fine
when the answer is short.

By perturbation kind (Claude/NQ): `magnitude` +0.044, `year` +0.024,
`decimal` +0.029. Detection never exceeds 4.5 % in any kind.

## 4. Length dilutes the signal — confirmed, and it is monotonic

| answer length | Claude Δ | GPT-4o-mini Δ |
|---|---|---|
| 6–12 words | — | +0.642 |
| 12–25 words | +0.084 | +0.411 |
| 25–50 words | +0.077 | +0.292 |
| 50+ words | +0.038 | +0.131 |

Median answer length: Claude **49** words, GPT-4o-mini **18**. A falsified value
is a smaller and smaller edit to the hypothesis as the surrounding correct text
grows, and the entailment judgement is dominated by the rest.

## 5. What it is *not*: verbatim quotation — hypothesis refuted

The obvious explanation was that Claude quotes the context back, so the answer
is largely a copy of the premise and NLI scores the copy.

Measured (fraction of answer word-5-grams appearing verbatim in the context):

| | mean overlap | corr(overlap, Δ) |
|---|---|---|
| Claude | 0.234 | **+0.041** |
| GPT-4o-mini | 0.234 | **+0.161** |

Identical between generators, and the correlation runs the *wrong* way — higher
overlap goes with a *larger* drop, not a smaller one. The hypothesis is dead.

**The cross-generator residual is unexplained.** At matched length (12–25 words)
and matched overlap (~0.21) the gap persists: Claude +0.084, GPT +0.411, a 5×
difference. Length accounts for the trend *within* each generator, not for the
gap *between* them.

Leading remaining hypothesis, untested: Claude writes **multi-claim** answers
(the Röntgen answer carries the prize, the year, the country and the 150,782 SEK
sum), so falsifying one value leaves the other claims intact and the
whole-answer entailment barely moves. That is testable by claim-level scoring —
which is the fix regardless.

## 6. Consequences

**a. The ±0.05 TOST margin is not defensible as "negligible."** On Claude's
answers, outright falsifying a fact moves `nli_max` by 0.033–0.094. The
equivalence margin is therefore the same size as a deliberate factual error.
"Equivalent within ±0.05" cannot be read as "equally faithful" when 0.05 is
roughly one falsified fact. This is the concrete version of the review's
concern that an uncalibrated evaluator makes small differences uninterpretable.

**b. Cross-generator faithfulness comparison is not sound as measured.** H3 asks
whether the faithfulness ranking survives a change of generator, but the
metric's sensitivity to real errors differs ~10× between the two generators.
The H3 null was already reported as failed; this says the comparison was not
measuring a stable quantity in the first place.

**c. The within-generator null is less affected** — verbosity is roughly
constant across embedders for a fixed generator and prompt — but its
*magnitude* now reads differently. Differences of 0.009–0.027 sit well below the
0.03–0.09 that a falsified fact produces.

**d. Independent convergence with the review.** The ChatGPT review Berend
commissioned independently flagged max-over-chunks aggregation as major red flag
#2, on multi-claim grounds. This is empirical support for that criticism, and
for claim-level scoring as the fix.

## 7. Next

1. **Claim-level faithfulness** — `min` over claims of `max` over chunks. Free,
   tests §5's remaining hypothesis, and answers review red flag #2.
2. **Literal value grounding** — every number/date/entity in the answer must
   appear in the context. Deterministic, near-instant, and catches exactly what
   NLI misses here.
3. **AlignScore** — a second evaluator; already wired, never run.
4. Re-derive the TOST margin from §6a rather than asserting ±0.05.
