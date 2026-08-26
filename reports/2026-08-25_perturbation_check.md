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

---

# Part 2 — claim-level scoring and AlignScore (2026-08-26)

Both open questions from §5 and §7 are now answered. Same 200 falsification
cases per condition, rebuilt deterministically, so all three evaluators saw
byte-identical answers and contexts.

## 8. Three evaluators on the same falsifications

| dataset / generator | scorer | orig | falsified | random | delta | caught @0.5 |
|---|---|---|---|---|---|---|
| **NQ / Claude** | nli_max | 0.918 | 0.880 | 0.522 | +0.038 | **4 %** |
| | claim_min | 0.686 | 0.535 | 0.249 | +0.151 | **23 %** |
| | AlignScore | 0.818 | 0.586 | 0.342 | +0.233 | **29 %** |
| **NQ / GPT-4o-mini** | nli_max | 0.893 | 0.436 | 0.507 | +0.456 | 58 % |
| | claim_min | 0.866 | 0.391 | 0.493 | +0.475 | 63 % |
| | AlignScore | 0.918 | 0.266 | 0.173 | +0.652 | **76 %** |
| **HotpotQA / Claude** | nli_max | 0.739 | 0.660 | 0.468 | +0.079 | **13 %** |
| | claim_min | 0.367 | 0.256 | 0.149 | +0.111 | 33 % |
| | AlignScore | 0.770 | 0.466 | 0.498 | +0.304 | **45 %** |
| **HotpotQA / GPT-4o-mini** | nli_max | 0.578 | 0.185 | 0.222 | +0.393 | 70 % |
| | claim_min | 0.571 | 0.182 | 0.225 | +0.389 | 70 % |
| | AlignScore | 0.773 | 0.115 | 0.223 | +0.658 | **90 %** |

**AlignScore is the best evaluator in all four cells**, by a wide margin on
Claude: 29 % against 4 % on NQ, 45 % against 13 % on HotpotQA. On Claude/NQ its
delta is **six times** nli_max's.

## 9. The severity is DeBERTa-specific; the phenomenon is not

Two separable claims, and only the first narrows:

- **`nli_max` is a poor faithfulness evaluator.** It is the worst of the three
  in every cell. The paper's headline metric would miss 96 % of falsified
  values on Claude/NQ where AlignScore misses 71 %.
- **Verbose multi-claim answers defeat every evaluator tested.** Under
  AlignScore the Claude/GPT gap is still 29 % vs 76 % (NQ) and 45 % vs 90 %
  (HotpotQA). Reduced, not removed.

So the reportable finding is the second one, stated across evaluators, with the
first as a concrete measurement of how much the choice matters.

## 10. Claim-level: mechanism confirmed, partial fix

`nli_max` delta by claim count — falls monotonically for **both** generators,
which is the dilution mechanism §5 could not identify:

| claims/answer | Claude nli | Claude claim | GPT nli | GPT claim |
|---|---|---|---|---|
| 1 | +0.086 | +0.098 | +0.492 | +0.492 |
| 2 | +0.056 | +0.131 | +0.336 | +0.452 |
| 3 | +0.018 | +0.188 | +0.137 | +0.277 |
| 5+ | +0.007 | +0.149 | — | — |

Claude writes **2.99** claims/answer, GPT **1.18** — that is the cross-generator
residual §5 left open. Single-claim GPT answers score **identically** under both
scorers (0.4922 vs 0.4922), confirming claim-level reduces to whole-answer when
there is one claim.

**But it is not a repair.** Scale-free, the median normalised drop on Claude
goes only 0.01 → 0.03, and `min`-over-claims moves in the correct direction just
**55 %** of the time: when the falsified claim is not already the weakest, the
minimum does not move. It converts a small-frequent signal into a
large-infrequent one.

**A second dilution mechanism remains.** At matched claim count (1 claim) Claude
still shows +0.098 against GPT's +0.492. Claude's individual claims average
**20.5 words** against GPT's **13.8** — sentence splitting removes dilution
*across* claims, not *within* one.

## 11. Consequence for the paper's main result

Every faithfulness number in the Rung 2 results — the four nulls, the spreads of
0.009–0.027, the 22/24 TOST equivalences — was computed with `nli_max`, now
measured to be the least discriminating of the three evaluators tested. The
main embedder comparison should be re-run with AlignScore before any
equivalence claim is reported. That is the review's "essential, not optional",
with a number attached.

## 12. Next

1. **Re-run the embedder faithfulness comparison under AlignScore** — free,
   and it gates the paper's headline.
2. **Literal value grounding** — every number/date/entity in the answer must
   appear in the context. Deterministic. 29 % detection is still not a usable
   product gate.
3. Sub-claim decomposition, to address the within-claim dilution in §10.
