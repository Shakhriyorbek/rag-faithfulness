# The embedder comparison depends on which faithfulness evaluator you use

**Run date:** 2026-08-26 · **Cost:** $0 · **Scope:** `checkpoints/n1000_v3`
16,000 generations re-scored with AlignScore; comparison answered-only, paired
on a common subset, paired bootstrap n=10,000.

---

## 1. The headline changes

The Rung 2 result was **four nulls** — no detectable faithfulness difference
between embedders in any dataset × generator cell. Every one of those numbers
used `nli_max`, which the 2026-08-25 perturbation experiment then measured as
the least discriminating evaluator available (4 % detection of a falsified
value on Claude/NQ, against AlignScore's 29 %).

Re-scored with AlignScore, **two of the four nulls break**:

| dataset / generator | `nli_max` spread | p | **AlignScore** spread | p |
|---|---|---|---|---|
| NQ / Claude | 0.0090 | 0.223 null | **0.0179** | **0.0005 differs** |
| NQ / GPT-4o-mini | 0.0129 | 0.099 null | 0.0080 | 0.118 null |
| HotpotQA / Claude | 0.0190 | 0.270 null | 0.0126 | 0.151 null |
| HotpotQA / GPT-4o-mini | 0.0267 | 0.065 null | **0.0461** | **0.0005 differs** |

Same queries, same answers, same abstention handling, same test. Only the
evaluator changed.

**This is the paper's most defensible finding.** Whether you conclude that
embedding choice affects faithfulness is decided by which faithfulness
evaluator you pick — and the field's common choice is the one that says no.

## 2. But "retrieval quality predicts faithfulness" is still not supported

Spearman between NDCG@5 and faithfulness across the four embedders. With n=4
this is direction, not proof — but the pattern is consistent across both
generators *and* both evaluators:

| dataset | generator | metric | rho | reading |
|---|---|---|---|---|
| NQ | Claude | nli | **−1.00** | inverts |
| NQ | Claude | align | −0.40 | unrelated |
| NQ | GPT-4o-mini | nli | **−0.80** | inverts |
| NQ | GPT-4o-mini | align | 0.00 | unrelated |
| HotpotQA | Claude | nli | +0.40 | unrelated |
| HotpotQA | Claude | align | **+0.80** | tracks NDCG |
| HotpotQA | GPT-4o-mini | nli | **+0.80** | tracks NDCG |
| HotpotQA | GPT-4o-mini | align | **+1.00** | tracks NDCG |

**The direction splits by dataset, not by evaluator or generator.** All four
HotpotQA cells are positive; all four NQ cells are zero or negative. On
multi-hop questions better retrieval does appear to improve grounding — the
answer must compose several pieces, so more of the needed evidence being
present matters. On single-hop factoids it does not.

That is close to what H2 predicted, though H2 was framed in terms of RFG. It
is **not** the general law the v5 title asserted.

## 3. Effect sizes stay small

Even where significant: 0.0179 and 0.0461. These are significant because
n=792 and n=510, not because they are large. Against retrieval-quality
spreads of 4.4 (NQ) and 11.9 (HotpotQA) NDCG points, faithfulness still moves
very little. The revised claim is "detectably, slightly" — not "substantially".

## 4. TOST across margins, replacing the asserted ±0.05

The review flagged the unjustified ±0.05 margin. A fixed margin also does not
transfer across evaluators, which do not share a scale. Pairs equivalent of 6
tested:

| dataset / generator | metric | ±0.01 | ±0.02 | ±0.03 | ±0.05 | ±0.10 |
|---|---|---|---|---|---|---|
| NQ / Claude | nli | 0/6 | 4/6 | 6/6 | 6/6 | 6/6 |
| | align | 2/6 | 3/6 | 6/6 | 6/6 | 6/6 |
| NQ / GPT | nli | 0/6 | 3/6 | 6/6 | 6/6 | 6/6 |
| | align | 1/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| HotpotQA / Claude | nli | 0/6 | 0/6 | 0/6 | 6/6 | 6/6 |
| | align | 0/6 | 3/6 | 6/6 | 6/6 | 6/6 |
| HotpotQA / GPT | nli | 0/6 | 0/6 | 1/6 | 4/6 | 6/6 |
| | align | 0/6 | 0/6 | 0/6 | 4/6 | 6/6 |

The old "22/24 equivalent at ±0.05" is recoverable from the ±0.05 column, but
it is clearly a statement about that margin rather than about the models. At
±0.02 the picture is 0/6 to 6/6 depending on the cell.

**The margin now has an external anchor:** falsifying one grounded value moves
`nli_max` by 0.033–0.094 on Claude's answers. A margin of 0.05 is therefore
about the size of one fabricated fact, which is not negligible. Report the
curve, and say what the bar means.

## 5. What to write

Supported:
- The embedder faithfulness verdict is **evaluator-dependent** — 2 of 4 cells
  flip between `nli_max` and AlignScore (§1).
- `nli_max` is the least sensitive of three evaluators tested, by a wide
  margin on verbose answers (perturbation report §8).
- On multi-hop, faithfulness tracks retrieval quality; on single-hop it does
  not (§2). Consistent across generators and evaluators.

Not supported:
- "Retrieval quality predicts correctness, not faithfulness" as a general
  claim — the v5 title. It is dataset-dependent and evaluator-dependent.
- Any equivalence claim stated without naming both the evaluator and the
  margin.

## 6. Next

1. **Re-run correctness with an LLM judge** — still blocks every 2×2 number.
2. **An open-weight generator (Qwen/Gemma)** — Berend asked, and under this
   framing it is load-bearing: the effect is mediated by answer verbosity, so
   a third generator with a different verbosity profile tests the mechanism.
3. **Claim-level over the full grid** — only the perturbation subset has been
   scored; `claim_scores_*` checkpoints do not exist yet.
