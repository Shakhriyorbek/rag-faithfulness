# Rung 2 results — NQ + HotpotQA, n=1000, 4 embedders, 2 generators

**Run date:** 2026-08-14 · **Cost:** $13.46 ($8.50 NQ + $4.96 HotpotQA) · **API errors:** 0
**Scope:** `checkpoints/n1000_v3` · 16,000 generations

---

## 1. The headline: the title claim is not supported

*"Beyond Retrieval Quality: How Embedding Architecture Affects Faithfulness"* —
on this evidence, embedding architecture does **not** detectably affect
faithfulness, on either dataset, with either generator.

Faithfulness (NLI-max) per embedder, **paired on the queries every model
answered**, abstentions excluded:

| dataset | generator | common n | spread | p (paired bootstrap) | significant |
|---|---|---|---|---|---|
| NQ | Claude Haiku 4.5 | 792 | 0.009 | 0.223 | no |
| NQ | GPT-4o-mini | 628 | 0.013 | 0.099 | no |
| HotpotQA | Claude Haiku 4.5 | 538 | 0.019 | 0.270 | no |
| HotpotQA | GPT-4o-mini | 510 | 0.027 | 0.065 | no |

Four cells, four nulls. The largest spread is 2.7 NDCG-equivalent points of
faithfulness against retrieval-quality spreads of 4.4 (NQ) and **11.9**
(HotpotQA) points.

## 2. ⚠️ The abstention trap — this nearly became a false headline

The `anchor_table` output, which pools all rows, showed HotpotQA faithfulness
tracking retrieval quality **rank for rank**:

| | all-mpnet | text-emb-3-small | BGE-M3 | E5-instruct |
|---|---|---|---|---|
| NDCG@5 | 0.705 | 0.767 | 0.809 | **0.824** |
| faithfulness, **pooled** | 0.548 | 0.599 | 0.628 | **0.638** |
| faithfulness, **answered-only** | 0.737 | 0.727 | 0.731 | **0.746** |

Pooled spread **0.090**, perfectly ordered — the title claim, apparently
confirmed on the multi-hop dataset exactly as H2 predicts.

It is an artifact. Abstentions ("I cannot answer based on the provided
context") are correctly *not* entailed by the context and score ~0.25–0.32
NLI. Abstention rate is itself driven by retrieval quality:

| model | NDCG@5 | abstention | accuracy |
|---|---|---|---|
| all-mpnet-base-v2 | 0.705 | **0.376** | 0.599 |
| text-embedding-3-small | 0.767 | 0.279 | 0.655 |
| BGE-M3 | 0.809 | 0.224 | 0.707 |
| E5-large-instruct | 0.824 | **0.207** | 0.734 |

So the worst retriever abstains most, and every abstention drags its pooled
mean down. Restricted to answered rows the spread collapses from 0.090 to
0.019, the ordering scrambles, and significance vanishes.

**Never report pooled faithfulness.** `results.faithfulness_by_model()`
excludes abstentions and pairs on a common subset by default.

## 3. What the embedder *does* control

Not faithfulness — accuracy and abstention. On HotpotQA, moving from
all-mpnet-base-v2 to E5-large-instruct is worth **+13.5 points of accuracy**
(0.599 → 0.734) and **−17 points of abstention** (0.376 → 0.207).

The defensible sentence: *better retrieval makes the model answer more often
and get it right more often; it does not make what the model asserts more
grounded in the context.*

## 4. Necessary / sufficient (Berend's reframing) — this holds

Claude, 4,000 query-rows per dataset:

| | NQ | HotpotQA |
|---|---|---|
| hit × correct | 71.8% | 67.2% |
| **hit × incorrect — NOT SUFFICIENT** | **23.3%** | **30.2%** |
| miss × correct — NOT NECESSARY | 0.5% | 0.2% |
| miss × incorrect | 4.4% | 2.4% |

Branch A (not sufficient) is strong and **grows on multi-hop**. Branch B (not
necessary) is negligible on both — 20 and 9 queries. Retrieval is very nearly
necessary here, and clearly not sufficient.

## 5. Matched retrieval quality — dead

Paired TOST, margin ±0.02 NDCG@5: **0 of 6 pairs matched on either dataset.**
On HotpotQA five of six differ at p<0.0001, with an 11.9-point spread. This is
no longer a margin-choice question.

## 6. Hypotheses

| | verdict |
|---|---|
| **H1** instruction-tuned < contrastive (RFG) | **failed** — E5-instruct worse than contrastive on NQ (−0.030) |
| **H2** RFG largest on HotpotQA | *inapplicable as stated* — RFG is negative throughout; but hit×incorrect does rise 23.3→30.2% |
| **H3** ranking consistent across generators | **failed** — Spearman of faithfulness ranking = **0.000**, p=1.0 |
| **H4** ESA higher for instruction-tuned | not run |
| **H5** re-ranking cuts RFG ≥15% | not run |

**H3 caveat that nearly inverted the result:** tested on nRFG the pipeline
reported rho = **1.00, supported**. nRFG = 1 − F/RQ, and RQ is
generator-invariant — both generators see identical retrieval — so with F
nearly flat the nRFG ranking collapses onto the NDCG ranking. It reproduced
the NDCG order exactly for both generators. Fixed: H3 now tests faithfulness;
the nRFG version is a flagged diagnostic.

## 7. RFG runs backwards

RFG = NDCG@5 − faithfulness is **negative for every model on NQ** (−0.028 to
−0.085): faithfulness (0.85) exceeds retrieval quality (0.79). The metric
assumes generation loses fidelity relative to retrieval; empirically it does
not. Independent support for demoting nRFG to a diagnostic.

## 8. Open measurement problem

**EM = 0.000 on all 4,000 Claude NQ rows.** Claude never emits a bare answer
span, so EM cannot serve as the lower bound the design intended, and accuracy
rests entirely on containment (an upper bound — a verbose answer can mention
the gold string incidentally). The honest interval is [0.000, 0.738], which is
useless. HotpotQA adds 59 yes/no answers where containment over-matches badly.
**An LLM judge is needed before the 2×2 accuracy numbers go in the paper.**

## 9. Recommended paper position

The necessary/sufficient framing is the spine; embedding-invariance of
faithfulness is a **negative result worth reporting**:

> Across four embedders spanning 11.9 NDCG@5 points and two generators,
> retrieval quality strongly predicts whether the model answers correctly and
> whether it abstains, but does not predict how faithful its assertions are.
> Meanwhile 23–30% of queries whose answer *was* retrieved are answered
> incorrectly.

That extends Salemi & Zamani (SIGIR 2024) rather than restating it: they show
relevance labels correlate weakly with downstream quality; this separates
*which* downstream property moves (correctness, abstention) from which does
not (faithfulness).

## 10. Next

1. **LLM-judge correctness** — blocks §8; cheap.
2. **C1/C2 floor and ceiling** — `pct_of_oracle` is NaN; the 2×2 has no anchors.
3. **QASPER** — third dataset, ~$5.
4. ESA, re-ranking, Llama-3 — H4/H5 untested.
