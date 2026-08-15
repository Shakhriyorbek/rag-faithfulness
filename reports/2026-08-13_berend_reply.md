# Reply to Dr. Berend — drafted 2026-08-13, rewritten 2026-08-14 with results

Responds to his letter of 2026-08-11 (necessary/sufficient framing, controlled
conditions, correctness conditioning, Shapley, sampling) — and now carries the
Rung 2 measurements, which changed what the paper can claim.

**Do not send before reading "Notes for me" at the bottom.**

---

**Subject:** Re: controlled conditions — results, and a title that no longer matches them

---

Dear Gábor,

Thank you — this reframing is more useful than anything else I have had on the
project. Restating the target claim as *good retrieval quality is necessary and
sufficient for a high-quality response* makes it falsifiable in a way "there is
a gap between retrieval and faithfulness" never was, and it told me what to
measure.

I have now implemented all of it and run it: two datasets, 1000 queries each,
four embedding models, two generators (Claude Haiku 4.5 and GPT-4o-mini),
16,000 generations, $13.46 of the $55 budget, no API errors. I am writing
before rewriting anything, because the results contradict my own title and I
would rather agree the response with you than present you with a rewrite.

## The main result: faithfulness does not vary with the embedder

Your branch A holds, and the mechanism is not the one I proposed.

I tested equivalence in both directions, using the paired TOST you prompted me
towards. Every embedder is evaluated on identical queries, so the test is
paired throughout.

|                                   | pairs equivalent | margin |
|-----------------------------------|------------------|--------|
| **Retrieval quality** (NDCG@5)    | **0 of 6**, each dataset | ±0.02 |
| **Faithfulness** (NLI entailment) | **22 of 24** across both datasets × both generators | ±0.05 |

On Natural Questions all six pairs are equivalent in faithfulness at
p_tost = 0.0000 for both generators. Meanwhile the retrieval spread between the
best and worst embedder is 4.4 NDCG@5 points on NQ and **11.9 points** on
HotpotQA — nowhere near matched.

So the honest sentence is the opposite of my title:

> Embedding models that differ substantially in retrieval quality produce
> statistically equivalent faithfulness.

What the embedder *does* control is whether the model answers at all and
whether it is right. On HotpotQA, moving from all-mpnet-base-v2 to
E5-large-instruct is worth **+13.5 points of accuracy** (0.599 → 0.734) and
**−17 points of abstention** (0.376 → 0.207), while the faithfulness of what is
actually asserted moves 0.019 and is not significant.

I think that is a better paper than the one I proposed, and it is a cleaner
extension of Salemi and Zamani (below) than "retrieval quality does not predict
downstream quality" would have been: it separates *which* downstream property
retrieval quality governs from which it does not.

## Why I nearly reported the opposite

This is worth a paragraph because it is the kind of error your letter is
designed to catch.

My first pass pooled all responses and showed HotpotQA faithfulness tracking
retrieval quality rank for rank:

| | all-mpnet | text-emb-3-small | BGE-M3 | E5-instruct |
|---|---|---|---|---|
| NDCG@5 | 0.705 | 0.767 | 0.809 | **0.824** |
| faithfulness, pooled | 0.548 | 0.599 | 0.628 | **0.638** |
| faithfulness, answered only | 0.737 | 0.727 | 0.731 | **0.746** |

A spread of 0.090, perfectly ordered — my hypothesis confirmed on the multi-hop
dataset, exactly where I predicted it.

It is an artifact of abstention. "I cannot answer based on the provided
context" is correctly *not* entailed by the context and scores about 0.25–0.32
NLI. Abstention rate is itself driven by retrieval quality, so the worst
retriever abstains most and every abstention drags its pooled mean down.
Restricted to answered responses the spread falls to 0.019, the ordering
scrambles, and significance disappears. I also had to pair the comparison on a
common subset: averaging each model over its own answered queries compares
different query sets, and which queries a model abstains on depends on its own
retrieval.

All faithfulness numbers in the paper will therefore be answered-only and
paired. I mention it because the pooled version is what a reader would compute
by default, and it points the wrong way.

## Your 2×2, with numbers

Claude, 4,000 query-rows per dataset:

| | NQ | HotpotQA |
|---|---|---|
| hit × correct | 71.8% | 67.2% |
| **hit × incorrect — not sufficient (A)** | **23.3%** | **30.2%** |
| miss × correct — not necessary (B) | 0.5% | 0.2% |
| miss × incorrect | 4.4% | 2.4% |

Branch A is substantial and grows on multi-hop. Branch B is negligible on both
— 20 and 9 queries respectively. On these two datasets retrieval is very nearly
necessary and clearly not sufficient, which is a sharper statement than I
expected to be able to make, and it is asymmetric in a way worth reporting.

`hit` here means the model was shown the answer, not merely the right document:
relevance is answer-bearing at the chunk level, and on HotpotQA only 2 of 1000
queries fell back to document-level relevance.

## What failed

- **H1** (instruction-tuned show a smaller gap than contrastive) — **failed**.
  E5-large-instruct is worse than the contrastive models on NQ.
- **H3** (ranking survives a change of generator) — **failed**. Spearman of the
  faithfulness ranking between Claude and GPT-4o-mini is **0.000**.
- **Matched retrieval quality**, the design premise — **not supported at any
  defensible margin**. On HotpotQA five of six pairs differ at p < 0.0001.

H3 also came with a trap I want to flag, since it bears on the metric question
below. Tested on nRFG, H3 came out *supported* at Spearman 1.00. That is an
artifact: nRFG = 1 − F/RQ, and RQ is a property of the retriever alone, so both
generators see the identical RQ term. With faithfulness nearly flat, the nRFG
ranking collapses onto the NDCG ranking — it reproduced the NDCG order exactly
for both generators. It was measuring agreement about retrieval and reporting
it as agreement about generators.

## nRFG — you were right, and the data now says so too

I asked in my previous draft whether nRFG should stay the primary metric. The
measurements answer it.

RFG = NDCG@5 − faithfulness is **negative for every model on NQ** (−0.028 to
−0.085), because faithfulness (~0.85) exceeds retrieval quality (~0.79). The
metric assumes generation *loses* fidelity relative to retrieval quality; it
does not. A "gap" that is reliably negative, whose ranking is driven by the
retrieval term, and which subtracts an entailment score from a ranking metric,
is not something I can defend as a headline.

My proposal is the second of the three options I put to you: **demote nRFG to a
diagnostic and make the necessary/sufficient grid the spine of the paper.** The
metric work stays, reported honestly including its negative sign, but stops
carrying weight it cannot bear.

Which leads to the title. *Beyond Retrieval Quality: How Embedding Architecture
Affects Faithfulness in RAG Systems* is contradicted by my own data. I would
propose something like **"Retrieval Quality Predicts Correctness, Not
Faithfulness, in Retrieval-Augmented Generation"**, but I would rather have
your view than pick one myself.

## The conditions you asked for

All implemented; the ones not yet run are marked.

**No retrieval (C1) — implemented, not yet run.** C1 cannot reuse the RAG
prompt: that template says "answer using ONLY the provided context" and
supplies a refusal string, so with an empty context it measures willingness to
refuse rather than parametric knowledge. C1 uses a closed-book prompt, with two
mitigations for the resulting two-way difference — a 100-query probe running
the RAG template *with* an empty context to size the prompt effect, and a
`noise_only` condition which is a floor measured through the real RAG prompt.

**Oracle (C2) — implemented, not yet run.** Defining it forced a choice I would
like to confirm. The qrels mark a chunk relevant if its source document is
gold; separately I store the annotated evidence text, which on HotpotQA is the
supporting sentences without their paragraph. An oracle built from the second
is strictly easier than perfect retrieval, so it would be a ceiling on
something the paper never measures. I default to the qrels definition and keep
the other available; the gap between them is interpretable as the cost of
retrieving paragraphs rather than sentences.

**Retrieval beating the oracle** — recorded and surfaced rather than clipped;
any cell above 100% of the C1→C2 span is called out.

**Subset and ordering — implemented, not yet run.** Holding relevance perfect
and varying only arrangement: order (gold chunks reversed), subset (best chunk
versus all), position at constant length (one gold chunk among four hard
negatives, placed first/middle/last), and noise only. Distractors are the
reference embedder's own top-20 misses, so they are what a real retriever
mistakes for relevant rather than random text.

**Shapley — implemented, not yet run.** The DOI you sent is Salemi and Zamani,
*Evaluating Retrieval Quality in Retrieval-Augmented Generation* (SIGIR 2024,
arXiv:2404.13781). Their eRAG scores each retrieved document alone and uses the
downstream result as its relevance label, which is precisely the singleton term
of a Shapley value. I implemented the exact Shapley value over the top-5, which
needs all 2⁵ = 32 subsets per query — eRAG and leave-one-out both live inside
that enumeration, so one run yields all three, about $3.40 per 100 queries. I
am particularly interested in documents with *negative* utility that the qrels
call relevant.

I should ask directly: did you intend that link as a positioning warning? Their
abstract already reports that relevance labels correlate only weakly with
downstream performance, which is close to my original premise. I now think the
result above is a genuine extension rather than a restatement — they show the
correlation is weak, I can show which downstream property moves and which is
equivalent — but I would like to know if that was your point.

## Sampling

`--emit-filter` implements your suggestion: retain only queries the model
answers incorrectly with no retrieval, since an instance already answered from
parametric knowledge cannot inform a study of retrieval quality. It needs C1,
so it is pending with C1.

How the sample is built is worth stating, because it is not "the first 1000".
On Natural Questions I draw until I have 1000 queries whose annotated short
answer actually appears in the retained context window. The window used to be
the first 500 non-HTML tokens, so the answer often fell outside it — the query
was then unanswerable from the corpus, yet the qrels still marked its document
relevant, manufacturing exactly the "retrieval succeeded, answer wrong" rows my
central claim rests on. Centring the window on the annotated span fixes almost
all of it; the residue is discarded. For the current sample I scanned 1857
candidates: 845 had no short answer (the usual NQ rate) and 12 had their answer
outside the window. These counts will be in the paper.

## The one measurement I do not yet trust

Exact match is **0.000 across all 4,000 Claude responses on NQ**. The model
never emits a bare answer span — it writes "Based on the provided context,
Wilhelm Conrad Röntgen of Germany received…" — so EM cannot serve as the lower
bound the design intended, and accuracy currently rests entirely on containment,
which is an upper bound because a long answer can mention the gold string while
asserting something else. The honest interval is [0.000, 0.738], which is too
wide to build on, and HotpotQA adds 59 yes/no answers where containment
over-matches.

So the accuracy figures and the 2×2 shares above should be read as provisional.
My plan is an LLM judge over the existing generations — no regeneration, so it
is cheap — before those numbers go in the paper. The faithfulness result is
unaffected, since it is NLI against the context and does not use the correctness
label.

## Where that leaves the remaining budget

$41 of the $55 is unspent. In priority order: the LLM judge; C1 and C2, without
which the 2×2 has no floor or ceiling anchors; then a decision I would like your
view on. Two of the paper's four stated contributions — the ESA geometric
analysis and the faithfulness-aware re-ranking — currently have no data at all.
ESA is free to run and is now more interesting than it was, since it might
explain *why* faithfulness is invariant to the embedder. Re-ranking is the
harder case: its motivation was to close a faithfulness gap that, on this
evidence, does not exist between embedders, and H5 was defined as a percentage
reduction in an RFG that turns out to be negative. I am inclined to run ESA and
either drop the re-ranking contribution or restate it as an accuracy
intervention rather than a faithfulness one.

Best regards,
Shakhriyorbek

---

## Notes for me (not for the email)

### What the contribution now is

The three things I listed on 2026-08-13 as surviving Salemi & Zamani have
changed. **"Comparison across embedding architectures at matched retrieval
quality" is dead** — 0 of 6 pairs matched at ±0.02, and the HotpotQA spread is
11.9 points. Do not write that sentence anywhere.

What is actually left, and it is better:

1. **Faithfulness is equivalent across embedders while retrieval quality is
   not** — an equivalence result, not a failure to reject. 22 of 24 pairs at
   ±0.05, both generators, both datasets.
2. **Which downstream property retrieval quality governs**: correctness and
   abstention yes, faithfulness no. This is the extension of eRAG.
3. The necessary/sufficient grid with the asymmetry (not sufficient 23–30%,
   necessary almost always).
4. The controlled floor/ceiling/arrangement conditions — still unrun.

### Presentation notes

- Report **the tightest margin at which all pairs are equivalent** rather than
  defending an arbitrary ±0.05. On NQ it will be very tight and the result is
  stronger stated that way.
- Never report pooled faithfulness. `results.faithfulness_by_model()` is
  answered-only and paired on a common subset by default.
- The equivalence claim is only as good as its power; state n per cell
  (538–792 after excluding abstentions) alongside every margin.

### Still outstanding

- Reference **[8]** still cites a non-existent "jina-embeddings-v5-text" →
  jina-embeddings-v3, arXiv:2409.10173.
- Add Salemi & Zamani as a reference with a real related-work paragraph, not a
  citation in a list. **Read the full PDF — only the abstract has been read.**
- §4.5.2 "cross-attention" — Llama-3 is decoder-only, and no code implements
  that analysis. Cut it rather than fix it.
- Every number in §6 is still simulated; replace with `reports/2026-08-14_rung2_results.md`.
- QASPER not run. Llama-3 not run (needs HF_TOKEN on gpu1).
- IEEE → ACL reformatting before any ARR submission.
- Rotate the API keys once the experiments finish — they have been on a shared
  university machine.

### Do not reverse these while rewriting

- Faithfulness is measured against the **generated** answer, never the gold
  one. ESA is the deliberate exception and §4.5.1 explains why.
- §4.5 is "Geometric Analysis", never "mechanistic".
- No "novel", "first to", "comprehensive", "causally", "state-of-the-art".
