# Reply to Dr. Berend — 2026-08-13

Draft. Responds to his letter of 2026-08-11 (necessary/sufficient framing,
controlled conditions, correctness conditioning, Shapley, sampling).

**Do not send before reading "Notes for me" at the bottom — there is one issue
in there that changes what the paper can claim, and it is better raised by me
than found by a reviewer.**

---

**Subject:** Re: controlled conditions — implemented, plus one positioning problem

---

Dear Gábor,

Thank you — this reframing is more useful than anything else I have had on the
project. Restating the target claim as *good retrieval quality is necessary and
sufficient for a high-quality response* makes it falsifiable in a way "there is
a gap between retrieval and faithfulness" never was, and it tells me what to
measure. I have implemented all of it. Nothing below has been run at scale yet;
I wanted the design settled with you first.

## The two branches, made measurable

Everything now reduces to one per-query table:

|                     | answer correct        | answer incorrect          |
|---------------------|-----------------------|---------------------------|
| **retrieval hit**   | as the premise expects | **not sufficient** (A)    |
| **retrieval miss**  | **not necessary** (B) | as the premise expects     |

Both of your branches are cells in it, and each cell also carries its mean
faithfulness. The *hit × incorrect* cell is the one I expect to matter: if
faithfulness is high there, the paper has direct evidence that an answer can be
well grounded in correctly retrieved text and still be wrong. That is a
narrower claim than the one I started with, and I think a much more defensible
one.

## Controlled conditions

**No retrieval (C1).** Implemented, embedder-independent, one pass per dataset.

One complication worth flagging: C1 cannot reuse the RAG prompt. That template
says "answer using ONLY the provided context" and supplies a refusal string, so
with an empty context it measures willingness to refuse rather than parametric
knowledge — the floor would come out near zero for entirely the wrong reason.
C1 therefore uses a closed-book prompt, which means C1 and the RAG condition
differ in two ways at once. Rather than only noting this, I have added two
mitigations: a 100-query probe that runs the RAG template *with* an empty
context so the size of the prompt effect is measured, and a `noise_only`
condition (below) that is a floor measured through the real RAG prompt with a
real context, so it has no prompt confound at all.

I also record abstentions separately from wrong answers. A floor made of "I do
not know" means something different from a floor made of confident errors, and
for the filter you suggested the distinction matters: a query the model
abstains on is a gap RAG can fill, whereas one it answers confidently and
wrongly requires RAG to overturn a belief.

**Oracle (C2).** Implemented, and defining it forced a choice I would like to
confirm with you. There are two notions of gold evidence in my pipeline, and
they are not the same object. The qrels mark a *chunk* relevant if its source
document is gold; separately I store the annotated evidence text, which on
HotpotQA is just the supporting sentences without the surrounding paragraph. An
oracle built from the second is strictly easier than perfect retrieval — no
retriever could return sentences stripped of their paragraphs — so it would be
a ceiling on something the paper never measures. I default to the qrels
definition so C2 is the ceiling of the task NDCG@5 actually scores, and keep
the other available; the gap between them is itself interpretable as the cost
of retrieving paragraphs rather than sentences.

I also found that for a fraction of QASPER questions the loader had been
substituting the paper's first three paragraphs when the annotated evidence was
a figure or table. Acceptable for qrels, not acceptable for a ceiling claim, so
those queries are now flagged and excluded from C2 and the exclusion rate is
reported.

**Retrieval beating the oracle.** Recorded and surfaced rather than clipped —
the report prints each embedder as a percentage of the C1→C2 span and calls out
any cell above 100%.

## Subset and ordering

Your third suggestion turned out to be the cheapest experiment with the
sharpest logic, so I built it out fully. Holding relevance perfect and varying
only the arrangement:

- **order** — the same gold chunks reversed. Identical information, so any
  difference is position sensitivity alone.
- **subset** — the single best chunk versus all of them, which asks whether
  more relevant context helps or distracts.
- **position at constant length** — one gold chunk plus four hard negatives,
  with the gold placed first, in the middle, or last. Relevance *and* context
  length are held constant and only position moves.
- **noise only** — five hard negatives, zero gold.

Distractors are drawn from the reference embedder's own top-20 misses, so they
are what a real retriever returns and mistakes for relevant rather than random
text. If the ordering conditions move accuracy at all, then relevance — which
is the entirety of what NDCG@5 measures — does not determine response quality,
and that is the paper's thesis argued from the oracle side rather than the
embedder side.

## Shapley, and the paper you linked

These turned out to be the same idea at two levels of generality, which I had
not seen until I read the paper.

The DOI you sent is Salemi and Zamani, *Evaluating Retrieval Quality in
Retrieval-Augmented Generation* (SIGIR 2024, arXiv:2404.13781). Their eRAG runs
the LLM on each retrieved document individually and uses the downstream score
as that document's relevance label — which is precisely the singleton term of a
Shapley value. Shapley generalizes it by also pricing what a document
contributes in the presence of the others.

I implemented the exact Shapley value over the top-5 retrieved documents. That
needs all 2⁵ = 32 subsets per query, but eRAG and leave-one-out are both
already inside that enumeration, so one run yields all three measures — about
$3.40 for 100 queries. Two value functions come off the same generations:
whether the answer is correct, and how faithful it is to that subset's own
context. The analysis then correlates each document's utility against its qrels
relevance label. A weak correlation there is "good retrieval is not sufficient"
stated per document rather than per system, and it needs no metric of my own
invention. I am particularly interested in documents with *negative* utility
that the qrels call relevant.

## Sampling

`--emit-filter` implements your suggestion directly: retain only the queries
the model answers incorrectly with no retrieval, on the grounds that an
instance already answered from parametric knowledge cannot inform a study of
retrieval quality. Every table can be recomputed on the filtered subset. The
retained fraction is worth reporting in its own right, since a low one is a
statement about benchmark contamination rather than about my sampling.

I should also state how the sample is built, because it is not simply "the
first 1000". On Natural Questions I draw until I have 1000 queries whose
annotated short answer actually appears in the retained context window, and
discard the rest. Two things forced this. The window used to be the first 500
non-HTML tokens, so the answer often fell outside it — the query was then
unanswerable from the corpus, yet the qrels still marked its document
relevant, which manufactured exactly the "retrieval succeeded, answer wrong"
rows my central claim rests on. Centring the window on the annotated span
fixes almost all of it; the residue is discarded. In the current 1000-query
sample I scanned 1857 candidates, of which 845 had no short answer at all
(the usual NQ rate) and 12 had their answer outside the window. I will report
these counts in the paper.

On scale: the whole programme above, including the conditions, comes to about
$55, which is what I have. I am keeping n=1000 rather than raising it, and I
want to be straightforward about the consequence. The conditional analysis
splits each model's queries into four cells, so per-cell n matters more than the
total; I handle that by pooling the three datasets for the 2×2 (3,000 queries
per model) and reporting the per-dataset breakdown as secondary. The filter you
suggested shrinks the set further, so I plan to report the necessity and
sufficiency shares on the filtered set, where the filter is conceptually
required, and the cell-level faithfulness means on the pooled unfiltered set,
where the sample size is needed — clearly labelled as to which is which rather
than silently mixed.

One more thing I would like to check with you, because I think it is the largest
hole in the current draft. The paper's premise is that these models have
near-identical retrieval quality, but nothing in it establishes that: my pilot
showed NDCG@5 between 0.908 and 0.956, which is a five-point spread. A
non-significant difference is not evidence of equivalence. I intend to add a
paired equivalence test (TOST) on per-query NDCG differences with a margin of
±0.02, which is reachable at n=1000 precisely because every embedder is
evaluated on identical queries. If no pair of models comes out equivalent, then
the "matched retrieval quality" framing cannot stand as written and the claim
becomes correlational — still worth reporting, but a different sentence. Does
±0.02 seem to you like the right margin?

## One thing I want your view on

Two of your remarks together undercut my own primary metric, and I would rather
say so now than defend it later.

nRFG is `(NDCG@5 − faithfulness) / NDCG@5`. It subtracts a faithfulness score
from a ranking metric — two different scales — and the result only carries
meaning under exactly the reading you questioned, where faithfulness stands in
for response quality independently of whether the answer is right. Under the
2×2 above, the quantity that actually matters is retrieval quality against
answer *correctness*, with faithfulness as a diagnostic conditioned on it.

So I see three options: keep nRFG as the headline and add the conditional
analysis around it; demote nRFG to a diagnostic and make the necessary/
sufficient grid the paper's spine; or drop nRFG and reframe the contribution
entirely around the controlled conditions. My own inclination is the second —
it keeps the metric work but stops it carrying weight it cannot bear. I would
value your view before I rewrite §3.

## Status

Implementation is complete for all of the above and the offline test suite
passes. The only measurements I have are still the 50-query pilot I sent on
26 July, which I am not treating as evidence. I have API budget from
10 August and plan to run the reduced version — three models on Natural
Questions at n=1000, plus C1, C2 and the ablations — before committing to the
full grid, so that we find out early whether the effect is there at all.

Best regards,
Shakhriyorbek

---

## Notes for me (not for the email)

### The thing to deal with before submission

**Salemi & Zamani already publish a version of this paper's premise.** Their
abstract states that query–document relevance labels show only a *small*
correlation with downstream RAG performance. That is the finding my paper is
built around, at SIGIR, in 2024. Berend linking it may be a gentle way of
pointing that out.

This does not kill the paper, but it does move the contribution. What is left
that is genuinely mine:

1. The comparison is **across embedding architectures at matched retrieval
   quality** — eRAG compares retrievers by a new metric, it does not ask
   whether models with *identical* NDCG diverge downstream. That is still open.
2. **Faithfulness** specifically, rather than answer quality generally, and
   conditioned on correctness.
3. The controlled floor/ceiling/arrangement conditions on three datasets.

What is no longer defensible as novel: "retrieval quality does not predict
downstream quality" as a headline claim. It must be positioned as *confirming
and extending* Salemi & Zamani, and their paper needs a real paragraph in
related work — not a citation dropped in a list. Read the full PDF before
writing it; I have only read the abstract.

Ask Berend directly whether he intended the link as a positioning warning. It
is better to have that conversation now.

### Also outstanding

- Reference **[8]** in the draft still cites a non-existent
  "jina-embeddings-v5-text". Fixed in `paper/RAG_Faithfulness_IEEE_v4.docx`,
  not yet in whatever he last read.
- Add Salemi & Zamani as a new reference.
- §4.5.2 "cross-attention" wording — Llama-3 is decoder-only. Fixed in v4.
- Every number in §6 is still simulated.
- IEEE → ACL reformatting before any ARR submission.
- The 2026-07-26 pilot table must never travel without its caveat paragraph.

### Do not reverse these while rewriting

- Faithfulness is measured against the **generated** answer, never the gold
  one. ESA is the deliberate exception and §4.5.1 explains why.
- §4.5 is "Geometric Analysis", never "mechanistic".
- No "novel", "first to", "comprehensive", "causally", "state-of-the-art".
