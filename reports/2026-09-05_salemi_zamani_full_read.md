# Salemi & Zamani — full PDF read (closes the open item in PAPER_TODO §2)

**Date:** 2026-09-05
**Paper:** A. Salemi and H. Zamani, "Evaluating Retrieval Quality in
Retrieval-Augmented Generation", SIGIR '24, pp. **2395–2400**.
arXiv:**2404.13781v1** (21 Apr 2024 — the only arXiv version).
DOI `10.1145/3626772.3657957` **verified via Crossref**: title, both authors,
container (Proc. 47th Int'l ACM SIGIR), pages and year all match. Berend's link
was correct; the arXiv PDF itself carries a placeholder DOI
(`10.1145/nnnnnnn.nnnnnnn`), so do not cite the PDF's DOI field.
Code: https://github.com/alirezasalemi7/eRAG

Until now only the abstract had been read (CLAUDE.md §4b). The full read
**confirms the positioning decision but narrows the threat**, and it makes one
sentence in the v6 draft (§II-A) an overstatement that a SIGIR-literate reviewer
would catch.

---

## 1. What the paper actually does

**Method (eRAG).** For a ranked list `R_k` and the RAG system's own LLM `M`,
label each retrieved document by the downstream score it produces *alone*:

```
G_q[d] = E_M( M(q, {d}), y )      for all d in R_k
```

where `y` is the gold downstream output and `E_M` is the task metric (EM,
accuracy, ROUGE). Aggregate the resulting document labels with any standard
ranking metric (P, R, MAP, MRR, NDCG, Hit Ratio) to score the list.

This confirms CLAUDE.md's reading: **eRAG is exactly the singleton term of a
Shapley value**, which is why points 5 and 6 of Berend's letter are one idea.
Eq. (1) is that singleton, verbatim.

**Claim.** eRAG correlates better with end-to-end downstream performance than
the alternative label sources, by +0.168 to +0.494 Kendall's tau, and is
2.5x faster / 7–48x more memory-efficient than end-to-end evaluation
(complexity argument: `O(lkd²)` vs `O(lk²d²)` for a vanilla transformer).

**Setup.** KILT versions of NQ, TriviaQA, HotpotQA, FEVER, WoW; KILT Wikipedia
dump, 100-word passages, title+passage as the document; retrievers BM25
(Pyserini) and Contriever (Faiss flat); reader **T5-small + Fusion-in-Decoder
(60M)**, also T5-base (220M), **fine-tuned 10 epochs**, k=50 by default;
Mistral-7B-Instruct-v0.2 as the LLM relevance annotator baseline.

---

## 2. The finding that matters to us — and the qualification the abstract omits

The abstract says relevance labels correlate weakly with downstream
performance. **Table 1 shows that is true of two of the three baselines, not
all three.** BM25 rows, Kendall's tau, best metric per block:

| Label source | NQ | TriviaQA | HotpotQA | FEVER | WoW |
|---|---|---|---|---|---|
| **Containing the answer** | **0.361** | **0.313** | **0.398** | 0.043 | 0.019 |
| KILT Provenance (human) | 0.181 | 0.151 | 0.139 | 0.021 | 0.003 |
| LLM relevance (Mistral-7B) | 0.060 | 0.189 | 0.034 | — | — |
| eRAG | 0.529 | 0.486 | 0.629 | 0.592 | 0.504 |

**Answer-containment labels are their strongest baseline on all three QA
datasets — 2x the human provenance labels.** The weak-correlation result is
specific to *human document-level relevance* and *LLM relevance judgments*.

**This is directly load-bearing for us.** After B6/B8b, our qrels are
**answer-bearing**: NQ relevance is decided by `gold_sentences` containment and
HotpotQA by supporting facts. Our retrieval metric is therefore the
*containment* row, not the *provenance* row — the one label source whose
correlation with downstream correctness they measured as moderate. So the
published result pre-empts less of our premise than §4b assumed: what is
published is that *human/LLM relevance labels* predict downstream correctness
poorly for a fine-tuned 60M FiD reader at k=50.

Second qualification, from their Figure 1: **correlation declines as k grows**,
because every retrieval metric scores documents independently while the LLM
consumes them jointly. Their headline is measured at **k=50**. We use **k=5** —
the regime where their own data puts the retrieval/downstream correlation at
its *highest*. A weak downstream signal at our k cannot be attributed to their
decline effect; it has to be argued on its own.

---

## 3. What they do *not* do — our contribution is intact

| Dimension | Salemi & Zamani | This project |
|---|---|---|
| Downstream property | **Correctness only** (EM / accuracy / F1) | **Faithfulness**, conditioned on correctness |
| Generator | T5-small/base + FiD, 60M/220M, **fine-tuned** | Claude Haiku 4.5, GPT-4o-mini, Qwen2.5-7B, **zero-shot prompted** |
| Object of study | the **retrieval metric** | the **faithfulness evaluator** |
| Retrievers compared | BM25 vs Contriever (2, different families) | 4 embedding models at **TOST-matched** retrieval quality |
| Evaluator sensitivity | not examined — one metric per dataset, fixed | the spine: same answers, 3 evaluators, different verdict |
| Controlled error injection | none | the falsification probe |

Two of these are worth stating explicitly in related work, because they are the
gaps our design fills:

1. **They never measure faithfulness.** `E_M` is always a match against gold
   `y`. Every number in the paper is a correctness number. Our v6 spine —
   that the *faithfulness* verdict is evaluator-dependent — is untouched by
   anything they publish.
2. **Their reader is a fine-tuned 60M encoder-decoder, not a prompted
   instruction-following model.** A fine-tuned FiD reader has no abstention
   behaviour, so the ~30% refusal rate that drives our Section VI cannot arise
   in their setup at all. Their conclusion about the retrieval-downstream link
   is a conclusion about that reader.

Also note: **eRAG requires the gold label `y`**, so it is an offline evaluation
method, not a signal available at retrieval time. That is the same asymmetry we
already committed to in §4.5.1 (`NLI(d, gold_a)` vs `NLI(d, q)`) and is worth
one clause — it is why eRAG cannot double as a re-ranking criterion.

---

## 4. Required change to the v6 draft — APPLIED 2026-09-05

`paper/build_v6.js` §II-A currently reads:

> "Salemi and Zamani [1] report that query-document relevance labels correlate
> only weakly with downstream retrieval-augmented generation performance, and
> propose evaluating each retrieved document by the downstream result it
> produces. Our starting point is downstream of theirs: we take as given that
> retrieval metrics do not directly predict generation quality, and ask which
> downstream property a given measurement instrument can resolve at all."

**"query-document relevance labels" is too broad** — it reports their abstract,
not their Table 1, and it describes our own qrels scheme as one of the things
they showed to be weak, which is the opposite of what they measured. Proposed
replacement:

> Salemi and Zamani [1] report that human document-level relevance labels, and
> LLM relevance judgments, correlate only weakly with downstream
> retrieval-augmented generation performance, and propose eRAG, which labels
> each retrieved document by the downstream result the system's own generator
> produces from it alone. Their strongest baseline is answer-containment
> labelling, the scheme our qrels use, which retains a moderate correlation
> (Kendall's tau 0.31-0.40 on the question-answering datasets); their reported
> correlations also decline as the number of retrieved documents grows, and are
> measured at k=50 against ours at k=5. Their downstream measure is correctness
> throughout, obtained from a fine-tuned 60M encoder-decoder reader. Our
> starting point is downstream of theirs: we take as given that retrieval
> metrics do not directly predict generation quality, and ask which downstream
> property a given measurement instrument can resolve at all -- for
> faithfulness rather than correctness, and for prompted instruction-following
> generators, which can abstain.

The reference entry `[1]` already carries the right pages (2395-2400) and arXiv
ID; **no bibliography change is needed.**

**Applied.** `paper/build_v6.js` §II-A now carries the replacement and
`paper/RAG_Faithfulness_v6_evaluators.docx` was rebuilt. Diffing the rendered
document text before and after shows **exactly one changed paragraph** and the
same 467 paragraphs total — nothing else in the paper moved. Rebuilding needed
Node, which had not survived the Windows -> Fedora migration; it is now
installed user-locally and the recipe is `DEVICE_MIGRATION.md` §6b. Note the
build script writes to the relative path `paper/...`, so it must be run **from
the repo root**.

---

## 5. Correction to CLAUDE.md §4b

§4b states: *"Its abstract states that query-document relevance labels correlate
only weakly with downstream RAG performance. **That is this paper's premise,
already published.**"* — accurate as a summary of the abstract, **too strong as
a summary of the paper**. What is published is narrower on three axes (label
source, k, and correctness-only), all three of which cut in our favour. The
positioning conclusion in §4b is otherwise unchanged and correct: comparison
across embedding architectures at matched retrieval quality, faithfulness
conditioned on correctness, and the controlled floor/ceiling conditions all
survive; "retrieval quality does not predict downstream quality" as a bare
headline still does not.

Local copy of the PDF: not committed (scratchpad only). Re-fetch with
`curl -sL https://arxiv.org/pdf/2404.13781` — note `.../pdf/2404.13781v3`
returns an HTML error page, there is no v3.
