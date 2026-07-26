# Update to Dr. Berend — 2026-07-26

Draft email. Attachment: `paper/RAG_Faithfulness_IEEE_v4.docx`

---

**Subject:** Progress update — pipeline running on gpu1, three corrections to the draft

---

Dear Gábor,

Thank you again for setting up the server access — everything works as you
described, and I have been running on gpu1 this week.

A short update, and two questions at the end.

## Implementation status

The experiment pipeline is now written as proper modules rather than a
notebook, and it runs end to end on gpu1. Phases implemented: corpus
construction and indexing, retrieval evaluation, generation, NLI
faithfulness, the entailment–similarity analysis, and the re-ranking
ablation. Every phase checkpoints and resumes.

## Errors I found while porting the notebook

Porting the code surfaced several problems that would have produced
plausible-looking but incorrect results. I think two are worth your time:

**1. The NLI entailment class was being read from the wrong index.** The
notebook took `probs[2]` as the entailment probability. For
`cross-encoder/nli-deberta-v3-large` the label order is
`{0: contradiction, 1: entailment, 2: neutral}`, so index 2 is *neutral*.
Every faithfulness score, the ESA correlation, and the re-ranking signal
would have been computed from the neutral probability. The index is now
resolved from the model configuration at load time; verified on the server:

```
labels: {0: contradiction, 1: entailment, 2: neutral} -> index 1
entailment case P(ent) = 0.9970 / contradiction case = 0.0000
```

**2. Retrieval quality would have evaluated to approximately zero.** The
qrels matched gold and retrieved passages by string equality between two
independent chunking passes, which essentially never coincide. NDCG@5 would
have been ~0 for every model, which is the retrieval half of RFG. Relevance
is now decided by chunk identity and document provenance.

Three smaller issues: the Natural Questions index contained only each
query's own gold context and no distractors; the NLI premise was truncated
at 512 tokens so retrieved chunks 3–5 never influenced the score (each chunk
is now scored separately); and the paired bootstrap did not re-centre the
resampling distribution, which made *p* ≈ 0.5 regardless of effect size — no
hypothesis could have reached significance.

## Change to the closed-source generator

I have replaced GPT-4o-mini with **Claude Haiku 4.5**, subject to your
agreement. My reasoning is that Haiku 4.5 sits in the same capability and
cost tier, so the design intent — a small, widely deployed closed-source
model — is preserved. A frontier model would likely be more faithful across
the board and could compress the very differences the study aims to measure.
Measured cost for the full grid is **$38.95** (metered on a pilot run), against
roughly $6 for GPT-4o-mini. Llama-3-8B-Instruct is unchanged as the
open-source generator, and both receive a byte-identical prompt.

If you would rather I keep GPT-4o-mini for comparability with the
literature, that is straightforward to revert.

## Corrections in the attached draft (v4)

Independent of any results:

- **Reference [8]** cited "jina-embeddings-v5-text", and Table 1 listed
  "jina-embeddings-v5-small". Neither exists. The model is
  **jina-embeddings-v3** (Sturua et al., arXiv:2409.10173), which is what the
  code uses.
- **§4.4 and H3** now name Claude Haiku 4.5.
- **§4.5.2** described analysing *cross-attention* in Llama-3. Llama-3 is
  decoder-only and has no encoder–decoder cross-attention; the text now
  refers to the attention mass that generated tokens place on context
  positions.

§6 is untouched — I have not put any measured numbers into the draft yet.

## Pilot run — pipeline validation, not results

A 50-query pilot on Natural Questions with three models completed cleanly
(150 generations, no errors, $0.28):

| Model | Paradigm | NDCG@5 | Faithfulness | nRFG |
|---|---|---|---|---|
| BGE-M3 | multilingual | 0.908 | 0.781 | 0.140 |
| all-mpnet-base-v2 | contrastive | 0.948 | 0.732 | 0.228 |
| E5-large-instruct | instruction-tuned | 0.956 | 0.691 | 0.278 |

**I want to be clear that this is not evidence.** n = 50, one dataset, one
generator, no confidence intervals, no significance test, and a corpus small
enough that retrieval is easy. I report it only to show the pipeline produces
coherent output end to end.

With that caveat: the retrieval and faithfulness orderings are inverted, and
**H1 currently points the wrong way** — the instruction-tuned model has the
*highest* nRFG, where H1 predicts the lowest. If that holds at n = 1000 I
intend to report H1 as unsupported rather than reframe it.

## Timeline and questions

I plan to run the full grid on **10 August**, once I have the API budget in
place. Before then I will run the 1,000-query, three-model version on
Natural Questions (about $6) to see whether the pattern above survives.

Two questions:

1. **Venue** — the EMNLP ARR deadline in May has passed. Would you prefer the
   July ARR cycle, or COLING 2026?
2. **Generator** — is the substitution above acceptable to you, or would you
   rather I keep GPT-4o-mini?

Best regards,
Shakhriyorbek

---

## Notes for me (not for the email)

- Attach `paper/RAG_Faithfulness_IEEE_v4.docx`; keep v3 as the record of what
  he last reviewed.
- Do **not** send the n=50 table without the caveat paragraph attached to it.
- If he asks why this took until late July: the honest answer is that the
  inherited pipeline had five defects that would each have produced confident
  wrong numbers, and finding them was the work. Running on 2 July would have
  produced a complete, plausible, invalid results table.
- Still open and not mentioned: AlignScore (phase E) not yet run; ESA and the
  re-ranking ablation implemented but not executed at scale; IEEE → ACL
  reformatting required before any ARR submission.
