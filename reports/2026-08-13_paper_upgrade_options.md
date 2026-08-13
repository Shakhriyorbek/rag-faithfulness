# Paper upgrade options — after Berend's 2026-08-11 letter

Decision document. Berend will let you choose, so these are framed as choices
with what each one buys and what it costs, not as a single plan.

**The code for every option below already exists and is committed.** Nothing
here is blocked on implementation. What is blocked is running the experiments,
which needs the API budget, and the writing.

---

## The situation in three sentences

The paper's current spine is a metric (RFG/nRFG) that Berend has now questioned
twice — first for not distinguishing both-high from both-low, now implicitly by
pointing out that faithfulness only matters when the answer is correct. The
paper he linked, Salemi & Zamani (SIGIR 2024), already publishes the premise the
paper is built on: relevance labels correlate weakly with downstream RAG
performance. The controlled conditions he proposed are stronger evidence than
anything currently in the draft, and they are cheap.

So the question is not "what do we add" but **"what carries the paper"**.

---

## Decision 1 — What is the paper's spine? (choose one)

### Option A — Keep RFG as the headline, add the conditions as support

Add a §5.4 reporting C1/C2. §3 untouched. nRFG stays the primary metric.

| | |
|---|---|
| Writing effort | **S** — about a week |
| Experiment cost | ~$53 (full grid + conditions) |
| What it buys | Least disruption; the existing draft survives mostly intact |
| **Risk** | **High.** A reviewer who knows Salemi & Zamani asks what is new, and the answer is a metric the supervisor has twice found wanting. nRFG subtracts a faithfulness score from a ranking metric — different scales — which is an easy attack in review. |

### Option B — Necessary/sufficient spine; RFG demoted to a diagnostic ⭐ recommended

The 2×2 becomes the paper's central table. Correctness comes first, faithfulness
is reported conditioned on it, RFG/nRFG survive as a subsection describing an
aggregate diagnostic rather than as the contribution.

| | |
|---|---|
| Writing effort | **M** — two to three weeks, mostly §1, §3, §6 |
| Experiment cost | ~$53 |
| What it buys | Answers Berend's letter directly; the claims become falsifiable per query; the metric work is kept without asking it to carry weight it cannot bear |
| Risk | Moderate — depends on the conditions actually separating. If C1 ≈ C2 ≈ embedders, there is no paper, but you would learn that in Rung 5 for ~$15 |

### Option C — Rebuild around per-document context utility (Shapley)

Drop RFG. The central object becomes each retrieved document's measured
contribution to correctness and faithfulness.

| | |
|---|---|
| Writing effort | **L** — a substantial rewrite |
| Experiment cost | ~$53 + more Shapley queries (2⁵ per query is the binding constraint) |
| What it buys | The most theoretically distinct contribution. eRAG only measures documents in isolation; Shapley prices them in company, and nobody has done that conditioned on *faithfulness* |
| Risk | **Highest.** Abandons two years of framing, and if Shapley is noisy at n=100 there is no fallback. Also the least mature analysis |

**Recommendation: B, with C's Shapley as a section inside it.** You keep the
existing structure, answer the supervisor, and still get the novel measurement —
without betting the paper on the newest and least tested piece.

---

## Decision 2 — How to handle Salemi & Zamani (choose one)

This one matters more than it looks. Handled badly it is a desk-reject risk;
handled well it is free credibility.

| Option | What it means | Verdict |
|---|---|---|
| **A** Cite it in the related-work list | One line among many | **Not enough.** Looks like you did not read it |
| **B** Full positioning paragraph | Explicitly state what they showed and what is left | Minimum acceptable |
| **C** Replicate their finding on your data, then extend ⭐ | `doc_utility.py --report` already correlates qrels relevance against measured utility — that *is* their headline, on 3 datasets and 7 embedders | **Recommended** |

Option C converts the threat into a strength. You own the prior result instead
of being caught by it: *"We reproduce Salemi & Zamani's finding on three
datasets and seven embedding models, and then ask the question their setup
cannot: do models with statistically indistinguishable retrieval quality
diverge downstream?"*

Ask Berend directly whether he meant the link as a positioning warning. If he
did, saying so first is worth a great deal.

---

## Decision 3 — Title

Current: *Beyond Retrieval Quality: How Embedding Architecture Affects
Faithfulness in RAG Systems*

**Constraint: do not put a result in the title before the results exist.** This
is exactly the overclaiming Berend flagged in round one.

| | Title | Fits spine |
|---|---|---|
| 1 | Keep as is | Any |
| 2 | *Matched Retrieval Quality, Divergent Faithfulness: Embedding Architecture in RAG Systems* | B — but asserts a result you have not measured |
| 3 | *Necessary, Sufficient, Neither: Auditing the Retrieval–Response Link in RAG* ⭐ | B — names the question, not the answer |
| 4 | *What Do Retrieved Documents Contribute? Per-Document Utility in RAG* | C |

Option 3 is the safest under his overclaiming rule: it states what the paper
investigates and commits to nothing. Decide the title **after** Rung 5.

---

## Decision 4 — The hypotheses

H1–H5 were written for the RFG frame. H1 is already contradicted at n=50, and
H3 still names GPT-4o-mini. Under spine B they need restating.

**Proposed replacement set** (keep the old ones as secondary where still
meaningful):

| | Hypothesis | Measured by |
|---|---|---|
| **N1** | Among queries where retrieval succeeds, a non-trivial share are answered incorrectly → retrieval is **not sufficient** | `conditional.py` 2×2, hit × incorrect cell |
| **N2** | A non-trivial share of correct answers occur without a retrieval hit → retrieval is **not necessary** | 2×2, miss × correct cell; C1 for the parametric part |
| **N3** | Faithfulness on incorrect answers is **no lower** than on correct ones | `faith_gap` — if positive, pooled faithfulness is not reportable |
| **N4** | Embedders with statistically indistinguishable NDCG@5 differ in accuracy and faithfulness | the original core claim, now stated per query with CIs |
| **N5** | Rearranging *identical* gold snippets changes accuracy | `context_ablation.py`: `gold_reversed − gold_all`, and the position spread |
| **N6** | Measured per-document utility correlates weakly with the qrels relevance label | `doc_utility.py` — the Salemi & Zamani replication |

Retained from the old set: **H2** (HotpotQA highest gap), **H4** (ESA higher for
instruction-tuned), **H5** (re-ranking). **H1 should be reported as tested and
unsupported** if the n=50 direction holds — Berend has said explicitly he values
that over an inflated claim.

**H3 must be rewritten regardless** — it still says GPT-4o-mini.

---

## Decision 5 — The evaluation set (this is his point about n=1000)

| Option | What it means | Cost |
|---|---|---|
| **A** Keep 1000 random, justify by resources | What the draft does now | $0 |
| **B** Primary results on the **filtered** set, full set in appendix ⭐ | Retain only queries the model gets wrong with no retrieval — his own suggestion. `conditions.py --emit-filter` then `conditional.py --filtered` | ~$1.50 extra |
| **C** Scale to full dataset splits | Removes the objection entirely | 10×+ the grid cost |

**Recommended: B.** He pre-empted the reviewer objection himself, and the
retained fraction is a publishable observation in its own right — a low
retention rate is a statement about **benchmark contamination**, not about your
sampling. Report it as a table.

Say the quiet part in the limitations section: resource constraints are real,
and B is a principled filter rather than a rationalization.

---

## Decision 6 — Section-level map (if you take spine B)

| Section | Change | Effort |
|---|---|---|
| §1 Intro | Reframe around necessary + sufficient. Add the Salemi & Zamani positioning paragraph | **M** |
| §2 Related work | New paragraph on eRAG and downstream-grounded relevance; connect to Shapley/SHAP | **M** |
| §3.1 Faithfulness | Keep the generated-answer definition. **Add:** faithfulness is reported conditioned on correctness, with the reason | **S** |
| §3.2 RFG/nRFG | Demote. Retitle as an aggregate diagnostic; state plainly that it mixes scales and what it is and is not for | **M** |
| **§3.3 NEW** Correctness | EM + token-F1, the 0.6 threshold, abstention handling | **S** |
| **§3.4 NEW** The 2×2 | Define the grid; both branches as falsifiable statements | **M** |
| §4.4 Generator | Already fixed in v4 (Claude Haiku 4.5). Tell him | done |
| §4.5 Geometric analysis | Unchanged. **Never** call it mechanistic | — |
| **§4.7 NEW** Controlled conditions | C1 floor, C2 oracle, the prompt-parity problem and how it is handled | **M** |
| **§4.8 NEW** Context arrangement | The 8 ablation conditions; hard negatives; slot-count control | **M** |
| **§4.9 NEW** Document utility | eRAG / LOO / Shapley; note that 2⁵ subsets yield all three | **M** |
| §5.3 Robustness | Unchanged — this already answers his round-one point 3 | — |
| §6 | **Rewrite from "Expected Results" into Results + Discussion.** Report failures as failures | **L** |
| §7 Limitations | Prompt asymmetry in C1; oracle definition choice; QASPER fallback exclusions; n and the filter | **M** |

New tables and figures:

- **Table N** — the 2×2, counts + mean faithfulness per cell *(the paper's centrepiece)*
- **Table O** — anchors: floor → embedders → ceiling, with % of oracle recovered
- **Figure X** — the 8 arrangement conditions, accuracy and faithfulness
- **Figure Y** — measured document utility vs qrels relevance label
- Keep the 9-variant robustness heatmap and the ESA figure

---

## Two things to raise with him explicitly

**1. Does nRFG survive?** Your recommended position is that it becomes a
diagnostic, not the contribution. He has questioned it twice; better to propose
demoting it yourself than to defend it through review.

**2. Is the oracle definition right?** There are two notions of gold evidence in
the pipeline and they are not the same object. C2 now defaults to the qrels
definition — the ceiling of the task NDCG@5 actually scores — rather than the
annotated evidence text, which on HotpotQA is supporting sentences stripped of
their paragraphs and is strictly easier than perfect retrieval. Running both is
cheap and the gap between them is interpretable. Ask which he wants as primary.

---

## What a reviewer will attack, and what covers it

| Attack | Cover |
|---|---|
| "Salemi & Zamani showed this in 2024" | Decision 2 option C — replicate, then state the delta |
| "nRFG mixes a ranking metric with a classifier score" | Decision 1 option B — demote it |
| "Why only 1000 instances?" | Decision 5 option B — principled filter, retention rate reported |
| "Faithfulness on wrong answers is meaningless" | §3.1 + the 2×2 — it is reported conditioned, and the incorrect cell is the *finding* |
| "The no-RAG baseline uses a different prompt" | `--prompt-probe` measures it; `noise_only` is a floor with no prompt confound |
| "Oracle is not a real ceiling" | qrels-based definition; QASPER stand-ins excluded and the exclusion rate reported |
| "H1 failed, so the framing is post-hoc" | Report H1 as unsupported, prominently. This is a strength with this supervisor |

---

## Sequencing — what to do in what order

1. **Rung 5 free steps first.** `correctness.py` and `conditional.py` run
   against existing checkpoints at zero cost. Even on the n=50 pilot they show
   you the *shape* of the 2×2.
2. **Rung 2 + Rung 5 paid steps** (~$21 total) — 3 models on NQ at n=1000 plus
   the conditions. **This is the decision gate.** If floor ≈ embedders ≈
   ceiling, the paper needs rethinking and you will know for $21 rather than
   $53.
3. **Send the reply and settle Decisions 1, 2 and 5 with Berend** — these change
   what you write, so settle them before writing.
4. **Full grid** only after the gate passes.
5. **Write §6 last**, from real numbers, with the failures in it.

**Do not choose the title until step 2 is done.**
