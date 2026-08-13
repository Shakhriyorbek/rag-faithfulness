# Paper plan — after Berend's 2026-08-11 letter

**Revision 3 (2026-08-13).** Budget is the previously requested envelope:
**~$38.95 for the grid plus ~$15 for the conditions ≈ $55**, with the existing
`budget_limit = $60` guard in `generate.py` as the hard ceiling. Revision 2
assumed an unconstrained budget and planned ~$290; that is withdrawn.

Decisions are locked before running, so this is a fixed-price plan with a
priority order — if something overruns, you cut from the bottom, not the middle.

---

## The envelope

Measured unit costs, from the 26 July cost probe (1,465 in-tok, 78 out-tok,
$0.001855/query):

| Request type | $/request |
|---|---|
| RAG generation, 5 chunks | 0.001855 |
| Closed-book (C1), no context | 0.000475 |
| Oracle (C2), ~2–3 chunks | ~0.0012 |
| Shapley subset, mean 2.5 chunks | ~0.0012 |
| GPT-4o-mini, same RAG prompt | ~0.000267 |

### Core plan — everything Berend asked for

| # | Experiment | Requests | Cost |
|---|---|---|---|
| 1 | Main grid — 7 models × 3 datasets × n=1000 | 21,000 | **$38.96** |
| 2 | C1 no-retrieval floor (embedder-independent) | 3,000 | $1.43 |
| 3 | C2 oracle ceiling, qrels definition | 3,000 | $3.60 |
| 4 | Prompt-confound probe | 300 | $0.14 |
| 5 | Context ablation — 8 conditions × 300 × NQ | 2,400 | $4.45 |
| 6 | Shapley — 32 subsets × 100 × NQ | 3,200 | $3.84 |
| | **Core total** | **32,900** | **$52.42** |

That leaves **~$7.50** under the $60 guard. Two things fit in it, in this order:

| Add-on | Requests | Cost | Why it is first |
|---|---|---|---|
| **GPT-4o-mini third generator**, full grid | 21,000 | **$5.61** | Cheapest upgrade available by a wide margin. Two generators is the bare minimum that can be called a comparison, and it restores the literature comparability lost in July |
| **Re-ranking λ=0.6, worst model, NQ only** | 1,000 | $1.86 | Keeps H5 alive; §4.6 exists in the draft and cutting it entirely weakens the paper |

Core + both add-ons = **$59.89**. That is right at the ceiling, so treat the
add-ons as genuinely last and re-check the running total before starting them.

**Leave `budget_limit = 60.0` exactly where it is.** It was an arbitrary number
when written; under this plan it is the correct guard.

### Cut in this order if you overrun

1. Re-ranking (H5 becomes future work)
2. GPT-4o-mini (back to two generators)
3. Shapley → eRAG only, 6 requests/query instead of 32 (saves ~$3.20, loses the
   interaction terms but keeps the Salemi & Zamani replication)
4. Context ablation from 8 conditions to 4 — keep `gold_all`, `gold_reversed`,
   `gold1_middle`, `noise_only`, which retain the order test and the floor

**Never cut**: the main grid, C1, C2. Those are the paper.

### What is free and unaffected

All scoring and analysis: `correctness.py`, `conditional.py`, the NLI passes,
ESA, the robustness matrix, TOST, every table and figure. Llama-3-8B is GPU
time, not API spend. Roughly 34,000 requests sequential is **6–9 hours** in
`tmux`, which is tolerable — so concurrency is now a nice-to-have, not a
blocker.

---

## The decisions

### D1. Spine → **B, locked.** Necessary/sufficient 2×2, RFG demoted

Unchanged, and the reason is what makes it safe to commit to before seeing data:

> **The 2×2 is a frame, not a result.** It reports whatever is in each cell. If
> retrieval turns out to be both necessary and sufficient, the 2×2 says so and
> the paper is an honest negative result that still answers the supervisor.
> Spine A needs RFG to be interesting; spine C needs Shapley to be clean. Both
> can fail and leave nothing.

### D2. Salemi & Zamani → replicate as a **focused case study**, not a broad sweep

Revision 2 proposed replicating across 7 embedders × 3 datasets. That does not
fit. At this budget the replication is **one embedder × NQ × 100 queries**.

Write it as what it is: a focused replication that reproduces their qualitative
finding on a new dataset, plus the interaction terms their singleton design
cannot express. Do not claim breadth you did not buy.

If item 4 in the cut list frees money, spend it on **eRAG across two more
embedders** rather than deeper Shapley — eRAG is 6 requests/query against
Shapley's 32, so breadth is roughly five times cheaper than depth here.

### D3. Title → decide **after** the run. Unchanged

*Necessary, Sufficient, Neither: Auditing the Retrieval–Response Link in RAG*
remains the placeholder. It names the question and commits to nothing, which is
what the overclaiming rule requires.

### D4. Hypotheses → N1–N6 with **pre-registered thresholds**

Since you are deciding before running, writing these down costs nothing and
stops §6 reading as fitted after the fact. Thresholds widened from revision 2 to
match n=1000.

| | Hypothesis | Supported if | Measured by |
|---|---|---|---|
| **N1** | Good retrieval is not sufficient | ≥10% of queries in *hit × incorrect*, 95% CI excluding 5% | `conditional.py` 2×2 |
| **N2** | Good retrieval is not necessary | ≥5% of queries in *miss × correct* | 2×2 + C1 |
| **N3** | Faithfulness does not track correctness | `faith_gap ≥ 0` for a majority of models; paired bootstrap p<0.05 for ≥1 | `conditional_faithfulness` |
| **N4** | Matched retrieval ⇒ divergent downstream | ≥2 models equivalent on NDCG@5 (paired TOST, ±0.02) yet differing downstream at p<0.05 | D8 |
| **N5** | Arrangement matters at fixed relevance | \|gold_reversed − gold_all\| ≥ 0.03 accuracy, or position spread ≥ 0.04 | `context_ablation.py` |
| **N6** | Relevance labels underpredict utility | Spearman ρ(qrel_relevant, utility) < 0.3 | `doc_utility.py` |

Secondary: **H2**, **H4**, **H5**. **H1 reported as tested and unsupported** if
the pilot direction holds — say it early and plainly. **H3 must be rewritten**;
it still names GPT-4o-mini as the only closed generator.

### D5. Sampling → **n=1000**, and pool across datasets for the 2×2

n=3000 is withdrawn. The per-cell power problem it was solving is real, so solve
it for free instead:

> The conditional analysis splits each model's queries four ways. **Pool the
> three datasets for the 2×2** — 3,000 queries per model — and report the
> per-dataset breakdown as secondary. That recovers per-cell power at zero cost.

**A tension to handle honestly:** the eval filter *shrinks* the set. If retention
is ~60%, the filtered primary analysis runs on ~600 queries per dataset, and
cells get thin again. Resolution: report the **necessity/sufficiency shares on
the filtered set** (where the filter is conceptually required) but the
**cell-level faithfulness means on the pooled unfiltered set** (where n is
needed), and state which is which. Do not silently mix them.

n=1000 with the filter applied is then a stated limitation, not a hidden one.

### D6. Third generator → **optional add-on at $5.61**, recommended

Cheapest remaining upgrade per unit of value. Three generators — one closed
cheap, one closed cheap-alternative, one open — is a defensible axis. Run it
only after confirming the core completed inside budget.

### D7. Concurrency → **downgraded to nice-to-have**

At ~34,000 requests the sequential loop is 6–9 hours, which `tmux` handles. Not
worth the risk of touching the fail-fast guards and resume ordering before a
budgeted run. Revisit only if you later scale up.

### D8. TOST equivalence test → **keep. Free, and still the biggest hole**

The paper's premise is *near-identical retrieval quality*. Nothing establishes
it, and the pilot showed NDCG@5 spanning **0.908–0.956 — a 5-point spread, which
is not matched.** "No significant difference" is absence of evidence, not
evidence of equivalence.

**Use the paired form.** Every embedder is evaluated on identical queries, so
the test is on per-query NDCG *differences*, whose standard deviation is far
smaller than the raw per-query SD. That is what makes a ±0.02 margin reachable
at n=1000 — the unpaired version would not be.

If no pair comes out equivalent, **N4 cannot be claimed as written** and the
framing shifts to "retrieval quality differences do not predict downstream
differences" — still interesting, but a different sentence. Better found by
design than in review.

---

## Order of operations

1. **Free analysis on the existing pilot, today.** `python src/correctness.py`
   then `python src/conditional.py`. Costs nothing, takes minutes, confirms the
   2×2 renders before you generate 33,000 answers into it.
2. **Implement TOST** in `results.py` (free, no API).
3. **Send the reply; settle D1, D2, D5, D8 with Berend.** These change what you
   write, not what you run, so they do not block step 4.
4. **Run the core, items 1–6, in `tmux`**, checking the running cost at each
   phase boundary.
5. **Check the total.** If under ~$54, add GPT-4o-mini; then, if still under,
   re-ranking on NQ.
6. **Scoring and analysis passes** — all free.
7. **Write §6 last**, from real numbers, with the failures in it.
8. **Choose the title.**

---

## Section-level writing plan (spine B)

| Section | Change | Effort |
|---|---|---|
| §1 Intro | Reframe around necessary + sufficient; Salemi & Zamani positioning | **M** |
| §2 Related work | eRAG, downstream-grounded relevance, Shapley/SHAP | **M** |
| §3.1 Faithfulness | Keep generated-answer definition; add the correctness conditioning and why | **S** |
| §3.2 RFG/nRFG | **Demote** to an aggregate diagnostic; state that it mixes scales | **M** |
| §3.3 NEW Correctness | EM + token-F1, the 0.6 threshold, abstention handling | **S** |
| §3.4 NEW The 2×2 | Both branches falsifiable; the pre-registered thresholds | **M** |
| §4.4 Generator | Claude Haiku 4.5 (+ GPT-4o-mini if run) | **S** |
| §4.5 Geometric analysis | Unchanged. **Never** call it mechanistic | — |
| §4.7 NEW Controlled conditions | C1, C2, the prompt-parity problem and its two mitigations | **M** |
| §4.8 NEW Context arrangement | 8 conditions, hard negatives, slot-count control | **M** |
| §4.9 NEW Document utility | eRAG / LOO / Shapley; scope it honestly as one embedder × NQ | **M** |
| §5.x NEW Matched-quality test | Paired TOST, margin, which pairs qualify | **M** |
| §5.3 Robustness | Unchanged — already answers round-one point 3 | — |
| §6 | **Rewrite from "Expected Results" into Results + Discussion** | **L** |
| §7 Limitations | Prompt asymmetry; oracle choice; QASPER exclusions; n; utility scope | **M** |

New tables and figures: the 2×2 *(centrepiece)*; anchors with % of oracle; the
TOST equivalence table; the 8-condition arrangement figure; utility vs relevance
label; plus the existing robustness heatmap and ESA figure.

---

## What a reviewer attacks, and what covers it

| Attack | Cover |
|---|---|
| "Salemi & Zamani showed this in 2024" | D2 — replicate, then state the delta. Scope it honestly |
| "nRFG mixes a ranking metric with a classifier score" | D1 — demoted to a diagnostic |
| "Why only 1000 instances?" | D5 — the principled filter, pooling for cell power, stated as a limitation |
| **"You never showed retrieval quality was matched"** | **D8 — paired TOST. Currently uncovered** |
| "Faithfulness on wrong answers is meaningless" | §3.1 + the 2×2 — reported conditioned; the incorrect cell is the finding |
| "The no-RAG baseline uses a different prompt" | `--prompt-probe` measures it; `noise_only` has no prompt confound |
| "Oracle is not a real ceiling" | qrels definition; QASPER stand-ins excluded, rate reported |
| "H1 failed, so the framing is post-hoc" | Pre-registered thresholds; report H1 unsupported, prominently |

---

## Open questions for Berend

1. **Does nRFG survive as the primary metric?** Recommended: demote it.
2. **Which oracle definition is primary** — qrels-based, or annotated evidence?
   Only the qrels one is funded; the other is ~$3.60 if he wants both.
3. **Is ±0.02 NDCG@5 the right equivalence margin?** This determines whether N4
   is claimable at all.
4. **Given a fixed budget, would he rather have** the third generator, the
   re-ranking ablation, or broader eRAG coverage? Roughly $6 buys one of them.
5. **Did he intend the Salemi & Zamani link as a positioning warning?**
