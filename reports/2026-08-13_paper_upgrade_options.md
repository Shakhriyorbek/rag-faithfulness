# Paper plan — after Berend's 2026-08-11 letter

**Revision 2 (2026-08-13).** Rewritten after confirming funds and infrastructure
are not a constraint, and that the decision comes before the run.

Revision 1 was a cost ladder built around learning cheaply before committing.
That framing is now wrong: if money is available, **the binding constraints
become wall-clock time, API concurrency, and statistical design** — and the
right move is to decide the whole experiment set now, pre-register how the
results will be read, and run it once, properly.

---

## What changes when budget stops mattering

| Was the constraint | Is now the constraint |
|---|---|
| ~$38 grid, decide what to skip | ~225,000 API requests at **sequential** speed — roughly 40 h in a loop that does one request at a time |
| n=1000 justified by resources | n justified by **per-cell** statistical power |
| Shapley limited to 100 queries | Shapley limited by 2ᵏ, so by wall-clock |
| One generator to save money | A third generator is now ~$17 and buys real generalization |

The whole revised programme costs roughly **$260–300**. That is no longer the
interesting number. The interesting numbers are 40 hours of sequential API time
and about 8 hours of NLI scoring.

---

## Deciding before running: how to do it without guessing

You want the plan settled first. That is achievable, but only if the chosen
spine is **robust to the outcome** — otherwise you are picking a story before
seeing the data, which is exactly what Berend's anti-overclaiming stance is
against.

This is the argument for spine B, and it is stronger than the cost argument I
made in revision 1:

> **The necessary/sufficient 2×2 is a frame, not a result.** It reports whatever
> is in each cell. If retrieval turns out to be both necessary and sufficient,
> the 2×2 says so — the paper becomes a negative result, honestly reported, and
> it is still publishable and still answers the supervisor. Spines A and C do
> not have this property: A needs RFG to be interesting, C needs Shapley to be
> clean. Both can fail and leave nothing.

So: **commit to B now**, and pre-register the reading rules below. That is
deciding first, without betting on an outcome.

---

## The decisions — revised

### D1. Spine → **B, locked.** Necessary/sufficient 2×2, RFG demoted

Unchanged recommendation, but now for a better reason: it is the only
outcome-robust option. Shapley becomes a full section inside it rather than a
token one, since scale is affordable.

### D2. Salemi & Zamani → **replicate, then extend** — now at full scale

Revision 1 proposed replicating their finding on one dataset with one embedder
because Shapley was expensive. Run it across **all 7 embedders × 3 datasets**
instead. That turns a defensive citation into a contribution: their result was
shown on their setup; you show it holds across seven embedding architectures and
three task types, and then ask the question their design cannot.

### D3. Title → still decide **after** the run. No change

Putting a result in the title before measuring it is the overclaiming he
flagged in round one. *Necessary, Sufficient, Neither: Auditing the
Retrieval–Response Link in RAG* remains the safest placeholder — it names the
question and commits to nothing.

### D4. Hypotheses → N1–N6, **with pre-registered thresholds**

This is the substantive upgrade over revision 1. Write down now what counts as
support, so §6 cannot be accused of being fitted after the fact.

| | Hypothesis | Supported if | Measured by |
|---|---|---|---|
| **N1** | Good retrieval is not sufficient | ≥10% of all queries fall in *hit × incorrect*, CI excluding 5% | `conditional.py` 2×2 |
| **N2** | Good retrieval is not necessary | ≥5% of all queries fall in *miss × correct* | 2×2 + C1 |
| **N3** | Faithfulness does not track correctness | `faith_gap ≥ 0` for a majority of models, paired bootstrap p<0.05 for at least one | `conditional_faithfulness` |
| **N4** | Matched retrieval ⇒ divergent downstream | ≥2 models **equivalent** on NDCG@5 (TOST, ±0.02) yet differing in accuracy or faithfulness at p<0.05 | new equivalence test, see D8 |
| **N5** | Arrangement matters at fixed relevance | \|gold_reversed − gold_all\| ≥ 0.02 accuracy, or position spread ≥ 0.03 | `context_ablation.py` |
| **N6** | Relevance labels underpredict utility | Spearman ρ(qrel_relevant, utility) < 0.3 | `doc_utility.py` |

Retained as secondary: **H2**, **H4**, **H5**. **H1 is reported as tested and
unsupported** if the n=50 direction holds — state it plainly and early.
**H3 must be rewritten regardless**; it still names GPT-4o-mini.

### D5. Sampling → **n = 3,000 per dataset**, filtered set as primary

Revision 1 recommended the filter mainly to deflect the "why 1000?" objection.
With funds available there is a better justification, and it is statistical
rather than rhetorical:

> **The conditional analysis splits each model's queries into four cells.**
> Total n is not what matters — per-cell n is. At n=1000, if the *hit ×
> incorrect* cell holds 15% of queries, that is 150 per model, which is thin for
> a faithfulness mean with a confidence interval. At n=3,000 it is 450.

That is an answer Berend will respect more than "reviewers wanted a bigger
number", and it happens to also close the reviewer objection.

Report **both**: filtered as primary (his suggested principle), unfiltered as a
robustness appendix. The retention rate stays a reported finding — a low one is
a statement about benchmark contamination.

QASPER will cap out below 3,000; report the actual n per dataset rather than
implying a uniform figure.

### D6. NEW — **Add a third generator** (~$17)

Now affordable, and it strengthens the weakest part of the design. H3/N4 rest on
generalization across generators, and two generators is the minimum that can be
called a comparison.

Bring **GPT-4o-mini** back alongside Claude Haiku 4.5 and Llama-3-8B. It also
restores comparability with the literature, which was the one real loss when the
generator changed in July. Cost is trivial: 63,000 queries ≈ **$17** at
GPT-4o-mini pricing.

Three generators — one closed frontier-adjacent, one closed cheap, one open —
is a defensible axis rather than an accident of budget.

### D7. NEW — **Add concurrency to the API path** before scaling

`generate.py` issues one request at a time. At ~225,000 requests that is roughly
40 hours of wall-clock, most of it waiting on network.

This is now the single highest-value engineering change: a bounded thread pool
(8–16 in flight) with the existing checkpoint cadence should cut it to under 5
hours. It needs care — the fail-fast guards and cost tracker assume sequential
execution, and the ordering of `generations` must stay stable for resume.

**Measure your actual rate limit first** rather than assuming a tier.

### D8. NEW — **Test that retrieval quality is actually matched** (TOST)

The paper's premise is *near-identical retrieval quality*. Nothing in the draft
or the code establishes that. The pilot showed NDCG@5 of 0.908–0.956, which is a
**5-point spread — not matched**.

This is a real hole. "No significant difference" is not evidence of equivalence;
it is absence of evidence. The correct instrument is **two one-sided tests
(TOST)** with a pre-specified equivalence margin (propose ±0.02 NDCG@5), which
requires large n — another independent reason for D5.

If no pair of models is statistically equivalent on retrieval quality, **N4
cannot be claimed as stated** and the framing must shift to "retrieval quality
differences do not predict downstream differences" — a correlational claim,
still interesting, but a different sentence. Better to find this out by design
than in review.

---

## The experiment manifest

Everything below is decided now and run in one campaign.

| # | Experiment | Requests | Cost | Notes |
|---|---|---|---|---|
| 1 | Main grid — 7 models × 3 datasets × n=3000 | 63,000 | ~$117 | Claude Haiku 4.5 |
| 2 | Third generator — GPT-4o-mini, same grid | 63,000 | ~$17 | D6 |
| 3 | Llama-3-8B — 3 models × 3 datasets × 1000 | 9,000 | GPU | ~10 h on the V100 |
| 4 | C1 no-retrieval floor | 9,000 | ~$4 | embedder-independent |
| 5 | C2 oracle ceiling (qrels) | 9,000 | ~$11 | |
| 6 | C2 oracle (gold_context variant) | 9,000 | ~$11 | the gap is interpretable |
| 7 | Prompt-confound probe | 300 | ~$0.15 | |
| 8 | Context ablation — 8 conditions × 1000 × 3 | 24,000 | ~$45 | D-decision N5 |
| 9 | Shapley — 32 subsets × 500 × 3 | 48,000 | ~$53 | yields eRAG + LOO free |
| 10 | Re-ranking λ sweep — 6 λ × worst model × 3 | 18,000 | ~$33 | H5 |
| | **Total** | **~252,000** | **~$290** | |

Free but time-consuming, on top: NLI scoring of everything (~1.4M forward
passes, roughly 8 h on the V100), AlignScore, ESA, and all the analysis passes.

**Realistic wall-clock: 2–3 days** with concurrency, most of it unattended in
`tmux`. Without concurrency, closer to a week.

---

## Order of operations

Nothing here is a "gate" in the revision-1 sense of deciding whether to
continue. The decisions are locked. These are dependencies.

1. **Settle D1, D2, D5, D6, D8 with Berend.** Send the reply. These change what
   you *write*, so settle them before writing — but they do not block running.
2. **Engineering, before any scaled run:**
   - concurrency in `generate.py` (D7)
   - raise the `budget_limit` guard — it currently raises at $60
   - implement the TOST equivalence test (D8) in `results.py`
   - measure actual API rate limits and throughput on 200 requests
3. **Free analysis on existing pilot checkpoints** — `correctness.py` then
   `conditional.py`. Costs nothing, takes minutes, and tells you the 2×2 renders
   correctly before you generate 250,000 answers into it. Do this today.
4. **Run experiments 1–10**, in that order, in `tmux`. Checkpoint everything.
5. **Scoring and analysis passes** — NLI, correctness, conditional, ablation
   report, utility report.
6. **Write §6 last**, from real numbers, failures included.
7. **Choose the title.**

---

## Section-level writing plan (spine B)

| Section | Change | Effort |
|---|---|---|
| §1 Intro | Reframe around necessary + sufficient; Salemi & Zamani positioning paragraph | **M** |
| §2 Related work | eRAG and downstream-grounded relevance; Shapley/SHAP connection | **M** |
| §3.1 Faithfulness | Keep generated-answer definition. Add: reported conditioned on correctness, with reasoning | **S** |
| §3.2 RFG/nRFG | **Demote.** Retitle as an aggregate diagnostic; state plainly that it mixes scales | **M** |
| §3.3 NEW Correctness | EM + token-F1, the 0.6 threshold, abstention handling | **S** |
| §3.4 NEW The 2×2 | Both branches as falsifiable statements; the pre-registered thresholds | **M** |
| §4.4 Generator | Now **three** generators; already partly fixed in v4 | **S** |
| §4.5 Geometric analysis | Unchanged. **Never** call it mechanistic | — |
| §4.7 NEW Controlled conditions | C1, C2 (both oracle definitions), the prompt-parity problem | **M** |
| §4.8 NEW Context arrangement | 8 conditions; hard negatives; slot-count control | **M** |
| §4.9 NEW Document utility | eRAG / LOO / Shapley; 2ᵏ yields all three | **M** |
| §5.x NEW Matched-quality test | TOST, margin, which pairs qualify | **M** |
| §5.3 Robustness | Unchanged — already answers round-one point 3 | — |
| §6 | **Rewrite from "Expected Results" into Results + Discussion** | **L** |
| §7 Limitations | Prompt asymmetry; oracle choice; QASPER exclusions; n and filter | **M** |

New tables and figures: the 2×2 *(centrepiece)*; the anchor table with % of
oracle; the TOST equivalence table; the 8-condition arrangement figure;
utility vs relevance-label; plus the existing robustness heatmap and ESA figure.

---

## What a reviewer attacks, and what covers it

| Attack | Cover |
|---|---|
| "Salemi & Zamani showed this in 2024" | D2 — replicate across 7 embedders × 3 datasets, then state the delta |
| "nRFG mixes a ranking metric with a classifier score" | D1 — demoted to a diagnostic |
| "Why only 1000 instances?" | D5 — n=3,000, justified by per-cell power, plus the principled filter |
| **"You never showed retrieval quality was matched"** | **D8 — TOST. This one is currently uncovered** |
| "Faithfulness on wrong answers is meaningless" | §3.1 + the 2×2 — reported conditioned; the incorrect cell is the finding |
| "The no-RAG baseline uses a different prompt" | `--prompt-probe` measures it; `noise_only` has no prompt confound |
| "Oracle is not a real ceiling" | Both definitions run; QASPER stand-ins excluded and the rate reported |
| "Two generators is not generalization" | D6 — three generators |
| "H1 failed, so the framing is post-hoc" | Pre-registered thresholds in D4; report H1 unsupported, prominently |

---

## Open questions for Berend

1. **Does nRFG survive as the primary metric?** Recommended position: demote it.
2. **Which oracle definition is primary** — qrels-based, or annotated evidence?
   Both are being run; the gap between them is interpretable.
3. **Is ±0.02 NDCG@5 the right equivalence margin** for the matched-quality
   claim? This determines whether N4 is claimable at all.
4. **Is a third generator worth it**, or does he prefer depth on two?
5. **Did he intend the Salemi & Zamani link as a positioning warning?**
