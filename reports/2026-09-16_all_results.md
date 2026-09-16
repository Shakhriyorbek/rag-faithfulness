# Every result we have, and what is now closed

2026-09-16. Covers 2026-09-08 through 2026-09-16 — the stretch after the B22/B23
paper update, which has had no report until now. Every number here was
re-derived from `~/rag-backup/checkpoints/n1000_v3` in this session, not copied
from an earlier report.

**The short version: every experiment is run.** Six of them have results that
appear nowhere in `paper/acl/main.tex`. All 13 external-review red flags and all
14 of Berend's points are now closed or explicitly stated as limitations.

---

## 1. What ran since the last report

| arm | scope | cost | in paper? |
|---|---|---|---|
| qwen LLM judge | 8 cells | ~$15 | ✅ yes |
| jina-embeddings-v3 | NQ, HotpotQA, QASPER × 3 generators × 3 evaluators | $0 | ❌ **no — and the paper says it was not run** |
| QASPER | 5 embedders × 3 generators × 3 evaluators, 819 queries | $5.22 | ❌ no |
| C1 no-RAG floor | NQ, HotpotQA, Claude only | ~$2 | ❌ no |
| C2 oracle ceiling | NQ, HotpotQA, Claude only | ~$2 | ❌ no |
| context ablation | 8 conditions × 2 datasets, n=300 | ~$3 | ❌ no |
| ESA | 4 embedders × 2 datasets, n=200 | $0 | ❌ no |
| re-ranking | all-mpnet × 2 datasets, λ=0.6 | ~$1 | ❌ no |

---

## 2. jina-embeddings-v3 — Berend's point 2, and a positive control

`main.tex:335` currently states that jina "configured and runnable but **not run
at this scale**". That is false as of this week. It ran on all three datasets,
all three generators, all three evaluators.

**Retrieval quality (NDCG@5):**

| system | NQ | HotpotQA | QASPER |
|---|---|---|---|
| jina-embeddings-v3 | **0.6845** | 0.7724 | 0.0902 |
| all-mpnet-base-v2 | 0.7861 | 0.7051 | 0.0616 |
| BGE-M3 | 0.7943 | 0.8093 | 0.1217 |
| E5-large-instruct | 0.8061 | 0.8237 | 0.1062 |
| text-embedding-3-small | 0.8299 | 0.7666 | 0.1146 |

On NQ jina is the **weakest system by 10 points** and triples the spread the
whole paper rests on: **0.0438 → 0.1454**. On HotpotQA it lands inside the
existing band and the spread does not move (0.1186 either way).

**It fails the matched-quality precondition on NQ, and TOST says so at maximum
strength.** Paired TOST on per-query NDCG at ±0.02, jina against each of the
four: mean differences 0.102, 0.110, 0.122, 0.145, **p_tost = 1.0000 in all
four**. Non-equivalence could not be more decisive. On HotpotQA jina vs
text-embedding-3-small is the **only equivalent pair in the entire table**
(mean diff 0.0059, p_tost 0.0083, matched = True).

**What happens to the headline if jina is included as a fifth system:**

| family | 4 systems (published) | 5 systems (with jina) |
|---|---|---|
| table-wide Holm over 18 | **2 of 6** | **6 of 6** |
| within-evaluator | 4 | 8 |
| within-cell | 4 | 10 |

Every cell "differs" once jina is in the set. **This is not a stronger result —
it is the premise breaking**, and it should be reported as such. The paper's
claim is about systems that retrieve *equally well* and are nonetheless judged
differently. Add a system that retrieves genuinely worse and the comparison
detects a real difference, under every evaluator, as it should.

**That makes jina the positive control the paper currently lacks.** A reviewer
can reasonably ask whether the 2-of-6 null is a power failure. It is not: same
test, same bootstrap, same n, and it fires in 6 of 6 cells the moment a
genuinely weaker retriever is present. This is worth more to the paper than a
fifth row in the matched comparison would have been.

**Recommendation:** report jina in the retrieval table and in its own
subsection as the control. Keep the matched comparison on the four TOST-checked
systems. Delete the false Limitations sentence.

---

## 3. QASPER — retrieval fails, and the evaluator ordering inverts

Pooled over 5 embedders, answered rows only:

| generator | NLI-max | AlignScore | claim-min | refusal |
|---|---|---|---|---|
| Claude | 0.712 | 0.629 | **0.240** | 35.4% |
| qwen | 0.757 | 0.798 | 0.536 | 27.8% |
| GPT-4o-mini | 0.856 | 0.874 | 0.790 | 45.0% |

Embedder spreads 0.011–0.075, comparable to the main grid. Retrieval essentially
failed — NDCG@5 0.062–0.122 against 0.705–0.830 on NQ/HotpotQA, Recall@5 8–18%.
QASPER is single-document QA and pooling 819 papers into one 13,876-chunk corpus
leaves questions ("which multilingual approaches do they compare with?") with no
paper-identifying signal.

**Two things worth reporting.** First, the generator ordering **inverts**: on
QASPER GPT-4o-mini is the most faithful and Claude the least, the reverse of
NQ/HotpotQA under NLI-max. Second, and more important, the **evaluator ordering
inverts too**. B23 established AlignScore > claim-min > NLI-max in all six cells
of the main grid. On QASPER, Claude reads NLI 0.712 > Align 0.629 > claim 0.240,
and only GPT-4o-mini preserves the old order.

So the evaluator ordering is not a property of the evaluators. It is a property
of the evaluators **in a given retrieval regime**. That is a genuine finding and
it is not the one the run was bought to produce.

**Recommendation:** a Discussion subsection framing QASPER as a
retrieval-collapse stress condition, not a third dataset in the main grid.
Claude's claim-min of 0.240 — its weakest assertion barely supported — is the
number to lead with.

---

## 4. C1 / C2 — Berend's letter-2 points 1, 2 and 3

Claude arm only; this is a real limitation and must be stated.

| | NQ | HotpotQA |
|---|---|---|
| **C1 floor** (no retrieval) | **0.322** acc, 15.5% refuse | **0.209** acc, 70.4% refuse |
| retrieval (4 embedders) | 0.760 – 0.796 | 0.528 – 0.703 |
| **C2 oracle ceiling** | **0.770** acc, 12.9% refuse | **0.868** acc, 8.0% refuse |

**Point 3 fires: retrieval beats the oracle on NQ.** `pct_of_oracle` for Claude:

| system | NQ | HotpotQA |
|---|---|---|
| text-embedding-3-small | **103.4%** | 71.9% |
| all-mpnet-base-v2 | **102.6%** | 60.8% |
| E5-large-instruct | **102.2%** | 81.0% |
| BGE-M3 | 98.7% | 78.2% |

Three of four systems exceed the ceiling. Berend asked explicitly that this be
surfaced and never clipped, and it happened. The explanation is in the design:
C2's oracle is `--oracle-source qrels`, the answer-bearing chunks only, which is
*narrower* context than top-5 retrieval. A narrower context triggers more
refusals on questions needing surrounding material. The oracle is a ceiling on
*evidence selection*, not on accuracy, and on a single-hop dataset retrieval's
extra context is worth more than its extra noise costs. On HotpotQA, where
composition matters, the ceiling behaves like a ceiling — nothing exceeds it and
the best system reaches 81%.

**The C1 floor is also a measured validity threat.** NQ and HotpotQA are
Wikipedia, and on NQ Claude answers **32.2%** correctly with no retrieved context
at all, refusing only 15.5% of the time. A third of the NQ result is reachable
from parametric memory. On HotpotQA the model mostly declines instead (70.4%
refusal, 20.9% correct), which is the better-behaved case.

---

## 5. Context ablation — Berend's letter-2 point 4

n=300 per condition. `faithfulness` was never scored for these rows; accuracy and
refusal are what we have.

| condition | NQ acc | NQ refuse | HotpotQA acc | HotpotQA refuse |
|---|---|---|---|---|
| gold_all | 0.790 | 12.7% | 0.877 | 7.3% |
| gold_reversed | 0.773 | 12.3% | 0.890 | 6.3% |
| gold_shuffled | 0.790 | 12.3% | 0.880 | 7.0% |
| gold_top1 | 0.770 | 17.0% | **0.517** | **75.7%** |
| gold1_first | 0.753 | 13.7% | 0.530 | 69.3% |
| gold1_middle | 0.757 | 16.3% | 0.503 | 68.3% |
| gold1_last | 0.727 | 15.3% | 0.510 | 72.3% |
| noise_only | **0.090** | 66.7% | **0.150** | 91.3% |

Three clean findings:

1. **Order does not matter.** Reversing or shuffling the gold context moves
   accuracy by at most 1.7 points on either dataset. At k=5 there is no
   "lost in the middle" effect: first/middle/last at constant length 5 spans
   0.727–0.757 on NQ and 0.503–0.530 on HotpotQA.
2. **Subset matters, and only where composition does.** Dropping to one gold
   chunk costs NQ 2 points (0.790 → 0.770) and HotpotQA **36 points**
   (0.877 → 0.517). The mechanism is visible in the refusal column: HotpotQA
   refusal jumps 7.3% → 75.7%. The model does not guess when half a multi-hop
   chain is missing — it declines. That is the behaviour you want, and it is
   the same refusal machinery that Section 5 shows contaminates the evaluation
   population.
3. **noise_only is a second floor** and it is not zero: 9.0% / 15.0% correct
   from hard negatives alone, consistent with the C1 parametric-memory result.

---

## 6. ESA — supervisor point 6, closed with data

The paper promised to compute both correlations: `corr(cos(q,d), NLI(d, gold))`,
which is what ESA measures, and `corr(cos(q,d), NLI(d, q))`, which is the signal
actually available at re-rank time. Both are now computed, n=200 per cell.

| system | dataset | ESA (gold) r | p | NLI(d,q) r | p |
|---|---|---|---|---|---|
| all-mpnet-base-v2 | NQ | 0.1464 | 0.039 | 0.1308 | 0.065 |
| BGE-M3 | NQ | 0.1893 | 0.007 | 0.2109 | 0.003 |
| E5-large-instruct | NQ | 0.2174 | 0.002 | 0.2256 | 0.001 |
| text-embedding-3-small | NQ | 0.1682 | 0.017 | 0.1655 | 0.019 |
| all-mpnet-base-v2 | HotpotQA | −0.1073 | 0.130 | 0.0771 | 0.278 |
| BGE-M3 | HotpotQA | −0.0859 | 0.226 | 0.0380 | 0.593 |
| E5-large-instruct | HotpotQA | −0.1148 | 0.106 | 0.0373 | 0.600 |
| text-embedding-3-small | HotpotQA | −0.0920 | 0.195 | 0.0698 | 0.600 |

**The geometric signal exists on NQ and not on HotpotQA.** On NQ both
correlations are positive and significant in 3 of 4 systems, and the two agree
closely — so on single-hop data the re-rank-time signal is about as good as the
gold-answer signal. On HotpotQA the ESA correlation is **negative in all four
systems** and the re-rank-time signal is indistinguishable from zero.

This is the honest answer to "re-ranking is not justified by the analysis": on
the dataset where it would matter most, the analysis does not justify it.

---

## 7. Re-ranking — H5 fails

Claude / all-mpnet-base-v2, λ=0.6, re-rank top-20 by NLI(d,q) then regenerate.

| | NQ | HotpotQA |
|---|---|---|
| refusal, baseline → reranked | 15.4% → **19.0%** | 39.3% → 40.4% |
| nli_max answered, baseline → reranked | 0.9192 → 0.9271 | 0.7158 → 0.7075 |
| paired delta (answered) | **+0.0079** | **−0.0083** |
| 95% CI | [−0.0067, +0.0225] | [−0.0285, +0.0116] |

**H5 predicted ≥15% improvement. The measured effect is under one point in
either direction and both CIs contain zero.** Re-ranking on NLI(d,q) does not
buy faithfulness. What it does buy is a higher refusal rate on NQ, 15.4% → 19.0%
— it filters context the generator was using, and the generator declines rather
than answering from less.

Consistent with §6: the re-rank signal correlates with faithfulness on NQ
(r≈0.13–0.23) but that correlation is far too weak to convert into a gain, and
on HotpotQA there is no signal to exploit at all.

---

## 8. Red flags — all 13

| | red flag | status |
|---|---|---|
| 🔴 1 | Single NLI judge, AlignScore not run | ✅ three evaluators everywhere |
| 🔴 2 | Max-over-chunks aggregation | ✅ claim-min over the full grid |
| 🔴 3 | Conditioning on "all models answered" (collider) | ✅ measured 09-12; Limitations bullet states the bound |
| 🔴 4 | Selective NQ sampling | ⚠️ **stated, not fixed** — Limitations bullet; unchanged |
| 🔴 5 | Causal language exceeds identification | ✅ swept in the v6 rewrite |
| 🔴 6 | ±0.05 TOST margin unjustified | ✅ margin curve + falsification anchor |
| 🔴 7 | Correctness metric inadequate | ✅ LLM judge, all 24 cells |
| 🟠 8 | Only 4 embedders | ✅ **5 now, and the 5th is a positive control** |
| 🟠 9 | Only 2 English benchmarks | ✅ **3 now** (QASPER); still all English |
| 🟠 10 | Only 2 closed-source generators | ✅ qwen open-weight arm |
| 🟡 11 | Multiple-testing presentation | ✅ all three families printed every run |
| 🟡 12 | RFG/nRFG conceptually weak | ✅ retired and reported as a result |
| 🟡 13 | Many experiments mentioned, not run | ✅ **all of them have now run** |

Red flag 4 is the only one not closed by an experiment, and it cannot be — it is
a property of how the NQ sample is drawn. It stays a limitation.

---

## 9. Berend's points — all 14

**First letter (7):**

| # | point | status |
|---|---|---|
| 1 | Notation precision, don't overclaim | ✅ |
| 2 | Add Jina embeddings | ✅ **actually run now**, §2 |
| 3 | RFG volatile across measurement choices | ✅ generalised into the paper's spine |
| 4 | RFG can't distinguish both-high from both-low | ✅ nRFG, then demoted |
| 5 | Faithfulness vs generated or gold answer | ✅ explicit in Definitions |
| 6 | Re-ranking not justified; "mechanistic" overclaims | ✅ **closed with data**, §6 and §7 |
| 7 | ESA doesn't extend Zhu et al. | ✅ claim and citation removed |

**Second letter (7):**

| # | ask | status |
|---|---|---|
| 1 | Zero-retrieval generation | ✅ C1, §4 — Claude only |
| 2 | Oracle RAG glass ceiling | ✅ C2, §4 — Claude only |
| 3 | Watch for retrieval beating the oracle | ✅ **it happens**, 3 of 4 on NQ, §4 |
| 4 | Subset/ordering of relevant snippets | ✅ 8 conditions, §5 |
| 5 | Faithfulness matters only when correct | ✅ conditional.py with judge correctness |
| 6 | Shapley per-document utility | ➖ dropped at his direction |
| 7 | Justify n=1000 or scale up | ⚠️ `--emit-filter` built; no scale-up run |

---

## 10. What is actually left

1. **Delete the false jina sentence** at `main.tex:335`. This is the only
   outright error in the manuscript.
2. **Draft the six missing sections** — §2 through §7 above.
3. **State the two arm limitations**: C1/C2 and the ablation are Claude-only;
   ESA and rerank are all-mpnet-only.
4. `faithfulness.py`, `claim_faithfulness.py`, `compare_evaluators.py`,
   `perturbation_check.py`, `perturb_report.py`, `copying_check.py` and
   `markdown_check.py` all hardcode `--datasets default='NQ,HotpotQA'`. That is
   what silently skipped QASPER and printed "complete". Point them at
   `config.DATASETS`.
5. Context-ablation rows were never scored for faithfulness. Free to fix if the
   ablation section wants a faithfulness column.

**Process note.** `results.equivalence_table()` writes `equivalence_table.pkl` on
every call and takes no model filter, so running it exploratorily overwrote the
published 4-model artifact in the backup. It was regenerated exactly — TOST is
analytic, not bootstrapped — but this is the same trap as `conditional.py`
overwriting `faith_gap_significance_*.pkl`. Copy before exploring.
