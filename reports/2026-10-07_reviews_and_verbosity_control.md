# The review cycle, and the verbosity control

2026-10-07. Covers 2026-09-16 through 2026-10-07: two automated review passes, a
code check, a claim-validity check, the W2 verbosity control, and the paper
changes that came out of all of it. Every number re-derived from checkpoints.

**State: the paper is 22 pages, builds clean (0 undefined, 0 overfull boxes),
230 tests pass, everything committed and pushed.**

---

## 1. What the reviews did to the score

| | first pass | second pass |
|---|---|---|
| Overall | 3 — Findings | **3.5 — Borderline Conference** |
| Soundness | 3.5 | **4 — Strong** |
| Excitement | 3.0 | **3.5** |
| Desk reject | pass | pass (incl. length, at 22 pp) |

The length warning I raised twice was wrong: the checker passed the 22-page
version both times. ARR's 8-page limit is a submission rule, not something this
reviewer applied.

The second pass names its own price for 4.0: engage ARES/RAGVUE and say whether
the diagnosed problems reach past NLI-style evaluators, and give practitioners a
decision pathway. Both were done (§6). Its stated downside risk is generator
coverage, which writing cannot fix.

---

## 2. The verbosity control — W2

The review's second weakness: assertion count is confounded with model family,
since the three generators differ in provider and training as well as length.
The review offered the cheap fix itself, "or prompting configuration", and that
is the better instrument. A fourth model adds a fourth confounded point;
varying the prompt on one model varies only the thing in question.

Qwen2.5-7B, run twice over identical queries, weights and retrieved contexts,
with **one line** added to the prompt. Grounding instruction and refusal
sentence byte-identical, so `is_abstention` still matches. 10 cells (5
embedders x 2 datasets), 10,000 answers, **$0**.

**The manipulation took, and overshot.**

| | assertions | words | refusals |
|---|---|---|---|
| qwen NQ | 1.77 | 34 | 11.8% |
| verbose NQ | **4.90** | 108 | 9.8% |
| qwen HotpotQA | 1.54 | 26 | 29.6% |
| verbose HotpotQA | **4.99** | 110 | 23.8% |

4.9 assertions overshoots Claude's 3.08, so the arm **extends** the verbosity
axis rather than interpolating on it.

**Detection falls in all six cells. No confidence interval overlaps.**

| evaluator | dataset | qwen | verbose | Δ (std → verb) |
|---|---|---|---|---|
| NLI-max | NQ | 35.1% | **10.4%** | 0.261 → 0.072 |
| NLI-max | HotpotQA | 54.8% | **17.4%** | 0.270 → 0.088 |
| Claim-min | NQ | 48.1% | **28.9%** | 0.297 → 0.054 |
| Claim-min | HotpotQA | 66.8% | **43.6%** | 0.263 → 0.025 |
| AlignScore | NQ | 51.8% | **16.0%** | 0.436 → 0.149 |
| AlignScore | HotpotQA | 74.5% | **18.5%** | 0.494 → 0.127 |

### Three things the run was not bought for

**AlignScore is not exempt, and it is not an NLI model.** It falls furthest in
proportional terms (74.5% → 18.5%) with mean response 0.494 → 0.127. That is the
single most useful result here, because it establishes that the structural bias
attaches to **scalar aggregation over a multi-assertion answer**, not to
entailment. A fine-tuned judge inherits it. This became the backbone of the
beyond-NLI argument the second review asked for.

**Claim-min keeps its sensitivity and loses its threshold.** It degrades least,
as its construction predicts — now shown by manipulation rather than inferred
across models. But its absolute means collapse: 0.693 → 0.234 on NQ and
0.424 → 0.076 on HotpotQA. A minimum over five assertions sits near zero
whatever the answer says. Resisting the structural bias and remaining
thresholdable are separate properties, and the Discussion now says so.

**The scale floor rises steeply.** An untouched answer against an *unrelated*
document goes 0.470 → 0.581 on NQ and **0.245 → 0.509** on HotpotQA. On
HotpotQA that puts the unrelated-document baseline **above the fixed 0.5 gate**.
Verbosity degrades a fixed threshold twice over, moving the falsified answer up
and the floor up with it. That is the paper's existing argument for a
floor-anchored gate, reached here by manipulation.

### The prediction failed, usefully

Registered before the run: an arm at 4.9 assertions should fall **below** the
5.1% of a generator at 3.08. It did not — **10.4%**. Answer structure is causal
and explains most of the gap, not all, and the residual is generator-specific.
That reproduces B23 from the opposite direction (conditioning on assertion count,
Claude was the outlier). On HotpotQA the residual vanishes: 0.088 against 0.085.

**Confound to carry forward:** refusals fall in both datasets, so the two arms
attempt slightly different questions. Stated, not corrected for.

---

## 3. Entity substitution — W1, free

The probe had always built entity cases (`build_entity_case`) and the
checkpoints had always held the results. They had never been reported. The
review's heaviest weakness cost nothing to answer.

The verbosity effect survives the change of error type. NQ under NLI-max: 0.082
for the most verbose generator against 0.403 for the tersest, versus 0.054 and
0.451 for numbers. Generator ordering reproduces in **5 of 6** dataset-evaluator
combinations; claim-min on NQ inverts the two more verbose generators
(0.163 vs 0.112).

The two error types are **not interchangeable** and neither is uniformly harder.
They agree within 0.05 for the tersest generator and diverge in opposite
directions for the other two. So neither licenses extrapolation to the other,
let alone to error types we did not test.

---

## 4. Power — W4, and it partly vindicates the reviewer

Minimum detectable effect at 80% power, two-sided paired, α = 0.05:
**0.012 to 0.060, median 0.034. In 16 of 18 cells the MDE exceeds the spread
actually observed.** Both significant cells clear their own MDE.

The part the paper had not said: **AlignScore has the lowest MDE in all six
cells** (0.012–0.037 against NLI-max's 0.021–0.060), because its paired
per-query differences are far less noisy (sd 0.12–0.30 against 0.20–0.48) on
identical answers. So part of the evaluator-dependence is **resolving power at
fixed n**, not only response to a known error. The design does not separate the
two and Section 7 now says that.

---

## 5. The population inconsistency found 2026-10-07

Chasing a loose end from the previous session: the Limitations bullet gave the
generator span as **1.12 to 2.73** assertions, while the abstract gives
**3.08, 1.77 and 1.25**. Those are two different populations for the same
quantity — 1.117 and 2.728 are computed over **all rows including refusals**,
and the abstract's figures are **answered-only** (verified to match exactly).

In a paper whose Section 6 argues that pooling refusals changes conclusions,
reporting an all-rows figure beside an answered-only one is the error the paper
warns about. Limitations now reads **1.17 to 2.77 assertions per attempted
answer**, and every assertion figure in the manuscript is answered-only.

---

## 6. Paper changes from the two reviews

**Added**
- §5.2 entity substitution + Table 2 (W1)
- §5.5 verbosity control + Table 5 (W2)
- §7.1 jina as a positive control, moved to the **front** of §7 (C7) — a reader
  met four nulls and only later learned the test fires
- §2.4 "Does this reach past entailment scorers?" (W8, the named score-raiser)
- §12.1 "What to do with a verbose generator" — four-step pathway (the second
  named score-raiser)
- §12.2 "Choosing among the three" + Table 6 selection table (W10)
- Appendix H power analysis + Table 12 (W4)
- Deterministic value check **promoted** from a hedged aside to the first
  recommendation, with its 100% recall stated up front (W10)

**Citations added, each verified against source rather than taken from a review**
| | venue |
|---|---|
| Wallat et al. | ICTIR 2025, doi 10.1145/3731120.3744592 |
| Ju, Wang & Zhang | arXiv:2604.25313 |
| Xu et al. | Findings of EMNLP 2025, arXiv:2510.13272 |
| Steen et al. | ACL 2023 short, pp. 914–924 |
| Saad-Falcon et al. (ARES) | **NAACL 2024**, 2024.naacl-long.20 — the review dated it 2023, which is the preprint |
| Murugaraj et al. (RAGVUE) | EACL 2026 demos, 2026.eacl-demo.35 |
| Dasigi et al. (QASPER) | NAACL 2021 — **still unverified against the Anthology** |

**Sinha repositioned.** The preprint was supporting a background claim in
Related Work. Its actual finding — embedding detection at 0% FPR on constructed
hallucinations and 100% on natural ones — is the best available caveat on our
own probe, which constructs its errors. Now cited in Limitations for that,
flagged as unreplicated. Citing a weak source to weaken your own result costs
nothing if it is wrong.

**Defects fixed**
- `\author` used single backslashes, so the preprint's title page read
  "orbek Boltabaev" with an email ending "@gma"
- Table 6 was a single-column table overflowing **into the Limitations text**
- abstract's "5-14% against 57-71%" was NLI-max across both datasets while the
  preceding clause was NQ-only; a reader checking the NQ column finds 5.1-25.1%
- the nulls read as findings where first stated; the power result is cited there
  now
- "its 512-token window" had an ambiguous referent
- document now has **zero** overfull boxes

**Code**
- `build_prompt(style=)` + `PROMPT_STYLES`; default byte-identical and pinned by
  `tests/test_prompt_styles.py`, because a prompt difference *between*
  generators would confound H3
- `run_pipeline --open-label --open-prompt-style`
- `faithfulness.py` and `claim_faithfulness.py` gain `--generators`. They read a
  hardcoded constant, so the control arm would have been **silently skipped** —
  B10 and the QASPER `--datasets` trap for the third time
- seven CLIs now take `--datasets` from `config.DATASETS`
- `CLAIMS.md` maps all 28 method claims to file and function, after a code check
  found no discrepancies but could not verify 22 of 23 claims
- `scripts/`: entity, power, refusal-score and verbosity-control tables

---

## 7. What is actually left

1. **Generator coverage (n=3)** — the review's named downside risk, and the only
   weakness writing cannot address. Needs a fourth model end to end.
2. **Verify `dasigi-etal-2021`** against the ACL Anthology. It is the one
   citation added in this period that has not been through the B3 check, and §6b
   of CLAUDE.md is a list of four wrong titles and five wrong attributions.
3. Constructed-vs-natural error gap is unquantified (review comment 3 suggests
   50–100 hand-annotated natural errors).
4. Re-ranking is one λ on one system (review weakness 10).
5. Context-ablation rows were never scored for faithfulness.

## Process notes worth keeping

- `results.equivalence_table()` writes its checkpoint on every call and takes no
  model filter. An exploratory run overwrote the published 4-model artifact;
  regenerated exactly, but **copy before exploring**. Same trap as
  `conditional.py` and `faith_gap_significance_*.pkl`.
- The probe run rewrote `perturb_summary{,_align,_claim}.pkl` on gpu1 to describe
  `qwen_verbose`. The main-grid versions are preserved at
  `n1000_v3/pre_w2verbose_2026-09-18/`.
- Three times now a run has done nothing and reported success because a
  hardcoded default excluded the thing being asked for. Any new arm needs its
  label or dataset threaded through **and printed at startup**.
