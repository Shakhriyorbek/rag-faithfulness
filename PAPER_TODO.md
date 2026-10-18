# What still has to change to finish the paper

Written 2026-08-26, after the AlignScore and claim-level runs. Read alongside
`reports/2026-08-25_perturbation_check.md` and
`reports/2026-08-26_evaluator_dependence.md` — every number cited here comes
from those.

**v6 is written.** Source is **`paper/acl/main.tex`** since the ACL conversion
of 2026-09-05 — `paper/build_v6.js` and its .docx are superseded and an edit
there will not reach the manuscript (see §7). It rewrites the
paper around the measurement spine described below: new title, evaluator
sensitivity as the lead result, three evaluators reported everywhere, the TOST
margin as a curve, and correctness explicitly flagged provisional.

`paper/RAG_Faithfulness_v5_results.docx` is superseded; its title is
contradicted by its own data. Kept for diffing only.

**All four items this header used to list are now closed:** the LLM judge ran
2026-08-30, the open-weight arm ran 2026-09-05, claim-level covers the full grid
as of 2026-09-04, and all 24 references were verified 2026-09-03. The paper was
updated for the last two on 2026-09-08. What is still open is in the block
below — page length first.

---

## Next session starts here — rewritten 2026-10-16

**The fourth generator is in and the paper is updated for it.** Mistral-7B ran
2026-10-15 (CLAUDE.md B28) and closed the review's one weakness that writing
could not fix. Latest reports: `reports/2026-10-07_reviews_and_verbosity_control.md`,
then the B28 block in CLAUDE.md.

| | |
|---|---|
| Paper | `paper/acl/main-preprint.pdf` — **send this one to Berend** |
| Build | 4 variants, **0 undefined refs, 0 overfull boxes, 0 rendered ??**; 230 tests pass |
| Headline | **4 comparisons in 3 of 8 cells, Holm over 24** (was 2 of 6 over 18) |
| Grid | **32,000 answers**, 8,055 refusals (25.2%) |
| Generators | four: Claude Haiku 4.5, GPT-4o-mini, Qwen2.5-7B, Mistral-7B |
| Experiments | all run; nothing paid remains |

### What is actually left, in priority order

1. ~~Berend's sentence-level simplification pass~~ — **done 2026-10-16.**
   Measured, not eyeballed: sentences of 45+ words went **28 to 0**, 40+ went
   **49 to 11** (the rest are dense table captions). 61 edits, no number moved;
   the rule was to split clauses and never drop a hedge, a population label or a
   scope qualifier. The scan is reproducible — see B30.
2. ~~Review items C4, C9, C10~~ — **done 2026-10-16.** C4 reframes the spread
   as a deliberately extreme sensitivity check (the old practitioner
   justification was simply wrong) and drops a sentence that contradicted the
   omnibus appendix; C9 puts claim-min's thresholdability cost into the
   recommendation and states its aggregation as a design choice; C10 reconciles
   constructed-vs-natural where the margin is anchored, and names the direction
   of the error — permissive, never strict. The review text is in
   `Review of When the Evaluator.txt`; **every C item is now closed.**
3. ~~Verify `dasigi-etal-2021`~~ — **done 2026-10-16** against ACL Anthology
   2021.naacl-main.365. Title, six authors in order, booktitle, pages, publisher
   and DOI all match; `month = jun` added. All 31 entries have now had the B3
   check.
4. **Decide the venue.** ⚠️ **The only item above that is still open, and it is
   yours rather than a task.** A hard 8-page limit is served by
   `main-8page.tex`, which already exists as the compact build of the same
   source (30 pages, body ~14).

### ~~One layout decision left behind by B30~~ — decided 2026-10-18

`tab:2` now carries all four generators, as a six-row `table*` at
`\footnotesize` with four column blocks. The row-per-generator-evaluator
alternative was tried first and reverted: it overflowed the column by 20.6pt
and cost three pages, against one page for the wide form. Both falsification
tables are ordered by assertions per answer, most first, so the bolded extremes
are the outer blocks. All 72 rendered values re-checked against the
checkpoints, 0 mismatches.

### ~~Two appendix tables are still three-generator~~ — closed 2026-10-18

Both now carry four generators, ordered by assertions per answer like the two
falsification tables. Every previously published value was reproduced from
`~/rag-backup/checkpoints/n1000_v3` before a new row was added — all 30 values
of `tab:3` and all 90 of `tab:9` matched to the last digit, and so did the whole
24-comparison Holm family behind `tab:7` (the four survivors at 0.0072, 0.0115,
0.0242 and 0.0378). After the edit, 180 rendered values re-checked against the
reference: 0 mismatches.
1. **`tab:3`** (`app:assertions`) — Mistral-7B at 1, 2, 3, 4, 5+ assertions
   responds by +0.385, +0.135, +0.077, +0.031, +0.082 under NLI-max and
   +0.385, +0.246, +0.080, +0.075, +0.139 under claim-min, on n = 422, 201, 97,
   50, 30.
2. **`tab:9`** (`app:tost`) — NQ/Mistral 0/6, 0/6, 4/6, 6/6, 6/6 under NLI-max;
   HotpotQA/Mistral is the only cell in the table where $\pm$0.10 still leaves
   a pair non-equivalent, under NLI-max and claim-min both.

Five prose figures were stale behind them and are fixed: the abstract's
three-generator single-assertion list, the median answer lengths (55, 25, 20,
15 words), the single-assertion word means (23.9, 19.5, 17.2, 16.1), the
"third generator" paragraph, and the single-assertion identity count
(2,472 → 3,249, max 2.33e-05; the three-generator 2,472 at 1.66e-05 was
confirmed first). **The `tab:9` body sentence was wrong independently of
Mistral** — see error 9 in CLAUDE.md B30.
5. ~~Mistral's jina cells have no NLI~~ — **done 2026-10-16**. The control now
   covers four generators and five systems: 8 comparisons in 5 of 8 cells,
   against 4 in 3 of 8 for the matched four.
   ~~Still open and paid: the LLM judge was never run on Mistral~~ — **done
   2026-10-16**: 8,000 rows, $13.74, 0 errors, 0 ungradable. Every
   correctness-conditioned result is now a four-generator result over 32 judged
   cells, and `faith_gap` over that family gives 8 significant, all negative
   (was 5 of 24). See CLAUDE.md B29. **Nothing paid remains open.**
6. Optional: quantify the constructed-vs-natural error gap (50-100 hand-annotated
   natural errors). This is the caveat Sinha supplies and we cannot size.
7. Optional, free: context-ablation rows were never scored for faithfulness;
   re-ranking is still one lambda on one system.

### Do not regress

- **Prompt parity.** `build_prompt`'s default is byte-identical across all four
  generators, pinned by `tests/test_prompt_styles.py`. The two open-weight arms
  differ only in weights; `qwen_verbose` shares qwen's weights and is a control,
  not a generator.
- **Assertion figures are answered-only** (B27), and `perturb_report`'s
  assertion table is **all rows** — do not read one for the other. The paper's
  single-assertion figures (0.102/0.409/0.485) come from the **NQ-only
  identity-check subset**, n=91/413/680, not the pooled assertion bucket.
- **State the population and the aggregation for every figure.** Three slips of
  this class are on record (B13, B27, and the two caught in B28).
- **Copy derived checkpoints before exploring.** `results.equivalence_table()`
  and `conditional.py` overwrite artifacts describing a different population.
  Mistral's probe checkpoints are preserved at
  `checkpoints/n1000_v3/preserve_mistral_2026-10-16/`.
- **`~/rag-backup` and gpu1 both hold the Mistral arm**, generation, scores and
  the 8 judged checkpoints (verified 8,000 judged rows on each, 2026-10-16).
  The judged files were written laptop-side and pushed **up** to gpu1, which is
  the reverse of the usual direction — check both trees after a judge run.

---

## 0. The one-line summary

The v5 paper says *"Retrieval Quality Predicts Correctness, Not Faithfulness."*
The data no longer supports that. What the data does support is that **the
answer to that question depends on which faithfulness evaluator you use** — and
that is a better paper.

---

## 1. Framing — do this first, everything else follows

### 1.1 The title must change

`"Retrieval Quality Predicts Correctness, Not Faithfulness in RAG"` is dead:

- it is **false on HotpotQA**, where faithfulness tracks NDCG (rho +0.40 to
  +1.00 across all four generator × evaluator combinations);
- it is **evaluator-dependent** — 2 of 4 cells flip between `nli_max` and
  AlignScore;
- the external review objected to precisely this sentence, calling it a
  general law where the evidence is an empirical result about four models.

### 1.2 The new spine

Make the paper about **measurement validity in RAG faithfulness evaluation**,
with the embedder grid as the *testbed* rather than the research question.
Nothing gets thrown away — the grid, the conditions, the 2×2 all stay; they
stop being the claim.

Six results support this spine, all measured:

| # | result | where |
|---|---|---|
| 1 | The embedder verdict flips with the evaluator — 2/4 cells | evaluator_dependence §1 |
| 2 | `nli_max` detects 4 % of falsified values on Claude/NQ; AlignScore 29 % | perturbation §8 |
| 3 | Verbose multi-claim answers defeat every evaluator tested | perturbation §9 |
| 4 | Pooling abstentions manufactures a retrieval–faithfulness relationship | rung2 §2 |
| 5 | nRFG's cross-generator agreement is an algebraic artifact | rung2 §6 |
| 6 | RFG runs negative — its core assumption is empirically backwards | rung2 §7 |

### 1.3 State each of those as a general result, never as autobiography

This is Berend's "dead ends rarely belong in a paper", and it is a rewriting
instruction, not a deletion instruction:

> ❌ "We initially used nRFG, obtained a spurious result, and then fixed it."
> ✅ "Any metric of the form 1 − F/RQ reproduces the retrieval ranking when F
>    is near-constant, so apparent cross-generator agreement in such metrics
>    is uninformative."

Same content. The second is a finding another researcher can hit; the first is
a lab notebook. Apply this to all six.

### 1.4 Motivation

Open with the document-QA case — an invoice says $1,000 and the system must not
say $1,500. One paragraph. It gives the paper a concrete stake and makes the
perturbation experiment intuitive. Do **not** make the product the subject.

---

## 2. Section-by-section

| section | change |
|---|---|
| **Title / abstract** | Rewrite per §1.1–1.2. Lead with evaluator dependence. |
| **§2 related work** | ✅ Salemi & Zamani (SIGIR 2024): **full PDF read and §II-A rewritten 2026-09-05** — the old paragraph overstated them (their weak-correlation result is about human/LLM relevance labels; answer-containment, which is our qrels scheme, holds tau 0.31-0.40, and their downstream measure is correctness throughout). See `reports/2026-09-05_salemi_zamani_full_read.md`. Add RAGAS (Es et al., EACL 2024) and distinguish it: your contribution is conditioning on retrieval success and the attribution decomposition, not the faithfulness measure itself. Verify both citations. |
| **§3.2 RFG / nRFG** | Keep the v5 demotion. Add §1.3-style general statements. |
| **§3.3** | Keep necessity/sufficiency, but flag it as provisional pending the LLM judge (see §3 below). |
| **§4.3 faithfulness** | Must now **name the evaluator** in the definition. Add claim-level: `F = min over claims of max over chunks`. Report all three evaluators wherever a faithfulness number appears. |
| **§4.4 correctness** | Keep the EM = 0.000 admission. Replace containment numbers once the judge runs. |
| **§4.6 TOST** | Replace the asserted ±0.05 with the margin sweep (0.01–0.10) from `compare_evaluators.py`. Anchor it: falsifying one grounded value moves `nli_max` by 0.033–0.094, so 0.05 ≈ one fabricated fact. |
| **§6 results** | Rewrite entirely. The four-null table is superseded. |
| **§6.3 abstention** | Keep the finding, restate per §1.3 — the review called this the paper's best contribution. |
| **hypotheses** | H1 not supported. H2 *partially supported* — the retrieval–faithfulness correlation is positive on HotpotQA, absent on NQ (this is new; H2 was framed via RFG). H3 not supported, and note the comparison was ill-posed. H4/H5 not run. |
| **limitations** | Add: 4 embedders, 2 datasets, 2 closed generators, one language; the NQ sampling filter; effect sizes small even where significant. |

---

## 3. Experiments still needed

Ordered by how much they unblock. Items 1–3 are the ones a reviewer will
demand.

1. **LLM-judge correctness** — *cheap, blocks the most.*
   EM = 0.000 on all 4,000 Claude/NQ rows, so every accuracy figure and the
   whole 2×2 rests on containment, an upper bound. Until this runs, §3.3 and
   §6.4 cannot be stated as measurements. This has been the top item since
   2026-08-14.

   **✅ DONE 2026-08-30** — 16,000 rows, 0 errors, $29.27. Results and the
   consequences for the draft: `reports/2026-08-30_judge_full_grid.md`.
   Headlines: containment understated accuracy by 5-15 points everywhere;
   "more faithful when wrong" is **false on NQ for both generators**; the
   "not sufficient" cell is ~80% refusals, so the grounded-and-wrong rate is
   4.5-7.9%, not 23-45%. Use `--correct-source judge` from here on.

   **Unblocked 2026-08-30.** The judge previously wrote `*_judged_*` that
   nothing read, so the ~$32 run would have moved no number. `conditional.py`
   and `results.py` now go through `correctness.load_correctness`:

   ```bash
   python3 src/conditional.py --datasets NQ,HotpotQA --correct-source judge
   ```

   It overlays the judge per row, falls back to containment where the judge has
   not run, and prints `judged N, heuristic N, ungradable N` so the difference
   is visible. Calibrate before spending: the only judged checkpoint in the
   backup is **4 rows of `AuthenticationError 401`** — the $0.01 calibration in
   the 2026-08-28 status report is not evidenced by any artifact on this
   machine, so re-run the 200-row calibration and `--compare-only` first.

2. **An open-weight generator — Qwen or Gemma.** *Free on the V100.*
   Berend asked for it for reproducibility, and under the new framing it is
   also load-bearing: the effect is mediated by answer verbosity, so a third
   generator with a different verbosity profile is a direct test of the
   mechanism, not just a robustness check.

   **Unblocked 2026-08-30.** `Llama3Generator` was hardcoded; it is now
   `LocalHFGenerator(model_id)` with `config.OPEN_MODEL_ID =
   'Qwen/Qwen2.5-7B-Instruct'` — **ungated, so `HF_TOKEN` is no longer a
   blocker** — and `config.OPEN_MODEL_LABEL = 'qwen'` scoping the checkpoint
   key. `qwen` is already in `GENERATORS` in all five analysis modules, so the
   arm gets scored rather than generated and forgotten. Run:
   `python3 -u src/run_pipeline.py --full --phases llama,d --datasets NQ`

3. **Claim-level over the full grid.** *Free.*
   Only the 200-case perturbation subset has been scored. `claim_scores_*`
   checkpoints do not exist. Run:
   `python3 -u src/claim_faithfulness.py --scope-n 1000`
   Then `compare_evaluators.py --metrics nli,align,claim` gives the full
   three-way table.

4. **Literal value grounding.** *Free, not yet built.*
   Every number/date/entity in the answer must appear in the context.
   Deterministic. `src/textnorm.py` already has the right normalisation.
   Needed for the document-QA argument: 29 % detection is not a usable gate.

5. **C1 / C2 floor and ceiling.** ✅ **RUN 2026-09-12** (Claude arm only); in
   the paper as Section 9. *The "never run" text below is history.*
   `pct_of_oracle` is NaN and the 2×2 has no anchors. C1 also tests a validity
   threat: NQ and HotpotQA are Wikipedia, so the generators may answer from
   memory rather than from the retrieved context.

6. **QASPER** — third dataset, ~$5. Also the least Wikipedia-like of the three.

7. **ESA and re-ranking** — ✅ **both RUN 2026-09-12**, both negative, both in
   the paper as Section 10. *The text below is history.* Originally: two of the
   four original contributions, never run.
   **Decide with Berend whether they stay in the paper at all.** Under the new
   spine they are optional; under the old one they were required.

**Dropped:** Shapley / `doc_utility.py`. Berend explicitly released you from it
("not that important... an underspecified rough idea").

**⚠️ "More faithful when wrong" is dead (2026-08-30).** Permutation-tested in
all 16 cells with Holm correction: **3 significant, all NEGATIVE**; zero
positive cells reach significance and none has a CI excluding zero. The
surviving effects say answers are more grounded when they are *correct*.
NQ/E5-large-instruct replicates across both generators (-0.113, -0.115) and is
the most defensible number in the grid. Remove the claim from the draft; do not
soften it. `reports/2026-08-30_faith_gap_significance.md`.

**Two numbers changed on 2026-08-30, both against the paper.** `faith_gap` used
a one-sided baseline: containment grades ~7-12% of Claude's *correct* rows as
correct while they are actually abstentions, and those sat in the baseline but
not in the numerator. Fixed, **Claude/NQ falls from +0.018..+0.050 to
-0.006..+0.018** — the "more faithful when wrong" line does not survive on NQ,
only on HotpotQA. Separately, the tables pooled NQ and HotpotQA, which made the
remaining effect look like a generator difference when it is a dataset one.
Anything in the draft resting on either has to be restated per dataset, and the
"more faithful when wrong" sentence has to be scoped to HotpotQA or dropped.

**Also settled 2026-08-30, and it changes a reported number:**
`conditional.GENERATORS` omitted `gpt4omini`, so every table in §3.3/§6.4
described the Claude arm only while ignoring 8,000 paid GPT-4o-mini rows. With
both arms in the frame, *hit × incorrect* on NQ is 23.3 % for Claude but
**46.5 %** for GPT-4o-mini. Whatever §8 says about that cell has to be restated
per generator, not pooled — and the near-doubling is itself a result worth a
sentence, since it is the same retrieval feeding both.

---

## 4. External review red flags — status

From the ChatGPT review Berend commissioned (2026-08-19).

| | red flag | status |
|---|---|---|
| 🔴 1 | Single NLI judge, AlignScore not run | ✅ **done** — AlignScore run, changes the conclusion |
| 🔴 2 | Max-over-chunks aggregation | ✅ **done** — claim-level built, mechanism confirmed |
| 🔴 3 | Conditioning on "all models answered" (collider) | ✅ **done** — measured 2026-09-12, bound stated in Limitations; the 3-way table was deliberately not built (see memory) |
| 🔴 4 | Selective NQ sampling | ✅ **stated** — Limitations bullet; cannot be fixed without redrawing the corpus. The C1 floor now quantifies the related Wikipedia dependence at 32.2% answerable with no retrieval |
| 🔴 5 | Causal language exceeds identification | ✅ **done** — swept in the v6 rewrite |
| 🟠 6 | ±0.05 TOST margin unjustified | ✅ **done** — margin sweep + empirical anchor |
| 🟠 7 | Correctness metric inadequate | ✅ **done** — LLM judge over all 32 matched cells (24 on 2026-08-30, Mistral 2026-10-16) |
| 🟠 8 | Only 4 embedders | ✅ **five now** — and the fifth is a positive control, not a survey entry |
| 🟠 9 | Only 2 English benchmarks | ✅ **three now** (QASPER); all still English, stated as a limitation |
| 🟠 10 | Only 2 closed-source generators | ✅ **done** — Qwen2.5-7B open-weight arm |
| 🟡 11 | Multiple-testing presentation | ✅ **done** — all three families printed on every run |
| 🟡 12 | RFG/nRFG conceptually weak | ✅ becomes a result rather than a liability |
| 🟡 13 | Many experiments mentioned, not run | ✅ **all of them have now run** and all are in the draft |

---

## 5. Berend's own points

| his point | action |
|---|---|
| "the paired TOST you pointed me towards" | **Correct the misattribution explicitly in the next email.** TOST was not his suggestion; the reply draft wrongly credited him. |
| Wall of text | The reply was 2,444 words. Cut it hard. |
| Heavy genAI, "surface without substance" | See §6 below. |
| No coherent narrative | §1.2 supplies one. |
| Too-strong language / overclaiming | Sweep for: "governs", "predicts", "shows that", "demonstrates". Prefer "across the models tested". |
| Dead ends don't belong | §1.3 — restate, don't delete. |
| Open-weight generator | §3.2. |
| Shapley unimportant | Dropped. |
| Non-archival workshop | Reasonable target. The review's verdict was "major revision"; a workshop gets feedback without spending a submission. |

---

## 6. The prose problem

The review named the fingerprints precisely: `"not X, but Y"`, `"nearly
necessary, clearly not sufficient"`, and a section progression it called
"almost too tidy". Those are real and they are in the v5 draft.

Its conclusion is the useful part: *human-designed experiment + LLM-assisted
writing*, saved by concrete substance — the NLI label-index bug, the abstention
artifact, the nRFG algebraic trap. **The substance is real; the polish reads as
hollow.**

So: rewrite the prose in your own voice, and deliberately break those rhythms.
Do not ask a model to "make it sound less AI" — that produces the same problem
one layer down. Write the sections from the reports, in your own words, and use
the model to check facts and numbers rather than to phrase them.

---

## 7. Formatting — ✅ done 2026-09-05, but the paper is now over length

The paper is **ACL format**, source at `paper/acl/main.tex`, built with
`paper/acl/build.sh` (tectonic; no TeX Live needed). It compiles clean — 0
undefined citations, 0 overfull boxes — and all 234 prose fragments were checked
to survive the conversion verbatim. `build.sh` emits both the anonymous
line-numbered ARR file (`main.pdf`) and a named reading copy
(`main-preprint.pdf`). `paper/build_v6.js` and its docx are superseded.

**⚠️ The blocking formatting problem is now length, not style.** Content
(Sections 1–10) was **9.4 pages against ARR's 8-page limit** when this was
written; the body is now ~14 pages and the limit is deliberately not enforced
(see the header). Limitations and
References do not count, so ignore the 12-page total. Roughly **1.5 pages must
come out** — over-length is a desk reject, not a reviewer complaint.

**Venue is decided** (see `reports/2026-09-05_berend_report.md` §2.3, pending
Berend's agreement): target **ARR 12 October 2026**, which feeds both
NAACL 2027 (CORE A) and COLING 2027 (CORE B) from one submission, with the
primary venue chosen at commitment on 20–23 December once reviews are in. The
author is comfortable with a B/C-tier outcome, which is an argument *for* this
deadline rather than for a smaller venue now — COLING is the B-tier outcome and
costs nothing extra to keep in reach. ESA and re-ranking are dropped rather
than delaying a cycle for them.

---

## 8. Numbers that must be replaced wherever they appear

- The **four-null** faithfulness table → now 2 nulls, 2 significant (AlignScore).
- **"22/24 equivalent at ±0.05"** → the margin sweep table.
- Every **accuracy / 2×2** figure → ~~provisional until the LLM judge runs~~
  **judged, all 32 matched cells** (2026-10-16). The 2×2 is now generator-major
  with eight rows; cell size 17-48%, committed-and-wrong 4.5-11.4%.
- **H3 "not supported"** → keep, but add that the comparison was not measuring
  a stable quantity, since evaluator sensitivity differs ~10× between
  generators.
- The **42,000-triple** claim was already cut in v5. Current: **16,000
  generations**, 4 embedders × 2 datasets × 2 generators × 1,000 queries.
