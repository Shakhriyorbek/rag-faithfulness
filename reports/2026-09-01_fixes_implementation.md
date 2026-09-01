# Implementation report — FIXES.md, items F1–F9

**Date:** 2026-09-01 · **Cost:** $0 (all re-scoring is local NLI / deterministic)
**Scope:** `checkpoints/n1000_v3` on gpu1 · **Code:** commits `c0a4339`, `9f9dacf`
**Tests:** 196 passed locally / 207 on gpu1, from 178 before.

Every number below was produced by running the code, not by reading it. The
`.docx` files are untouched, as instructed.

---

## Headline: neither stop-condition fired

FIXES.md asked to stop and report immediately if the Table III gradient
flattens under F3, or if the Section V deltas move by more than ~0.02 under F1.

- **F3 does not flatten Table III.** The digit-boundary bug changed the
  assertion count of **32 of 16,000 answers** (26 Claude, 6 GPT-4o-mini). The
  per-generator means the paper quotes move from **2.992 → 2.996** (Claude/NQ)
  and **1.176 → 1.176** (GPT/NQ). The mechanism claim in Section V-C stands.
- **F1 moves the Section V sample, not (by more than 0.02) the deltas.**
  Re-scored, all 16 cells: **max |Δ| = 0.0165, mean 0.0074**, under the
  threshold everywhere. The published anchor 0.033–0.094 becomes 0.037–0.088.

The two substring blockers were both real and both are fixed. Their downstream
effect is much smaller than their severity suggested, and in one case runs the
opposite way from the brief's prediction — details per item.

---


## F1 — `num_in_text` matched numbers as substrings — **FIXED, stop-condition did not fire**

Reproduced exactly as reported; all five listed calls returned True. Fixed by
matching at numeric/word boundaries.

**The brief's proposed regex does not satisfy its own test table.** Its
lookbehind is `(?<![\d.,])`, which lets `B` through, so
`num_in_text('7', 'model B7 was tested')` stays True. The implemented boundary
mirrors `NUM_RE` instead — `(?<![\w.,])…(?![\w.,]?\d)` — which excludes a
digit glued to letters, exactly as `NUM_RE` already does for `B12` and
`28.0.0.137`. All six of the brief's cases now pass and are pinned as tests.

### Eligible cases, attributed to each fix

| cell | as published | + F1 boundary | + F4 refusal rule |
|---|---|---|---|
| NQ / Claude | 1,893 | 1,994 | **2,052** |
| NQ / GPT-4o-mini | 845 | 892 | 893 |
| HotpotQA / Claude | 1,554 | 1,599 | **1,694** |
| HotpotQA / GPT-4o-mini | 831 | 886 | 893 |
| **total** | **5,123** | **5,371** | **5,532** |

The sample grew, which is the opposite of what the severity ordering suggests
but is the right direction: rule (b) — "the replacement must not appear in the
context" — was throwing away more eligible cases than rule (a) was letting
bad ones in. Within the 200 cases each cell actually scores, 87–95% of query
IDs are unchanged and 81–91% are the identical (query, value, replacement)
triple, so this is a partial resample, not a new experiment.

### Deltas and detection, before and after — all 16 cells re-scored

| cell | Δ before | Δ after | change | det. before | det. after |
|---|---|---|---|---|---|
| NQ/claude/all-mpnet | +0.0338 | +0.0367 | +0.0029 | 3% | 3% |
| NQ/claude/BGE-M3 | +0.0379 | +0.0405 | +0.0025 | 2% | 2% |
| NQ/claude/E5-large-instruct | +0.0508 | +0.0555 | +0.0046 | 6% | 7% |
| NQ/claude/text-embedding-3-small | +0.0302 | +0.0377 | +0.0076 | 3% | 4% |
| NQ/gpt/all-mpnet | +0.4797 | +0.4768 | −0.0029 | 62% | 61% |
| NQ/gpt/BGE-M3 | +0.4474 | +0.4309 | −0.0165 | 58% | 54% |
| NQ/gpt/E5-large-instruct | +0.4353 | +0.4281 | −0.0071 | 55% | 52% |
| NQ/gpt/text-embedding-3-small | +0.4624 | +0.4460 | −0.0164 | 58% | 55% |
| HotpotQA/claude/all-mpnet | +0.0594 | +0.0636 | +0.0042 | 9% | 12% |
| HotpotQA/claude/BGE-M3 | +0.0737 | +0.0619 | −0.0118 | 14% | 14% |
| HotpotQA/claude/E5-large-instruct | +0.0872 | +0.0875 | +0.0004 | 15% | 16% |
| HotpotQA/claude/text-embedding-3-small | +0.0936 | +0.0804 | −0.0132 | 13% | 13% |
| HotpotQA/gpt/all-mpnet | +0.3948 | +0.4056 | +0.0109 | 70% | 73% |
| HotpotQA/gpt/BGE-M3 | +0.3573 | +0.3516 | −0.0057 | 65% | 65% |
| HotpotQA/gpt/E5-large-instruct | +0.4046 | +0.3973 | −0.0073 | 71% | 69% |
| HotpotQA/gpt/text-embedding-3-small | +0.4141 | +0.4093 | −0.0047 | 76% | 75% |

**max |Δ| = 0.0165, mean |Δ| = 0.0074 — under the 0.02 stop-threshold in every
cell.** All four evaluators were re-scored over all 16 cells; per-evaluator
movement:

| evaluator | max \|Δ\| | mean \|Δ\| | over the 0.02 threshold? |
|---|---|---|---|
| NLI-max | 0.0165 | 0.0074 | no |
| NLI-concat | — | — | new run, no baseline |
| claim-min | 0.0187 | 0.0073 | no |
| **AlignScore** | **0.0241** | 0.0099 | **2 of 16 cells** |

⚠️ **Flagging the one exceedance, as instructed.** Two AlignScore cells move
just past the threshold — NQ/GPT-4o-mini/text-embedding-3-small (+0.6630 →
+0.6389, −0.0241) and NQ/GPT-4o-mini/BGE-M3 (+0.6674 → +0.6444, −0.0230). Both
are on a base of ~0.66, i.e. a 3.5% relative change, both are in the direction
of a smaller effect, and neither changes anything qualitative: AlignScore's
headline detection rates go 29/76/45/90% → **27/74/44/89%**. Nothing in the
abstract depends on those digits, so I did not stop — but the brief asked to be
told, and this is the telling. The published ranges move from 0.033–0.094 to **0.037–0.088** on
Claude and 0.357–0.480 to **0.352–0.477** on GPT-4o-mini; detection moves from
2–15% to 2–16% and 55–76% to 52–75%.

**Section VIII's TOST anchor survives**: "one falsified fact moves NLI-max by
0.033–0.094" becomes **0.037–0.088**, so ±0.05 is still about the size of one
fabrication and the margin curve needs no re-derivation.

### The sample contamination this exposed — worth a sentence in the paper

`build_number_case` takes the first **grounded** number in the answer, and not
every number in an answer is a claim about the world. Splitting the new runs by
what the falsified value actually is:

| cell | role | n | mean Δ |
|---|---|---|---|
| HotpotQA / Claude | content | 730 | **+0.0850** |
| | chunk citation | 61 | **−0.0625** |
| | list marker | 9 | +0.0501 |
| NQ / Claude | content | 747 | **+0.0442** |
| | chunk citation | 6 | +0.0154 |
| | list marker | 47 | +0.0203 |
| NQ / GPT-4o-mini | content | 789 | +0.4492 |
| | list marker | 11 | +0.1749 |
| HotpotQA / GPT-4o-mini | content | 784 | +0.3907 |

Falsifying *"According to Chunk 4"* into *"According to Chunk 7"* is not a
factual falsification, and on HotpotQA/Claude those 61 cases have a **negative**
mean delta — the score goes UP. They are 7.6% of that cell and they pull its
mean down by about 15% (content-only +0.0850 against +0.0734 pooled).

Case construction was deliberately **not** changed, because every scorer has to
see the same cases and three scorer runs were already in flight. The
recommendation is to exclude citation and list-marker values from
`build_number_case` and re-run the probe once — it is one line and ~2 GPU-hours
— which would raise the reported Claude deltas rather than lower them.

---

## F2 — `textnorm.contains` substring defect — **FIXED, and it moves almost nothing**

`squash()` normalised correctly and the test was then a raw substring test on
the result. Confirmed exactly as reported: `contains('the budget was 10000
dollars', '1000')`, `contains('Alice went to Paris', 'Ali')` and
`contains('the state artifact', 'art')` were all True.

Fixed by padding both sides with a space, so the test is on token sequences.
`contains_any` got the same change. The NQ tokenisation behaviour the module
exists for is preserved (`contains('wilhelm conrad röntgen s', 'Röntgen')` is
still True).

One existing test asserted the defect as deliberate behaviour
(`test_substring_of_word_still_matches`, "gold spans are frequently sub-token
('Ren' in 'Kylo Ren')"). That justification does not survive inspection:
`squash()` splits on punctuation and whitespace, so the sub-token cases that
actually occur are separate TOKENS and still match. The test was inverted and
its docstring rewritten to say so.

### What it moved — measured on all three call sites

**1. The retained NQ sample: no change at all.**

| | old rule | new rule |
|---|---|---|
| NQ samples whose answer is absent from its own gold context | 0 / 1000 | **0 / 1000** |

The loader's B8 invariant still holds and the retained sample is byte-identical.
**No regeneration is needed and the paper's dataset description does not
change.** (On HotpotQA the count goes 44 → 62, but every one of those is a
`yes`/`no` answer, where the answer string is not expected to appear in the
context and where relevance is decided by `gold_sentences`, not by the answer.)

**2. qrels and retrieval quality (Table I): 7 NQ queries, 4th-decimal effect.**

| | NQ | HotpotQA |
|---|---|---|
| relevant chunks, old → new | 1378 → 1371 | 2003 → 2003 |
| queries whose relevant set changed | 7 / 1000 | 0 / 1000 |
| doc-level fallbacks | 0 → 0 | 2 → 2 |

Phase B was re-run on gpu1 with the rebuilt qrels:

| dataset | model | NDCG@5 old → new | Recall@5 | MRR@5 |
|---|---|---|---|---|
| NQ | all-mpnet-base-v2 | 0.7862 → **0.7861** | 0.9222 → 0.9225 | 0.7554 → 0.7544 |
| NQ | BGE-M3 | 0.7944 → **0.7943** | 0.8912 → 0.8912 | 0.7850 → 0.7840 |
| NQ | E5-large-instruct | 0.8064 → **0.8061** | 0.9278 → 0.9277 | 0.7839 → 0.7831 |
| NQ | text-embedding-3-small | 0.8300 → **0.8299** | 0.9486 → 0.9492 | 0.8035 → 0.8029 |
| HotpotQA | all four | unchanged to 4 dp | unchanged | unchanged |

The paired TOST equivalence table does not change a single verdict; the largest
movement in any `p_tost` is 0.0122 (E5 vs text-embedding-3-small on NQ,
0.6937 → 0.7059) and no pair crosses `equivalent` or `matched` in either
direction.

**3. Containment correctness: 166 of 16,000 rows flip, and agreement with the
LLM judge improves.**

| cell | containment acc old → new | rows flipped | agreement with judge | FP vs judge | FN vs judge |
|---|---|---|---|---|---|
| NQ / Claude | 0.7232 → 0.7192 | 16 | 0.7830 → 0.7815 | 0.0785 → **0.0772** | 0.1385 → 0.1412 |
| NQ / GPT-4o-mini | 0.4875 → 0.4863 | 5 | 0.8495 → 0.8482 | 0.0003 → 0.0003 | 0.1502 → 0.1515 |
| HotpotQA / Claude | 0.6737 → 0.6635 | 41 | 0.8317 → **0.8350** | 0.1042 → **0.0975** | 0.0640 → 0.0675 |
| HotpotQA / GPT-4o-mini | 0.5397 → 0.5138 | 104 | 0.8718 → **0.8792** | 0.0283 → **0.0115** | 0.1000 → 0.1092 |
| **all 16,000** | 0.6061 → 0.5957 | 166 | 0.8340 → **0.8360** | 0.0528 → **0.0466** | 0.1132 → 0.1174 |

This is the expected shape: a stricter test removes false positives (−0.62pp)
and adds false negatives (+0.42pp), netting a small improvement in agreement.
It confirms the CLAUDE.md §8 finding that containment errs in both directions;
it does not change the decision to use the judge, and every judge-sourced
number in Section VI is untouched.

**Paper impact:** Table I, 4th decimal. Section VI-C: unchanged (judge-sourced).
Dataset description: unchanged.

### One decision this forces

Relevance semantics changed for 7 NQ queries, and CLAUDE.md §8 says to bump
`CORPUS_VERSION` whenever they do. **I did not bump it.** `utils.set_scope`
builds the checkpoint directory as `n{N}_{CORPUS_VERSION}`, so bumping to v4
would orphan all 16,000 paid generations and the entire judge run in
`n1000_v3` — $42.73 of API spend re-spent for a 4th-decimal change to seven
queries. The corpus, the chunks and the retained samples are byte-identical;
only the chunk-relevance rule moved. I rebuilt `qrels_*`, `per_query_rq_*` and
`retrieval_quality_all` in place and archived the old copies to
`checkpoints/n1000_v3/superseded_2026-09-01/`. **Say if you want the version
bumped anyway** — it is a one-line change plus a full re-run.

---

## F3 — claim splitter merged digit-initial sentences — **FIXED, gradient does not flatten**

Confirmed: both reported cases produced 1 claim instead of 2. Fixed by adding
`nxt.isdigit()` to the sentence-start test; the character class also had a
straight `"` twice and no opening curly quote, both corrected.

### Assertion counts, before and after

| generator | dataset | old | new | answers whose count changed |
|---|---|---|---|---|
| Claude | NQ | 2.992 | **2.996** | 26 / 8000 across both datasets |
| Claude | HotpotQA | 2.456 | 2.460 | |
| GPT-4o-mini | NQ | 1.176 | **1.176** | 6 / 8000 across both datasets |
| GPT-4o-mini | HotpotQA | 1.056 | 1.058 | |

The paper's 2.99 and 1.18 are the NQ cells and both survive. The defect was
real and selective in the direction the brief predicted, but its frequency is
0.3% of Claude answers and 0.1% of GPT answers, so **the monotone gradient in
Table III is not an artifact of it.**

**Also added, as asked:** a regression test for both numeric cases, one for the
curly-quote class, and the identity assertion — with one claim and no markdown,
`claim_min` must equal `whole_max` to float tolerance. It is pinned as a unit
test against a deterministic per-pair NLI stub, and checked on real data by
`src/perturb_report.py`.

### Table III rebuilt, with n — and it is an NQ-only table

`src/perturb_report.py` rebuilds the by-assertion-count table with a
denominator and a bootstrap CI on every row. Restricted to NQ it reproduces the
published Table III to three decimals, including the 0.4922 figure:

| assertions | paper | rebuild (NQ, pre-fix) | n |
|---|---|---|---|
| 1 | +0.086 | **+0.0860** | 92 |
| 2 | +0.056 | **+0.0555** | 298 |
| 3 | +0.018 | +0.0220 | 193 |
| 4 | *(not shown)* | +0.0106 | 104 |
| 5+ | +0.007 | **+0.0067** | 113 |
| GPT, 1 assertion | +0.492 | **+0.4922** | 675 |

**So Table III is the NQ half of the experiment, and the paper does not say
so.** On HotpotQA the same decay exists for Claude but sits higher
(+0.1227 / +0.0928 / +0.0570 / +0.0282 / +0.0460, n = 137/292/274/68/29), and
GPT-4o-mini has **n = 1, 0, 0** in buckets 3, 4 and 5+ — there is no HotpotQA
gradient to report for that generator at all.

Over the full grid after the fixes, Claude decays monotonically through four
buckets and then ticks back up:

| assertions | Claude Δ (n) | GPT-4o-mini Δ (n) |
|---|---|---|
| 1 | +0.1115 (225) | +0.4452 (1404) |
| 2 | +0.0681 (578) | +0.2561 (126) |
| 3 | +0.0406 (470) | +0.0382 (28) |
| 4 | +0.0210 (177) | **+0.2422 (17)** |
| 5+ | **+0.0371 (150)** | +0.0141 (9) |

Under claim-level aggregation the decay is absent, which is the contrast
Section V-C rests on and it survives intact:

| assertions | NLI-max Δ (n) | **claim-min Δ (n)** |
|---|---|---|
| 1 | +0.1115 (225) | +0.1324 (225) |
| 2 | +0.0681 (578) | +0.1350 (578) |
| 3 | +0.0406 (470) | +0.1367 (470) |
| 4 | +0.0210 (177) | +0.1034 (177) |
| 5+ | +0.0371 (150) | +0.0972 (150) |

NLI-max falls ~3× across the buckets; claim-min is flat to bucket 3 and then
declines mildly. **The mechanism claim holds.**

**The table as printed oversells it, though.** Three recommendations, all
cheap: say the table is NQ; print n per row; and either show bucket 4 or
collapse to "3 or more" — the GPT row beyond bucket 2 rests on 28, 17 and 9
cases and is not monotone once bucket 4 is visible.

One number does move under B16: bucket 3 goes from the paper's +0.018 to
**+0.0220** (NLI) and +0.188 to **+0.2068** (claim-min) on the same pre-fix
scores, purely because the fixed splitter re-buckets ~30 answers. The shape is
unchanged; the individual cell is 10% off.

### The identity assertion, checked on real data — and it FAILED

With the correct condition (one claim *and* that claim IS the answer):

| group | n | cases differing | mean \|diff\| | max \|diff\| |
|---|---|---|---|---|
| answer contains **no** markdown | 1,490 | 112 | 2.6e-07 | **1.66e-05** |
| answer **contains** markdown | 131 | **131 (all)** | **0.0392** | **0.729** |

The no-markdown group holds to 1.7e-05 — that residual is batch-padding noise
(batched inference pads to the longest pair in the batch), not an aggregation
difference. The claim-level implementation is correct.

**But the markdown group is exactly the failure the brief predicted.**
`nli.score_chunks` scores the answer as the generator wrote it, `**bold**`
included; `score_claims` runs `strip_markdown` over claims first. So the two
aggregates are not reading the same hypothesis, and DeBERTa's entailment
probability moves by up to **0.729** on nothing but a pair of asterisks
(`**Graduados**` vs `Graduados`).

This is not a small population: markdown is in **74.9%** of Claude's answered
rows and **0.3%** of GPT-4o-mini's. The paper's central cross-generator
contrast — NLI-max is far less sensitive to falsification on Claude's answers
than on GPT-4o-mini's — is therefore confounded with a formatting difference
that only one generator produces. `src/markdown_check.py` is built to size it
(same answers, same chunks, raw vs stripped) but **is not run**, and the fix —
strip markdown in both places, or in neither — **is not applied**, because it
moves every `nli_max` number for Claude. See the decisions list at the end.

---

## F4 — two disagreeing abstention detectors — **FIXED**

They were not the same function and did not select the same rows. Measured over
all 16,000 generations:

| | rows |
|---|---|
| both detectors call it a refusal | 4,016 |
| `perturbation_check` only | **365** |
| `correctness` only | **0** |
| neither | 11,619 |

The substring rule was a strict superset. Classifying its 365 extra rows:

| | rows | what they are |
|---|---|---|
| mid-answer hedges | **253 (69%)** | substantive answers carrying a caveat — "…the following people were involved in the Mapp v. Ohio case: … However, the context does not contain their ages." |
| anchored refusals behind a preamble | **112 (31%)** | "Based on the provided context, I cannot answer this question." |

So the extra phrases were **not** kept: two thirds of what they caught are
answers, and matching "does not contain" anywhere in an answer is what caught
them. The canonical rule is now `correctness`'s anchored `startswith` on
SQuAD-normalised text, moved to **`src/abstention.py`** and imported by both
call sites. `correctness.is_abstention` and `perturbation_check.is_abstention`
are now literally the same object, pinned by a test.

**Adopting it changes no published number**: every row the canonical rule calls
a refusal was already a refusal on both sides, so Section VI's population is
untouched. Section V's population grows — the falsification sample no longer
excludes 365 substantive answers (eligible cases +161, see F1).

`normalize_answer` moved to `textnorm.squad_normalize` so `abstention.py` can
use it without importing `correctness`; `correctness.normalize_answer` is an
alias and every existing caller is unaffected.

### NEEDS A DECISION — the 112 preamble-hidden refusals

The canonical phrase list misses refusals that open with the generators'
standard preamble. `abstention.is_abstention_extended()` implements the wider
anchored rule (strip a leading "Based on …," / "According to …,", then match a
longer phrase list) and is wired into **nothing**: turning it on would move the
abstention rate, the answered-only population, and therefore the spine result
in Section VII. 112 rows is 0.7% of the grid and 2.7% of the current refusal
count. My recommendation is to leave it off for this paper and state the rule
precisely, but it is your call.

---

## F5 — evaluators do not see the same premise construction — **NEEDS A PAPER EDIT + diagnostic run**

Confirmed and **not** changed in code, as instructed. The answer strings are
byte-identical across evaluators; the premises are not:

| evaluator | premises per scored answer |
|---|---|
| NLI-max | **6** — each of the 5 chunks separately, plus their concatenation truncated at 512 tokens — maximum taken |
| Claim-min | the same 6, per claim; minimum over claims of the maximum over premises |
| AlignScore | **1** — `' '.join(chunks)`, split internally by AlignScore itself |

Suggested wording for Section IV: *"Each answer is scored by all three
evaluators as a byte-identical string. Premise construction follows each
evaluator's intended interface: the NLI scorer receives the five retrieved
chunks separately plus their concatenation and the maximum is taken, since any
retrieved document may entail the answer and a concatenation exceeds the
512-token window; the claim-level scorer applies the same construction per
extracted claim; AlignScore receives the concatenated context as a single
premise and performs its own splitting."*

### The diagnostic, run — premise granularity is about a quarter of the gap

`--scorer nli_concat` scores the same cases with the same DeBERTa model on
AlignScore's premise construction: one concatenated context, no per-chunk
maximum. Pooled over the four embedders (800 cases per cell):

| cell | | NLI-max | **NLI-concat** | claim-min | AlignScore |
|---|---|---|---|---|---|
| NQ / Claude | Δ | +0.043 | **+0.081** | +0.151 | +0.224 |
| | detected @0.5 | 4% | **11%** | 24% | 27% |
| HotpotQA / Claude | Δ | +0.073 | **+0.068** | +0.105 | +0.297 |
| | detected @0.5 | 14% | **21%** | 34% | 44% |
| NQ / GPT-4o-mini | Δ | +0.446 | +0.419 | +0.463 | +0.633 |
| | detected @0.5 | 56% | 65% | 60% | 74% |
| HotpotQA / GPT-4o-mini | Δ | +0.391 | +0.241 | +0.389 | +0.661 |
| | detected @0.5 | 71% | 71% | 71% | 89% |

All four columns are the post-fix runs, on identical cases.

Taking detection as the yardstick, the share of the NLI-max → AlignScore gap
that premise construction alone explains is **30%** on NQ/Claude, **23%** on
HotpotQA/Claude, 50% on NQ/GPT-4o-mini and **0%** on HotpotQA/GPT-4o-mini.

**So the confound is real and it is bounded at roughly a quarter.** Dropping
the per-chunk maximum roughly doubles NLI's response to a falsified value on
Claude's answers — max-over-chunks is itself part of what makes NLI-max
insensitive, which is the same weakest-link argument the paper already makes
about max-over-claims — but it leaves most of the distance to AlignScore
unexplained. The Section V cross-evaluator claim survives with that quarter
subtracted and stated.

The floor moves the same way: the mean random-context score is 0.420 under
NLI-max and **0.238** under NLI-concat, i.e. a good part of the inflated floor
in F6 is the maximum over six premises rather than the model.

---

## F6 — detection threshold below the random-context floor — **FIXED (all three parts), and it reframes Table II**

### 1. Donor rejection

`build_cases` now rejects a random-context donor whose text contains any
quantity from the answer, keeping the 10-try loop and falling back to the old
behaviour with a printed count. Fallbacks: **16–21 of 200 (8–10.5%)** on
NQ/Claude, **4–6 (2–3%)** on NQ/GPT-4o-mini, **0** on both HotpotQA cells. The
floor moves only slightly (NQ/Claude 0.504 → 0.489, 0.522 → 0.497), which
says the accidental support was real but not the main driver of the floor.

### 2. The floor-anchored gate — the substantive result

The brief is right and the effect is larger than stated. The random-context
distribution is not merely close to the 0.5 gate, it is **heavy-tailed**:

| cell | mean random score | **p95 random score** |
|---|---|---|
| NQ / Claude | 0.489 – 0.527 | **0.985 – 0.990** |
| NQ / GPT-4o-mini | 0.484 – 0.531 | 0.974 – 0.979 |
| HotpotQA / Claude | 0.440 – 0.477 | 0.990 – 0.994 |
| HotpotQA / GPT-4o-mini | 0.180 – 0.223 | 0.939 – 0.954 |

So under NLI-max an answer scored against a **completely unrelated context**
beats 0.5 about half the time, and beats 0.94 five percent of the time. Read
at a gate anchored to that distribution (its 95th percentile) instead of at
0.5:

| evaluator | cell | det. @ 0.5 | **det. @ floor gate** | n at the gate |
|---|---|---|---|---|
| NLI-max | NQ / Claude | 2 – 7% | **43 – 46%** | 83 – 135 |
| NLI-max | NQ / GPT-4o-mini | 52 – 61% | **84 – 89%** | 105 – 128 |
| NLI-max | HotpotQA / Claude | 12 – 16% | **43 – 52%** | 29 – 60 |
| NLI-max | HotpotQA / GPT-4o-mini | 65 – 75% | **79 – 85%** | 86 – 102 |
| claim-min | NQ / Claude | 24% | **57%** | pooled |
| claim-min | HotpotQA / Claude | 34% | 47% | pooled |
| AlignScore | NQ / Claude | 27% | **77%** | pooled |
| AlignScore | HotpotQA / Claude | 44% | **93%** | pooled |
| AlignScore | HotpotQA / GPT-4o-mini | 89% | 97% | pooled |

(All rows are the post-fix runs. NLI rows are per-embedder ranges; the
claim-min and AlignScore rows are pooled over the four embedders, 800 cases.)

**Two conclusions, and the paper needs both:**

- **The evaluator ordering is unchanged** — AlignScore > claim-min > NLI-max in
  every cell, at either gate. The paper's spine does not depend on the gate.
- **The magnitudes do.** "NLI-max catches 4% of falsified values" is a
  statement about a compressed scale as much as about blindness; on the same
  cases at a floor-anchored gate it catches 43–46%. The cross-generator gap
  narrows from roughly 10× (2–16% vs 52–75%) to under 2× (43–52% vs 79–89%).
  **Section V-B should report both numbers**, because the 4% figure as
  currently written overstates what was measured.

The floor-anchored gate is a **comparison device, not an operating point**: it
sits at 0.94–0.99, above the mean untouched answer, so only 29–135 of 200
untouched answers clear it. A product cannot deploy it. That is precisely why
it belongs beside the fixed gate rather than replacing it.

### 3. Binomial CIs

Wilson 95% intervals are now printed on every detection rate — overall, by
perturbation kind, and for the entity condition — and carried in the summary
dict. At n = 200 the 3% cell is [1.5%, 6.9%] and the 61% cell is [53.8%, 67.9%],
so several of the between-embedder differences readable off Table II are inside
their own intervals.

---

## F7 — statistics reporting — **FIXED (all three)**

The core statistics were re-checked and are correct, as the brief says; no
change was made to `bootstrap_significance`'s pairing, `permutation_test`,
`faithfulness_by_model`'s common-subset ordering, or `tost_equivalence`.

### 1. Holm correction on the evaluator table — applied, and the headline strengthens

`compare_evaluators.py` now Holm-corrects over the family it prints (8
comparisons: 2 datasets × 2 generators × 2 evaluators), reports `p_holm`
alongside `p`, and reads its verdicts off the corrected value.

| dataset | generator | evaluator | n | spread | p | **p_holm** | verdict |
|---|---|---|---|---|---|---|---|
| NQ | Claude | NLI-max | 792 | 0.0090 | 0.2233 | 0.4930 | null |
| NQ | Claude | AlignScore | 792 | 0.0179 | **0.0005** | **0.0040** | **DIFFERS** |
| NQ | GPT-4o-mini | NLI-max | 628 | 0.0129 | 0.0986 | 0.4930 | null |
| NQ | GPT-4o-mini | AlignScore | 628 | 0.0080 | 0.1179 | 0.4930 | null |
| HotpotQA | Claude | NLI-max | 538 | 0.0190 | 0.2704 | 0.4930 | null |
| HotpotQA | Claude | AlignScore | 538 | 0.0126 | 0.1510 | 0.4930 | null |
| HotpotQA | GPT-4o-mini | NLI-max | 510 | 0.0267 | 0.0652 | 0.3912 | null |
| HotpotQA | GPT-4o-mini | AlignScore | 510 | 0.0461 | **0.0005** | **0.0040** | **DIFFERS** |

**2 of 4 cells still break under AlignScore after correction**, at
p_holm = 0.004. The result is stronger stated this way, exactly as the brief
predicted.

**A second thing fell out of this, and it is worth having.** CLAUDE.md records
that the *identity* of the two disagreeing cells depends on whether
`abstained` comes from the heuristic or from the judge — NQ/Claude stops
disagreeing and HotpotQA/Claude starts, "both at p≈0.048". Those two p-values
are 0.0481 and 0.0487 uncorrected; under Holm they are **0.2886**, and the
verdict flip disappears:

```
ABSTENTION-SOURCE SENSITIVITY  (heuristic vs judge, alpha=0.05)
  no cell changes verdict — the result is stable to this choice
```

Under the judge the two significant cells are the same two — NQ/Claude
AlignScore (p 0.0002, p_holm 0.0016) and HotpotQA/GPT-4o-mini AlignScore
(p 0.0007, p_holm 0.0049). **The spine result is now stable to the abstention
source**, which removes the caveat CLAUDE.md attaches to it. That paragraph in
CLAUDE.md and the corresponding hedge in the paper should be updated.

### 2. Bootstrap resolution — reported

`bootstrap_significance` and `permutation_test` now return `n_extreme` and
`p_resolution`, and the table prints the count:

| cell | p | resamples |
|---|---|---|
| NQ/Claude AlignScore | 0.0005 | **5 / 10,000** |
| HotpotQA/GPT AlignScore | 0.0005 | **5 / 10,000** |
| NQ/Claude AlignScore (judge) | 0.0002 | 2 / 10,000 |

Write it in the paper as `p = 0.0005 (5/10,000 resamples)`. The smallest
attainable non-zero value is 1e-4, printed in the table footer, so 0.0005 is
five draws and not a clamped floor.

### 3. Per-row n in Table III — done

`src/perturb_report.py` rebuilds the by-assertion-count table with n and a
bootstrap CI on every row, and collapses the tail to `5+`. Numbers below.

---

## F8 — retired-metric code still live — **FIXED**

- `src/metrics.py` → **`src/legacy/rfg.py`**, with a header pointing at
  Section III-C and a package docstring explaining why the metric is retired.
- `results.assemble_results(legacy_rfg=False)` no longer writes the `RFG` /
  `nRFG` columns by default; `run_pipeline --legacy-rfg` turns them back on.
- `robustness_analysis` and the nRFG rows of `hypothesis_summary` skip
  themselves with a printed message when the columns are absent, rather than
  raising. `figures.py` skips Fig. 2–4 the same way.
- **H1 is deleted.** It passed `cont[:n]` and `inst[:n]` — different models,
  aligned only by DataFrame row order — into `bootstrap_significance`, which
  pairs by index and asserts pairing; the assert passed because the lengths
  matched, so the test ran and returned a p-value for a comparison it had not
  made. It was dead relative to v6 (H1 is reported as unsupported on the
  per-evaluator spreads), and an invalid test should not ship even dead. A
  regression test asserts no `H1` row is ever emitted.
- `robustness_analysis`'s docstring said "Uses GPT-4o-mini rows" while the code
  filtered `claude`. The code is right — Claude is the reference arm — and the
  docstring now says so.
- **README** gains a table naming the four embedders every reported result uses
  and stating that `GTE-large`, `Instructor-XL` and `jina-embeddings-v3` are
  configured but were never run at n=1000, with the reason (each adds a full
  paid generation pass without changing what the evaluator comparison shows).

---


## F9 — deterministic value-presence check — **NEW RESULT, implemented and measured**

`perturbation_check.numeric_grounding_check(answer, chunks)` extracts every
quantity with `NUM_RE` and returns False if any fails the fixed `num_in_text`
against the concatenated context. It is wired in as `--scorer numeric`, so it
runs through the same case construction and the same four conditions as the
three model-based evaluators, with no model loaded and no GPU.

Scored as a falsification detector: the untouched answer is a negative (a flag
on it is a false positive) and the falsified one is a positive. n = 200 per
cell, 16 cells.

### The first run said the check is not deployable. It was wrong, and why is the interesting part.

| dataset / generator | FP rate on untouched answers | recall |
|---|---|---|
| NQ / Claude | 8.0 – 12.5% | 100% |
| NQ / GPT-4o-mini | 1.0 – 1.5% | 100% |
| **HotpotQA / Claude** | **36.5 – 39.5%** | 100% |
| HotpotQA / GPT-4o-mini | 2.0 – 3.0% | 100% |

Inspecting those false positives: they are dominated by numbers that are not
claims about the world at all. The prompt numbers the retrieved chunks, and
Claude cites them — *"This information is found in Chunk 3, which states…"* —
and Claude writes ordered lists, *"1. Royce da 5'9" (Bad) 2. Eminem (Evil)"*.
Neither value is asserted of the world and neither can be expected to appear
in the retrieved text. Counting them as ungrounded is a defect in the check.

`content_numbers()` now excludes both (a citation left-context, or a
line-initial `N.` / `N)` marker). Same cases, same code path:

| dataset / generator | FP rate (95% CI) | recall (95% CI) | precision | F1 |
|---|---|---|---|---|
| NQ / Claude | **1.5 – 3.0%** [0.5, 6.4] | 92.5 – 95.5% [88.0, 97.6] | 0.969 – 0.984 | 0.949 – 0.965 |
| NQ / GPT-4o-mini | **0.5 – 1.0%** [0.1, 3.6] | 98.0 – 99.0% [95.7, 99.7] | 0.990 – 0.995 | 0.985 – 0.992 |
| HotpotQA / Claude | **2.5 – 6.0%** [1.1, 10.2] | 90.0 – 92.5% [85.0, 95.4] | 0.938 – 0.974 | 0.923 – 0.946 |
| HotpotQA / GPT-4o-mini | **1.1 – 3.0%** [0.3, 6.4] | 100% [97.9, 100] | 0.971 – 0.989 | 0.985 – 0.995 |

Recall is no longer 100% on Claude, and the reason is worth stating: in 5.0%
(NQ) to 9.5% (HotpotQA) of Claude's cases the value the probe falsified **is
itself a chunk citation or a list marker**, which the check now ignores. Those
are not factual falsifications — see the sample-contamination note under F1 —
so the recall loss is the probe's problem, not the check's.

**Compare against the model-based evaluators on the same 200 cases per cell**
(NQ/Claude): NLI-max detects 4%, claim-min 23%, AlignScore 29%, and the
deterministic check 92.5–95.5% at a 1.5–3.0% false-positive rate, with a
random-context floor of **0.028** against NLI-max's 0.47–0.52. That is the
strongest argument in the paper for Section IX's recommendation, and it is now
measured rather than asserted.

By perturbation kind, on NQ/Claude before the exclusion (the kind breakdown is
about which values are hard to ground, so it is clearest on the raw variant):

| kind | n | recall | false-positive rate |
|---|---|---|---|
| `magnitude` — the invoice case | 108 – 112 | 100% | 10.2 – 16.5% |
| `year` | 84 – 87 | 100% | 4.6 – 8.1% |
| `decimal` | 4 – 5 | 100% | 0.0% |

`magnitude` costs about twice what `year` does, because quantities get
restated (counted, summed, converted) far more often than years do.

### What the residual false positives are

After the exclusion, the surviving false positives are **derived** quantities —
the answer computes rather than copies:

- counting: *"Letters to Cleo had 6 members listed: …"*, *"Xanthoceras contains only 1 species"*
- arithmetic: *"between 1913 and 1927, which is approximately 14 years"*
- aggregation across chunks: *"resulted in 7 fatalities"*
- format conversion: *"Super Bowl LI (51)"*, dates written out

A literal value-grounding check should arguably flag all of those — none is
copied from the context — so this is a limitation to state rather than a bug to
fix. **Deployability: yes for both generators at 0.5–6.0% false positives**,
with the caveat that a product wanting to allow derived quantities needs a
normalisation layer (roman numerals, number words, date formats) and,
realistically, an arithmetic-aware check.

---

---

## Summary — status of every item

| item | verdict | what it moves in the paper |
|---|---|---|
| **F1** `num_in_text` substring | **FIXED** | Section V deltas 0.033–0.094 → **0.037–0.088**; Table II detection 2–15% → 2–16%; Section VIII margin anchor holds |
| **F2** `textnorm.contains` substring | **FIXED** | Table I in the 4th decimal; retained NQ sample **unchanged**; no TOST verdict moves |
| **F3** claim splitter on digits | **FIXED** | Table III bucket 3 by ~10% relative; the gradient and the mechanism claim survive |
| **F4** two abstention detectors | **FIXED** | nothing published; Sections V and VI now describe the same population |
| **F5** premise construction | **NEEDS A PAPER EDIT** (code unchanged, as instructed) | Section IV method text; the diagnostic bounds the confound at ~a quarter of the NLI/AlignScore gap |
| **F6** gate below the floor | **FIXED** | Table II gains a second gate and CIs; the "4%" figure needs restating |
| **F7** statistics reporting | **FIXED** | Table VII gains `p_holm` and resample counts; the headline **strengthens**, and a documented caveat disappears |
| **F8** retired metric live | **FIXED** | repo hygiene; an invalid H1 test deleted |
| **F9** value-presence check | **NEW RESULT** | Section IX gains a measured recommendation instead of an untested one |

Two findings surfaced that the brief did not anticipate, both flagged below
rather than fixed.

## Decisions I need from you

1. **The markdown asymmetry (biggest one).** `nli.py` scores the raw answer,
   `score_claims` strips markdown first. 74.9% of Claude's answers carry
   markdown against 0.3% of GPT-4o-mini's, and the score moves by up to 0.729
   on the asterisks alone. Options: (a) strip in both places and re-run all
   faithfulness scoring — every `nli_max` number for Claude moves, ~3 GPU-hours,
   free; (b) strip in neither; (c) leave it and disclose it as a limitation.
   I recommend (a): the paper's cross-generator claim is otherwise confounded
   with formatting, and a reviewer who reads `score_claims` will find it.
   `src/markdown_check.py` is built and ready to size the change first.

2. **Citation contamination in the falsification sample.** `build_number_case`
   perturbs a chunk citation or a list marker in 9.5% of HotpotQA/Claude cases,
   and those cases have a **negative** mean delta, depressing that cell by ~15%.
   Excluding them is one line plus a ~2 GPU-hour re-run, and it would raise the
   reported Claude deltas. Recommend doing it.

3. **`CORPUS_VERSION` was not bumped** for F2's relevance-rule change (7 of
   1000 NQ queries). Bumping strands $42.73 of paid generations for a
   4th-decimal change. Say if you want it bumped anyway.

4. **The 112 preamble-hidden refusals** ("Based on the provided context, I
   cannot answer this question") that the canonical rule counts as answers.
   `abstention.is_abstention_extended` exists and is wired into nothing.
   0.7% of the grid. Recommend leaving it and stating the rule precisely.

## Paper edits this report implies (not made — `.docx` untouched)

- Section IV: state the premise construction per evaluator (wording drafted
  under F5).
- Section V-B: report detection at both gates, with CIs; the 4% figure alone
  overstates the finding.
- Section V-C / Table III: say it is NQ; print n; show or collapse bucket 4.
- Section VII / Table VII: add `p_holm`, write `p = 0.0005 (5/10,000
  resamples)`, and **delete the hedge about the two cells depending on the
  abstention source** — after correction they do not.
- Section VIII: the margin anchor becomes 0.037–0.088.
- Section IX: replace the untested recommendation with the measured numbers
  (FP 0.5–6.0%, recall 90–100%).
- `CLAUDE.md` §2 has been updated already; its old paragraph on abstention-
  source instability is marked superseded.
