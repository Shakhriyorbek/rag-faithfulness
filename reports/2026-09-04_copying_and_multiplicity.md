# Closing the two items flagged at the end of the 2026-09-03 session

**Run date:** 2026-09-04 · **Cost:** $0 (pure arithmetic on existing checkpoints)
**Scope:** `checkpoints/n1000_v3` on gpu1, post-D1/D2 (written 2026-09-03 19:41–19:46)
**Code:** new `src/copying_check.py`; new `_report_holm_scope_sensitivity` in
`src/compare_evaluators.py`. Tests 223 → **238 passed** on gpu1.

The 2026-09-03 session ended by flagging two things it had not closed. Both are
now closed. One confirms a published result and strengthens it; the other does
not change the headline but makes a hidden choice visible.

---

## 1. Verbatim copying — the Section V-B control

### The problem

Section V-B rejects the obvious explanation for the paper's central asymmetry:
that Claude quotes the retrieved text back, so the evaluator scores a copy of
its own premise and one substituted value cannot move it. The figures behind
that rejection — overlap 0.234 for both generators, corr(overlap, Δ) +0.041 and
+0.161 — were produced once, on 2026-08-25, by a script that was never
committed. Because the measure had no home in `src/`, it was the only number in
Section V not regenerated after the 2026-09-03 decisions, **both of which move
the quantity it correlates against**: D1 changed every score, D2 changed which
cases exist.

### What was rebuilt

`src/copying_check.py`. Same measure — the fraction of the answer's word
5-grams occurring in the retrieved context as a contiguous token sequence,
correlated against `orig − number` from the `perturb_*` checkpoints — with two
previously implicit choices now explicit, because both decide the number:

**Tokenisation.** `textnorm.squash` maps every non-word character to a space,
so it already neutralises markdown: `**Graduados**` and `Graduados` squash to
the same token. A raw `.split()` does not. The two disagree on exactly the
answers D1 was about, and markdown is 74.9% of Claude's answers against 0.3% of
GPT-4o-mini's. `--variants` reports all four combinations.

**Short answers.** An answer under 5 tokens has no 5-gram and is now excluded,
not scored 0. GPT-4o-mini's median answer is 18 words against Claude's 49, so
counting them as "copied nothing" would manufacture a cross-generator overlap
difference out of answer length alone — the exact confound this control exists
to rule out. 11 of 3,184 cases are excluded on this rule.

### Result — the conclusion holds, and the published figures barely moved

3,184 cases across 16 cells; 3,173 scored.

| cell | n | mean overlap | Pearson r | 95% CI | Spearman |
|---|---|---|---|---|---|
| NQ / Claude | 800 | 0.248 | +0.010 | [−0.075, +0.094] | +0.069 |
| NQ / GPT-4o-mini | 791 | 0.274 | +0.149 | [+0.082, +0.214] | +0.088 |
| HotpotQA / Claude | 800 | 0.203 | +0.095 | [+0.022, +0.166] | +0.088 |
| HotpotQA / GPT-4o-mini | 782 | 0.194 | +0.172 | [+0.099, +0.241] | +0.112 |
| **ALL / Claude** | **1600** | **0.226** | **+0.046** | **[−0.004, +0.096]** | +0.089 |
| **ALL / GPT-4o-mini** | **1573** | **0.235** | **+0.166** | **[+0.118, +0.212]** | +0.108 |

Against the published 0.234 / 0.234 and +0.041 / +0.161: the correlations
reproduce to within 0.005 and the overlaps to within 0.01. **The result is
confirmed on the current data.**

Three refinements the original did not have:

- **The overlaps are not identical.** 0.226 (Claude) against 0.235
  (GPT-4o-mini) — close, but the paper's "0.234 for both" overstated the
  coincidence. The direction matters and is worth stating plainly: the
  generator whose scores barely move copies **slightly less**, which is the
  wrong way round for the hypothesis.
- **Claude's correlation is indistinguishable from zero**, CI [−0.004, +0.096].
  The draft called both correlations "positive, where the explanation predicts
  negative". That is right for GPT-4o-mini and too strong for Claude. Corrected
  in the paper.
- **The gap survives at matched overlap**, which the correlation alone does not
  show and which is the stronger refutation:

| overlap band | Claude n | Claude Δ | GPT n | GPT Δ |
|---|---|---|---|---|
| [0.0, 0.1) | 424 | +0.060 | 660 | +0.394 |
| [0.1, 0.2) | 322 | +0.075 | 206 | +0.326 |
| [0.2, 0.3) | 338 | +0.052 | 228 | +0.396 |
| [0.3, 1.0] | 516 | +0.085 | 479 | +0.507 |

At least a fivefold separation in every band, including above 0.3 where both
generators are demonstrably reusing a large part of the retrieved wording. If
copying explained the asymmetry the two would converge inside a band. They do
not.

### The measure is provably insensitive to D1

`raw/squash` and `stripped/squash` are **byte-identical** in every cell, which
is what the tokenisation argument predicts. The whitespace variants sit much
lower (Claude 0.127–0.136, GPT 0.144) because markdown tokens can never match,
and the published 0.234 is a squash-style figure. So this number could not have
been changed by D1 — but that was worth demonstrating rather than assuming,
since the reason it could not is a property of the tokeniser and not of the
data.

---

## 2. Multiplicity scope — which family Holm corrects over

### The problem

The headline is a **count**: k of 4 nulls break under a more sensitive
evaluator. k is not a property of the data. NQ/Claude under AlignScore sits at
p = 0.0081 uncorrected and lands either side of α = 0.05 depending on the
family:

| family | tests | breaks | cells |
|---|---|---|---|
| table-wide | 8 | **1** | HotpotQA/GPT-4o-mini |
| within-evaluator | 4 | **2** | + NQ/Claude |
| within-cell | 2 | **2** | + NQ/Claude |

NQ/Claude/align: p_holm **0.0567** table-wide, **0.0243** within evaluator,
**0.0162** within cell. Every other cell keeps its verdict under all three.

### The decision: table-wide stays

The within-evaluator argument is at least as strong on the statistics. The four
NLI tests establish *which cells are null* — they are a precondition for the
question being asked, not competing discoveries — so the family actually at
risk of producing a false positive is the four AlignScore rejections. Nothing
about conditioning on an NLI non-rejection inflates the AlignScore Type I rate;
if anything, given the two evaluators are positively correlated, it makes the
test conservative.

**That is not sufficient reason to switch, and we do not.** Table-wide was
fixed in advance (decision A5) and is the most conservative of the three.
Adopting a smaller family *after* observing that it restores a cell is a
forking path whatever its a-priori merit, and a reviewer would be right to say
so. The a-priori argument would have had to be made a-priori.

What changes is that the choice is no longer invisible.
`compare_evaluators._report_holm_scope_sensitivity` now prints all three
families, their counts, and any cell whose verdict depends on the choice, on
every run. Section VII states the alternative and the reasoning.

This is worth reporting rather than burying for the same reason Section VII-B
is: the paper's subject is analysis decisions that change a verdict while
leaving every summary statistic intact. A multiplicity family that moves the
headline count from one to two is another instance of it, inside the paper's
own statistics.

---

## 3. Side finding — the backup does not contain the 2026-09-03 re-run

`~/rag-backup/checkpoints` on Fedora was last refreshed **2026-09-01 15:39**.
The D1/D2 re-run of the probe and the evaluator comparison landed on gpu1 on
2026-09-03 19:41–19:46 and **is not in the backup**. The convention in
CLAUDE.md ties the refresh to paid runs, and the 09-03 re-run was free, so
nothing was violated — but the backup is currently a pre-D1 snapshot, and the
post-decision checkpoints behind every number in the current paper exist in one
place only.

```bash
rsync -avz szte-gpu:rag_faithfulness/checkpoints/ ~/rag-backup/checkpoints/
```

Recommend widening the rule from "after every paid run" to "after any run that
changes a number in the paper".

---

## Paper edits made

- **Section V-B** rewritten: 0.226 / 0.235 rather than 0.234 for both; Claude's
  correlation given as indistinguishable from zero with its CI rather than as
  positive; the matched-overlap bands added as the primary refutation; a note
  that the measure is unaffected by the Section VII-B normalisation. No new
  table, so numbering is unchanged.
- **Section VII** gains a paragraph on the multiplicity family: the three
  adjusted p-values for the swing cell, why within-evaluator is defensible, why
  table-wide is kept anyway.
- Rebuilt: `paper/RAG_Faithfulness_v6_evaluators.docx` (33.9 KB). No other
  number in the paper cites the V-B figures; Table VII's caption already
  describes the eight-comparison family and needed no change.

## What did NOT change

- The headline count stays **1 of 4 cells**.
- No checkpoint was rewritten; both items are read-only analyses.
- Section V-B's conclusion is unchanged — the copying hypothesis is still
  rejected, now on stronger evidence than the correlation alone.
