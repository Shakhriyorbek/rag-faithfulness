# Implementation report — revisions Part A/B, and the four open decisions

**Date:** 2026-09-03 · **Cost:** $0 (all re-scoring is local NLI/AlignScore; no API spend)
**Scope:** `checkpoints/n1000_v3` on gpu1 · **Code:** commits `7ff405f`, `083f576`, `ba7e82f`
**Tests:** locally 196 → 205 → **207**; on gpu1 207 → **223** (the strip test skips without torch)
**Paper:** `paper/RAG_Faithfulness_v6_evaluators.docx`, 24.6 KB → 28.5 → 31.1 → **33.2 KB**

Written 2026-09-04, from the session record and the artifacts it produced. Every
number below came out of a run, not out of the previous report — the Part A pass
deliberately re-derived the September 1 figures from the checkpoints rather than
transcribing them, and they reproduced exactly.

---

## Headline: the spine dropped from 2 of 4 cells to 1 of 4

The four decisions left open by `reports/2026-09-01_fixes_implementation.md`
were all taken, and two of them required re-scoring the whole grid. That
happened in **two full passes**, because D2 and D4 change which cases exist and
D1 changes what every scorer reads:

| pass | commit | what it carried | outcome |
|---|---|---|---|
| 1 (~5 h) | `7ff405f` | D2 content-only cases, D4 preamble-stripping refusal rule | spine **still 2 of 4** |
| 2 (~7 h) | `083f576` | D1 markdown stripped for every evaluator | spine **1 of 4** |

**NQ/Claude under AlignScore is the cell that fell.** Its spread went 0.0179 →
**0.0114** and its Holm-adjusted p 0.0040 → **0.0567**. It remains significant
uncorrected (p = 0.0081). Only HotpotQA/GPT-4o-mini survives correction, at
p_holm 0.0040.

That cell's apparent evaluator-dependence was substantially a **formatting
artifact produced by one generator**. It is written up as Section VII-B — a
correction reported as a finding — rather than absorbed silently, because the
failure mode generalises: a preprocessing step applied to one aggregate and not
another is invisible in every summary statistic, survives every significance
test, and is detectable only through an identity the two aggregates must
satisfy.

Two things were caught along the way that would each have produced a wrong
conclusion on their own — the frozen `abstained` field (§5) and the citation
whose arXiv ID was right and whose title was invented (§1).

---

## 1. Part B — B3, verify every citation

### Where the work had actually stopped

B1, B2, B4, B5 and B6 were already drafted in the uncommitted working tree, and
the body text had been renumbered to a new `[1]`–`[24]` scheme. **The
bibliography still held the old 13-entry list.** Eleven citations therefore
pointed at nothing and the rest pointed at the wrong papers. Work had stopped
exactly at B3.

### Four entries were wrong, and §6 could not have caught them

The earlier §6 pass fixed **authors**. It never checked **titles**. All 24
entries were verified directly against arXiv and the ACL Anthology — ID
resolves, venue and status confirmed, abstract read to confirm it supports what
it is cited for.

| Ref | Was | Actually |
|---|---|---|
| [13] | "Each to Their Own: Matching Retrievers to Generators", Chen, S. | **arXiv:2507.17442 is a different paper entirely** — S. Chen, Z. Zhao, J. Chen, "Confident RAG: … Mathematics Question Answering through Multi-Embedding and Confidence Scoring". The ID was right; the title was invented |
| [4] | "FaithJudge" as the paper title | FaithJudge is the **framework**. Paper: Tamber, Kazi, Sourabh, Lin, "Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards", EMNLP 2025 Industry Track, pp. 799–811, arXiv:2505.04847 |
| [12] | "The Semantic Illusion in Retrieval-Augmented Generation" | Truncated. Full: "The Semantic Illusion: **Certified Limits of Embedding-Based Hallucination Detection in RAG Systems**", D. Sinha, arXiv:2512.15068 |
| [14] | DeBERTa — He, Liu, Gao, Chen, ICLR 2021 | The scorer is `nli-deberta-**v3**-large` → **DeBERTaV3**, He, Gao, Chen, ICLR 2023, arXiv:2111.09543 — a different paper |

[13] is the one the revisions file had warned about ("at least one could not be
independently confirmed"). Because the real paper is maths-QA-specific, the
in-text sentence was qualified rather than left implying general-purpose RAG.

The claims made **about** each work were checked too, not just the identifiers:
TRUE's eleven datasets, AGGREFACT's stratification, RAGTruth's ~18,000
word-level-annotated responses, FaithBench's ~50% detector accuracy, and Wallace
et al.'s subword finding all match the body text as written. The
"jina-embeddings-v5-text" entry flagged in CLAUDE.md §2 — a model that does not
exist — is resolved as **[24] jina-embeddings-v3**, Sturua et al.,
arXiv:2409.10173.

Recorded in **CLAUDE.md §6b**, because the existing §6 table would otherwise
have reintroduced two of them. Citation audit on the built `.docx`: `[1]`–`[24]`
all cited and all defined, nothing dangling either way.

---

## 2. Part A — ten edits, numbers re-derived rather than copied

| | change |
|---|---|
| **A1** | Abstract: "explained by" → "largely accounted for by", with the residual named at a matched single assertion (+0.087 vs +0.481 when the edit was made; **+0.102 vs +0.485** after D1). Dropped **"monotonically"** — post-F3 the decay is not strictly monotone in either generator |
| **A2** | Both cells reported rather than one |
| **A3** | Table II rebuilt: floor-anchored gate beside the fixed 0.5, Wilson 95% CIs on every rate, and a paragraph on why rates need floor normalisation. NLI-max on Claude/NQ is **5.1% at gate 0.5 but 50.2% at the floor gate**; the cross-generator gap narrows from ~13× to under 2×; evaluator ordering is unchanged at either gate |
| **A4** | Table III carries **n per row**, adds bucket 4, and is labelled **NQ** — it always was NQ-only and the paper never said so. GPT's tail rests on 26/10/7 cases |
| **A5** | Table VII gains Holm-adjusted p; headline **p = 0.0005 (Holm 0.0040)**, caption states 5 of 10,000 resamples so it cannot read as a clamped floor |
| **A6** | Method states the premise construction per evaluator (6 premises vs 1) and points at the concat diagnostic that bounds the confound |
| **A7** | One refusal rule stated once, with per-cell exclusions |
| **A8** | Prefix sampling disclosed: cases are the first eligible per cell, not a random sample |

Three further edits were forced by the same re-runs, not by the A-list: the
Section VIII margin anchor, the conclusion's headline figures, and **Section IX,
which still claimed the deterministic value check was "not evaluated here"** when
F9 had measured it — a false statement that the previous re-run cycle had
created.

**An API error interrupted this pass mid-way.** A1, A2, A6, A7, A8 had landed;
A3, A4, A5 had not. This was verified against the file rather than assumed.
Nothing was half-written, because each edit is an exact-match replacement that
asserts it matched once — a partial batch fails loudly instead of corrupting the
file.

One item was deliberately **not** resolved here: the V-C identity sentence was
rewritten to say that single-assertion answers *with no markdown* score
identically under both aggregates, rather than the old blanket "0.4922 in each
case", which was no longer true. That was decision D1, and it was left stated
accurately rather than pre-empted.

---

## 3. The four decisions

### D1 — markdown is stripped for every evaluator, and it cost a cell

`nli.score_chunks` scored the answer exactly as written, `**bold**` included,
while `claim_faithfulness.score_claims` stripped markdown first. The two
aggregates were not reading the same hypothesis.

**Sized before deciding** (`markdown_check`, n = 300/cell). Stripping *lowers*
Claude's NLI-max:

| | mean change | mean abs | > 0.05 | max |
|---|---|---|---|---|
| NQ / Claude | −0.005 to −0.012 | 0.013–0.027 | 4.7–9.7% | 0.77 |
| HotpotQA / Claude | −0.021 to −0.043 | 0.045–0.059 | **17.7–19.3%** | 0.95 |

So markdown *inflates* Claude's scores, and on HotpotQA roughly a fifth of
answers move by more than 0.05 — comparable to the size of a whole falsification
effect.

**Implementation.** `strip_markdown` moved to `textnorm` (`claim_faithfulness`
imports `nli`, so `nli` could not import it back — the same constraint that
moved `squad_normalize` in B17) and is applied in `nli.score_chunks`, in the
AlignScore arm of `faithfulness.py`, and once at `_Scorer.score` in
`perturbation_check`. That last point matters: `nli_concat` and `align` call
their models directly and would otherwise still have read the raw string. It is
idempotent.

**The identity check is the direct verification**, and it is why this defect was
findable at all:

| single-assertion, markdown group | before | after |
|---|---|---|
| cases differing | **131 of 131** | **7 of 140** |
| largest difference | **0.729** | **1.5e-05** |

That residual now equals the batch-padding noise in the unformatted group.
Table III confirms it independently: the single-assertion row is numerically
**equal** under both aggregates (+0.102 Claude, +0.485 GPT), the degenerate case
they must reduce to. Pooled over both groups the identity holds to 1.7e-05 on
n = 1,640.

**Consequence — the headline.** NQ/Claude under AlignScore: spread 0.0179 →
0.0114, p_holm 0.0040 → **0.0567**. 2 of 4 cells → **1 of 4**. Claude's deltas
rose across the board, because stripping removes score inflation from the
unfalsified side (NQ/Claude +0.0442 → **+0.0541**; HotpotQA +0.0817 →
**+0.0845**). GPT-4o-mini is unchanged, as expected at 0.3% markdown.

> **Do not un-strip one arm without re-running phases D and E and the whole
> probe.** Superseded checkpoints:
> `checkpoints/n1000_v3/superseded_2026-09-03_markdown/`.

### D2 — case construction is now content-only

`build_number_case` took the first **grounded** number in the answer, which is a
chunk citation or an ordered-list marker in 9.5% of HotpotQA/Claude cases and
5.0% of NQ/Claude ones. Falsifying "Chunk 4" into "Chunk 7" is not a
falsification, and those cases carried a **negative** mean delta (−0.0625),
depressing the cell.

The direction was established *before* committing to a re-run, by filtering the
existing rows on `value_role()`: HotpotQA/Claude +0.0734 → +0.0850, NQ/Claude
+0.0426 → +0.0442, GPT cells unmoved. Case construction now takes the first
**content** number, which also restores n to 200 per cell instead of discarding
6.6–8.8% of Claude's sample.

**A second result fell out of it.** The deterministic numeric check now detects
**100% in all 16 cells**, up from 90.9–98.6%. That residual was never the check
failing — it was the probe falsifying chunk citations that the check correctly
ignores. Removing them removed it entirely.

### D3 — `CORPUS_VERSION` stays at v3

A deliberate, documented deviation from the §8 convention, which says to bump
whenever corpus or relevance semantics change. The F2 fix changed the
chunk-relevance rule for **7 of 1000** NQ queries; NDCG moves in the 4th decimal
and **no TOST verdict moves**. The corpus, the chunks and the retained samples
are byte-identical.

Bumping re-keys `checkpoints/n{N}_{CORPUS_VERSION}` and would strand 16,000 paid
generations plus the judge run — **$42.73 re-spent to change a 4th decimal**.
Old checkpoints archived at `checkpoints/n1000_v3/superseded_2026-09-01/`. If
the relevance rule changes again for a reason that moves a reported number, bump
then.

### D4 — the refusal rule strips the preamble

Both generators prepend "Based on the provided context," to everything, refusals
included, so the bare anchored rule graded genuine refusals as attempts.
`is_abstention` now strips a leading "Based on …," / "According to …," before the
anchored test, over the wider phrase list.

**Audited over all 16,000 generations before adoption:** 107 rows move from
attempt to refusal (0.67%), **all of them Claude's** — 80 HotpotQA, 27 NQ, none
GPT-4o-mini. 105 open with "I cannot answer this question". Mid-answer hedges
are still answers. The pre-change rule survives as `is_abstention_bare` so the
difference stays measurable.

Per-cell refusals, before → after: NQ/Claude 560 → **587**, NQ/GPT-4o-mini
1,162 → 1,162, HotpotQA/Claude 1,086 → **1,166**, HotpotQA/GPT-4o-mini 1,208 →
1,208. Total **4,016 → 4,123 of 16,000 (25.8%)**.

> The 09-01 report predicted 112. The audit measures **107**; the difference is
> rows excluded as `[ERROR]`/None before the rule is applied. 107 is the number
> the adopted code produces over the grid.

---

## 4. What the two passes moved

| quantity | 09-01 | after pass 1 | after pass 2 (final) |
|---|---|---|---|
| cells breaking under AlignScore | 2 of 4 | 2 of 4 | **1 of 4** |
| NQ/Claude AlignScore p (Holm) | 0.0005 (0.0040) | 0.0007 (0.0049) | **0.0081 (0.0567)** |
| NQ/Claude NLI-max delta | +0.0426 | +0.0442 | **+0.0541** |
| HotpotQA/Claude NLI-max delta | +0.0734 | +0.0817 | **+0.0845** |
| Section VIII margin anchor | 0.037–0.088 | 0.037–0.093 | **0.049–0.095** |
| refusals excluded | 4,016 | 4,123 | 4,123 |
| numeric-check recall | 90.9–98.6% | 100% (16/16) | 100% (16/16) |

Eligible cases 5,272, sampled 3,184 (60%) — unaffected by D1.

---

## 5. A bug caught mid-run: `abstained` is frozen at scoring time

After pass 1, **Table VII came back byte-identical** — same n, same p — despite
D4 having moved 107 rows out of the answered population. That was wrong, and the
reason matters.

`correctness.py:216` computes `abstained` **once, at scoring time**, and stores
it in the `*_scored` checkpoints. Sections VI–VIII read that stored value.
The D4 rule change therefore never reached them. Only Section V updated, because
the perturbation probe calls `is_abstention` live.

Re-scoring correctness (free, no API) made Table VII move properly.
`run_phase_correctness` overwrites unconditionally, so a re-run refreshes it.

**Had this not been checked, the session would have reported "the spine is
unchanged" when the change had simply not propagated.** A code path that stores
a derived predicate rather than recomputing it will silently ignore a change to
the rule that produced it.

The judged checkpoints were checked too: the judge's `abstained` comes from its
own verdict rather than from `is_abstention`, so **no re-judging was needed —
$29 not spent.**

---

## 6. Operational notes worth keeping

- **`node` is not on `PATH`.** It lives at `~/.local/node/bin/node`, and
  `build_v6.js` writes to the relative path `paper/…`, so it must be run from
  the repo root: `PATH="$HOME/.local/node/bin:$PATH" node paper/build_v6.js`.
- **An SSH timeout does not kill the remote process.** Four concurrent
  `markdown_check` runs accumulated this way, three of them writing to the same
  log. CLAUDE.md already says to use `tmux` for long jobs; this is why. Launch
  detached, then poll.
- **`pkill` on a pattern can match the tmux server's own argv**, because the
  server carries the session command. It killed the server and both sessions.
  Match on a narrower pattern, or kill by PID.
- **A stale `ControlMaster` socket looks exactly like a gateway block.** The
  banner-exchange timeout was diagnosed as rate-limiting; TCP 22 was in fact
  open. Check the socket before backing off — and note that a retry loop firing
  every 120 s would keep a real block alive.
- **`ssh -n` nulls stdin**, which silently turns a heredoc upload into a 0-byte
  file.

---

## 7. Still open after this session

Unchanged by the above:

- Open-weight generator (free, the Qwen arm is wired and ready)
- Claim-level over the full grid (free — only the perturbation subset is scored,
  which is why the evaluator table is a family of 8 and not 12)
- C1/C2 floor and ceiling (~$3–4), QASPER (~$5)
- IEEE → ACL conversion
- The Salemi & Zamani PDF, still read only in abstract
- ESA / re-ranking — decision pending with Berend

Two items flagged at the close of this session were **closed on 2026-09-04**:
the Section V-B verbatim-copying figures (not regenerated by either pass, since
the measure had no home in `src/`) and the multiplicity-family choice behind the
"1 of 4" count. See `reports/2026-09-04_copying_and_multiplicity.md`.
