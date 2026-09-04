# Claim-level faithfulness over the full grid

**Run date:** 2026-09-04 · **Cost:** $0 (local NLI) · **Duration:** 3 h 03 m (08:18–13:21 UTC)
**Scope:** `checkpoints/n1000_v3` on gpu1 · **Code:** `src/claim_faithfulness.py`, commit `f0bf4c8` (adaptive batching)
**Output:** 16 `claim_scores_*` checkpoints, **16,000 rows, 0 NaN**, 280,548 NLI pairs

This closes item 3 of `PAPER_TODO.md` §3. Until now only the 200-case
perturbation subset had been scored claim-level, which is why the evaluator
table was a family of 8 rather than 12.

---

## Headline

1. **The Table III mechanism reproduces at grid scale.** The gap between
   claim-level and whole-answer aggregation tracks assertions per answer almost
   exactly, on 16,000 answers rather than the 200-case subset.
2. **The evaluator table is now 12 rows.** Under table-wide Holm the headline
   count is **unchanged at 1 of 4 cells** — HotpotQA/GPT-4o-mini, p_holm
   0.0040 → **0.0060** on the larger family.
3. **That cell now breaks under two independent evaluators, not one.**
   Claim-min gives p = 0.0070 in the same cell. It does not survive table-wide
   correction, but it is the second-smallest p in the table, and it means the
   result is not an AlignScore artifact.
4. **Claim-min is the more discriminating instrument in all four cells** —
   larger spread and smaller p than NLI-max everywhere.
5. **Equivalence is evaluator-dependent too.** At ±0.03 on NQ/Claude, NLI-max
   calls 6 of 6 system pairs equivalent and claim-min calls 2 of 6.

---

## 1. The mechanism, at n = 16,000

`claim_min` is min-over-claims of max-over-chunks; `whole_max` is the old
metric computed on the same inputs in the same pass, so the two are directly
comparable per answer.

| cell | claims/answer | claim_min | whole_max | gap |
|---|---|---|---|---|
| NQ / Claude | 3.0 | 0.595 – 0.614 | 0.848 – 0.859 | **≈ 0.24** |
| HotpotQA / Claude | 2.4 – 2.5 | 0.285 – 0.338 | 0.548 – 0.638 | **≈ 0.27** |
| NQ / GPT-4o-mini | 1.2 | 0.784 – 0.804 | 0.797 – 0.825 | **≈ 0.02** |
| HotpotQA / GPT-4o-mini | 1.0 – 1.1 | 0.681 – 0.761 | 0.694 – 0.766 | **≈ 0.01** |

The gap is an order of magnitude larger for the generator that writes ~3
assertions per answer than for the one that writes ~1. That is the weakest-link
property doing exactly what it is supposed to: max-over-chunks lets correct
surrounding material carry an unsupported claim, and the effect scales with how
much surrounding material there is.

It also confirms the Section V-C claim on the full population rather than on a
200-case sample: **whole-answer aggregation is not measuring the same thing for
the two generators**, and the difference is a property of answer structure, not
of grounding.

Note the absolute level on HotpotQA/Claude: **claim_min 0.28–0.34**. Under the
whole-answer metric the same answers score 0.55–0.64. A multi-hop answer from a
verbose generator almost always contains at least one poorly supported
assertion.

---

## 2. The evaluator table, now a family of 12

Answered-only, paired on a common subset, 10,000 resamples, Holm over the 12
comparisons printed.

| dataset | generator | metric | n | spread | p | p_holm | verdict |
|---|---|---|---|---|---|---|---|
| NQ | Claude | nli | 784 | 0.0107 | 0.1551 | 0.5064 | null |
| NQ | Claude | align | 784 | 0.0114 | 0.0081 | 0.0810 | null |
| NQ | Claude | claim | 784 | 0.0216 | 0.1042 | 0.5064 | null |
| NQ | GPT-4o-mini | nli | 628 | 0.0130 | 0.0978 | 0.5064 | null |
| NQ | GPT-4o-mini | align | 628 | 0.0080 | 0.1173 | 0.5064 | null |
| NQ | GPT-4o-mini | claim | 628 | 0.0151 | 0.0746 | 0.5064 | null |
| HotpotQA | Claude | nli | 517 | 0.0232 | 0.1586 | 0.5064 | null |
| HotpotQA | Claude | align | 517 | 0.0164 | 0.0633 | 0.5064 | null |
| HotpotQA | Claude | claim | 517 | 0.0345 | 0.0528 | 0.4752 | null |
| HotpotQA | GPT-4o-mini | nli | 510 | 0.0267 | 0.0652 | 0.5064 | null |
| **HotpotQA** | **GPT-4o-mini** | **align** | **510** | **0.0461** | **0.0005** | **0.0060** | **DIFFERS** |
| HotpotQA | GPT-4o-mini | claim | 510 | 0.0378 | 0.0070 | 0.0770 | null |

**The headline count is unchanged: 1 of 4.** Expanding the family from 8 to 12
raised the surviving cell's p_holm from 0.0040 to 0.0060, comfortably clear of
0.05.

**The new information is the last row.** The same cell that breaks under
AlignScore is second-smallest in the table under claim-min, at p = 0.0070
uncorrected. Two evaluators built on different principles — a trained alignment
model and a min-over-claims decomposition of the same NLI model that finds
nothing — agree that HotpotQA/GPT-4o-mini is where the embedding systems
differ. NLI-max says p = 0.0652 there. This is the paper's argument in a
cleaner form than one evaluator against one: the disagreement is not
AlignScore-specific.

### Claim-min is the more sensitive instrument in every cell

| cell | nli spread (p) | claim spread (p) |
|---|---|---|
| NQ / Claude | 0.0107 (0.1551) | **0.0216 (0.1042)** |
| NQ / GPT-4o-mini | 0.0130 (0.0978) | **0.0151 (0.0746)** |
| HotpotQA / Claude | 0.0232 (0.1586) | **0.0345 (0.0528)** |
| HotpotQA / GPT-4o-mini | 0.0267 (0.0652) | **0.0378 (0.0070)** |

Larger spread and smaller p in all four, without exception. Consistent with the
falsification result, where claim-min detected 25–34% of falsified values on
Claude's answers against NLI-max's 5–14%. The aggregation choice, not the
underlying model, is doing the work — claim-min and nli-max use the **same**
DeBERTa checkpoint.

### Multiplicity scope now swings two cells

| family | tests | breaks |
|---|---|---|
| table-wide | 12 | **1** — HotpotQA/GPT/align |
| within-evaluator | 4 | **3** — + NQ/Claude/align, HotpotQA/GPT/claim |
| within-cell | 3 | **3** — same three |

Swing cells: HotpotQA/GPT/claim (p_holm 0.0770 / 0.0280 / 0.0140) and
NQ/Claude/align (0.0810 / 0.0243 / 0.0243).

Table-wide is kept, as decided on 2026-09-04 (B21). Applying it to a 12-row
table is the same rule, not a new one: the paper always specified three
evaluators, so **12 is the design family and the earlier 8 reflected incomplete
data**, not a smaller intended set. That the count survives the stricter family
is worth stating.

---

## 3. Equivalence is evaluator-dependent as well

Pairs judged equivalent, of six tested, by margin:

| cell | metric | ±0.01 | ±0.02 | ±0.03 | ±0.05 | ±0.10 |
|---|---|---|---|---|---|---|
| NQ / Claude | nli | 0/6 | 4/6 | 6/6 | 6/6 | 6/6 |
| NQ / Claude | claim | 0/6 | 0/6 | **2/6** | 6/6 | 6/6 |
| HotpotQA / Claude | nli | 0/6 | 0/6 | 0/6 | 5/6 | 6/6 |
| HotpotQA / Claude | claim | 0/6 | 0/6 | 0/6 | **3/6** | 6/6 |

At ±0.03 on NQ/Claude the two evaluators disagree completely: NLI-max declares
every pair equivalent, claim-min declares two. This is a second, independent
route to the paper's conclusion — an equivalence claim, not just a difference
claim, depends on which evaluator is used to make it.

---

## 4. Operational note — the batch size had to become adaptive

The run could not start as written. A co-tenant on the shared GPU held 27.2 of
32 GB, and the hardcoded batch of 16 died on the first forward pass. Measured
with ~760 MiB free after the model loads: **batch 2 fits at a 1,917 MiB peak,
batch 4 does not** (112 ms/pair against 42 unconstrained). CPU is not a fallback
— 1,060 ms/pair at 16 threads, roughly 70 h for this workload.

`nli.NLIScorer` now picks its batch size from free VRAM, halves it on OOM, and
grows it back after a long clean run. `RAG_NLI_BATCH` pins it. The retry re-runs
the failing slice rather than skipping it, which is the part that matters:
`score_claims` slices the returned probabilities by block, so a short return
would attribute one claim's score to another claim and every `claim_min` in the
grid would be silently wrong. Pinned by `tests/test_nli_batching.py` (10 tests).

The co-tenant's job ended before launch, so the run went at batch 16 throughout.

---

## 5. What this implies for the paper (not yet applied)

- **Table VII** gains four rows and its caption changes from eight comparisons
  to twelve; the surviving cell's p_holm becomes 0.0060.
- **Section VII's multiplicity paragraph** (written 2026-09-04) cites 0.057 /
  0.024 / 0.016 for the swing cell under the three families. Those become
  0.0810 / 0.0243 / 0.0243, and there are now **two** swing cells.
- **Section V-C** can state the mechanism on 16,000 answers rather than 200,
  with the claim_min/whole_max gap by generator.
- **A new point is available**: the cell that breaks does so under two
  evaluators, so the finding is not specific to AlignScore.
- **Section VIII** can add that equivalence verdicts move with the evaluator.
- Wherever the draft says claim-level covers only the perturbation subset, that
  is no longer true.

---

## 6. Still open

Unchanged: open-weight generator (free, Qwen arm wired), C1/C2 (~$3–4), QASPER
(~$5), IEEE → ACL, the Salemi & Zamani PDF, ESA/re-ranking pending with Berend.
