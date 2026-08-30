# faith_gap significance — 2026-08-30

Two-sample permutation test, 10,000 iterations, seed 42, Holm-Bonferroni across
the 8 tests in each generator arm. Correctness from the LLM judge; abstentions
excluded from both sides. `conditional.py --correct-source judge` prints it.

**Result: 3 of 16 cells are significant after correction, and all three are
NEGATIVE. No positive cell survives, and no positive cell has a confidence
interval excluding zero.**

"More faithful when wrong" — the draft's sharpest claim — has no statistical
support anywhere in the grid.

---

## 1. Claude

| dataset | model | n_wrong | gap | 95% CI | p_raw | p_holm | sig |
|---|---|---|---|---|---|---|---|
| NQ | E5-large-instruct | 58 | **-0.1131** | [-0.207, -0.028] | 0.0015 | **0.0120** | ✅ |
| NQ | text-embedding-3-small | 61 | **-0.1018** | [-0.185, -0.026] | 0.0027 | **0.0189** | ✅ |
| NQ | all-mpnet-base-v2 | 52 | -0.0755 | [-0.163, +0.003] | 0.0174 | 0.1044 | — |
| NQ | BGE-M3 | 63 | -0.0511 | [-0.124, +0.011] | 0.0723 | 0.3455 | — |
| HotpotQA | text-embedding-3-small | 44 | -0.1100 | [-0.239, +0.015] | 0.0691 | 0.3455 | — |
| HotpotQA | BGE-M3 | 44 | -0.0782 | [-0.209, +0.049] | 0.1973 | 0.5919 | — |
| HotpotQA | E5-large-instruct | 45 | -0.0490 | [-0.172, +0.068] | 0.4116 | 0.6228 | — |
| HotpotQA | all-mpnet-base-v2 | 50 | +0.0580 | [-0.040, +0.149] | 0.3114 | 0.6228 | — |

## 2. GPT-4o-mini

| dataset | model | n_wrong | gap | 95% CI | p_raw | p_holm | sig |
|---|---|---|---|---|---|---|---|
| NQ | E5-large-instruct | 71 | **-0.1153** | [-0.206, -0.028] | 0.0017 | **0.0136** | ✅ |
| NQ | text-embedding-3-small | 73 | -0.0705 | [-0.146, +0.003] | 0.0321 | 0.2247 | — |
| NQ | BGE-M3 | 67 | -0.0616 | [-0.138, +0.010] | 0.0591 | 0.3546 | — |
| NQ | all-mpnet-base-v2 | 72 | -0.0565 | [-0.136, +0.014] | 0.0828 | 0.3980 | — |
| HotpotQA | E5-large-instruct | 77 | +0.0928 | [-0.006, +0.187] | 0.0796 | 0.3980 | — |
| HotpotQA | BGE-M3 | 84 | +0.0129 | [-0.089, +0.114] | 0.7988 | 1.0000 | — |
| HotpotQA | text-embedding-3-small | 81 | +0.0033 | [-0.100, +0.106] | 0.9527 | 1.0000 | — |
| HotpotQA | all-mpnet-base-v2 | 82 | -0.0310 | [-0.137, +0.070] | 0.5522 | 1.0000 | — |

## 3. What holds

**The direction that survives is the opposite of the claim.** Every significant
cell is negative: answers are *more* grounded when they are correct. That is the
intuitive result, and it is what the corrected pipeline produces.

**NQ / E5-large-instruct replicates across generators** — Claude -0.1131 and
GPT-4o-mini -0.1153, effectively the same effect measured on two different
generators over the same retrieval. That is the single most robust number in
this table, and the only one that would survive a hostile reviewer.

**The largest positive gap in the grid is HotpotQA / E5-large-instruct /
GPT-4o-mini at +0.093**, and its CI is [-0.006, +0.187] — it *almost* excludes
zero, raw p=0.080, adjusted 0.398. It is the one cell where "more faithful when
wrong" might yet be real, and it is a candidate for a targeted follow-up, not
a result.

## 4. What must not be claimed

- That any embedder is more faithful when wrong. Zero of eight positive-signed
  cells reach significance, corrected or raw-with-CI.
- That HotpotQA behaves differently from NQ in this respect. The HotpotQA cells
  are uniformly non-significant with CIs 0.10-0.25 wide; they are uninformative,
  not evidence of a different direction.
- That 8 of 16 raw p-values below 0.10 mean anything collectively. Across 16
  tests at alpha=0.05 roughly one raw positive is expected by chance, which is
  why Holm is reported.

## 5. The honest limitation

`n_wrong` is **44-84** in every cell — refusals are excluded and judged accuracy
is high, so few wrong-but-answered rows remain. The CIs are correspondingly wide
(±0.08 to ±0.13). These tests are well powered to reject "more faithful when
wrong" as a general claim, and poorly powered to resolve individual cells. Read
the CI width, not the p-value.

Scaling n would mean more queries per dataset, not more embedders — the wrong
answers are the scarce resource.
