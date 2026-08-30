# LLM-judge calibration — 2026-08-30

200 rows of `generated_claude_all-mpnet-base-v2_NQ`, judge `claude-opus-5`.
**$0.4085, 200 requests, 0 errors, $0.00199/row.** Log:
`gpu1:~/rag_faithfulness/logs/judge_calib.log`.

## 1. The judge does not track containment

n = 205 graded rows.

| signal | accuracy | agreement with judge |
|---|---|---|
| judge | 0.8000 | — |
| containment | 0.7756 | 0.8585 |
| exact match | 0.0000 | 0.2000 |

Disagreement is 14.1% at row level, in **both** directions:

- containment correct, judge wrong — **5.85%**
- containment wrong, judge correct — **8.29%**

That matters more than the 2.4pp gap in the aggregate rate suggests, because
the necessary/sufficient 2x2 is built per row from `correct` x `hit`. A row that
flips moves between cells even when the marginal totals barely move.

## 2. Containment is not an upper bound

`correctness.py` and `CLAUDE.md` both said EM is the lower bound and containment
the upper, with the truth in between. Measured, the truth is **above both**:
judge 0.800 > containment 0.776 > EM 0.000. The false-negative rate (8.29%)
exceeds the false-positive rate (5.85%), so containment understates on net.

The upper-bound reasoning was only ever half the story — it accounted for long
answers mentioning the gold string incidentally, but not for paraphrases,
aliases and unit differences that containment cannot see. Both files corrected.

## 3. B12 confirmed directly

Rows containment graded **correct** that are actually abstentions: **10**.
The judge overturns **9 of them (90%)**.

These are exactly the rows that were corrupting the `faith_gap` baseline (B12):
a short gold answer string appearing inside "I cannot answer based on the
provided context". At 6.7-11.7% of correct rows across the Claude cells, they
were large enough to flip the sign of the paper's headline on NQ.

Abstention detection itself is close: heuristic and judge agree on 31, with 3
found only by the heuristic and 3 only by the judge — so `is_abstention` is
sound and the problem was never abstention detection, only the grading of those
rows as correct.

## 4. Does the full run pay for itself

Yes. 14.1% row-level disagreement is far above the level at which the judge
"adds nothing", and the one measurement it settles — the abstained-correct
false positives — already changed a reported sign.

Cost: $0.00199/row. 16 checkpoints x 1,000 rows = **~$32** for the full grid.
Cheaper partial options, in order of value:

| scope | rows | cost |
|---|---|---|
| the 4 Claude x NQ checkpoints | 4,000 | ~$8 |
| all 8 Claude checkpoints (NQ + HotpotQA) | 8,000 | ~$16 |
| full grid, both generators | 16,000 | ~$32 |

The GPT-4o-mini arm matters here despite the cost: it has **zero**
abstained-correct rows on NQ, so the B12 correction did not move it. Judging
only Claude would leave the two arms graded by different standards, which is
precisely the confound H3 exists to avoid.

## 5. Caveat

One model x dataset x generator cell, n=205, single judge. The 5.85/8.29 split
is not established for HotpotQA, which has different answer forms and roughly
double the abstention rate. Treat the direction as established and the
magnitudes as provisional.
