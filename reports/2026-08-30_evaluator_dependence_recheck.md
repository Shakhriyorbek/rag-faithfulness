# Evaluator dependence — re-checked after the judge run, 2026-08-30

`compare_evaluators.py --metrics nli,align --correct-source both --scope-n 1000`.
Free; no faithfulness score was recomputed. The question was whether the paper's
spine survives the correctness work done today.

**It does. The published numbers reproduce exactly, and the headline count holds
under both bases — but not in the same cells.**

---

## 1. The published result reproduces

Abstentions decided by the heuristic (`is_abstention`), which is what the
2026-08-26 report used and what the code now defaults to again:

| dataset / generator | n | NLI-max | AlignScore | |
|---|---|---|---|---|
| NQ / Claude | 792 | 0.0090 p=0.223 null | 0.0179 **p=0.0005** | **breaks** |
| NQ / GPT-4o-mini | 628 | 0.0129 p=0.099 null | 0.0080 p=0.118 null | — |
| HotpotQA / Claude | 538 | 0.0190 p=0.270 null | 0.0126 p=0.151 null | — |
| HotpotQA / GPT-4o-mini | 510 | 0.0267 p=0.065 null | 0.0461 **p=0.0005** | **breaks** |

Identical to `reports/2026-08-26_evaluator_dependence.md`. 2 of 4 nulls break
under the more sensitive evaluator.

## 2. It also holds when abstentions come from the judge — in different cells

| dataset / generator | n | NLI-max | AlignScore | |
|---|---|---|---|---|
| NQ / Claude | 773 | 0.0141 **p=0.048** | 0.0179 **p=0.0002** | agree |
| NQ / GPT-4o-mini | 628 | 0.0129 p=0.099 null | 0.0080 p=0.118 null | — |
| HotpotQA / Claude | 483 | 0.0071 p=0.679 null | 0.0180 **p=0.049** | **breaks** |
| HotpotQA / GPT-4o-mini | 503 | 0.0254 p=0.073 null | 0.0432 **p=0.0007** | **breaks** |

Still 2 of 4. But NQ/Claude stops disagreeing and HotpotQA/Claude starts.

Per-cell, 2 of 8 dataset x generator x metric combinations change verdict:

```
HotpotQA claude align   contains p=0.1510 (null, n=538) -> judge p=0.0487 (DIFFERS, n=483)
NQ       claude nli     contains p=0.2233 (null, n=792) -> judge p=0.0481 (DIFFERS, n=773)
```

Both land on 0.048-0.049. Knife-edge.

## 3. Why correctness touches a faithfulness result at all

`faithfulness_by_model(answered_only=True)` drops abstentions, so "which rows
are attempts" is an input to the comparison. Switching that decision from the
heuristic to the judge moves 19 rows on NQ/Claude and 55 on HotpotQA/Claude —
enough to cross alpha in two places.

GPT-4o-mini is untouched in both NQ cells (n=628 either way): the judge and the
heuristic agree perfectly on its refusals there. The movement is entirely in the
Claude arm, whose answers are longer and whose refusals are more often
paraphrased rather than templated.

## 4. What to report

The heuristic is the default and should stay the primary basis: it is
deterministic, and abstention detection is string matching that
`is_abstention()` does well (31 agreements against 3+3 disagreements on the
calibration sample). Making a headline depend on an LLM's abstention call buys
nothing.

But this belongs in the paper rather than in a footnote. The thesis is that
measurement choices decide the result; here a **second, independent**
measurement choice — how you decide which answers count as attempts — moves the
same conclusion. That is a stronger version of the argument than the evaluator
result alone.

What must not be claimed: that any specific cell shows an effect. The count
(2 of 4) is stable; the identity of the cells is not.

## 5. TOST is unchanged in substance

Equivalence still fails at +/-0.01 almost everywhere and holds at +/-0.05 nearly
everywhere, under both bases. HotpotQA/GPT-4o-mini under AlignScore is the
hardest cell — 0/6 equivalent pairs out of six until +/-0.05. Since one
falsified fact moves NLI-max by 0.033-0.094, a +/-0.05 margin is about the size
of a single fabrication, so "equivalent at +/-0.05" is a weak statement and the
curve should keep being reported instead of a point.
