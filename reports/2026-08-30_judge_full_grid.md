# LLM-judge correctness over the full grid — 2026-08-30

16 checkpoints x 1,000 rows, judge `claude-opus-5`. **15,795 requests, 0 errors,
$29.27** (~$0.00186/row), 09:57 to 16:43. Log
`gpu1:~/rag_faithfulness/logs/judge_full.log`. This closes `PAPER_TODO.md` §3.1,
the top item since 2026-08-14.

Every accuracy number below is now a measurement. Containment is retired as the
correctness signal; it survives only as a reported proxy.

---

## 1. Containment is biased, and not even in a consistent direction

⚠️ **Corrected 2026-08-30.** An earlier version of this report said containment
understated accuracy in all 16 cells. That was wrong — generalised from the NQ
rows. In the four HotpotQA cells under Claude it *overstates*.

| block | containment | judge | delta | FP | FN |
|---|---|---|---|---|---|
| Claude / NQ | 0.709-0.738 | 0.760-0.796 | **+0.051 to +0.066** | 6.9-8.7% | 13.4-14.5% |
| Claude / HotpotQA | 0.599-0.734 | 0.528-0.703 | **-0.028 to -0.071** | 8.9-13.0% | 5.8-7.3% |
| GPT-4o-mini / NQ | 0.479-0.495 | 0.627-0.646 | **+0.148 to +0.151** | 0.0-0.1% | 14.8-15.1% |
| GPT-4o-mini / HotpotQA | 0.458-0.607 | 0.515-0.689 | **+0.057 to +0.082** | 2.5-3.2% | 8.7-10.7% |

FP = containment says correct, judge says wrong. FN = the reverse.

Which error dominates depends on the cell. Claude's HotpotQA answers are long
and multi-hop, so they mention a reference string while asserting something
else — the classic containment false positive, at 8.9-13.0%. Everywhere else
the false negatives (paraphrase, alias, unit) dominate.

**Containment is therefore a bound in neither direction.** The "EM lower,
containment upper" framing was wrong, but so is "containment understates" — its
bias flips sign between dataset-generator blocks, which is worse for a proxy
than a consistent offset would be.

**GPT-4o-mini was penalised worst on NQ.** Its containment false-positive rate
there is essentially zero and its false-negative rate ~15%, so containment
understated it by about 15 points while understating Claude by about 6. Under containment the
Claude-vs-GPT accuracy gap on NQ/all-mpnet read 0.725 vs 0.496 (22.9 pts); judged
it is 0.790 vs 0.646 (**14.4 pts**). A third of that gap was measurement error.

## 2. B12 is fully resolved

`n_correct_abstained` is **0 in all 16 cells**. The judge never grades a refusal
as correct, so the containment false positives that were corrupting the
`faith_gap` baseline are gone rather than merely corrected for.

## 3. "More faithful when wrong" does not survive

This was the draft's sharpest claim. With judged correctness and abstentions
excluded from both sides:

| | NQ | HotpotQA |
|---|---|---|
| Claude | **-0.051 .. -0.113** (all 4 negative) | -0.110 .. +0.058 (1 of 4 positive) |
| GPT-4o-mini | **-0.057 .. -0.115** (all 4 negative) | -0.031 .. +0.093 (3 of 4 positive) |

**NQ is uniformly negative for both generators** — answers are *more* grounded
when they are right, which is the opposite of the claim. Only HotpotQA still
shows positive gaps, and only for 4 of its 8 cells.

Trajectory of this number, all on the same 16,000 answers:

| grading | Claude / NQ faith_gap |
|---|---|
| containment, one-sided baseline (pre-B12) | +0.018 .. +0.050 |
| containment, symmetric baseline (B12 fixed) | -0.006 .. +0.018 |
| **LLM judge, symmetric baseline** | **-0.051 .. -0.113** |

Each step was a correctness fix, and each moved the number the same way. The
claim was an artifact of the grading signal.

⚠️ `n_wrong_answered` is now only **44-84 per cell** — refusals are excluded and
judged accuracy is higher, so the cell shrank. These gaps need a paired test
before they are reported as anything but descriptive.

## 4. The "not sufficient" cell is ~80% refusals

| generator | dataset | hit x incorrect | of which abstained | `share_answered` |
|---|---|---|---|---|
| Claude | NQ | 0.1742 | 483 / 697 (69%) | **0.0535** |
| Claude | HotpotQA | 0.3402 | 1182 / 1361 (87%) | **0.0447** |
| GPT-4o-mini | NQ | 0.3172 | 1003 / 1269 (79%) | **0.0665** |
| GPT-4o-mini | HotpotQA | 0.3625 | 1133 / 1450 (78%) | **0.0793** |

Branch A — "good retrieval was not sufficient" — is what this cell is for. Once
refusals are removed, the rate of *grounded, committed, wrong* answers is
**4.5-7.9%**, not the 23-45% the raw cell suggests. That is still a real
phenomenon and still supports the branch, but it is an order of magnitude
smaller than the draft implies, and the raw cell mostly measures how readily a
generator declines.

## 5. What has to change in the paper

1. Every accuracy figure — judged values, containment as a footnoted proxy.
2. The "more faithful when wrong" sentence — false on NQ for both generators.
   Scope it to HotpotQA, and only after a significance test on n=44-84.
3. The "not sufficient" share — report `share_answered`, with the abstention
   share alongside.
4. The Claude-vs-GPT accuracy gap — 14.4 pts on NQ/all-mpnet, not 22.9.
5. §3.3 and §6.4 can now be stated as measurements rather than provisional.

## 6. Not yet done

Faithfulness scores are unchanged — the judge grades correctness only, so
nothing here touches the evaluator-dependence result, which remains the paper's
spine. Still open: significance tests on the new gaps, C1/C2 anchors
(`pct_of_oracle` is still NaN), the open-weight arm, QASPER, and the ESA /
re-ranking decision with Berend.
