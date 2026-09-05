# Report to Dr. Berend — 2026-09-05

> **Notes before sending — not part of the message.**
>
> The last reply (2026-08-28) worked because it was short. His three standing
> complaints were: walls of text, references to things never discussed, and
> prose that reads as genAI output with surface polish and no substance. The
> letter below is ~520 words and asks **two** questions. Do not add a third.
>
> **Rewrite any sentence that does not sound like you.** That matters more than
> the wording being optimal. This is a draft to edit, not a message to send
> as-is.
>
> Attach `paper/acl/main-preprint.pdf` (named, page-numbered). Not `main.pdf`
> — that one is the anonymous ARR version and will look odd to him.
>
> The letter deliberately leaves out: the six audit defects of 2026-09-01, the
> multiplicity-family analysis, the copying control, and the Salemi & Zamani
> full read. All are in `reports/`. Section 2 below has the full status if he
> asks for it.

---

## 1. The letter

**Subject:** Update — third generator, and a venue plan

Dear Dr. Berend,

Three things have changed since my last message.

**A third generator.** I ran Qwen2.5-7B on the university GPU, 8,000 answers,
no cost. It was the open-weight arm I had listed as a limitation. Two bugs had
to be fixed first — that code had never actually been run — and I have written
both up.

The result is the strongest thing in the project so far. Under AlignScore on
HotpotQA, all three generators rank the four embedding models in the same
order. Under the NLI metric, no two of them agree. The retrieval is shared
across the three, so this is three generators agreeing about a property of the
retrieval rather than three independent replications, and I say that in the
paper. But it makes the point sharper than the two-generator version did: which
evaluator you pick decides not just the significance but the ordering.

Adding the third generator made the multiple-comparison correction stricter,
over 18 tests instead of 8. The number of cells that break under AlignScore
still went up, from 1 to 2.

**One result of mine was an artifact, and I have written it up as a finding.**
Claude formats its answers with markdown; GPT-4o-mini almost never does. My NLI
scorer was reading the asterisks and my claim-level scorer was stripping them,
so the two were scoring different strings. Stripping markdown for both cost me
one of the two cells I had reported to you in August. The corrected headline is
now 2 of 6 rather than 2 of 4. The section explaining it is in the draft.

**Format.** The paper is now in ACL format rather than IEEE, so it can go to
ARR. Converting it showed the content runs to about 9.4 pages against ARR's
8-page limit, so roughly a page and a half has to come out.

On venue: I am comfortable aiming at a B-tier or C-tier conference rather than
holding out for a top one. As far as I can tell that does not mean submitting
somewhere smaller now. The next ARR deadline is **12 October 2026**, and that
one cycle feeds both NAACL 2027 and COLING 2027. COLING is CORE B. You choose
which one to commit the paper to in late December, after the reviews come back.
So the October deadline is the low-risk option, not the ambitious one.

That is five weeks. It is enough to update the results and cut the length. It
is not enough to also run the ESA and re-ranking experiments, which have never
been run and which the paper no longer depends on since the framing changed.

My two questions:

1. Do you agree with targeting 12 October, and with dropping ESA and
   re-ranking rather than delaying for them?
2. If a page and a half has to go, would you cut the definitions section or the
   discussion?

The current draft is attached.

Best regards,
Shakhriyorbek

---

## 2. Full status behind the letter — for reference, not for sending

### 2.1 What has been run since the 2026-08-28 message

| | result | cost |
|---|---|---|
| LLM judge over the full grid (08-30) | correctness measured, not proxied | $29.27 |
| Significance testing of `faith_gap` (08-30) | 3 of 16 cells significant, **all negative** | free |
| External audit fixes (09-01) | six defects, all measurement-affecting | free |
| The four open decisions (09-03) | D1–D4 taken, probe re-run | free |
| Copying control + multiplicity (09-04) | both close | free |
| Claim-level over the full grid (09-04) | evaluator family 8 → 12 rows | free |
| Open-weight arm (09-05) | Qwen2.5-7B, 8,000 answers | $0 |
| Salemi & Zamani read in full (09-05) | related work corrected | free |

Total paid to date: **$42.73**.

### 2.2 What died, and should stay dead

- **H1 is not supported.** Faithfulness differences between embedders are
  mostly null, and where they are not, the sign depends on the evaluator.
- **"More faithful when wrong" is false on NQ** for both API generators.
  It survived only because the baseline excluded refusals on one side.
- **The `hit × incorrect` cell is mostly refusals** — 25% to 60% depending on
  the cell — so its size is not evidence that retrieval was insufficient.
- **Containment is not a bound on accuracy in either direction.** It understates
  by 5–15 points in twelve cells and *overstates* by 3–7 in four.

Each of these was a claim in a draft he has read. They should be stated as
failures if he asks, not quietly dropped.

### 2.3 The venue analysis in full

The relevant fact is that ARR decouples review from venue.

| | |
|---|---|
| **Next ARR deadline** | **12 October 2026** (5 weeks from today) |
| Cycle feeds | NAACL 2027 **and** COLING 2027 |
| Commitment deadline | 20 December 2026 (COLING) / 23 December 2026 (NAACL) |
| NAACL 2027 | 1–5 June 2027, San Francisco — **CORE A** |
| COLING 2027 | 9–14 May 2027, Macau — **CORE B** |

You submit once, get reviews, and *then* pick the primary venue in December
with the scores in hand. A paper can be committed to only one primary venue and
that choice cannot be changed afterwards, so the decision is real — but it is
made after the information arrives, not before.

This is why "I am fine with B-tier" argues *for* the October deadline rather
than for a smaller venue now. COLING is the B-tier outcome and it is reachable
from the same submission as the A-tier one, at no extra cost and no extra
deadline. Committing does not guarantee acceptance — the conference still
decides — but the reviews are reusable either way.

**If October slips.** ARR runs roughly ten-week cycles, so the next deadline
after October should land around late December or January and would target
ACL 2027. That has not been posted yet, so treat the date as an estimate. The
cost of slipping is one cycle, about ten weeks, not a year.

**Not recommended:** a direct-submission C-tier venue now. It forgoes reusable
reviews, and the paper's problem is length and freshness of results, not that
it is aiming too high.

### 2.4 What has to happen before 12 October

Ordered by what blocks what.

1. **Update the results for B22 and B23.** Table VII goes from 8 rows to 18;
   the headline goes from "1 of 4" to "2 of 6"; Table III gains the qwen row.
   This is the bulk of the work and nothing else should start first.
2. **Cut ~1.5 pages.** Content is 9.4 pages against an 8-page limit. Going over
   is a desk reject, not a reviewer complaint.
3. **Remove the open-weight generator from the Limitations section** — it is
   listed there as missing and it is no longer missing.
4. **Judge correctness on the qwen arm**, ~$15. Optional: the arm's faithfulness
   results do not depend on it, but the correctness-conditioned analysis does.
5. Decide the two remaining paper items: IEEE→ACL is done, but §4.5.2 still
   says "cross-attention" where decoder-only models use self-attention over
   context tokens, and no code implements that analysis.

### 2.5 Open questions not asked in the letter

Held back deliberately — he answers what is asked, and a longer list gets a
shorter reply. Ask them next round.

- Whether the paper should report the within-evaluator Holm family alongside
  the table-wide one. The table-wide family is the conservative choice and is
  what the draft uses; the within-evaluator family is defensible and restores
  a cell. Both are printed on every run.
- Whether QASPER is worth ~$5 as a third dataset.
- Whether C1 (no-retrieval floor) and C2 (oracle ceiling) are worth ~$3–4.
  These came from his own August letter and have still never been run.
