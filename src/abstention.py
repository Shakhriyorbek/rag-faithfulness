"""
abstention.py — ONE refusal rule, used everywhere.

WHY THIS FILE EXISTS (found 2026-09-01)
    There were two `is_abstention()` functions and they did not select the
    same rows:

                        perturbation_check          correctness
        markers         9 phrases                   4 phrases
        normalisation   .lower() on raw text        SQuAD normalize_answer
        match rule      substring, ANYWHERE         startswith (anchored)

    Measured over all 16,000 generations in checkpoints/n1000_v3:

        both detectors agree it is a refusal   4,016
        perturbation_check only                  365
        correctness only                           0
        neither                               11,619

    So the substring rule is a strict superset, and its extra 365 rows are
    dominated by mid-answer hedges: "Based on the provided context, the
    following people were involved in the Mapp v. Ohio case: ... However, the
    context does not contain their ages." That is a substantive answer. It was
    excluded from the falsification sample (Section V) while counting as
    answered in the conditional analysis (Section VI), so the two sections
    described different populations as "answered".

THE CANONICAL RULE
    The anchored `startswith` test on SQuAD-normalised text, i.e. what
    correctness.py did. It is the stricter and better-justified of the two —
    a refusal is what the answer OPENS with, not a caveat it contains — and it
    is the rule whose agreement with the LLM judge is reported in the paper
    (Section VI-A). Adopting it changes no published number: every row it
    calls a refusal was already a refusal on both sides.

    The extra phrases from perturbation_check were NOT kept. Classifying the
    365 disagreements shows 253 (69%) are mid-answer hedges, which is what a
    substring rule buys you and why it is wrong.

THE PREAMBLE GAP — CLOSED 2026-09-03
    The remaining 112 (31%) are genuine refusals that open with a preamble
    ("Based on the provided context, I cannot answer this question."), which a
    bare anchored rule misses. That is a gap in the phrase list, not a reason
    to match hedges anywhere in an answer, so it is fixed where it belongs:
    the canonical rule now strips a leading "Based on ...," / "According to
    ...," preamble before the anchored test, over a wider phrase list.

    Audited over all 16,000 generations before adoption: 107 rows change from
    attempt to refusal (0.67% of the grid), ALL of them Claude's — 80 on
    HotpotQA, 27 on NQ, none on GPT-4o-mini. 105 of the 107 open with "I
    cannot answer this question"; the other two are "The context does not
    specify ..." and "I cannot determine ...". Every sampled row is a genuine
    refusal that explains itself, e.g. "Based on the provided context, I
    cannot answer this question. The context mentions that Home Alone 2 is set
    in New York ... but it does not contain ...".

    This moves the answered-only population behind Sections V through VIII, so
    every affected analysis was re-run rather than carried over.
"""
import re
from typing import Tuple

import textnorm

# Written as the prompts write them; normalised below so the comparison is
# apples to apples. textnorm.squad_normalize() strips articles, so a literal
# "...the provided context" would never match its own normalised form.
#
# The closed-book prompt (C1) explicitly offers "I do not know" and the RAG
# prompt offers "I cannot answer based on the provided context". Both are
# INCORRECT for grading purposes but are a different phenomenon from a
# confident wrong answer, and the C1 floor is uninterpretable without
# separating them.
ABSTENTION_PHRASES: Tuple[str, ...] = (
    'I do not know',
    "I don't know",
    'I cannot answer based on the provided context',
    "I can't answer based on the provided context",
)

# Preamble both generators prepend to everything, refusals included.
# Stripped by the canonical rule before the anchored test.
_PREAMBLE = r'^\s*(?:based on|according to)[^,.:]{0,60}[,:]\s*'
_PREAMBLE_RE = re.compile(_PREAMBLE, re.I)

# Refusal openings the bare ABSTENTION_PHRASES list does not cover.
_EXTENDED_PHRASES: Tuple[str, ...] = ABSTENTION_PHRASES + (
    'I cannot answer this question',
    'I cannot determine',
    'I am unable to answer',
    'The context does not',
    'The provided context does not',
    'There is no information',
)


def is_abstention(pred: str) -> bool:
    """
    THE canonical rule: True when the answer OPENS with a refusal.

    A leading "Based on the provided context," / "According to the context,"
    preamble is stripped first, because both generators prepend it to
    everything including their refusals. The test is still anchored — a
    caveat carried in the middle of a substantive answer is not a refusal.
    """
    if not pred:
        return False
    stripped = _PREAMBLE_RE.sub('', pred)
    p = textnorm.squad_normalize(stripped)
    return any(p.startswith(textnorm.squad_normalize(m))
               for m in _EXTENDED_PHRASES)


def is_abstention_bare(pred: str) -> bool:
    """
    The pre-2026-09-03 rule: anchored, no preamble strip, short phrase list.

    Kept so the 107-row difference stays measurable and the older reported
    numbers remain reproducible. Not used by any analysis.
    """
    if not pred:
        return False
    p = textnorm.squad_normalize(pred)
    return any(p.startswith(textnorm.squad_normalize(m))
               for m in ABSTENTION_PHRASES)


# Back-compat alias: the extended rule IS the canonical rule as of 2026-09-03.
is_abstention_extended = is_abstention
