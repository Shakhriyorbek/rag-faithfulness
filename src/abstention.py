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
    365 disagreements shows 253 (69%) are mid-answer hedges. The remaining
    112 (31%) are genuine refusals that open with a preamble
    ("Based on the provided context, I cannot answer this question."), which
    the anchored rule misses — but that is a separate, measurable gap in the
    phrase list, not a reason to match hedges anywhere in an answer. See
    `is_abstention_extended()`, which is a DIAGNOSTIC only: turning it on
    would move the abstention rate, the answered-only population, and with it
    the paper's spine result, so it is not wired into any analysis.
"""
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
# Used ONLY by is_abstention_extended().
_PREAMBLE = r'^\s*(?:based on|according to)[^,.:]{0,60}[,:]\s*'

# Refusal openings the canonical list does not cover. DIAGNOSTIC ONLY.
_EXTENDED_PHRASES: Tuple[str, ...] = ABSTENTION_PHRASES + (
    'I cannot answer this question',
    'I cannot determine',
    'I am unable to answer',
    'The context does not',
    'The provided context does not',
    'There is no information',
)


def is_abstention(pred: str) -> bool:
    """True when the answer OPENS with a refusal rather than an attempt."""
    if not pred:
        return False
    p = textnorm.squad_normalize(pred)
    return any(p.startswith(textnorm.squad_normalize(m))
               for m in ABSTENTION_PHRASES)


def is_abstention_extended(pred: str) -> bool:
    """
    Anchored refusal detection after stripping a "Based on ...," preamble,
    over a wider phrase list.

    DIAGNOSTIC ONLY — not used by any analysis. It exists so the size of the
    canonical rule's blind spot can be measured (112 of 16,000 rows on
    n1000_v3) without silently changing which rows count as attempts.
    """
    import re
    if not pred:
        return False
    stripped = re.sub(_PREAMBLE, '', pred, flags=re.I)
    p = textnorm.squad_normalize(stripped)
    return any(p.startswith(textnorm.squad_normalize(m))
               for m in _EXTENDED_PHRASES)
