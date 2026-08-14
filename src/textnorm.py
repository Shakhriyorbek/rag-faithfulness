"""
textnorm.py — one canonical containment test, shared by every place that
asks "does this text contain that span?":

  - datasets_loader: is the annotated answer inside its own context window?
  - retrieval_eval:  does this chunk carry one of the gold sentences?
  - correctness:     does the generated answer contain the gold answer?

Why this file exists (measured on NQ, n=1000, 2026-08-14).
NQ documents are token LISTS joined back together with spaces, so the stored
context reads `Wilhelm Conrad Röntgen 's` and `1,020 - 1,080 kg`, while the
annotated answer reads `Wilhelm Conrad Röntgen's` and `1,020–1,080 kg`. A
whitespace-only normalizer calls those different strings. A strict test
reported **248/1000 answers "missing from their own context"**; only **61**
were genuinely absent — the other **187 were this artifact**.

It mattered in two places at once:
  1. the loader over-reported missing answers, and
  2. build_qrels used the same strict test to pick relevant chunks, found
     none, and fell back to marking the WHOLE document relevant — silently
     restoring the B6 bug (qrels claiming relevance for a chunk that does
     not carry the answer) for a quarter of the data.

Normalization rule: lowercase, replace every non-word character with a
SPACE, collapse whitespace. Punctuation is replaced rather than deleted on
purpose — deleting glues neighbouring tokens together (`28.0.0.137` ->
`2800137`) and invents matches that are not there. `\\w` is unicode-aware, so
`Röntgen` keeps its `ö`; an ASCII-only class would mangle every non-English
answer.

Articles are deliberately NOT stripped here. This is a containment test over
a literal span, not answer-equivalence scoring — correctness.normalize_answer
keeps the field-standard SQuAD normalization (which deletes punctuation and
drops articles) for EM and token-F1.
"""
import re

_NONWORD = re.compile(r'[^\w\s]', re.UNICODE)


def squash(s: str) -> str:
    """Lowercase, punctuation -> space, whitespace collapsed."""
    if not s:
        return ''
    return ' '.join(_NONWORD.sub(' ', s.lower()).split())


def contains(haystack: str, needle: str) -> bool:
    """True if `needle` appears in `haystack` under squash() normalization.

    An empty needle is never contained: callers treat "no gold span" as
    "cannot be verified", never as a trivially satisfied match.
    """
    n = squash(needle)
    return bool(n) and n in squash(haystack)


def contains_any(haystack: str, needles) -> bool:
    h = squash(haystack)
    return any(n and n in h for n in (squash(x) for x in needles or ()))
