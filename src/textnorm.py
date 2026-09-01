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
import string

_NONWORD = re.compile(r'[^\w\s]', re.UNICODE)
_ARTICLES = re.compile(r'\b(a|an|the)\b', re.UNICODE)
_PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def squad_normalize(s: str) -> str:
    """SQuAD/NQ normalization: lowercase, DELETE punctuation, drop articles.

    Distinct from squash() on purpose. This is the field-standard rule for EM
    and token-F1, where both sides get the same treatment, and it is also what
    the abstention test runs on. squash() replaces punctuation with a space
    instead, because for a containment test deleting punctuation glues
    neighbouring tokens together and invents matches.

    Lives here rather than in correctness.py so abstention.py can use it
    without importing the correctness module. correctness.normalize_answer is
    an alias for it.
    """
    if s is None:
        return ''
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(' ', s)
    return ' '.join(s.split())


def squash(s: str) -> str:
    """Lowercase, punctuation -> space, whitespace collapsed."""
    if not s:
        return ''
    return ' '.join(_NONWORD.sub(' ', s.lower()).split())


def _pad(s: str) -> str:
    """Space-pad so a token-sequence test can be written as a substring test."""
    return f' {s} '


def contains(haystack: str, needle: str) -> bool:
    """True if `needle` appears in `haystack` as a whole token sequence.

    MATCHING IS ON TOKENS, NOT CHARACTERS (fixed 2026-09-01). squash() already
    reduces both sides to space-separated tokens, but the test used to be a raw
    substring test on the result, which matched inside words and inside longer
    numbers:

        contains('the budget was 10000 dollars', '1000')          -> True
        contains('he was born in 19051 census district', '1905')  -> True
        contains('Alice went to Paris', 'Ali')                    -> True

    Every false positive here is load-bearing: in the NQ loader it retains a
    query whose answer is not really in its own window, in build_qrels it marks
    a chunk relevant that does not carry the gold span (the B6 bug this module
    exists to prevent), and in correctness.contains_answer it grades an answer
    correct on a coincidental substring. Short gold answers — years, counts,
    single tokens — are exactly the ones that matched spuriously.

    Padding both sides with a space turns the substring test into a
    token-sequence test while keeping the NQ tokenisation behaviour the module
    docstring documents: gold "Röntgen's" squashes to `röntgen s` and the
    context's "Röntgen 's" squashes to `röntgen s`, so it still matches.

    An empty needle is never contained: callers treat "no gold span" as
    "cannot be verified", never as a trivially satisfied match.
    """
    n = squash(needle)
    return bool(n) and _pad(n) in _pad(squash(haystack))


def contains_any(haystack: str, needles) -> bool:
    h = _pad(squash(haystack))
    return any(n and _pad(n) in h for n in (squash(x) for x in needles or ()))
