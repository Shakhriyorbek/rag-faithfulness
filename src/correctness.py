"""
correctness.py — answer correctness scoring (Berend 2026-08-11, point 1).

WHY THIS EXISTS
    Faithfulness only carries meaning when the answer is correct. Before this
    module the pipeline had no correctness metric of any kind: the gold answer
    (`sample.answer`) reached the retrieval records and ESA, but never the
    faithfulness path. The reported `faithfulness` mean therefore pooled all
    four cells of the correctness x retrieval-quality grid, so a model that
    confidently grounded a WRONG answer in retrieved text scored high.

WHAT IT DOES
    Scores generation checkpoints that already exist. This is a pure re-scoring
    pass: no regeneration, no API calls, no cost.

    Adds three fields to every generation record:
        correct_em  - exact match after SQuAD normalization (strict)
        correct_f1  - token-level F1 against the gold answer (graded)
        correct     - bool; EM, or F1 >= CORRECT_F1_THRESHOLD

USAGE
    python src/correctness.py                 # score every generation checkpoint
    python src/correctness.py --dry-run       # report coverage, write nothing

Downstream, report faithfulness CONDITIONED on `correct` rather than pooled.
The `correct == False` cell is not a throwaway: high faithfulness on wrong
answers is direct evidence that retrieval quality does not imply answer
quality (Berend's branch A).
"""
from __future__ import annotations

import argparse
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent))

import config
import textnorm
from utils import load_checkpoint, save_checkpoint

# Graded-match threshold. QASPER answers are free-form, so strict EM is far too
# harsh there; 0.6 token-F1 is the conventional operating point for "the model
# said the right thing". Report EM alongside so the choice stays visible.
CORRECT_F1_THRESHOLD = 0.6

# Which signal `correct` is derived from. Default changed from 'f1' to
# 'contains' on 2026-08-13 after the pilot showed EM/F1 scoring the model at 2%
# accuracy where it was actually right 58% of the time — see contains_answer().
CORRECT_MODES = ('contains', 'f1', 'em')
CORRECT_MODE = 'contains'

_ARTICLES = re.compile(r'\b(a|an|the)\b', re.UNICODE)
_PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD/NQ normalization: lowercase, strip punctuation, articles, extra ws."""
    if s is None:
        return ''
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(' ', s)
    return ' '.join(s.split())


def exact_match(pred: str, golds: Sequence[str]) -> bool:
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in golds if g is not None)


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 (SQuAD definition), max over gold alternatives by caller."""
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        # Both empty -> perfect; exactly one empty -> zero.
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def best_f1(pred: str, golds: Sequence[str]) -> float:
    vals = [token_f1(pred, g) for g in golds if g is not None]
    return max(vals) if vals else 0.0


def contains_answer(pred: str, golds: Sequence[str]) -> bool:
    """
    True when a normalized gold answer appears verbatim inside the prediction.

    WHY THIS EXISTS (measured on the 2026-07-26 pilot, 2026-08-13)
        EM and token-F1 assume the prediction is roughly the length of the gold
        span. These generations are not: the RAG prompt produces
        "Based on the provided context, **Wilhelm Conrad Rontgen** of Germany
        received the first Nobel Prize in Physics in 1901. He received 150,782
        SEK for his discovery..." against a gold answer of
        "Wilhelm Conrad Rontgen, of Germany".

        That answer is RIGHT. EM scores it 0 and token-F1 scores it 0.28,
        because precision is destroyed by every extra token. Across the pilot,
        EM/F1-threshold accuracy was 2% while the model was actually correct on
        58% of queries. The metric was measuring verbosity, not correctness.

    HOW TO REPORT IT
        Containment is an UPPER bound: a long answer can mention the gold
        string incidentally while asserting something else. EM is a LOWER
        bound. Report both and state that the true value lies between them —
        do not quietly present containment alone as "accuracy".

    NORMALIZATION (changed 2026-08-14)
        Containment uses textnorm, not normalize_answer. normalize_answer
        DELETES punctuation, which glues tokens together: gold "Röntgen's"
        becomes "röntgens" while the generated "Röntgen 's" becomes
        "röntgen s", and the substring test fails on an answer that is
        plainly right. textnorm replaces punctuation with a space instead.
        normalize_answer stays in use for EM and token-F1, where both sides
        get the same treatment and the SQuAD-standard behaviour is wanted.
    """
    if not pred:
        return False
    return textnorm.contains_any(pred, [g for g in golds if g is not None])


def _gold_answers(record: Dict) -> List[str]:
    """
    Pull gold answer(s) off a generation record.

    Generation records are built as {**retrieval_record, 'generated_answer': ...},
    and the retrieval records carry 'answer' (embed_index puts `sample.answer`
    there). Accept a list or a single string; tolerate the alternative keys that
    the loaders may emit for multi-answer datasets.
    """
    for key in ('answer', 'answers', 'gold_answer', 'gold_answers'):
        if key in record and record[key] is not None:
            val = record[key]
            if isinstance(val, str):
                return [val]
            if isinstance(val, (list, tuple)):
                return [v for v in val if isinstance(v, str)]
    return []


# Abstentions are model behaviour, not failures: the closed-book prompt (C1)
# explicitly offers "I do not know", and the RAG prompt offers "I cannot answer
# based on the provided context". Both are INCORRECT for grading purposes, but
# they are a different phenomenon from a confident wrong answer and the C1
# floor is uninterpretable without separating them.
# Written as the prompts write them; normalized below so the comparison is
# apples to apples. normalize_answer() strips articles, so a literal
# "...the provided context" would never match its own normalized form.
_ABSTENTION_PHRASES = (
    'I do not know',
    "I don't know",
    'I cannot answer based on the provided context',
    "I can't answer based on the provided context",
)


def is_abstention(pred: str) -> bool:
    """True when the answer is a refusal/abstention rather than an attempt."""
    if not pred:
        return False
    p = normalize_answer(pred)
    return any(p.startswith(normalize_answer(m)) for m in _ABSTENTION_PHRASES)


def is_ungradable(record: Dict) -> bool:
    """
    True when this row carries no usable prediction, so correctness is
    UNDEFINED rather than False.

    Two cases, both of which the first version of this module silently graded
    as wrong answers:
      - '[ERROR: ...]' — an API failure. Counting a 429 as a wrong answer
        deflates the accuracy of whichever condition happened to hit rate
        limits, and does it invisibly.
      - generated_answer is None — the C2 oracle records queries with no gold
        evidence this way rather than generating under an 'oracle' label.
        Grading them as incorrect would understate the oracle CEILING, which
        is the one number the whole condition exists to establish.
    """
    pred = record.get('generated_answer')
    if pred is None:
        return True
    return isinstance(pred, str) and pred.startswith('[ERROR')


def score_records(records: Iterable[Dict],
                  threshold: float = CORRECT_F1_THRESHOLD,
                  mode: str = CORRECT_MODE) -> List[Dict]:
    """
    Return records with correctness fields added. Pure; does not mutate input.

    Always writes all three signals — `correct_em`, `correct_f1`,
    `correct_contains` — so the choice stays visible and reversible. `mode`
    only decides which one `correct` is derived from:

        'contains'  gold appears in the prediction        (default; see
                    contains_answer() for why, and for the upper-bound caveat)
        'f1'        EM, or token-F1 >= threshold          (SQuAD convention;
                    valid only for short-form output)
        'em'        exact match after normalization       (strictest)
    """
    if mode not in CORRECT_MODES:
        raise ValueError(f'mode must be one of {CORRECT_MODES}, got {mode!r}')
    out = []
    for r in records:
        rec = dict(r)
        golds = _gold_answers(rec)
        pred = rec.get('generated_answer') or ''
        rec['abstained'] = is_abstention(pred)
        if not golds or is_ungradable(rec):
            # No gold, or no usable prediction -> correctness is undefined.
            # Do NOT default to False: that would silently move rows into the
            # 'incorrect' cell and bias every conditional mean computed over it.
            rec['correct_em'] = None
            rec['correct_f1'] = None
            rec['correct_contains'] = None
            rec['correct'] = None
        else:
            em = exact_match(pred, golds)
            f1 = best_f1(pred, golds)
            con = contains_answer(pred, golds)
            rec['correct_em'] = bool(em)
            rec['correct_f1'] = round(float(f1), 4)
            rec['correct_contains'] = bool(con)
            rec['correct'] = bool({'contains': con or em,
                                   'f1': em or f1 >= threshold,
                                   'em': em}[mode])
        rec['correct_mode'] = mode
        out.append(rec)
    return out


# Every family of checkpoint that holds generated answers. `generated_*` alone
# would miss the controlled conditions, which is exactly where correctness
# matters most: the C1 floor and the C2 ceiling ARE accuracy numbers.
GENERATION_GLOBS = (
    'generated_*.pkl',    # main grid, per embedder
    'norag_*.pkl',        # C1 parametric floor
    'oracle_*.pkl',       # C2 oracle ceiling
    'ctx_*.pkl',          # context_ablation.py: subset / order / noise
    'util_*.pkl',         # doc_utility.py: eRAG / LOO / Shapley subsets
)


def _generation_checkpoint_names() -> List[str]:
    """Every generation checkpoint on disk, ignoring *_partial resume files."""
    names = []
    for pattern in GENERATION_GLOBS:
        for p in sorted(Path(config.CHECKPOINT_DIR).glob(pattern)):
            stem = p.stem
            if stem.endswith('_partial') or stem.endswith('_scored'):
                continue
            names.append(stem)
    return sorted(set(names))


def run_phase_correctness(dry_run: bool = False,
                          threshold: float = CORRECT_F1_THRESHOLD,
                          mode: str = CORRECT_MODE) -> Dict[str, Dict]:
    """
    Score every generation checkpoint; write `<name>_scored`.

    Returns {checkpoint_name: {n, n_scored, em_rate, f1_mean, correct_rate,
                               abstention_rate, n_ungradable}}.
    """
    summary: Dict[str, Dict] = {}
    names = _generation_checkpoint_names()
    if not names:
        print(f'[correctness] no generation checkpoints in {config.CHECKPOINT_DIR}')
        return summary

    for name in names:
        records = load_checkpoint(name)
        if not records:
            continue
        scored = score_records(records, threshold=threshold, mode=mode)
        graded = [r for r in scored if r['correct'] is not None]
        n_ok = sum(1 for r in graded if r['correct'])
        n_g = len(graded)
        stats = {
            'n': len(scored),
            'n_scored': n_g,
            'n_ungradable': len(scored) - n_g,
            'mode': mode,
            'em_rate': (sum(1 for r in graded if r['correct_em']) / n_g) if n_g else None,
            'f1_mean': (sum(r['correct_f1'] for r in graded) / n_g) if n_g else None,
            'contains_rate': (sum(1 for r in graded if r['correct_contains']) / n_g)
                             if n_g else None,
            'correct_rate': (n_ok / n_g) if n_g else None,
            # Abstention is measured over graded rows only, so an API outage
            # cannot masquerade as the model declining to answer.
            'abstention_rate': (sum(1 for r in graded if r['abstained']) / n_g)
                               if n_g else None,
        }
        summary[name] = stats

        if n_g:
            # EM is the lower bound, containment the upper. Printing them
            # together keeps the gap visible instead of letting one number
            # stand in for "accuracy".
            line = (f'  {name}: n={stats["n"]:>5}  '
                    f'correct[{mode}]={stats["correct_rate"]:.3f}  '
                    f'(EM={stats["em_rate"]:.3f} .. contains='
                    f'{stats["contains_rate"]:.3f})  '
                    f'F1={stats["f1_mean"]:.3f}  '
                    f'abstain={stats["abstention_rate"]:.3f}')
        else:
            line = f'  {name}: n={stats["n"]:>5}  correct=n/a (nothing gradable)'
        if stats['n_ungradable']:
            line += f'  [!] {stats["n_ungradable"]} ungradable'
        print(line)

        if not dry_run:
            save_checkpoint(f'{name}_scored', scored)

    if dry_run:
        print('[correctness] dry run — nothing written')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true',
                    help='report coverage without writing *_scored checkpoints')
    ap.add_argument('--threshold', type=float, default=CORRECT_F1_THRESHOLD,
                    help=f'token-F1 threshold for `correct` (default {CORRECT_F1_THRESHOLD})')
    ap.add_argument('--mode', choices=list(CORRECT_MODES), default=CORRECT_MODE,
                    help=f'signal `correct` derives from (default {CORRECT_MODE}; '
                         f"'f1' is the SQuAD convention and only valid for "
                         f'short-form output)')
    args = ap.parse_args()
    print(f'=== phase: correctness scoring, mode={args.mode} (no API cost) ===')
    run_phase_correctness(dry_run=args.dry_run, threshold=args.threshold,
                          mode=args.mode)


if __name__ == '__main__':
    main()
