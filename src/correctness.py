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
from utils import load_checkpoint, save_checkpoint

# Graded-match threshold. QASPER answers are free-form, so strict EM is far too
# harsh there; 0.6 token-F1 is the conventional operating point for "the model
# said the right thing". Report EM alongside so the choice stays visible.
CORRECT_F1_THRESHOLD = 0.6

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
                  threshold: float = CORRECT_F1_THRESHOLD) -> List[Dict]:
    """Return records with correctness fields added. Pure; does not mutate input."""
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
            rec['correct'] = None
        else:
            em = exact_match(pred, golds)
            f1 = best_f1(pred, golds)
            rec['correct_em'] = bool(em)
            rec['correct_f1'] = round(float(f1), 4)
            rec['correct'] = bool(em or f1 >= threshold)
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
                          threshold: float = CORRECT_F1_THRESHOLD) -> Dict[str, Dict]:
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
        scored = score_records(records, threshold=threshold)
        graded = [r for r in scored if r['correct'] is not None]
        n_ok = sum(1 for r in graded if r['correct'])
        n_g = len(graded)
        stats = {
            'n': len(scored),
            'n_scored': n_g,
            'n_ungradable': len(scored) - n_g,
            'em_rate': (sum(1 for r in graded if r['correct_em']) / n_g) if n_g else None,
            'f1_mean': (sum(r['correct_f1'] for r in graded) / n_g) if n_g else None,
            'correct_rate': (n_ok / n_g) if n_g else None,
            # Abstention is measured over graded rows only, so an API outage
            # cannot masquerade as the model declining to answer.
            'abstention_rate': (sum(1 for r in graded if r['abstained']) / n_g)
                               if n_g else None,
        }
        summary[name] = stats

        if n_g:
            line = (f'  {name}: n={stats["n"]:>5}  '
                    f'correct={stats["correct_rate"]:.3f}  '
                    f'EM={stats["em_rate"]:.3f}  F1={stats["f1_mean"]:.3f}  '
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
    args = ap.parse_args()
    print('=== phase: correctness scoring (no API cost) ===')
    run_phase_correctness(dry_run=args.dry_run, threshold=args.threshold)


if __name__ == '__main__':
    main()
