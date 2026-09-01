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
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent))

import abstention
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

# The SQuAD normalizer now lives in textnorm alongside squash(), so
# abstention.py can share it without importing this module. Re-exported here
# because EM, token-F1 and every caller of correctness.normalize_answer expect
# it at this name.
normalize_answer = textnorm.squad_normalize


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
        ⚠️ NOT A CLEAN UPPER BOUND — measured 2026-08-30. The reasoning below
        is sound in one direction only: a long answer CAN mention the gold
        string incidentally while asserting something else, and on the
        200-row Claude/all-mpnet/NQ judge calibration that happens 5.85% of
        the time. But containment also MISSES correct answers — paraphrases,
        aliases, different units — at 8.29%, which is more. Net, containment
        UNDERSTATED accuracy: judge 0.800 vs containment 0.776 vs EM 0.000.

        So "the truth lies between EM and containment" is false. Nor is
        "containment understates" reliable: over the full grid (2026-08-30)
        its bias FLIPS SIGN by block — it understates by 5-15 points in 12
        cells but OVERSTATES by 3-7 points in the four HotpotQA cells under
        Claude, whose long multi-hop answers mention a reference string while
        asserting something else (false positives 8.9-13.0% there).

        Containment is a cheap proxy with error in both directions and no
        stable sign, not a bound. Report it as a proxy, give the judge number
        where one exists, and never present containment alone as "accuracy".

        90% of the abstentions that containment graded correct were overturned
        by the judge — that is the dominant false-positive mode, and it is why
        conditional.py reports n_correct_abstained.

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


# THE refusal rule — one definition, shared with perturbation_check.py, which
# used to carry a second and looser one (substring match over 9 phrases,
# anywhere in the answer). The two disagreed on 365 of 16,000 rows, so
# Section V's "answered" population was not Section VI's. See abstention.py
# for the measurement and for why this anchored rule is the canonical one.
_ABSTENTION_PHRASES = abstention.ABSTENTION_PHRASES
is_abstention = abstention.is_abstention


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


# ── choosing where `correct` comes from ──────────────────────────────────────
# score_records() writes all three heuristic signals; llm_judge.py writes a
# fourth, better one to a SEPARATE checkpoint. Until 2026-08-30 nothing read
# that fourth one: conditional.py and results.py went straight to `_scored`,
# so grading the grid with an LLM judge would have cost ~$32 and changed no
# number in the paper. load_correctness() is the single place that decides.
CORRECT_SOURCES = ('judge', 'contains', 'em', 'f1')
DEFAULT_JUDGE = 'claude'


def _heuristic_correct(rec: Dict, source: str,
                       threshold: float = CORRECT_F1_THRESHOLD) -> Optional[bool]:
    """
    Derive `correct` from the three stored signals rather than trusting the
    stored `correct` field, which was fixed at scoring time by whatever
    --mode happened to be passed then. None stays None: see score_records().
    """
    em = rec.get('correct_em')
    f1 = rec.get('correct_f1')
    con = rec.get('correct_contains')
    if em is None and f1 is None and con is None:
        return None
    if source == 'em':
        return bool(em)
    if source == 'f1':
        return bool(em or (f1 is not None and f1 >= threshold))
    return bool(con or em)          # 'contains', and the fallback for 'judge'


def load_correctness(name: str, source: str = 'judge',
                     judge: str = DEFAULT_JUDGE,
                     threshold: float = CORRECT_F1_THRESHOLD):
    """
    Per-query correctness for one generation checkpoint, from the chosen source.

    Returns (by_qid, coverage) where
        by_qid    {query_id: {'correct', 'abstained', 'correct_f1',
                              'correct_source'}}
        coverage  Counter — 'judge', 'heuristic', 'ungradable', 'total'.
                  Print it. A run that quietly fell back to containment for
                  every row is indistinguishable from a judged run in the
                  output tables, and that is exactly the mistake worth $32.

    source='judge' overlays the judge on the heuristic PER ROW, with three
    cases that must not be collapsed:
      - no judged row at all              -> heuristic (judge has not been run
                                             on this checkpoint yet)
      - judged, source_ungradable=True    -> None. The ROW has no usable
                                             prediction; the heuristic says
                                             None here too. Never False.
      - judged, correct=None, not
        source_ungradable                 -> a judge-side failure (rate limit,
                                             unparseable reply). Retryable, so
                                             fall back to the heuristic rather
                                             than dropping the query.
    """
    if source not in CORRECT_SOURCES:
        raise ValueError(f'source must be one of {CORRECT_SOURCES}, got {source!r}')

    base = load_checkpoint(f'{name}_scored')
    if not base:
        raw = load_checkpoint(name)
        if not raw:
            return {}, Counter()
        # Not scored yet — score in memory so callers still get an answer.
        # 'judge' grades on top of containment, which is the default mode.
        mode = source if source in CORRECT_MODES else CORRECT_MODE
        base = score_records(raw, threshold=threshold, mode=mode)

    judged = {}
    if source == 'judge':
        judged = {r['query_id']: r
                  for r in (load_checkpoint(f'{name}_judged_{judge}') or [])}

    out, cov = {}, Counter()
    for rec in base:
        qid = rec.get('query_id')
        if qid is None:
            continue
        cov['total'] += 1
        correct = _heuristic_correct(rec, source, threshold)
        abstained = rec.get('abstained')
        used = source if source != 'judge' else 'contains'

        j = judged.get(qid)
        if j is not None:
            if j.get('correct') is not None:
                correct = bool(j['correct'])
                # parse_verdict() returns abstention alongside the verdict; it
                # sees the same string is_abstention() does but understands
                # paraphrases, so prefer it when present.
                if j.get('abstained') is not None:
                    abstained = bool(j['abstained'])
                used = f'judge:{judge}'
            elif j.get('source_ungradable') or j.get('ungradable'):
                # 'ungradable' is the pre-2026-08-28 field name. Checkpoints
                # written before the resume fix used it for BOTH cases, so a
                # rate-limited row was frozen as permanently ungradable; the
                # backup's only judged file is 4 rows of 401 stored that way.
                # Reading it as ungradable is still safe: the heuristic returns
                # None for those rows too, because the source row is unusable.
                correct = None
            # else: judge-side failure -> keep the heuristic value above

        if correct is None:
            cov['ungradable'] += 1
        elif used.startswith('judge'):
            cov['judge'] += 1
        else:
            cov['heuristic'] += 1

        out[qid] = {'correct': correct, 'abstained': abstained,
                    'correct_f1': rec.get('correct_f1'),
                    'correct_source': used}
    return out, cov


def format_coverage(cov: Counter) -> str:
    """One-line summary of where a frame's correctness labels came from."""
    if not cov:
        return 'no rows'
    return (f'{cov["total"]} rows: judged {cov["judge"]}, '
            f'heuristic {cov["heuristic"]}, ungradable {cov["ungradable"]}')


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
            # '_judged_*' matches 'generated_*.pkl' but holds verdicts, not
            # generations — scoring it produces a junk '*_judged_*_scored'.
            if (stem.endswith('_partial') or stem.endswith('_scored')
                    or '_judged_' in stem):
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
