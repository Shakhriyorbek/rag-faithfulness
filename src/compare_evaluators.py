"""
compare_evaluators.py — the embedder faithfulness comparison, under each
available evaluator.

WHY THIS EXISTS
    Every faithfulness number in the Rung 2 results was computed with
    `nli_max`. The 2026-08-26 perturbation experiment then measured `nli_max`
    as the least discriminating of the three evaluators available: on
    Claude/NQ it detected 4% of deliberately falsified values where AlignScore
    detected 29%. An equivalence claim resting on the least sensitive
    instrument is the weakest form of that claim, so the comparison is re-run
    here under each evaluator and the results placed side by side.

    A null that survives a more sensitive evaluator is a much stronger result.
    A null that does not survive it was an artifact of the instrument.

MARGIN SWEEP
    The paper asserted a TOST margin of +/-0.05 without justification, which
    the external review flagged. A fixed margin is also not transferable
    across evaluators: AlignScore and nli_max do not share a scale, so "0.05"
    does not mean the same thing in both. This reports equivalence across a
    range of margins instead, so the reader picks the bar.

    The perturbation result gives that bar an external anchor: on Claude's
    answers, outright falsifying one grounded value moves nli_max by
    0.033-0.094. A margin near 0.05 is therefore roughly "one falsified fact",
    which is not a negligible difference.

Read-only over existing checkpoints. No API spend.
"""
import argparse
import sys
from typing import List

import numpy as np

import config
from results import (FAITH_SOURCES, bootstrap_significance,
                     faithfulness_by_model, tost_equivalence)
from utils import set_scope

MARGINS = [0.01, 0.02, 0.03, 0.05, 0.10]


def _pairwise_tost(per_query: dict, margin: float) -> tuple:
    """(n_equivalent, n_pairs) over every unordered pair of models."""
    names = sorted(per_query)
    eq = tot = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = per_query[names[i]], per_query[names[j]]
            n = min(len(a), len(b))
            if n < 2:
                continue
            tot += 1
            if tost_equivalence(a[:n], b[:n], margin=margin).get('equivalent'):
                eq += 1
    return eq, tot


def compare(dataset: str, generator: str, models: List[str],
            metrics: List[str], correct_source: str = 'contains') -> list:
    out = []
    for metric in metrics:
        r = faithfulness_by_model(dataset, generator, models,
                                  answered_only=True, metric=metric,
                                  correct_source=correct_source)
        if not r:
            print(f'  [{dataset}/{generator}/{metric}] no checkpoints — skipped')
            continue

        pq = r['per_query']
        worst, best = r['worst'], r['best']
        n = min(len(pq[worst]), len(pq[best]))
        sig = bootstrap_significance(pq[worst][:n], pq[best][:n])
        p = sig.get('p_value', float('nan'))

        row = {'dataset': dataset, 'generator': generator, 'metric': metric,
               'correct_source': correct_source,
               'n_common': r['n_common'], 'means': r['means'],
               'spread': r['spread'], 'best': best, 'worst': worst,
               'p_value': p, 'tost': {}}
        for mg in MARGINS:
            row['tost'][mg] = _pairwise_tost(pq, mg)
        out.append(row)
    return out



def _report_source_sensitivity(rows, alpha: float = 0.05):
    """
    Where the significance verdict moves when `abstained` is decided by the
    judge instead of the heuristic.

    This exists because it does move. On 2026-08-30 the switch flipped
    NQ/Claude from evaluator-disagreement to agreement and created a fresh
    disagreement at HotpotQA/Claude, with p-values landing on 0.048 and 0.049.
    The paper's spine is that measurement choices decide the result; this is a
    second, independent measurement choice doing exactly that, so it belongs in
    the output rather than in a footnote.
    """
    by = {}
    for r in rows:
        by.setdefault((r['dataset'], r['generator'], r['metric']),
                      {})[r['correct_source']] = r
    moved = []
    for key, d in sorted(by.items()):
        if len(d) < 2:
            continue
        c, j = d.get('contains'), d.get('judge')
        vc, vj = c['p_value'] < alpha, j['p_value'] < alpha
        if vc != vj:
            moved.append((key, c, j))
    print('\n' + '=' * 92)
    print('ABSTENTION-SOURCE SENSITIVITY  (heuristic vs judge, alpha=%.2f)' % alpha)
    print('=' * 92)
    if not moved:
        print('  no cell changes verdict — the result is stable to this choice')
        return
    for (ds, gen, metric), c, j in moved:
        print('  %-9s %-10s %-6s  contains p=%.4f (%s, n=%d)  ->  '
              'judge p=%.4f (%s, n=%d)'
              % (ds, gen, metric, c['p_value'],
                 'DIFFERS' if c['p_value'] < alpha else 'null', c['n_common'],
                 j['p_value'], 'DIFFERS' if j['p_value'] < alpha else 'null',
                 j['n_common']))
    print('  -> %d of %d cells are not stable to how "abstained" is decided.'
          % (len(moved), len(by)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets', default='NQ,HotpotQA')
    ap.add_argument('--generators', default='claude,gpt4omini')
    ap.add_argument('--models', default=None)
    ap.add_argument('--metrics', default='nli,align,claim')
    ap.add_argument('--correct-source', default='contains',
                    choices=['contains', 'judge', 'both'],
                    help='which signal decides `abstained`, and so which rows '
                         'count as attempts. Default "contains" is the '
                         'deterministic heuristic and is what the published '
                         'result uses; "judge" is the robustness check; "both" '
                         'runs each and reports where the verdict moves')
    ap.add_argument('--scope-n', type=int, default=1000)
    args = ap.parse_args()

    set_scope(args.scope_n)
    models = ([m.strip() for m in args.models.split(',')] if args.models
              else [c['name'] for c in config.EMBEDDING_MODELS])
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    for m in metrics:
        if m not in FAITH_SOURCES:
            raise SystemExit(f'unknown metric {m!r}')

    sources = (['contains', 'judge'] if args.correct_source == 'both'
               else [args.correct_source])
    rows = []
    for src in sources:
        for ds in [d.strip() for d in args.datasets.split(',')]:
            for gen in [g.strip() for g in args.generators.split(',')]:
                rows += compare(ds, gen, models, metrics, correct_source=src)

    if not rows:
        print('No results — check the scope and that phase e has run.')
        return 1

    if len(sources) > 1:
        _report_source_sensitivity(rows)
    print('\n' + '=' * 92)
    print('EMBEDDER FAITHFULNESS SPREAD, answered-only, paired on a common subset')
    print('=' * 92)
    hdr = ('%-10s %-10s %-6s %-9s %6s %8s %8s %8s %9s'
           % ('dataset', 'generator', 'metric', 'abstain', 'n', 'worst',
              'best', 'spread', 'p'))
    print(hdr); print('-' * len(hdr))
    for r in rows:
        print('%-10s %-10s %-6s %-9s %6d %8.4f %8.4f %8.4f %9.4f'
              % (r['dataset'], r['generator'], r['metric'], r['correct_source'], r['n_common'],
                 r['means'][r['worst']], r['means'][r['best']],
                 r['spread'], r['p_value']))

    print('\n' + '=' * 92)
    print('TOST EQUIVALENCE across margins  (pairs equivalent / pairs tested)')
    print('A margin near 0.05 on nli_max is about the size of ONE falsified')
    print('fact (perturbation report, 2026-08-26) — not a negligible difference.')
    print('=' * 92)
    hdr2 = '%-10s %-10s %-6s %-9s' % ('dataset', 'generator', 'metric', 'abstain')
    hdr2 += ''.join('%12s' % ('+/-%.2f' % m) for m in MARGINS)
    print(hdr2); print('-' * len(hdr2))
    for r in rows:
        line = '%-10s %-10s %-6s %-9s' % (r['dataset'], r['generator'],
                                          r['metric'], r['correct_source'])
        for mg in MARGINS:
            eq, tot = r['tost'][mg]
            line += '%12s' % ('%d/%d' % (eq, tot))
        print(line)

    print('\n' + '=' * 92)
    print('PER-MODEL MEANS')
    print('=' * 92)
    for r in rows:
        print('\n%s / %s / %s  (n=%d)'
              % (r['dataset'], r['generator'], r['metric'], r['n_common']))
        for m, v in sorted(r['means'].items(), key=lambda kv: kv[1]):
            print('    %-28s %.4f' % (m, v))

    # Does the conclusion depend on the instrument?
    print('\n' + '=' * 92)
    print('DOES THE NULL SURVIVE A MORE SENSITIVE EVALUATOR?')
    print('=' * 92)
    # Keyed by abstention source too: with --correct-source both, omitting it
    # let the second pass silently overwrite the first and the section showed
    # one set of numbers under no label at all.
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r['dataset'], r['generator'],
                            r['correct_source']), {})[r['metric']] = r
    for (ds, gen, src), d in sorted(by_cell.items()):
        parts = []
        for metric in ('nli', 'align', 'claim'):
            if metric in d:
                verdict = 'null' if d[metric]['p_value'] >= 0.05 else 'DIFFERS'
                parts.append('%s: spread %.4f p=%.3f n=%d -> %s'
                             % (metric, d[metric]['spread'],
                                d[metric]['p_value'], d[metric]['n_common'],
                                verdict))
        print('  %s / %s   [abstained by: %s]' % (ds, gen, src))
        for p in parts:
            print('      ' + p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
