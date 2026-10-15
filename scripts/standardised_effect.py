#!/usr/bin/env python3
"""Appendix: spread in units of each evaluator's own paired SD.

Answers the reviewer's central question: does AlignScore break more cells
because it responds more, or because its per-query differences are quieter and
the same difference clears significance sooner? Dividing the spread by the
paired SD removes the variance term. Read-only. Run from the repo root.
"""
import sys

import numpy as np

sys.path.insert(0, 'src')
from utils import set_scope  # noqa: E402

set_scope(1000)
from results import faithfulness_by_model  # noqa: E402

M4 = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
      'text-embedding-3-small']
EVALUATORS = [('nli', 'NLI-max'), ('claim', 'Claim-min'), ('align', 'Align')]


def standardised(dataset, generator, metric):
    r = faithfulness_by_model(dataset, generator, M4, answered_only=True,
                              metric=metric, correct_source='contains')
    if not r:
        return None
    pq = r['per_query']
    best, worst = r['best'], r['worst']
    n = min(len(pq[best]), len(pq[worst]))
    d = np.asarray(pq[best][:n], float) - np.asarray(pq[worst][:n], float)
    return r['spread'] / d.std(ddof=1)


def main():
    print('%-9s %-12s %9s %9s %9s' % ('dataset', 'generator', *[l for _, l in EVALUATORS]))
    align_wins = total = 0
    for ds in ['NQ', 'HotpotQA']:
        for g in ['claude', 'qwen', 'gpt4omini']:
            vals = {lab: standardised(ds, g, m) for m, lab in EVALUATORS}
            if any(v is None for v in vals.values()):
                continue
            total += 1
            if vals['Align'] > vals['NLI-max']:
                align_wins += 1
            print('%-9s %-12s %9.3f %9.3f %9.3f'
                  % (ds, g, vals['NLI-max'], vals['Claim-min'], vals['Align']))
    print('\nAlignScore standardised effect > NLI-max in %d of %d cells'
          % (align_wins, total))
    print('If this were purely a variance difference, the two would be equal.')


if __name__ == '__main__':
    main()
