#!/usr/bin/env python3
"""Section 5.5 / Table 5: falsification response under a prompt-varied arm.

Compares the standard open-weight arm against the same model under a verbose
prompt, holding weights, queries and retrieved context fixed. Read-only over
perturb_* checkpoints. Run from the repo root.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'src')

B = os.environ.get('RAG_CHECKPOINT_DIR', 'checkpoints/n1000_v3') + '/'
M4 = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
      'text-embedding-3-small']
EVALUATORS = [('', 'NLI-max'), ('claim_', 'Claim-min'), ('align_', 'AlignScore')]
ARMS = ['qwen', 'qwen_verbose']


def wilson(k, n, z=1.96):
    """Wilson interval — the normal approximation is wrong at these rates."""
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def cell(prefix, arm, dataset):
    rows = []
    for m in M4:
        f = f'{B}perturb_{prefix}{arm}_{m}_{dataset}.pkl'
        if os.path.exists(f):
            rows += pickle.load(open(f, 'rb'))
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d = d[d['orig'].notna() & d['number'].notna()]
    elig = int((d['orig'] >= 0.5).sum())
    det = int(((d['orig'] >= 0.5) & (d['number'] < 0.5)).sum())
    lo, hi = wilson(det, elig)
    return dict(n=len(d), orig=d['orig'].mean(), fals=d['number'].mean(),
                delta=d['orig'].mean() - d['number'].mean(),
                det=det / elig, lo=lo, hi=hi,
                floor=d['random'].mean() if 'random' in d else float('nan'))


def main():
    print('%-11s %-9s %-9s %5s %7s %8s %7s  %s'
          % ('evaluator', 'dataset', 'arm', 'n', 'orig', 'delta', 'det',
             '95% CI'))
    for prefix, label in EVALUATORS:
        for ds in ['NQ', 'HotpotQA']:
            for arm in ARMS:
                c = cell(prefix, arm, ds)
                if c is None:
                    print(f'  MISSING {label} {arm} {ds}')
                    continue
                print('%-11s %-9s %-9s %5d %7.3f %+8.3f %6.1f%%  [%.1f%%, %.1f%%]'
                      % (label, ds, arm.replace('qwen_', ''), c['n'], c['orig'],
                         c['delta'], c['det'] * 100, c['lo'] * 100,
                         c['hi'] * 100))

    print('\nscale floor (untouched answer vs an unrelated document), NLI-max:')
    for ds in ['NQ', 'HotpotQA']:
        for arm in ARMS:
            c = cell('', arm, ds)
            if c:
                print('  %-9s %-13s %.3f' % (ds, arm, c['floor']))


if __name__ == '__main__':
    main()
