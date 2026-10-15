#!/usr/bin/env python3
"""C7: is the probe's non-random prefix exchangeable with the eligible set?

Cases are the first 200 eligible answers in dataset order, not a random 200,
but the Wilson intervals reported beside the detection rates assume random
sampling. If detection does not trend with position inside the prefix, the
prefix behaves like a random sample and the intervals stand.

Reports detection rate by quartile of position, plus a Spearman trend test of
per-case detection against position. Read-only. Run from the repo root.
"""
import os
import pickle
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, 'src')

B = os.environ.get('RAG_CHECKPOINT_DIR', 'checkpoints/n1000_v3') + '/'
M4 = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
      'text-embedding-3-small']
GENS = ['claude', 'qwen', 'gpt4omini']


def cases(generator, dataset):
    """Per-embedder case lists, each kept in its original prefix order."""
    out = []
    for m in M4:
        f = f'{B}perturb_{generator}_{m}_{dataset}.pkl'
        if os.path.exists(f):
            out.append(pickle.load(open(f, 'rb')))
    return out


def main():
    print('C7: detection rate by position within the 200-case prefix (NLI-max)')
    print('%-9s %-10s %7s %7s %7s %7s   %8s %7s'
          % ('dataset', 'generator', 'Q1', 'Q2', 'Q3', 'Q4', 'rho', 'p'))
    all_rho = []
    for ds in ['NQ', 'HotpotQA']:
        for g in GENS:
            pos, det = [], []
            for rows in cases(g, ds):
                for i, r in enumerate(rows):
                    o, n = r.get('orig'), r.get('number')
                    if o is None or n is None or not np.isfinite(o):
                        continue
                    if o < 0.5:          # not eligible for the fixed gate
                        continue
                    pos.append(i)
                    det.append(1 if n < 0.5 else 0)
            if len(det) < 40:
                continue
            pos, det = np.array(pos), np.array(det)
            qs = []
            for lo, hi in [(0, 50), (50, 100), (100, 150), (150, 1 << 30)]:
                sel = (pos >= lo) & (pos < hi)
                qs.append(det[sel].mean() if sel.sum() else float('nan'))
            rho, p = stats.spearmanr(pos, det)
            all_rho.append((f'{ds}/{g}', rho, p))
            print('%-9s %-10s %6.1f%% %6.1f%% %6.1f%% %6.1f%%   %+8.3f %7.3f'
                  % (ds, g, *[q * 100 for q in qs], rho, p))
    print()
    sig = [(n, r, p) for n, r, p in all_rho if p < 0.05]
    print('cells with a significant position trend at alpha=0.05: %d of %d'
          % (len(sig), len(all_rho)))
    for n, r, p in sig:
        print('   %-20s rho=%+.3f p=%.4f' % (n, r, p))
    if not sig:
        print('   none -- detection does not trend with position, so the prefix')
        print('   is exchangeable with the eligible set on this measure.')


if __name__ == '__main__':
    main()
