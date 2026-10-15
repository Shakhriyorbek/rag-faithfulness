#!/usr/bin/env python3
"""C3b: an omnibus test, and the system x evaluator interaction.

Two objections the reviewer raises about the headline count:

1. The spread is max-minus-min over four systems, so the two systems entering
   it are chosen for being extreme, and its p-value is not adjusted for that
   choice. Fixed here by permuting system labels WITHIN each query and
   re-selecting the extremes in every resample, so the selection is part of
   the null. A Friedman test is reported alongside as a standard omnibus.

2. The count depends on treating cells as separate tests. A model with
   evaluator and system as factors asks the paper's actual question directly:
   does the pattern across systems depend on which evaluator scored them?
   Scores are z-standardised within evaluator first, because the three do not
   share a scale or a variance.

Read-only over scored checkpoints. Run from the repo root.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from utils import set_scope  # noqa: E402

set_scope(1000)
from correctness import load_correctness  # noqa: E402
from results import FAITH_SOURCES, holm_bonferroni  # noqa: E402
from utils import load_checkpoint  # noqa: E402

M4 = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
      'text-embedding-3-small']
GENS = ['claude', 'qwen', 'gpt4omini']
METRICS = [('nli', 'NLI-max'), ('align', 'AlignScore'), ('claim', 'Claim-min')]
NRESAMP = 10000
SEED = 42


def paired_matrix(dataset, generator, metric):
    """Queries x systems matrix of faithfulness, attempts only, common subset."""
    prefix, field = FAITH_SOURCES[metric]
    per = {}
    for m in M4:
        scores = load_checkpoint(f'{prefix}_{generator}_{m}_{dataset}')
        if not scores:
            continue
        graded, _ = load_correctness(
            f'generated_{generator}_{m}_{dataset}', source='contains')
        drop = {q for q, r in graded.items() if r.get('abstained')}
        per[m] = {r['query_id']: r[field] for r in scores
                  if r['query_id'] not in drop
                  and r.get(field) is not None and np.isfinite(r[field])}
    if len(per) < 2:
        return None, None
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    cols = [m for m in M4 if m in per]
    X = np.array([[per[m][q] for m in cols] for q in common], float)
    return X, common


def omnibus(X, rng):
    """Friedman, plus a permutation test on max-min that re-selects extremes."""
    fried = stats.friedmanchisquare(*[X[:, j] for j in range(X.shape[1])])
    obs = X.mean(axis=0)
    obs_range = obs.max() - obs.min()
    n, k = X.shape
    count = 0
    for _ in range(NRESAMP):
        # permute system labels independently within each query
        idx = np.argsort(rng.random((n, k)), axis=1)
        permuted = np.take_along_axis(X, idx, axis=1)
        mu = permuted.mean(axis=0)
        if (mu.max() - mu.min()) >= obs_range - 1e-12:
            count += 1
    return fried.pvalue, obs_range, (count + 1) / (NRESAMP + 1)


def main():
    rng = np.random.default_rng(SEED)
    print('=' * 86)
    print('PART 1  Omnibus per cell: do the four systems differ at all?')
    print('=' * 86)
    print('%-9s %-10s %-11s %7s %10s %12s' %
          ('dataset', 'generator', 'evaluator', 'range', 'Friedman p', 'perm p'))
    rows = []
    for ds in ['NQ', 'HotpotQA']:
        for g in GENS:
            for metric, lab in METRICS:
                X, _ = paired_matrix(ds, g, metric)
                if X is None or len(X) < 30:
                    continue
                fp, rng_obs, pp = omnibus(X, rng)
                rows.append((f'{ds}/{g}/{lab}', rng_obs, fp, pp))
                print('%-9s %-10s %-11s %7.4f %10.4f %12.4f'
                      % (ds, g, lab, rng_obs, fp, pp))
    for name, ps in (('Friedman', [r[2] for r in rows]),
                     ('permutation (extremes re-selected)', [r[3] for r in rows])):
        adj = holm_bonferroni(ps, alpha=0.05)
        adj_p = [a[1] if isinstance(a, (list, tuple)) else a for a in adj]
        brk = [rows[i][0] for i, p in enumerate(adj_p) if p < 0.05]
        print('\n  %s: %d of %d cells significant after Holm over %d'
              % (name, len(brk), len(rows), len(rows)))
        for b in brk:
            print('      ', b)

    print('\n' + '=' * 86)
    print('PART 2  Mixed model: does the system pattern depend on the evaluator?')
    print('=' * 86)
    from statsmodels.regression.mixed_linear_model import MixedLM
    for ds in ['NQ', 'HotpotQA']:
        for g in GENS:
            recs = []
            for metric, lab in METRICS:
                X, common = paired_matrix(ds, g, metric)
                if X is None:
                    continue
                z = (X - X.mean()) / X.std(ddof=1)      # standardise per evaluator
                cols = [m for m in M4][:X.shape[1]]
                for qi, q in enumerate(common):
                    for sj, sysname in enumerate(cols):
                        recs.append((q, sysname, lab, z[qi, sj]))
            if not recs:
                continue
            d = pd.DataFrame(recs, columns=['query', 'system', 'evaluator', 'z'])
            # REML likelihoods are NOT comparable across different fixed-effects
            # structures, so the LR test is fitted by ML. The Wald test on the
            # interaction terms is reported too, since it is valid either way
            # and does not depend on the optimiser reaching the same basin.
            full = MixedLM.from_formula('z ~ C(system) * C(evaluator)',
                                        groups='query', data=d).fit(reml=False)
            add = MixedLM.from_formula('z ~ C(system) + C(evaluator)',
                                       groups='query', data=d).fit(reml=False)
            lr = 2 * (full.llf - add.llf)
            names = [n for n in full.params.index if ':' in n]
            df = len(names)
            p_lr = stats.chi2.sf(lr, df) if (df > 0 and lr > 0) else float('nan')
            R = np.zeros((df, len(full.params)))
            for r, nm in enumerate(names):
                R[r, list(full.params.index).index(nm)] = 1.0
            w = full.wald_test(R, scalar=True)
            p_w = float(w.pvalue)
            print('%-9s %-10s n=%6d  LR=%7.2f df=%d p=%-8.4g  Wald chi2=%7.2f p=%.4g'
                  % (ds, g, len(d), lr, df, p_lr, float(w.statistic), p_w))
    print('\nA significant interaction means the ordering or spacing of the four')
    print('systems is not the same under every evaluator -- the paper\'s claim,')
    print('tested without selecting extremes or splitting into cells.')


if __name__ == '__main__':
    main()
