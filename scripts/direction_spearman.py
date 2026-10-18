#!/usr/bin/env python3
"""Direction of the embedder effect, and what pooling refusals does to it.

Produces two things the paper reports and nothing in src/ computed, which is
how the 2026-08-25 copying control went stale (CLAUDE.md B20):

  1. Section "direction": Spearman of mean NDCG@5 against mean faithfulness
     across the four matched systems, for every dataset x generator x
     evaluator combination. With four systems and no ties rho can only take
     multiples of 0.2, so these coefficients carry no individual significance
     and are read as a sign pattern.

  2. Table 4: the same correlation computed on the POOLED population
     (refusals included) beside the answered-only one. Pooling is what makes
     the column track retrieval quality, and the sign of that tracking is
     generator-dependent, so both populations are printed side by side and
     every figure names which it is.

Read-only. Run from the repo root with RAG_CHECKPOINT_DIR set to the data tree.
"""
import os
import pickle
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, 'src')
from utils import set_scope  # noqa: E402

set_scope(1000)
from results import faithfulness_by_model  # noqa: E402

M4 = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
      'text-embedding-3-small']
GENS = ['claude', 'qwen', 'gpt4omini', 'mistral']
EVALS = [('nli', 'NLI-max'), ('claim', 'Claim-min'), ('align', 'AlignScore')]
CKPT = (os.environ.get('RAG_CHECKPOINT_DIR')
        or os.path.expanduser('~/rag-backup/checkpoints/n1000_v3')).rstrip('/')


def mean_ndcg(model, dataset):
    f = '%s/per_query_rq_%s_%s.pkl' % (CKPT, model, dataset)
    if not os.path.exists(f):
        return None
    d = pickle.load(open(f, 'rb'))
    return float(np.mean([v['NDCG@5'] for v in d.values()]))


def rho(dataset, generator, metric, answered_only):
    r = faithfulness_by_model(dataset, generator, M4,
                              answered_only=answered_only, metric=metric,
                              correct_source='contains')
    if not r:
        return None, None
    nd, fa = [], []
    for m in M4:
        n = mean_ndcg(m, dataset)
        if n is None or m not in r['means']:
            continue
        nd.append(n)
        fa.append(r['means'][m])
    if len(nd) < 3:
        return None, None
    rr = stats.spearmanr(nd, fa)
    return float(rr.statistic), r['spread']


def main():
    print('checkpoints:', CKPT)
    print('\nNDCG@5 per system (fixed across generators):')
    for ds in ['NQ', 'HotpotQA']:
        parts = ['%s %.4f' % (m, mean_ndcg(m, ds)) for m in M4
                 if mean_ndcg(m, ds) is not None]
        print('  %-9s %s' % (ds, '   '.join(parts)))

    print('\n' + '=' * 78)
    print('DIRECTION  rho(NDCG@5, mean faithfulness), ANSWERED-ONLY, 4 systems')
    print('=' * 78)
    print('%-9s %-10s %-11s %8s %9s' % ('dataset', 'generator', 'evaluator',
                                        'rho', 'spread'))
    pos = neg = zero = 0
    rows = []
    for ds in ['NQ', 'HotpotQA']:
        for g in GENS:
            for mk, ml in EVALS:
                r, sp = rho(ds, g, mk, True)
                if r is None:
                    continue
                rows.append((ds, g, ml, r))
                print('%-9s %-10s %-11s %8.2f %9.4f' % (ds, g, ml, r, sp))
                if r > 0:
                    pos += 1
                elif r < 0:
                    neg += 1
                else:
                    zero += 1
    print('\n  %d combinations: %d positive, %d negative, %d zero'
          % (len(rows), pos, neg, zero))
    for ds in ['NQ', 'HotpotQA']:
        sub = [r for d, g, m, r in rows if d == ds]
        print('  %-9s positive %d/%d, zero-or-negative %d/%d'
              % (ds, sum(1 for r in sub if r > 0), len(sub),
                 sum(1 for r in sub if r <= 0), len(sub)))
    for mk, ml in EVALS:
        sub = [(d, r) for d, g, m, r in rows if m == ml]
        hq = [r for d, r in sub if d == 'HotpotQA']
        nq = [r for d, r in sub if d == 'NQ']
        clean = all(r > 0 for r in hq) and all(r <= 0 for r in nq)
        print('  %-11s HotpotQA all positive and NQ all zero-or-negative: %s'
              % (ml, 'YES' if clean else 'NO'))

    print('\n' + '=' * 78)
    print('TABLE 4  pooling refusals: same correlation, both populations')
    print('=' * 78)
    print('%-9s %-10s %-11s %9s %9s %9s %9s'
          % ('dataset', 'generator', 'evaluator', 'rho_pool', 'rho_ans',
             'sprd_pool', 'sprd_ans'))
    for ds in ['NQ', 'HotpotQA']:
        for g in GENS:
            for mk, ml in EVALS:
                rp, spp = rho(ds, g, mk, False)
                ra, spa = rho(ds, g, mk, True)
                if rp is None or ra is None:
                    continue
                print('%-9s %-10s %-11s %9.2f %9.2f %9.4f %9.4f'
                      % (ds, g, ml, rp, ra, spp, spa))


if __name__ == '__main__':
    main()
