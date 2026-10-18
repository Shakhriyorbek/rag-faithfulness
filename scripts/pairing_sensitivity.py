#!/usr/bin/env python3
"""Red flag 3: does pairing on the commonly-answered subset drive the result?

Pairing on the queries every system answered drops exactly those queries where
retrieval quality decided whether an answer was produced, so the comparison is
conditional on a subset the compared systems helped select. The check is to
recompute each system's mean over ITS OWN answered rows -- which removes the
selection but gives up the pairing -- and ask whether the spread and the
ordering move.

Prints, per dataset x generator x evaluator: paired spread, own-rows spread,
their difference, and whether the system ordering changes. Read-only.
Run from the repo root with RAG_CHECKPOINT_DIR set to the data tree.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, 'src')
from abstention import is_abstention  # noqa: E402
from utils import set_scope  # noqa: E402

set_scope(1000)
from results import FAITH_SOURCES, faithfulness_by_model  # noqa: E402

M4 = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
      'text-embedding-3-small']
GENS = ['claude', 'qwen', 'gpt4omini', 'mistral']
EVALS = [('nli', 'NLI-max'), ('claim', 'Claim-min'), ('align', 'AlignScore')]
CKPT = (os.environ.get('RAG_CHECKPOINT_DIR')
        or os.path.expanduser('~/rag-backup/checkpoints/n1000_v3')).rstrip('/')


def own_rows_means(dataset, generator, metric):
    """Mean per system over that system's own answered rows (no pairing)."""
    prefix, field = FAITH_SOURCES[metric]
    out = {}
    for m in M4:
        fs = '%s/%s_%s_%s_%s.pkl' % (CKPT, prefix, generator, m, dataset)
        fg = '%s/generated_%s_%s_%s.pkl' % (CKPT, generator, m, dataset)
        if not (os.path.exists(fs) and os.path.exists(fg)):
            continue
        ans = {r['query_id']: r.get('generated_answer')
               for r in pickle.load(open(fg, 'rb'))}
        vals = [r[field] for r in pickle.load(open(fs, 'rb'))
                if r.get(field) is not None and np.isfinite(r[field])
                and ans.get(r['query_id'])
                and not str(ans[r['query_id']]).startswith('[ERROR')
                and not is_abstention(ans[r['query_id']])]
        if vals:
            out[m] = float(np.mean(vals))
    return out


def main():
    print('checkpoints:', CKPT)
    hdr = ('%-9s %-10s %-11s %10s %10s %8s  %s'
           % ('dataset', 'generator', 'evaluator', 'paired', 'own-rows',
              'diff', 'ordering'))
    print(hdr); print('-' * len(hdr))
    worst = 0.0
    reorder = total = 0
    by_eval = {}
    for ds in ['NQ', 'HotpotQA']:
        for g in GENS:
            for mk, ml in EVALS:
                r = faithfulness_by_model(ds, g, M4, answered_only=True,
                                          metric=mk, correct_source='contains')
                own = own_rows_means(ds, g, mk)
                if not r or len(own) < len(M4):
                    continue
                p_sp = r['spread']
                o_sp = max(own.values()) - min(own.values())
                p_ord = sorted(r['means'], key=lambda m: r['means'][m])
                o_ord = sorted(own, key=lambda m: own[m])
                same = p_ord == o_ord
                total += 1
                if not same:
                    reorder += 1
                    by_eval[ml] = by_eval.get(ml, 0) + 1
                worst = max(worst, abs(o_sp - p_sp))
                print('%-9s %-10s %-11s %10.4f %10.4f %8.4f  %s'
                      % (ds, g, ml, p_sp, o_sp, o_sp - p_sp,
                         'same' if same else 'CHANGED'))
    print('\n  spread moves by at most %.4f across %d combinations' % (worst, total))
    print('  ordering changes in %d of %d combinations' % (reorder, total))
    for ml in [l for _, l in EVALS]:
        n = sum(1 for ds in ['NQ', 'HotpotQA'] for g in GENS)
        print('    %-11s changed in %d of %d cells' % (ml, by_eval.get(ml, 0), n))


if __name__ == '__main__':
    main()
