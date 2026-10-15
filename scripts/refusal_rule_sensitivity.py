#!/usr/bin/env python3
"""C6: the headline count under each defensible refusal rule.

Every result from Section 5 onward is computed on answers that a rule calls
attempts. The rule is a researcher degree of freedom, and this paper's own
argument is that such choices move conclusions -- so the count is reported
under each rule rather than only under the one we adopted.

Rules compared:
  canonical  is_abstention           -- adopted; strips a "Based on ..." preamble
  bare       is_abstention_bare      -- the pre-D4 rule, no preamble stripping
  judge      the LLM judge's call    -- read from the judged checkpoints

Read-only. Run from the repo root.
"""
import sys

import numpy as np

sys.path.insert(0, 'src')
from utils import set_scope  # noqa: E402

set_scope(1000)
import config  # noqa: E402
from abstention import is_abstention, is_abstention_bare  # noqa: E402
from correctness import load_correctness  # noqa: E402
from results import (FAITH_SOURCES, bootstrap_significance,  # noqa: E402
                     holm_bonferroni)
from utils import load_checkpoint  # noqa: E402

M4 = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
      'text-embedding-3-small']
GENS = ['claude', 'qwen', 'gpt4omini']
METRICS = [('nli', 'NLI-max'), ('align', 'AlignScore'), ('claim', 'Claim-min')]
ALPHA = 0.05


def drop_set(generator, model, dataset, rule):
    """Query ids whose answer the given rule calls a refusal."""
    if rule == 'judge':
        graded, _ = load_correctness(
            f'generated_{generator}_{model}_{dataset}', source='judge')
        return {q for q, r in graded.items() if r.get('abstained')}
    if rule == 'canonical':
        graded, _ = load_correctness(
            f'generated_{generator}_{model}_{dataset}', source='contains')
        return {q for q, r in graded.items() if r.get('abstained')}
    # `bare` is not precomputed anywhere, so recompute from the raw answers.
    gens = load_checkpoint(f'generated_{generator}_{model}_{dataset}') or []
    out = set()
    for g in gens:
        a = g.get('generated_answer')
        if not a or str(a).startswith('[ERROR'):
            continue
        if is_abstention_bare(a):
            out.add(g['query_id'])
    return out


def cell(dataset, generator, metric, rule):
    prefix, field = FAITH_SOURCES[metric]
    per = {}
    for m in M4:
        scores = load_checkpoint(f'{prefix}_{generator}_{m}_{dataset}')
        if not scores:
            continue
        drop = drop_set(generator, m, dataset, rule)
        per[m] = {r['query_id']: r[field] for r in scores
                  if r['query_id'] not in drop
                  and r.get(field) is not None
                  and np.isfinite(r[field])}
    if len(per) < 2:
        return None
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    if not common:
        return None
    means = {m: float(np.mean([per[m][q] for q in common])) for m in per}
    order = sorted(means, key=means.get)
    best, worst = order[-1], order[0]
    a = [per[worst][q] for q in common]
    b = [per[best][q] for q in common]
    sig = bootstrap_significance(a, b)
    return dict(spread=means[best] - means[worst], n=len(common),
                p=sig.get('p_value', float('nan')))


def run(rule):
    rows = []
    for ds in ['NQ', 'HotpotQA']:
        for g in GENS:
            for metric, lab in METRICS:
                c = cell(ds, g, metric, rule)
                if c:
                    rows.append((f'{ds}/{g}/{lab}', c))
    ps = [c['p'] for _, c in rows]
    adj = holm_bonferroni(ps, alpha=ALPHA)
    adj_p = [a[1] if isinstance(a, (list, tuple)) else a for a in adj]
    breaks = [(name, c, p) for (name, c), p in zip(rows, adj_p) if p < ALPHA]
    return rows, adj_p, breaks


def main():
    print('C6: headline count under each defensible refusal rule')
    print('family = all %d comparisons, Holm-corrected, alpha = %.2f\n'
          % (len(GENS) * 2 * len(METRICS), ALPHA))
    summary = {}
    for rule in ['canonical', 'bare', 'judge']:
        rows, adj_p, breaks = run(rule)
        summary[rule] = breaks
        print('%-10s %d of %d cells break:' % (rule, len(breaks), len(rows) // 3))
        for name, c, p in breaks:
            print('             %-34s spread %.4f  p_holm %.4f  n=%d'
                  % (name, c['spread'], p, c['n']))
        if not breaks:
            print('             (none)')
        print()
    names = {r: {n for n, _, _ in b} for r, b in summary.items()}
    stable = set.intersection(*names.values()) if names else set()
    allc = set.union(*names.values()) if names else set()
    print('cells that break under EVERY rule :', sorted(stable) or 'none')
    print('cells that break under SOME rule  :', sorted(allc - stable) or 'none')


if __name__ == '__main__':
    main()
