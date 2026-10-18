#!/usr/bin/env python3
"""faith_gap, Holm-corrected over the whole judged family rather than per arm.

conditional.py corrects within one generator's ten tests (four matched systems
plus jina). The paper reports the count over the matched family across every
judged arm, which is a different and stricter correction, and nothing computed
it -- it was assembled by hand. That is how the copying control went stale
(CLAUDE.md B20), so it lives here instead.

Each cell's raw p is a within-(dataset, model) permutation test and does not
depend on which other systems are in the frame, so the per-arm checkpoints
written by conditional.py can be pooled directly; only the correction changes.

Read-only over faith_gap_significance_* checkpoints. Run from the repo root
with RAG_CHECKPOINT_DIR pointing at the data tree, after conditional.py has
run for every generator.
"""
import sys

import pandas as pd

sys.path.insert(0, 'src')
from utils import set_scope  # noqa: E402

set_scope(1000)
import config  # noqa: E402
from results import holm_bonferroni  # noqa: E402
from utils import load_checkpoint  # noqa: E402

# The matched four only. jina fails the matched-quality precondition on NQ and
# is a control, so it is not a member of the family the paper corrects over.
M4 = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
      'text-embedding-3-small']


def main():
    rows = []
    missing = []
    for g in config.ACTIVE_GENERATORS:
        t = load_checkpoint(f'faith_gap_significance_{g}')
        if t is None or len(t) == 0:
            missing.append(g)
            continue
        t = t[t['model'].isin(M4)].copy()
        t['generator'] = g
        rows.append(t)

    print('generators in config.ACTIVE_GENERATORS: %s'
          % ', '.join(config.ACTIVE_GENERATORS))
    if missing:
        print('  [!] no faith_gap_significance checkpoint for: %s'
              % ', '.join(missing))
        print('      run: python3 src/conditional.py --generator <g> '
              '--correct-source judge')
    if not rows:
        raise SystemExit('no faith_gap_significance_* checkpoints found')

    t = pd.concat(rows, ignore_index=True)
    # Recompute the correction over the pooled family; the stored p_holm was
    # corrected within one arm and must not be reused here.
    t['p_holm_family'] = holm_bonferroni(t['p_raw'].tolist())
    t['sig_family'] = t['p_holm_family'] < 0.05
    t = t.sort_values('p_raw')

    pd.set_option('display.width', 200)
    cols = ['generator', 'dataset', 'model', 'n_wrong', 'n_correct',
            'faith_gap', 'ci_low', 'ci_high', 'p_raw', 'p_holm_family',
            'sig_family']
    print('\n=== faith_gap over the judged family (%d cells) ===' % len(t))
    print(t[cols].to_string(index=False))

    sig = t[t['sig_family']]
    pos = t[t['faith_gap'] > 0]
    print('\nsignificant after Holm over the %d cells: %d' % (len(t), len(sig)))
    if len(sig):
        print('  signs: %d negative, %d positive'
              % ((sig['faith_gap'] < 0).sum(), (sig['faith_gap'] > 0).sum()))
    print('positive-signed cells: %d; any with a CI excluding zero: %s'
          % (len(pos), bool((pos['ci_low'] > 0).any())))

    # The replication the paper names, stated per generator so a reader can
    # see it is the same system and dataset in each arm.
    e5 = t[(t['model'] == 'E5-large-instruct') & (t['dataset'] == 'NQ')]
    if len(e5):
        print('\nE5-large-instruct on NQ, by generator:')
        for r in e5.sort_values('generator').itertuples():
            print('  %-10s gap %+.3f  p_raw %.4f  p_holm %.4f  %s'
                  % (r.generator, r.faith_gap, r.p_raw, r.p_holm_family,
                     'survives' if r.sig_family else 'does not survive'))


if __name__ == '__main__':
    main()
