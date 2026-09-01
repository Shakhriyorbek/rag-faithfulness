"""
markdown_check.py — does the faithfulness score react to formatting characters?

WHY THIS EXISTS (found 2026-09-01 while adding the F3 identity assertion)
    `faithfulness.py`/`nli.py` score the answer EXACTLY as the generator wrote
    it, `**bold**` and all. `claim_faithfulness.score_claims` runs
    `strip_markdown` over the claims first. So on the same answer the two
    aggregates are not reading the same hypothesis, and the single-assertion
    identity check found the consequence: 244 of 1,615 single-claim cases
    differ, by up to 0.73.

    That matters beyond the identity check, because markdown is not evenly
    distributed. Of the answered rows in checkpoints/n1000_v3:

        Claude        4,760 / 6,354  (74.9%) contain markdown
        GPT-4o-mini      19 / 5,630  ( 0.3%)

    The paper's cross-generator comparison — NLI-max is far less sensitive to
    a falsified value on Claude's answers than on GPT-4o-mini's — is therefore
    confounded with a formatting difference that only one generator produces.
    This script measures how large that confound is: the same answers, the
    same chunks, one scored raw and one with the markdown removed.

READ-ONLY on existing checkpoints. NLI runs locally; no API spend.
"""
import argparse
import sys

import numpy as np

import abstention
import claim_faithfulness as cf
import config
from utils import load_checkpoint, save_checkpoint, set_scope


def run_cell(nli, gen, model, ds, limit):
    recs = load_checkpoint(f'generated_{gen}_{model}_{ds}')
    if not recs:
        return None
    rows = []
    for g in recs:
        a = g.get('generated_answer') or ''
        chunks = (g.get('retrieved_texts') or [])[:config.TOP_K]
        if not chunks or a.startswith('[ERROR') or abstention.is_abstention(a):
            continue
        stripped = cf.strip_markdown(a).strip()
        if stripped == a.strip():
            continue                        # nothing to measure
        raw = nli.score_chunks(chunks, a)['nli_max']
        clean = nli.score_chunks(chunks, stripped)['nli_max']
        rows.append({'query_id': g['query_id'], 'raw': float(raw),
                     'stripped': float(clean)})
        if len(rows) >= limit:
            break
    return rows


def summarise(rows, label):
    raw = np.array([r['raw'] for r in rows])
    st = np.array([r['stripped'] for r in rows])
    d = st - raw
    print(f'\n--- {label} (n={len(rows)}) ---')
    print(f'  nli_max raw          {raw.mean():.4f}')
    print(f'  nli_max stripped     {st.mean():.4f}   '
          f'mean change {d.mean():+.4f}')
    print(f'  |change|: mean {np.abs(d).mean():.4f}  median '
          f'{np.median(np.abs(d)):.4f}  p95 {np.percentile(np.abs(d), 95):.4f}'
          f'  max {np.abs(d).max():.4f}')
    print(f'  answers moving more than 0.05: {(np.abs(d) > 0.05).mean():.1%}   '
          f'more than 0.10: {(np.abs(d) > 0.10).mean():.1%}')
    # The comparison that matters: one falsified fact moves nli_max by
    # 0.033-0.094 on Claude's answers (Section V). If deleting asterisks moves
    # it by the same order, the metric is reading formatting as strongly as it
    # reads content.
    return {'label': label, 'n': len(rows), 'raw': float(raw.mean()),
            'stripped': float(st.mean()), 'mean_change': float(d.mean()),
            'mean_abs': float(np.abs(d).mean()),
            'p95_abs': float(np.percentile(np.abs(d), 95)),
            'max_abs': float(np.abs(d).max()),
            'frac_over_05': float((np.abs(d) > 0.05).mean()),
            'frac_over_10': float((np.abs(d) > 0.10).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets', default='NQ,HotpotQA')
    ap.add_argument('--generators', default='claude')
    ap.add_argument('--models', default='all-mpnet-base-v2')
    ap.add_argument('--limit', type=int, default=300)
    ap.add_argument('--scope-n', type=int, default=1000)
    args = ap.parse_args()

    set_scope(args.scope_n)
    from nli import NLIScorer
    nli = NLIScorer()
    out = []
    for ds in [d.strip() for d in args.datasets.split(',')]:
        for gen in [g.strip() for g in args.generators.split(',')]:
            for m in [x.strip() for x in args.models.split(',')]:
                rows = run_cell(nli, gen, m, ds, args.limit)
                if rows:
                    out.append(summarise(rows, f'{ds}/{gen}/{m}'))
    if out:
        save_checkpoint('markdown_sensitivity', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
