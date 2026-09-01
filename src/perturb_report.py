"""
perturb_report.py — the cross-scorer tables over the falsification runs.

Reads the `perturb*` checkpoints that perturbation_check.py wrote (one per
cell per scorer) and joins them back to the generations, so every row carries
the answer it came from and therefore its assertion count. Pure arithmetic:
no model, no API, no GPU.

Two things it produces that the ad-hoc 2026-08-26 analysis did not:

  1. **n per row.** The by-assertion-count table in the paper (Table III)
     rests on four buckets, one of which was empty for GPT-4o-mini, and
     printed no denominators. A monotone gradient over three points with
     unknown n is not a result a reader can check.
  2. **The single-assertion identity, on real data.** With one claim and no
     markdown, claim-level aggregation must reduce EXACTLY to the
     whole-answer metric. If it does not, strip_markdown is changing the
     hypothesis for one scorer and not the other, and the two are not being
     compared on identical inputs.
"""
import argparse
import sys
from collections import defaultdict

import numpy as np

import claim_faithfulness as cf
import config
from utils import checkpoint_exists, load_checkpoint, set_scope

SCORERS = ('nli', 'nli_concat', 'claim', 'align', 'numeric')
BUCKETS = ((1, 1, '1'), (2, 2, '2'), (3, 3, '3'), (4, 4, '4'), (5, 10 ** 6, '5+'))


def _ck(scorer, gen, model, ds):
    suffix = '' if scorer == 'nli' else f'_{scorer}'
    return f'perturb{suffix}_{gen}_{model}_{ds}'


def load_cells(scorer, datasets, generators, models):
    """{(ds, gen, model): [row, ...]} for one scorer."""
    out = {}
    for ds in datasets:
        for gen in generators:
            for m in models:
                name = _ck(scorer, gen, m, ds)
                if checkpoint_exists(name):
                    out[(ds, gen, m)] = load_checkpoint(name)
    return out


def answer_index(datasets, generators, models):
    """{(ds, gen, model, query_id): answer}."""
    idx = {}
    for ds in datasets:
        for gen in generators:
            for m in models:
                for g in load_checkpoint(f'generated_{gen}_{m}_{ds}') or []:
                    idx[(ds, gen, m, g['query_id'])] = g.get('generated_answer') or ''
    return idx


def bucket_of(n):
    for lo, hi, label in BUCKETS:
        if lo <= n <= hi:
            return label
    return '5+'


def gradient(cells, answers, scorer):
    """Mean drop when a value is falsified, by assertion count, per generator."""
    acc = defaultdict(lambda: defaultdict(list))
    for (ds, gen, m), rows in cells.items():
        for r in rows:
            a = answers.get((ds, gen, m, r['query_id']))
            if a is None or r.get('number') is None:
                continue
            b = bucket_of(len(cf.split_claims(a)))
            acc[gen][b].append(float(r['orig']) - float(r['number']))
    print(f'\n--- drop by assertions per answer, scorer={scorer} ---')
    print(f'{"generator":<12}{"assertions":>11}{"n":>7}{"mean delta":>12}'
          f'{"95% CI":>22}')
    out = {}
    for gen in sorted(acc):
        for _, _, label in BUCKETS:
            v = np.asarray(acc[gen].get(label, []), dtype=float)
            if v.size == 0:
                print(f'{gen:<12}{label:>11}{0:>7}{"—":>12}{"—":>22}')
                continue
            lo, hi = (np.nan, np.nan)
            if v.size > 1:
                rng = np.random.default_rng(config.SEED)
                boot = v[rng.integers(0, v.size, size=(10000, v.size))].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
            out[(gen, label)] = (v.size, float(v.mean()), float(lo), float(hi))
            print(f'{gen:<12}{label:>11}{v.size:>7}{v.mean():>+12.4f}'
                  f'{("[%+.4f, %+.4f]" % (lo, hi)):>22}')
    return out


def claim_counts(answers):
    """Mean assertions per answer, per generator and dataset."""
    acc = defaultdict(list)
    for (ds, gen, _m, _q), a in answers.items():
        if a.startswith('[ERROR'):
            continue
        acc[(gen, ds)].append(len(cf.split_claims(a)))
    print('\n--- assertions per answer (all generated answers) ---')
    per_gen = defaultdict(list)
    for (gen, ds), v in sorted(acc.items()):
        print(f'  {gen:<12}{ds:<10} mean {np.mean(v):.3f}   n={len(v)}')
        per_gen[gen] += v
    for gen, v in sorted(per_gen.items()):
        print(f'  {gen:<12}{"ALL":<10} mean {np.mean(v):.3f}   n={len(v)}')


def identity_check(cells_nli, cells_claim, answers, tol=1e-6):
    """
    Single-assertion, markdown-free answers must score identically under
    whole-answer and claim-level aggregation.
    """
    diffs, n_checked, worst = [], 0, (0.0, None)
    for key, rows in cells_nli.items():
        claim_rows = {r['query_id']: r for r in cells_claim.get(key, [])}
        for r in rows:
            a = answers.get((*key, r['query_id']))
            if a is None:
                continue
            if len(cf.split_claims(a)) != 1 or cf.strip_markdown(a) != a.strip():
                continue
            c = claim_rows.get(r['query_id'])
            if c is None:
                continue
            n_checked += 1
            d = abs(float(r['orig']) - float(c['orig']))
            diffs.append(d)
            if d > worst[0]:
                worst = (d, (key, r['query_id']))
    print('\n--- single-assertion identity check (claim_min == whole_max) ---')
    if not diffs:
        print('  no eligible cases (need both the nli and claim runs)')
        return
    diffs = np.asarray(diffs)
    print(f'  cases checked           {n_checked}')
    print(f'  max |difference|        {diffs.max():.6g}')
    print(f'  cases beyond {tol:g}      {(diffs > tol).sum()}')
    if diffs.max() > tol:
        print(f'  worst: {worst[1]}  -> strip_markdown is changing the '
              f'hypothesis for one scorer only')
    else:
        print('  identity holds — the two aggregates see the same hypothesis')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets', default='NQ,HotpotQA')
    ap.add_argument('--generators', default='claude,gpt4omini')
    ap.add_argument('--models', default=None)
    ap.add_argument('--scorers', default='nli,nli_concat,claim,align')
    ap.add_argument('--scope-n', type=int, default=1000)
    args = ap.parse_args()

    set_scope(args.scope_n)
    datasets = [d.strip() for d in args.datasets.split(',')]
    generators = [g.strip() for g in args.generators.split(',')]
    models = ([m.strip() for m in args.models.split(',')] if args.models
              else [c['name'] for c in config.EMBEDDING_MODELS])
    scorers = [s.strip() for s in args.scorers.split(',') if s.strip()]

    answers = answer_index(datasets, generators, models)
    claim_counts(answers)

    loaded = {}
    for s in scorers:
        cells = load_cells(s, datasets, generators, models)
        if not cells:
            print(f'\n[{s}] no checkpoints — skipped')
            continue
        loaded[s] = cells
        gradient(cells, answers, s)

    if 'nli' in loaded and 'claim' in loaded:
        identity_check(loaded['nli'], loaded['claim'], answers)
    return 0


if __name__ == '__main__':
    sys.exit(main())
