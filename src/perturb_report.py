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


def by_value_role(cells, answers, scorer):
    """
    The falsified value split by what it is: a fact, a chunk citation, or an
    ordered-list marker.

    `build_number_case` takes the first GROUNDED number in the answer, and
    both generators emit numbers that are not claims about the world — chunk
    citations ("According to Chunk 4, ...") and list numbering. Perturbing
    "Chunk 4" to "Chunk 7" is not a falsification, and its near-zero delta
    enters the mean as if it were one. Measured on 200 eligible answers per
    cell, the falsified value is a citation or a list marker in 9.5% of
    HotpotQA/Claude cases and 5.0% of NQ/Claude ones (GPT-4o-mini: 0-1%), so
    this table says how much that costs.
    """
    import perturbation_check as pc
    acc = defaultdict(lambda: defaultdict(list))
    for (ds, gen, m), rows in cells.items():
        for r in rows:
            a = answers.get((ds, gen, m, r['query_id']))
            if a is None or r.get('number') is None or not r.get('num_from'):
                continue
            role = 'content'
            for mm in pc.NUM_RE.finditer(a):
                if mm.group(1) == r['num_from']:
                    role = pc.value_role(a, mm.start(1), mm.end(1))
                    break
            acc[(ds, gen)][role].append(float(r['orig']) - float(r['number']))
    print(f'\n--- drop by what the falsified value IS, scorer={scorer} ---')
    print(f'{"cell":<26}{"role":>13}{"n":>7}{"mean delta":>12}')
    for cell in sorted(acc):
        for role in ('content', 'citation', 'list-marker'):
            v = np.asarray(acc[cell].get(role, []), dtype=float)
            if v.size == 0:
                continue
            print(f'{("%s/%s" % cell):<26}{role:>13}{v.size:>7}{v.mean():>+12.4f}')


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
    A single-assertion answer must score identically under whole-answer and
    claim-level aggregation — IF both aggregates are handed the same string.

    Split into two groups on purpose, because they answer different questions:

      no markdown   the claim IS the answer, byte for byte. The identity is a
                    correctness check on the claim-level implementation and
                    must hold to float tolerance.
      markdown      the claim is the answer with `**`/`*`/backticks removed,
                    because score_claims runs strip_markdown on claims and
                    NOT on the whole-answer hypothesis. The two scorers are
                    then reading different strings, and the size of the
                    difference measures how much the evaluator reacts to
                    formatting characters alone.
    """
    groups = {'no markdown': [], 'markdown': []}
    worst = {'no markdown': (0.0, None), 'markdown': (0.0, None)}
    for key, rows in cells_nli.items():
        claim_rows = {r['query_id']: r for r in cells_claim.get(key, [])}
        for r in rows:
            a = answers.get((*key, r['query_id']))
            if a is None:
                continue
            claims = cf.split_claims(a)
            stripped = cf.strip_markdown(a).strip()
            # The condition is not merely "one claim": it is "the single claim
            # IS the answer, up to markdown". A numbered-list answer can yield
            # one claim while the splitter drops the lead-in and the short list
            # items, in which case the two aggregates are legitimately scoring
            # different content and the identity does not apply.
            if len(claims) != 1 or claims[0] != stripped:
                continue
            c = claim_rows.get(r['query_id'])
            if c is None:
                continue
            g = 'no markdown' if stripped == a.strip() else 'markdown'
            d = abs(float(r['orig']) - float(c['orig']))
            groups[g].append(d)
            if d > worst[g][0]:
                worst[g] = (d, (key, r['query_id']))
    print('\n--- single-assertion identity check (claim_min == whole_max) ---')
    for g, diffs in groups.items():
        if not diffs:
            print(f'  [{g}] no eligible cases')
            continue
        d = np.asarray(diffs)
        print(f'  [{g}] n={d.size}   beyond {tol:g}: {(d > tol).sum()}   '
              f'mean |diff| {d.mean():.4f}   max {d.max():.4f}')
        if g == 'no markdown' and (d > tol).any():
            print(f'      VIOLATION — worst {worst[g][1]}; the claim-level '
                  f'implementation is not reducing to the whole-answer metric')
        elif g == 'markdown' and (d > tol).any():
            print(f'      worst {worst[g][1]} — this is strip_markdown '
                  f'applied to one hypothesis and not the other, not a bug in '
                  f'the aggregation')


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
        by_value_role(cells, answers, s)

    if 'nli' in loaded and 'claim' in loaded:
        identity_check(loaded['nli'], loaded['claim'], answers)
    return 0


if __name__ == '__main__':
    sys.exit(main())
