"""
copying_check.py — is the weak falsification response just verbatim quotation?

WHY THIS EXISTS
    Section V-B of the paper rejects the obvious explanation for the paper's
    central asymmetry. NLI-max barely moves when a grounded value in one of
    Claude's answers is falsified, and moves a lot on GPT-4o-mini's. The
    natural reading is that one generator quotes the retrieved text back, so
    the evaluator is scoring a copy of its own premise and the copied bulk
    carries the entailment regardless of the edit.

    That claim was measured once, on 2026-08-25, by an ad-hoc script that was
    never committed: overlap 0.234 for both generators, corr(overlap, delta)
    +0.041 and +0.161. Because it had no home in `src/`, it was the one
    number in Section V not regenerated after the 2026-09-03 decisions (D1
    markdown stripping, D2 content-only case construction), both of which
    change the deltas it correlates against. This module gives it a home.

WHAT IT MEASURES
    overlap(answer, context) = fraction of the answer's word 5-grams that
    appear in the retrieved context as a contiguous token sequence.
    delta = orig - number, the drop in score when one grounded value is
    falsified, joined per query from the `perturb_*` checkpoints.

    The copying hypothesis predicts a NEGATIVE correlation: the more of the
    answer is copied, the less one substituted value should matter. It also
    predicts Claude's overlap should exceed GPT-4o-mini's, since Claude is
    the generator whose scores do not move.

TOKENISATION IS A REPORTED CHOICE, NOT AN INCIDENTAL ONE
    `textnorm.squash` maps every non-word character to a space, so it already
    neutralises markdown: `**Graduados**` and `Graduados` squash to the same
    token. A raw `.split()` tokenisation does not — it makes them different
    tokens that can never match the context. The two therefore disagree on
    exactly the answers D1 was about, and 74.9% of Claude's answers carry
    markdown against 0.3% of GPT-4o-mini's. `--variants` reports all four
    combinations so the published figure can be located and the primary one
    read against it.

READ-ONLY on existing checkpoints. No model, no GPU, no API spend.
"""
import argparse
import sys
from collections import defaultdict

import numpy as np

import config
import textnorm
from utils import checkpoint_exists, load_checkpoint, set_scope

NGRAM = 5


# ── overlap ────────────────────────────────────────────────────────────────

def _tokens(text, tokeniser):
    if tokeniser == 'squash':
        return textnorm.squash(text).split()
    return (text or '').split()


def ngram_overlap(answer, context, n=NGRAM, tokeniser='squash'):
    """Fraction of the answer's word n-grams that occur in the context.

    Returns None when the answer is shorter than n tokens and therefore has
    no n-gram to score. Those answers are excluded rather than counted as 0:
    a 4-word answer has not failed to copy, the question does not apply to
    it, and GPT-4o-mini writes far more of them (median 18 words against
    Claude's 49), so scoring them as 0 would manufacture exactly the
    cross-generator overlap difference this module is testing for.
    """
    a = _tokens(answer, tokeniser)
    if len(a) < n:
        return None
    c = _tokens(context, tokeniser)
    ctx = {tuple(c[i:i + n]) for i in range(len(c) - n + 1)}
    grams = [tuple(a[i:i + n]) for i in range(len(a) - n + 1)]
    return sum(g in ctx for g in grams) / len(grams)


# ── data ───────────────────────────────────────────────────────────────────

def generation_index(datasets, generators, models):
    """{(ds, gen, model, query_id): (answer, context)}."""
    idx = {}
    for ds in datasets:
        for gen in generators:
            for m in models:
                for g in load_checkpoint(f'generated_{gen}_{m}_{ds}') or []:
                    chunks = (g.get('retrieved_texts') or [])[:config.TOP_K]
                    idx[(ds, gen, m, g['query_id'])] = (
                        g.get('generated_answer') or '', ' '.join(chunks))
    return idx


def load_cells(datasets, generators, models, scorer='nli'):
    suffix = '' if scorer == 'nli' else f'_{scorer}'
    out = {}
    for ds in datasets:
        for gen in generators:
            for m in models:
                name = f'perturb{suffix}_{gen}_{m}_{ds}'
                if checkpoint_exists(name):
                    out[(ds, gen, m)] = load_checkpoint(name)
    return out


def collect(cells, gens, strip, tokeniser):
    """[(key, overlap, delta)] over every scored perturbation case."""
    rows = defaultdict(list)
    skipped_short = defaultdict(int)
    for (ds, gen, m), cases in cells.items():
        for r in cases:
            got = gens.get((ds, gen, m, r['query_id']))
            if got is None or r.get('number') is None:
                continue
            answer, context = got
            if strip:
                answer = textnorm.strip_markdown(answer)
            ov = ngram_overlap(answer, context, tokeniser=tokeniser)
            if ov is None:
                skipped_short[(ds, gen)] += 1
                continue
            rows[(ds, gen)].append((ov, float(r['orig']) - float(r['number'])))
    return rows, skipped_short


# ── statistics ─────────────────────────────────────────────────────────────

def _pearson(x, y):
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    def rank(v):
        order = v.argsort()
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(v.size, dtype=float)
        # average ties, or a generator that writes many identical-overlap
        # answers gets an arbitrary ordering baked into the statistic
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(counts.size)
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]
    return _pearson(rank(x), rank(y))


def _boot_ci(x, y, fn, n=10000, seed=config.SEED):
    if x.size < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    vals = np.array([fn(x[i], y[i]) for i in idx])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)
    return tuple(float(v) for v in np.percentile(vals, [2.5, 97.5]))


def report(rows, skipped, label, ci=True):
    print(f'\n--- {label} ---')
    print(f'{"cell":<26}{"n":>6}{"short":>7}{"mean overlap":>14}'
          f'{"pearson r":>12}{"95% CI":>22}{"spearman":>11}')
    out = {}
    for key in sorted(rows):
        v = np.asarray(rows[key], dtype=float)
        ov, d = v[:, 0], v[:, 1]
        r = _pearson(ov, d)
        rho = _spearman(ov, d)
        lo, hi = _boot_ci(ov, d, _pearson) if ci else (np.nan, np.nan)
        out[key] = {'n': int(ov.size), 'overlap': float(ov.mean()),
                    'pearson': r, 'lo': lo, 'hi': hi, 'spearman': rho}
        cell = f'{key[0]}/{key[1]}'
        ci_s = '—' if np.isnan(lo) else f'[{lo:+.3f}, {hi:+.3f}]'
        print(f'{cell:<26}{ov.size:>6}{skipped.get(key, 0):>7}'
              f'{ov.mean():>14.3f}{r:>+12.3f}{ci_s:>22}{rho:>+11.3f}')
    # pooled per generator, which is how the paper states it
    per_gen = defaultdict(list)
    for (ds, gen), v in rows.items():
        per_gen[gen] += v
    for gen in sorted(per_gen):
        v = np.asarray(per_gen[gen], dtype=float)
        ov, d = v[:, 0], v[:, 1]
        r = _pearson(ov, d)
        lo, hi = _boot_ci(ov, d, _pearson) if ci else (np.nan, np.nan)
        ci_s = '—' if np.isnan(lo) else f'[{lo:+.3f}, {hi:+.3f}]'
        out[('ALL', gen)] = {'n': int(ov.size), 'overlap': float(ov.mean()),
                             'pearson': r, 'lo': lo, 'hi': hi,
                             'spearman': _spearman(ov, d)}
        print(f'{("ALL/" + gen):<26}{ov.size:>6}{"":>7}{ov.mean():>14.3f}'
              f'{r:>+12.3f}{ci_s:>22}{_spearman(ov, d):>+11.3f}')
    return out


def matched_overlap_bands(rows, label, edges=(0.0, 0.1, 0.2, 0.3, 1.01)):
    """The delta by overlap band, per generator.

    The correlation is one number over a cloud; the paper's argument needs
    the stronger statement that the cross-generator gap survives AT MATCHED
    overlap. If copying explained the gap, the two generators would converge
    inside a band.
    """
    print(f'\n--- delta by overlap band, {label} ---')
    print(f'{"band":<14}{"generator":<14}{"n":>6}{"mean delta":>13}'
          f'{"mean overlap":>14}')
    per_gen = defaultdict(list)
    for (_ds, gen), v in rows.items():
        per_gen[gen] += v
    for lo, hi in zip(edges, edges[1:]):
        for gen in sorted(per_gen):
            v = np.asarray(per_gen[gen], dtype=float)
            sel = v[(v[:, 0] >= lo) & (v[:, 0] < hi)]
            band = f'[{lo:.1f}, {hi:.1f})'
            if sel.size == 0:
                print(f'{band:<14}{gen:<14}{0:>6}{"—":>13}{"—":>14}')
                continue
            print(f'{band:<14}{gen:<14}{sel.shape[0]:>6}'
                  f'{sel[:, 1].mean():>+13.4f}{sel[:, 0].mean():>14.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets', default='NQ,HotpotQA')
    ap.add_argument('--generators', default='claude,gpt4omini')
    ap.add_argument('--models', default=None)
    ap.add_argument('--scorer', default='nli',
                    help='which perturbation checkpoint supplies the delta')
    ap.add_argument('--scope-n', type=int, default=1000)
    ap.add_argument('--variants', action='store_true',
                    help='also report raw/whitespace tokenisation combinations')
    ap.add_argument('--no-ci', action='store_true')
    args = ap.parse_args()

    set_scope(args.scope_n)
    datasets = [d.strip() for d in args.datasets.split(',')]
    generators = [g.strip() for g in args.generators.split(',')]
    models = ([m.strip() for m in args.models.split(',')] if args.models
              else [c['name'] for c in config.EMBEDDING_MODELS])

    cells = load_cells(datasets, generators, models, args.scorer)
    if not cells:
        print(f'no perturb_{args.scorer} checkpoints found — nothing to do')
        return 1
    gens = generation_index(datasets, generators, models)
    print(f'{len(cells)} cells, {sum(len(v) for v in cells.values())} cases, '
          f'scorer={args.scorer}, n-gram={NGRAM}')

    # primary: the string the scorer actually reads after D1, tokenised by
    # the project's canonical containment rule
    rows, skipped = collect(cells, gens, strip=True, tokeniser='squash')
    primary = report(rows, skipped, 'PRIMARY  answer=stripped  tokens=squash',
                     ci=not args.no_ci)
    matched_overlap_bands(rows, 'stripped/squash')

    if args.variants:
        for strip in (True, False):
            for tok in ('squash', 'whitespace'):
                if strip and tok == 'squash':
                    continue
                r, s = collect(cells, gens, strip=strip, tokeniser=tok)
                report(r, s, f'answer={"stripped" if strip else "raw"}  '
                             f'tokens={tok}', ci=False)
    return 0 if primary else 1


if __name__ == '__main__':
    sys.exit(main())
