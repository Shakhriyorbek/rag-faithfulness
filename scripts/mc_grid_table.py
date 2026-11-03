#!/usr/bin/env python3
"""
mc_grid_table.py — Bespoke-MiniCheck-7B as a fourth evaluator in the system
comparison (third review: "I would raise my score if the evaluator-dependence
result were demonstrated beyond NLI-derived scorers").

Measures fixed on 2026-11-03, before any grid MiniCheck score existed.

PRIMARY
1. Per dataset x generator cell, the comparison every evaluator gets
   (answered-only by the string rule, paired on the queries all four systems
   answered, spread = best minus worst system mean, paired bootstrap p with
   10,000 resamples): MiniCheck on the whole answer.
2. Holm over the 32 comparisons of four evaluators x eight cells: how many
   survive and which. The published family of 24 is recomputed first and must
   return the four published survivors. The 24 stays the paper's design
   family; the 32 is reported beside it as the extension.
3. MDE at 80% power and the standardised effect (spread over the SD of the
   paired per-query difference), computed as for tab:7.
4. Omnibus: Friedman and the permutation test that re-selects the extremes,
   continuing the published script's RNG stream so its 24 values reproduce,
   then Holm over 32.
5. The system x evaluator interaction (mixed model, scores standardised within
   evaluator), refitted with MiniCheck as a fourth evaluator, beside the
   published three-evaluator fit.
6. tab:choose: MiniCheck's MDE range, and the number of generators whose
   refusals score higher than their attempts (gap averaged over the two
   datasets, as refusal_scores.py).

SECONDARY, reported but in no family: MiniCheck's sentence-minimum aggregate
under 1 and 3.

SENSITIVITY, printed and not adopted: Holm within MiniCheck's own 8 (B21).

    RAG_CHECKPOINT_DIR=~/rag-backup/checkpoints/n1000_v3 \
        python3 scripts/mc_grid_table.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from abstention import is_abstention                           # noqa: E402
from results import (bootstrap_significance, faithfulness_by_model,  # noqa: E402
                     holm_bonferroni)
from utils import checkpoint_exists, load_checkpoint, save_checkpoint  # noqa: E402
import omnibus_and_interaction as oi                           # noqa: E402

M4 = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
      'text-embedding-3-small']
DATASETS = ['NQ', 'HotpotQA']
GENS = ['claude', 'qwen', 'gpt4omini', 'mistral']     # the omnibus script's order
PUBLISHED = [('nli', 'NLI-max'), ('align', 'AlignScore'), ('claim', 'Claim-min')]
MC = ('minicheck', 'MiniCheck')
MC_SENT = ('minicheck_sent', 'MiniCheck-sent')
ZA, ZB = stats.norm.ppf(0.975), stats.norm.ppf(0.80)

# Published values every extension is checked against first.
PUB_SURVIVORS = {('HotpotQA', 'mistral', 'nli'): 0.0072,
                 ('HotpotQA', 'gpt4omini', 'align'): 0.0115,
                 ('HotpotQA', 'qwen', 'align'): 0.0242,
                 ('HotpotQA', 'mistral', 'claim'): 0.0378}
PUB_PERM = {'HotpotQA/gpt4omini/AlignScore', 'HotpotQA/mistral/NLI-max'}
PUB_FRIEDMAN = {'HotpotQA/mistral/NLI-max'}
PUB_MDE = {'nli': (0.021, 0.061), 'claim': (0.024, 0.064), 'align': (0.012, 0.037)}
PUB_REFUSAL_HIGHER = {'nli': 2, 'claim': 2, 'align': 0}


def comparison(ds, gen, metric):
    """One row of tab:7: spread, p, MDE and standardised effect."""
    r = faithfulness_by_model(ds, gen, M4, answered_only=True, metric=metric,
                              correct_source='contains')
    if not r:
        return None
    pq, best, worst = r['per_query'], r['best'], r['worst']
    n = min(len(pq[best]), len(pq[worst]))
    sig = bootstrap_significance(pq[worst][:n], pq[best][:n])
    d = np.asarray(pq[best][:n], float) - np.asarray(pq[worst][:n], float)
    sd = d.std(ddof=1)
    return {'dataset': ds, 'generator': gen, 'metric': metric,
            'n': r['n_common'], 'spread': r['spread'], 'p': sig['p_value'],
            'mde': (ZA + ZB) * sd / np.sqrt(n), 'std': r['spread'] / sd,
            'best': best, 'worst': worst, 'means': r['means']}


def holm(rows, key='p'):
    adj = holm_bonferroni([r[key] for r in rows], alpha=0.05)
    for r, a in zip(rows, adj):
        r['p_holm'] = a
    return [r for r in rows if r['p_holm'] < 0.05]


def _k(r):
    return (r['dataset'], r['generator'], r['metric'])


def fmt(r):
    return ('%-9s %-10s %-15s n=%4d spread %.4f p %.4f p_holm %-7s MDE %.4f '
            'Std %.3f' % (r['dataset'], r['generator'], r['metric'], r['n'],
                          r['spread'], r['p'], r.get('p_holm', ''), r['mde'],
                          r['std']))


def part_comparison(with_mc=True):
    print('=' * 100)
    print('1-3  The comparison, Holm, MDE and standardised effect')
    print('=' * 100)
    pub = [comparison(ds, g, m) for ds in DATASETS for g in GENS
           for m, _ in PUBLISHED]
    surv = {_k(r): r['p_holm'] for r in holm(pub)}
    print('published family of 24: %d survive' % len(surv))
    for k, v in sorted(surv.items()):
        print('   ', k, v)
    if surv != PUB_SURVIVORS:
        raise SystemExit('REPRODUCTION FAILED: the published 24-family '
                         'survivors do not reproduce; not extending.')
    for m, (lo, hi) in PUB_MDE.items():
        v = [r['mde'] for r in pub if r['metric'] == m]
        if (round(min(v), 3), round(max(v), 3)) != (lo, hi):
            raise SystemExit(f'REPRODUCTION FAILED: {m} MDE range '
                             f'{min(v):.3f}-{max(v):.3f}, published {lo}-{hi}')
    print('published survivors and MDE ranges reproduce.\n')
    if not with_mc:
        return pub, None, None, None

    mc = [comparison(ds, g, MC[0]) for ds in DATASETS for g in GENS]
    if any(r is None for r in mc):
        raise SystemExit('missing minicheck_scores checkpoints')
    ext = [dict(r) for r in pub + mc]
    surv32 = holm(ext)
    print('extended family of 32 (four evaluators): %d survive, in %d of 8 cells'
          % (len(surv32), len({(r['dataset'], r['generator']) for r in surv32})))
    for r in ext:
        print('   ' + fmt(r) + ('   *' if r in surv32 else ''))
    within = [dict(r) for r in mc]
    sw = holm(within)
    print('\nsensitivity, not adopted: Holm within MiniCheck\'s 8: %d survive %s'
          % (len(sw), [(r['dataset'], r['generator']) for r in sw]))
    print('MiniCheck MDE %.3f-%.3f (median %.3f); spread < MDE in %d of 8; '
          'Std %.3f-%.3f'
          % (min(r['mde'] for r in mc), max(r['mde'] for r in mc),
             np.median([r['mde'] for r in mc]),
             sum(r['spread'] < r['mde'] for r in mc),
             min(r['std'] for r in mc), max(r['std'] for r in mc)))
    for ds in DATASETS:
        for g in GENS:
            row = {m: next(r for r in ext if _k(r) == (ds, g, m))
                   for m in ('nli', 'claim', 'align', 'minicheck')}
            lowest = min(row, key=lambda m: row[m]['mde'])
            print('   %-9s %-10s lowest MDE: %-10s  Std MiniCheck %.3f vs NLI '
                  '%.3f, Align %.3f' % (ds, g, lowest, row['minicheck']['std'],
                                        row['nli']['std'], row['align']['std']))

    print('\nsecondary, in no family: MiniCheck sentence-minimum')
    sent = [comparison(ds, g, MC_SENT[0]) for ds in DATASETS for g in GENS]
    for r in sent:
        print('   ' + fmt(r))
    return pub, mc, sent, ext


def part_omnibus(with_mc=True):
    print('\n' + '=' * 100)
    print('4  Omnibus, continuing the published RNG stream')
    print('=' * 100)
    rng = np.random.default_rng(oi.SEED)
    rows = []
    for ds in DATASETS:
        for g in GENS:
            for metric, lab in PUBLISHED:
                X, _ = oi.paired_matrix(ds, g, metric)
                fp, rng_obs, pp = oi.omnibus(X, rng)
                rows.append((f'{ds}/{g}/{lab}', rng_obs, fp, pp))
    for name, col, pub in (('Friedman', 2, PUB_FRIEDMAN),
                           ('permutation', 3, PUB_PERM)):
        adj = holm_bonferroni([r[col] for r in rows])
        got = {rows[i][0] for i, p in enumerate(adj) if p < 0.05}
        print('published 24, %s: %s' % (name, sorted(got)))
        if got != pub:
            raise SystemExit(f'REPRODUCTION FAILED: {name} survivors {got}')
    if not with_mc:
        return rows, None
    for ds in DATASETS:
        for g in GENS:
            X, _ = oi.paired_matrix(ds, g, MC[0])
            fp, rng_obs, pp = oi.omnibus(X, rng)
            rows.append((f'{ds}/{g}/{MC[1]}', rng_obs, fp, pp))
            print('   %-28s range %.4f Friedman p %.4g perm p %.4g'
                  % (rows[-1][0], rng_obs, fp, pp))
    out = {}
    for name, col in (('Friedman', 2), ('permutation', 3)):
        adj = holm_bonferroni([r[col] for r in rows])
        got = sorted(rows[i][0] for i, p in enumerate(adj) if p < 0.05)
        out[name] = got
        print('extended 32, %s: %d survive %s' % (name, len(got), got))
    return rows, out


def interaction(ds, g, metrics):
    from statsmodels.regression.mixed_linear_model import MixedLM
    recs = []
    for metric, lab in metrics:
        X, common = oi.paired_matrix(ds, g, metric)
        z = (X - X.mean()) / X.std(ddof=1)
        for qi, q in enumerate(common):
            for sj, s in enumerate(M4[:X.shape[1]]):
                recs.append((q, s, lab, z[qi, sj]))
    d = pd.DataFrame(recs, columns=['query', 'system', 'evaluator', 'z'])
    full = MixedLM.from_formula('z ~ C(system) * C(evaluator)', groups='query',
                                data=d).fit(reml=False)
    add = MixedLM.from_formula('z ~ C(system) + C(evaluator)', groups='query',
                               data=d).fit(reml=False)
    lr = 2 * (full.llf - add.llf)
    df = len([n for n in full.params.index if ':' in n])
    return len(d), lr, df, (stats.chi2.sf(lr, df) if lr > 0 else float('nan'))


def part_interaction(with_mc=True):
    print('\n' + '=' * 100)
    print('5  System x evaluator interaction, three and four evaluators')
    print('=' * 100)
    rows = []
    for ds in DATASETS:
        for g in GENS:
            n3, lr3, df3, p3 = interaction(ds, g, PUBLISHED)
            n4, lr4, df4, p4 = (interaction(ds, g, PUBLISHED + [MC]) if with_mc
                                else (0, float('nan'), 0, float('nan')))
            rows.append((ds, g, p3, p4))
            print('   %-9s %-10s three: n=%6d LR %6.2f df %d p %.4f | four: '
                  'n=%6d LR %6.2f df %d p %.4f'
                  % (ds, g, n3, lr3, df3, p3, n4, lr4, df4, p4))
    p3s = [r[2] for r in rows]
    print('three evaluators: p %.2f-%.2f (published 0.31-0.96)'
          % (min(p3s), max(p3s)))
    if (round(min(p3s), 2), round(max(p3s), 2)) != (0.31, 0.96):
        raise SystemExit('REPRODUCTION FAILED: three-evaluator interaction range')
    if not with_mc:
        return rows
    p4s = [r[3] for r in rows]
    print('four evaluators:  p %.2f-%.2f, below 0.05 in %d of 8'
          % (min(p4s), max(p4s), sum(p < 0.05 for p in p4s)))
    return rows


def part_refusals():
    print('\n' + '=' * 100)
    print('6  Refusals against attempts (tab:choose bottom row)')
    print('=' * 100)
    sources = {'nli': ('nli_scores', 'nli_max'), 'claim': ('claim_scores', 'claim_min'),
               'align': ('align_scores', 'align_score'),
               'minicheck': ('minicheck_scores', 'minicheck'),
               'minicheck_sent': ('minicheck_scores', 'minicheck_sent')}
    out = {}
    for key, (pre, col) in sources.items():
        gaps = {}
        for g in GENS:
            per_ds = []
            for ds in DATASETS:
                ref, att = [], []
                for m in M4:
                    if not checkpoint_exists(f'{pre}_{g}_{m}_{ds}'):
                        continue
                    ans = {r['query_id']: r['generated_answer'] for r in
                           load_checkpoint(f'generated_{g}_{m}_{ds}')}
                    for r in load_checkpoint(f'{pre}_{g}_{m}_{ds}'):
                        v = r.get(col)
                        if v is None or not np.isfinite(v):
                            continue
                        (ref if is_abstention(ans[r['query_id']] or '')
                         else att).append(v)
                if ref:
                    per_ds.append(np.mean(ref) - np.mean(att))
            if per_ds:
                gaps[g] = float(np.mean(per_ds))
        out[key] = gaps
        print('   %-15s refusal minus attempt: %s  -> refusals higher in %d of %d'
              % (key, {g: round(v, 3) for g, v in gaps.items()},
                 sum(v > 0 for v in gaps.values()), len(gaps)))
    for m, k in PUB_REFUSAL_HIGHER.items():
        if sum(v > 0 for v in out[m].values()) != k:
            raise SystemExit(f'REPRODUCTION FAILED: refusal row for {m}')
    print('published refusal row (2, 2, 0 of 4) reproduces.')
    return out


def main():
    with_mc = '--reproduce-only' not in sys.argv
    pub, mc, sent, ext = part_comparison(with_mc)
    omni, omni_out = part_omnibus(with_mc)
    inter = part_interaction(with_mc)
    refusals = part_refusals()
    if not with_mc:
        print('\nreproduce-only: every published check passed; nothing saved')
        return
    save_checkpoint('mc_grid_table', {'published': pub, 'minicheck': mc,
                                      'minicheck_sent': sent, 'extended': ext,
                                      'omnibus': omni, 'omnibus_holm32': omni_out,
                                      'interaction': inter, 'refusals': refusals})
    print('\nsaved checkpoint mc_grid_table')


if __name__ == '__main__':
    main()
