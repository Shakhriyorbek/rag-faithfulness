"""
ragtruth_table.py — measures over the RAGTruth scores (third review, D5).

All fixed before the scores were read:

1. overall, per scorer: AUROC of clean against hallucinated responses (higher
   score = better supported), with a bootstrap 95% interval over responses;
   detection at the 0.5 gate (hallucinated responses scored below it) and
   false rejection (clean responses scored below it), Wilson intervals.
2. dilution, two ways, each split into tertiles computed over the population
   before any score is read:
   a. by assertions per response (claim_faithfulness.split_claims), with
      AUROC, detection and false rejection inside each tertile, so length
      cannot drive the AUROC itself;
   b. among hallucinated responses only, by the share of the response's
      characters inside an annotated span: detection per tertile. The probe's
      mechanism predicts a smaller share is caught less often.
3. per generator: AUROC, detection, false rejection.

Added after the above were read, and labelled so in the paper: the same
scorer's AUROC between untouched and falsified answers over the probe cases
of tab:2full, per dataset (a point of comparison for natural vs constructed
errors); the share-vs-assertion-count Spearman among hallucinated answers; and
mean assertions per generator.

    RAGTRUTH_DIR=... RAG_CHECKPOINT_DIR=~/rag-backup/checkpoints/n1000_v3 \
        python3 scripts/ragtruth_table.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import claim_faithfulness as cf                                 # noqa: E402
import ragtruth_check as rc                                     # noqa: E402
from perturbation_check import _wilson_ci                       # noqa: E402
from utils import checkpoint_exists, load_checkpoint            # noqa: E402

SCORERS = ['nli', 'claim', 'align', 'minicheck', 'minicheck_sent']
GATE = 0.5
B = 2000
rng = np.random.default_rng(42)


def auroc(clean, hall):
    """P(clean score > hallucinated score), ties counted half (Mann-Whitney)."""
    if len(clean) == 0 or len(hall) == 0:
        return float('nan')
    x = np.concatenate([clean, hall])
    ranks = np.argsort(np.argsort(x, kind='mergesort'), kind='mergesort') + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    r_clean = ranks[:len(clean)].sum()
    return (r_clean - len(clean) * (len(clean) + 1) / 2) / (len(clean) * len(hall))


def auroc_ci(clean, hall):
    vals = [auroc(rng.choice(clean, len(clean)), rng.choice(hall, len(hall)))
            for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5])


def measures(rows, s):
    sc = np.array([s[r['id']] for r in rows])
    h = np.array([r['hallucinated'] for r in rows])
    clean, hall = sc[~h], sc[h]
    k_d, k_f = int((hall < GATE).sum()), int((clean < GATE).sum())
    return {'n_clean': len(clean), 'n_hall': len(hall),
            'auroc': auroc(clean, hall),
            'auroc_ci': auroc_ci(clean, hall) if len(clean) and len(hall) else (np.nan,) * 2,
            'det': k_d / max(1, len(hall)), 'det_ci': _wilson_ci(k_d, len(hall)),
            'fr': k_f / max(1, len(clean)), 'fr_ci': _wilson_ci(k_f, len(clean))}


def fmt(m):
    return (f'n={m["n_clean"]:>4}/{m["n_hall"]:<4}  AUROC {m["auroc"]:.3f} '
            f'[{m["auroc_ci"][0]:.3f}-{m["auroc_ci"][1]:.3f}]  '
            f'det {m["det"]:6.1%} [{m["det_ci"][0]:.1%}-{m["det_ci"][1]:.1%}]  '
            f'falsereject {m["fr"]:6.1%} [{m["fr_ci"][0]:.1%}-{m["fr_ci"][1]:.1%}]')


def tertiles(values):
    lo, hi = np.percentile(values, [100 / 3, 200 / 3])
    return lo, hi


def main():
    rows = rc.load_rows()
    for r in rows:
        r['n_assert'] = len(cf.split_claims(textnorm_strip(r['response'])))
        span_chars = len({i for a, b, _ in r['spans'] for i in range(a, b)})
        r['hall_share'] = span_chars / max(1, len(r['response']))
    a_lo, a_hi = tertiles([r['n_assert'] for r in rows])
    hs = [r['hall_share'] for r in rows if r['hallucinated']]
    s_lo, s_hi = tertiles(hs)
    print(f'assertion tertile cuts: {a_lo:.1f} / {a_hi:.1f}; '
          f'hallucinated-share cuts: {s_lo:.3f} / {s_hi:.3f}')

    def a_bin(r):
        return 0 if r['n_assert'] <= a_lo else 1 if r['n_assert'] <= a_hi else 2

    def s_bin(r):
        return 0 if r['hall_share'] <= s_lo else 1 if r['hall_share'] <= s_hi else 2

    out = {}
    for sc in SCORERS:
        name = f'ragtruth_{sc}'
        if not checkpoint_exists(name):
            print(f'\n[{sc}] no checkpoint'); continue
        s = load_checkpoint(name)
        missing = [r['id'] for r in rows if r['id'] not in s]
        if missing:
            print(f'\n[{sc}] {len(missing)} responses unscored — skipped'); continue
        print(f'\n=== {sc} ===')
        out[sc] = {'all': measures(rows, s)}
        print(f'  all                   {fmt(out[sc]["all"])}')
        for b, lab in enumerate([f'assert <= {a_lo:.0f}', f'assert <= {a_hi:.0f}',
                                 f'assert >  {a_hi:.0f}']):
            sub = [r for r in rows if a_bin(r) == b]
            out[sc][('assert', b)] = m = measures(sub, s)
            print(f'  {lab:<21} {fmt(m)}')
        for b, lab in enumerate(['share low', 'share mid', 'share high']):
            sub = [r for r in rows if r['hallucinated'] and s_bin(r) == b]
            k = sum(s[r['id']] < GATE for r in sub)
            lo, hi = _wilson_ci(k, len(sub))
            out[sc][('share', b)] = (k / len(sub), (lo, hi), len(sub))
            print(f'  {lab:<21} n={len(sub):<4} det {k / len(sub):6.1%} [{lo:.1%}-{hi:.1%}]')
        for g in sorted({r['model'] for r in rows}):
            sub = [r for r in rows if r['model'] == g]
            out[sc][('gen', g)] = m = measures(sub, s)
            print(f'  {g:<21} {fmt(m)}')
    post_hoc(rows)
    return out


PROBE_PREFIX = {'nli': 'perturb', 'claim': 'perturb_claim',
                'align': 'perturb_align', 'minicheck': 'perturb_minicheck',
                'minicheck_sent': 'perturb_minicheck_sent'}
PROBE_GENS = ['claude', 'gpt4omini', 'qwen', 'mistral']
PROBE_MODELS = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
                'text-embedding-3-small']


def post_hoc(rows):
    from scipy.stats import spearmanr
    print('\n=== post hoc ===')
    hh = [r for r in rows if r['hallucinated']]
    rho = spearmanr([r['hall_share'] for r in hh], [r['n_assert'] for r in hh])
    print(f'  share vs assertions among hallucinated: rho {rho.statistic:+.3f} '
          f'p {rho.pvalue:.4f}')
    for g in sorted({r['model'] for r in rows}):
        print(f'  {g:<21} assertions/answer '
              f'{np.mean([r["n_assert"] for r in rows if r["model"] == g]):.2f}')
    for sc, pre in PROBE_PREFIX.items():
        for ds in ['NQ', 'HotpotQA']:
            o, f = [], []
            for g in PROBE_GENS:
                for m in PROBE_MODELS:
                    for r in load_checkpoint(f'{pre}_{g}_{m}_{ds}'):
                        o.append(r['orig']); f.append(r['number'])
            print(f'  probe AUROC {sc:<15} {ds:<9} n={len(o)} '
                  f'{auroc(np.array(o), np.array(f)):.3f}')


def textnorm_strip(t):
    import textnorm
    return textnorm.strip_markdown(t)


if __name__ == '__main__':
    main()
