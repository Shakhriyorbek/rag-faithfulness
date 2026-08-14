"""
results.py — assemble phase checkpoints into the results DataFrame,
statistical testing, robustness analysis, hypothesis summary.

Replaces the notebook's undefined SIMULATED_RESULTS (audit B2): everything
downstream (figures, tables) reads the DataFrame assembled here from real
checkpoints.

Metric conventions (CLAUDE.md §5):
  - nRFG is the PRIMARY metric; raw RFG is a secondary diagnostic and is
    always reported alongside absolute faithfulness so both-low stays visible.
  - RFG/nRFG reuse src/metrics.py (rfg, nrfg, robustness grid).
"""
from typing import Dict, List

import numpy as np
import pandas as pd

import config
from metrics import (compute_rfg_variants, nrfg, rfg,
                     robustness_correlation_matrix, summarize_robustness)
from utils import checkpoint_exists, load_checkpoint, save_checkpoint

GENERATORS = ['claude', 'llama3']


def bootstrap_significance(scores_a: List[float], scores_b: List[float],
                           n_bootstrap: int = 10_000,
                           alpha: float = 0.05) -> dict:
    """Paired bootstrap resampling test (Dror et al.), seed = config.SEED."""
    assert len(scores_a) == len(scores_b), 'samples must be paired'
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    n = len(a)
    observed = a.mean() - b.mean()
    rng = np.random.default_rng(config.SEED)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    # The bootstrap distribution is centered at the OBSERVED diff; the null
    # distribution is obtained by re-centering it at zero. (The notebook
    # compared the un-centered distribution against the observed diff, which
    # makes p ~= 0.5 for any true effect — nothing could ever be significant.)
    centered = diffs - diffs.mean()
    p_value = float(np.mean(np.abs(centered) >= np.abs(observed)))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return {
        'observed_diff': round(float(observed), 4),
        'p_value': round(p_value, 4),
        'significant': p_value < alpha,
        'ci_95': (round(float(ci_low), 4), round(float(ci_high), 4)),
        'n_bootstrap': n_bootstrap,
    }


# ── Equivalence testing (matched retrieval quality) ───────────────
# The paper's premise is that these embedders reach NEAR-IDENTICAL retrieval
# quality yet diverge downstream. A non-significant difference does not
# establish that: absence of evidence is not evidence of absence, and with
# n=1000 a t-test can fail to reject while the true gap is large.
#
# TOST (two one-sided tests) inverts the burden. The null is
# "the models DIFFER by at least the margin"; rejecting it at alpha licenses
# the claim of equivalence. Two one-sided tests are run against +/- margin
# and the LARGER p-value decides — equivalence needs both tails rejected.
#
# The test is PAIRED because every embedder is evaluated on exactly the same
# queries; pairing removes per-query difficulty, which is by far the largest
# source of variance here.
EQUIV_MARGIN_NDCG = 0.02   # what counts as "the same" retrieval quality


def tost_equivalence(scores_a, scores_b, margin: float = EQUIV_MARGIN_NDCG,
                     alpha: float = 0.05) -> dict:
    """Paired TOST on per-query scores. Equivalent iff p_tost < alpha.

    margin is on the metric's own scale (NDCG@5 points, so 0.02 = 2 points).
    Choosing it is a judgement call that belongs in the paper, not a default
    to hide behind — state the margin and why.
    """
    from scipy import stats

    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    assert len(a) == len(b), 'TOST is paired — samples must align'
    d = a - b
    n = len(d)
    mean_d = float(d.mean())
    # ddof=1 on a single observation is nan (and warns); n<2 is handled by
    # the zero-variance branch below, which refuses to claim equivalence.
    sd = float(d.std(ddof=1)) if n >= 2 else 0.0
    se = sd / np.sqrt(n) if sd > 0 else 0.0

    if se == 0 or not np.isfinite(se):
        # Every query differs by the same amount (often exactly zero): there
        # is no sampling uncertainty left, so the comparison is decided by
        # the offset itself. n=1 lands here too via sd=nan, and must NOT
        # pass — one query cannot establish anything.
        equivalent = n >= 2 and abs(mean_d) < margin
        return {'mean_diff': round(mean_d, 5), 'n': n, 'margin': margin,
                'p_lower': 0.0 if equivalent else 1.0,
                'p_upper': 0.0 if equivalent else 1.0,
                'p_tost': 0.0 if equivalent else 1.0,
                'equivalent': bool(equivalent),
                'ci_90': (round(mean_d, 5), round(mean_d, 5)),
                'note': 'zero variance in paired differences'}

    df = n - 1
    # H0_lower: diff <= -margin   (reject => diff is above -margin)
    t_lower = (mean_d + margin) / se
    p_lower = float(stats.t.sf(t_lower, df))
    # H0_upper: diff >= +margin   (reject => diff is below +margin)
    t_upper = (mean_d - margin) / se
    p_upper = float(stats.t.cdf(t_upper, df))
    p_tost = max(p_lower, p_upper)

    # The (1-2*alpha) CI is the interval TOST is equivalent to: contained in
    # +/-margin iff the test rejects. Reported because reviewers read it
    # faster than a p-value.
    crit = stats.t.ppf(1 - alpha, df)
    ci = (mean_d - crit * se, mean_d + crit * se)
    return {
        'mean_diff': round(mean_d, 5),
        'n': n,
        'margin': margin,
        'p_lower': round(p_lower, 4),
        'p_upper': round(p_upper, 4),
        'p_tost': round(p_tost, 4),
        'equivalent': bool(p_tost < alpha),
        'ci_90': (round(ci[0], 5), round(ci[1], 5)),
    }


def equivalence_table(dataset: str = None, metric: str = 'NDCG@5',
                      margin: float = EQUIV_MARGIN_NDCG) -> pd.DataFrame:
    """Pairwise paired TOST over every embedder pair, from Phase B output.

    Reads per_query_rq_{model}_{dataset} checkpoints. Also runs the ordinary
    paired difference test, because the honest sentence is a conjunction:
    "no detectable difference AND equivalent within +/-margin". A pair that
    is neither is simply not matched, and the paper must not describe it as
    such.
    """
    from scipy import stats

    ds_list = [dataset] if dataset else config.DATASETS
    rows = []
    for ds_name in ds_list:
        loaded = {}
        for cfg in config.EMBEDDING_MODELS:
            pq = load_checkpoint(f"per_query_rq_{cfg['name']}_{ds_name}")
            if pq:
                loaded[cfg['name']] = pq
        names = sorted(loaded)
        for i, m_a in enumerate(names):
            for m_b in names[i + 1:]:
                shared = sorted(set(loaded[m_a]) & set(loaded[m_b]))
                if len(shared) < 3:
                    continue
                a = [loaded[m_a][q][metric] for q in shared]
                b = [loaded[m_b][q][metric] for q in shared]
                t = tost_equivalence(a, b, margin=margin)
                _, p_diff = stats.ttest_rel(a, b)
                rows.append({
                    'dataset': ds_name, 'metric': metric,
                    'model_a': m_a, 'model_b': m_b,
                    'mean_a': round(float(np.mean(a)), 4),
                    'mean_b': round(float(np.mean(b)), 4),
                    'mean_diff': t['mean_diff'],
                    'n': t['n'],
                    'p_difference': round(float(p_diff), 4),
                    'p_tost': t['p_tost'],
                    'ci_90_low': t['ci_90'][0],
                    'ci_90_high': t['ci_90'][1],
                    'equivalent': t['equivalent'],
                    'matched': bool(t['equivalent'] and p_diff >= 0.05),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        save_checkpoint('equivalence_table', df)
    return df


def _mean(scores, key):
    vals = [s[key] for s in scores if key in s]
    return float(np.mean(vals)) if vals else None


def assemble_results(force: bool = False) -> pd.DataFrame:
    """
    One row per model × dataset × generator with retrieval quality,
    faithfulness (NLI + AlignScore), RFG and nRFG.
    """
    if not force and checkpoint_exists('final_results_df'):
        return load_checkpoint('final_results_df')

    retrieval_quality = load_checkpoint('retrieval_quality_all')
    if not retrieval_quality:
        raise RuntimeError('Run Phase B first (retrieval_quality_all missing).')

    paradigm = {c['name']: c['paradigm'] for c in config.EMBEDDING_MODELS}
    rows = []
    for model, per_ds in retrieval_quality.items():
        for ds_name, rq in per_ds.items():
            for gen in GENERATORS:
                nli_scores = load_checkpoint(
                    f'nli_scores_{gen}_{model}_{ds_name}')
                if not nli_scores:
                    continue
                align_scores = load_checkpoint(
                    f'align_scores_{gen}_{model}_{ds_name}')

                nli_max = _mean(nli_scores, 'nli_max')
                align = _mean(align_scores or [], 'align_score')
                # 'mean' faithfulness = mean of the two signals when both
                # exist, else whichever is available
                signals = [v for v in (align, nli_max) if v is not None]
                faith_mean = float(np.mean(signals))

                rows.append({
                    'model': model,
                    'paradigm': paradigm.get(model, '?'),
                    'dataset': ds_name,
                    'generator': gen,
                    'NDCG@5': rq['NDCG@5'],
                    'Recall@5': rq['Recall@5'],
                    'MRR@5': rq['MRR@5'],
                    'nli_max': nli_max,
                    'nli_mean_agg': _mean(nli_scores, 'nli_mean'),
                    'align_score': align,
                    'faithfulness': faith_mean,
                    'RFG': rfg(rq['NDCG@5'], faith_mean),
                    'nRFG': nrfg(rq['NDCG@5'], faith_mean),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('No scored generations found — run Phases C/D.')
    save_checkpoint('final_results_df', df)
    print(f'assembled {len(df)} result rows')
    return df


def robustness_analysis(df: pd.DataFrame) -> str:
    """
    Berend point 3: 3 retrieval × 3 faithfulness metrics = 9 RFG variants,
    Spearman correlation of the model rankings they induce. Reuses
    metrics.py. Uses GPT-4o-mini rows (full 7-model coverage).
    """
    sub = df[df['generator'] == 'claude']
    per_model = {}
    for model, g in sub.groupby('model'):
        per_model[model] = {
            'ndcg@5': g['NDCG@5'].mean(),
            'recall@5': g['Recall@5'].mean(),
            'mrr@5': g['MRR@5'].mean(),
            'alignscore': g['align_score'].mean(),
            'nli': g['nli_max'].mean(),
            'mean': g['faithfulness'].mean(),
        }
    # Drop faithfulness metrics with no data (e.g. AlignScore not yet run)
    faith_metrics = [m for m in config.FAITHFULNESS_METRICS
                     if not np.isnan(next(iter(per_model.values()))[
                         {'alignscore': 'alignscore', 'nli': 'nli',
                          'mean': 'mean'}[m]])]
    variants = compute_rfg_variants(per_model, config.RETRIEVAL_METRICS,
                                    faith_metrics)
    labels, corr = robustness_correlation_matrix(variants)
    summary = summarize_robustness(labels, corr)
    save_checkpoint('robustness_matrix', (labels, corr))
    print(summary)
    return summary


def hypothesis_summary(df: pd.DataFrame) -> pd.DataFrame:
    """H1–H5 against real data. Failures are reported as failures."""
    out = []
    claude_rows = df[df['generator'] == 'claude']

    # H1: instruction-tuned < contrastive (nRFG)
    inst = claude_rows[claude_rows['paradigm'] == 'instruction-tuned']['nRFG'].tolist()
    cont = claude_rows[claude_rows['paradigm'] == 'contrastive']['nRFG'].tolist()
    if inst and cont:
        n = min(len(inst), len(cont))
        h1 = bootstrap_significance(cont[:n], inst[:n])
        out.append({
            'hypothesis': 'H1 instruction-tuned < contrastive (nRFG)',
            'observed': h1['observed_diff'], 'p': h1['p_value'],
            'supported': bool(h1['significant'] and h1['observed_diff'] > 0),
        })

    # H2: HotpotQA has the highest nRFG
    by_ds = claude_rows.groupby('dataset')['nRFG'].mean()
    if len(by_ds) > 1:
        out.append({
            'hypothesis': 'H2 HotpotQA highest nRFG (multi-hop)',
            'observed': round(float(by_ds.max() - by_ds.drop(by_ds.idxmax()).mean()), 4),
            'p': None,
            'supported': by_ds.idxmax() == 'HotpotQA',
        })

    # H3: model ranking by nRFG consistent across generators
    llama = df[df['generator'] == 'llama3']
    shared = sorted(set(claude_rows['model']) & set(llama['model']))
    if len(shared) >= 3:
        from scipy.stats import spearmanr
        r_g = claude_rows[claude_rows['model'].isin(shared)].groupby('model')['nRFG'].mean()
        r_l = llama[llama['model'].isin(shared)].groupby('model')['nRFG'].mean()
        rho, p = spearmanr(r_g[shared], r_l[shared])
        out.append({
            'hypothesis': 'H3 nRFG ranking consistent across generators',
            'observed': round(float(rho), 4), 'p': round(float(p), 4),
            'supported': bool(rho > 0.7),
        })

    # H4: ESA higher for instruction-tuned
    esa = load_checkpoint('esa_analysis')
    if esa:
        para = {c['name']: c['paradigm'] for c in config.EMBEDDING_MODELS}
        inst_r, cont_r = [], []
        for model, per_ds in esa.items():
            for ds_res in per_ds.values():
                target = (inst_r if para.get(model) == 'instruction-tuned'
                          else cont_r if para.get(model) == 'contrastive'
                          else None)
                if target is not None:
                    target.append(ds_res['esa_gold']['spearman_r'])
        if inst_r and cont_r:
            diff = float(np.mean(inst_r) - np.mean(cont_r))
            out.append({
                'hypothesis': 'H4 ESA higher for instruction-tuned',
                'observed': round(diff, 4), 'p': None,
                'supported': diff > 0,
            })

    # H5: re-ranking cuts worst model's RFG by >= 15% relative
    worst = claude_rows.groupby('model')['nRFG'].mean().idxmax()
    base = claude_rows[claude_rows['model'] == worst]
    deltas = []
    for ds_name in base['dataset'].unique():
        rr = load_checkpoint(
            f'reranked_claude_{worst}_{ds_name}_lam{config.LAMBDA_RERANK}')
        if not rr:
            continue
        row = base[base['dataset'] == ds_name].iloc[0]
        rr_faith = float(np.mean([x['nli_max'] for x in rr]))
        rfg_base = rfg(row['NDCG@5'], row['nli_max'])
        rfg_rr = rfg(row['NDCG@5'], rr_faith)  # retrieval side unchanged
        if rfg_base > 0:
            deltas.append((rfg_base - rfg_rr) / rfg_base)
    if deltas:
        rel = float(np.mean(deltas))
        out.append({
            'hypothesis': f'H5 re-ranking cuts {worst} RFG >=15% rel.',
            'observed': round(rel, 4), 'p': None,
            'supported': rel >= 0.15,
        })

    hyp_df = pd.DataFrame(out)
    save_checkpoint('hypothesis_summary', hyp_df)
    return hyp_df
