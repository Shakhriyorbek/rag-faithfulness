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
