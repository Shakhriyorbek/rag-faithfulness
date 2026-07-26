"""
rerank.py — §4.6 faithfulness-aware re-ranking ablation (Eq. 5).

    score(d, q) = λ · cos(E(d), E(q)) + (1 − λ) · NLI(d, q)

NLI(d, q) — entailment of the QUERY by the document — is used because no
answer exists at retrieval time (the distinction supervisor point 6 forced
into the open; esa_analysis.py measures how well this proxy correlates).

The ablation: re-rank each query's top-RERANK_POOL candidates, regenerate
with the new top-5, re-score faithfulness, and compare RFG against the
cosine-only baseline. H5: ≥15% relative RFG reduction for the worst model.
"""
from typing import Dict, List, Tuple

import numpy as np

import config
from utils import checkpoint_exists, load_checkpoint, save_checkpoint


def rerank_candidates(cos_scores: List[float], nli_scores: List[float],
                      lam: float = config.LAMBDA_RERANK) -> List[int]:
    """Return candidate indices re-ordered by Eq. 5, best first."""
    combined = [lam * c + (1 - lam) * n
                for c, n in zip(cos_scores, nli_scores)]
    return sorted(range(len(combined)), key=lambda i: -combined[i])


def _query_nli_scores(nli, retrievals: List[dict],
                      pool: int) -> List[List[float]]:
    """NLI(d, q) for every candidate of every query (cached per call)."""
    all_scores = []
    for r in retrievals:
        pairs = [(t, r['question']) for t in r['retrieved_texts'][:pool]]
        all_scores.append(nli.entailment_probs(pairs))
    return all_scores


def run_rerank_experiment(model: str, datasets: Dict,
                          lam: float = config.LAMBDA_RERANK,
                          pool: int = config.RERANK_POOL):
    """
    Full ablation for one embedding model (typically the worst by nRFG):
    re-rank -> regenerate (GPT-4o-mini) -> NLI-score. Each stage checkpoints.
    Downstream comparison happens in results assembly (baseline vs reranked).
    """
    from generate import ClaudeGenerator
    from nli import NLIScorer
    nli = NLIScorer()
    gen = ClaudeGenerator()

    for ds_name in datasets:
        ck_out = f'reranked_claude_{model}_{ds_name}_lam{lam}'
        if checkpoint_exists(ck_out):
            print(f'  [skip] {ck_out}')
            continue
        retrievals = load_checkpoint(f'retrieval_{model}_{ds_name}')
        if not retrievals:
            print(f'  missing retrieval_{model}_{ds_name} — run Phase A')
            continue

        # 1. NLI(d, q) over the candidate pool (model-independent, cached)
        ck_nli = f'rerank_nli_dq_{model}_{ds_name}'
        nli_scores = load_checkpoint(ck_nli)
        if nli_scores is None:
            print(f'  NLI(d, q) over top-{pool} pool [{ds_name}]...')
            nli_scores = _query_nli_scores(nli, retrievals, pool)
            save_checkpoint(ck_nli, nli_scores)

        # 2. Re-rank, regenerate with new top-5, NLI-score the new answers
        print(f'  re-rank + regenerate [{model}] [{ds_name}] (λ={lam})...')
        results = load_checkpoint(ck_out + '_partial') or []
        done = len(results)
        for i, (r, nq) in enumerate(zip(retrievals[done:],
                                        nli_scores[done:]), start=done):
            order = rerank_candidates(
                r['retrieval_scores'][:pool], nq, lam)
            new_texts = [r['retrieved_texts'][j] for j in order][:config.TOP_K]
            new_ids = [r['retrieved_ids'][j] for j in order][:config.TOP_K]
            answer = gen.generate(r['question'], new_texts)
            faith = nli.score_chunks(new_texts, answer)
            results.append({
                'query_id': r['query_id'],
                'reranked_ids': new_ids,
                'generated_answer': answer,
                'nli_max': faith['nli_max'],
                'nli_mean': faith['nli_mean'],
                'nli_concat': faith['nli_concat'],
            })
            if (i + 1) % 100 == 0:
                save_checkpoint(ck_out + '_partial', results)
                print(f'    {i + 1}/{len(retrievals)}')
        save_checkpoint(ck_out, results)
        mean_f = np.mean([x['nli_max'] for x in results])
        print(f'  {model}/{ds_name} λ={lam}: mean nli_max = {mean_f:.4f}')
    print('[rerank] complete')


def lambda_sweep_ranking_only(model: str, ds_name: str,
                              lambdas: List[float] = None,
                              pool: int = config.RERANK_POOL
                              ) -> Dict[float, List[List[int]]]:
    """
    Cheap λ sensitivity: re-ranked orderings per λ WITHOUT regeneration
    (regeneration for every λ would multiply API cost). Retrieval-side
    metrics over these orderings guide which λ gets the full ablation.
    """
    lambdas = lambdas or config.RERANK_LAMBDAS
    retrievals = load_checkpoint(f'retrieval_{model}_{ds_name}')
    nli_scores = load_checkpoint(f'rerank_nli_dq_{model}_{ds_name}')
    if retrievals is None or nli_scores is None:
        raise RuntimeError('Run run_rerank_experiment first (needs the '
                           'cached NLI(d, q) scores).')
    out = {}
    for lam in lambdas:
        out[lam] = [
            rerank_candidates(r['retrieval_scores'][:pool], nq, lam)
            for r, nq in zip(retrievals, nli_scores)
        ]
    return out
