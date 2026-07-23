"""
esa_analysis.py — §4.5 Geometric Analysis: Entailment-Similarity Alignment.

ESA (Eq. 4) correlates cosine similarity cos(E(q), E(d)) with an NLI
entailment signal over gold evidence chunks. Two variants are computed,
closing supervisor point 6:

  esa_gold — NLI(d -> gold_answer): the paper's Eq. (4). DELIBERATELY uses
             the gold answer (unlike faithfulness F): ESA measures a static
             property of the embedding space, independent of any generator.
  esa_query — NLI(d -> q): the signal actually available at re-ranking time
             (no answer exists yet). Its correlation with cosine similarity
             is what connects §4.5 to the §4.6 re-ranking strategy.

Terminology note: this analysis is correlational/geometric — never call it
"mechanistic" (supervisor point 6).
"""
from typing import Dict, List

import numpy as np
from scipy import stats

import config
from datasets_loader import LoadedDataset
from embed_index import EmbeddingModelWrapper, chunk_document
from nli import NLIScorer
from utils import (checkpoint_exists, free_memory, load_checkpoint,
                   save_checkpoint)


def _gold_chunks(loaded: LoadedDataset, n: int):
    """(sample, first gold chunk text) for up to n samples."""
    out = []
    for s in loaded.samples:
        if len(out) >= n:
            break
        chunks = chunk_document(f'{s.query_id}_gold', s.gold_context)
        if chunks:
            out.append((s, chunks[0].text))
    return out


def _correlations(cos_scores: List[float], nli_scores: List[float]) -> dict:
    pearson_r, p_p = stats.pearsonr(cos_scores, nli_scores)
    spearman_r, p_s = stats.spearmanr(cos_scores, nli_scores)
    return {
        'pearson_r': round(float(pearson_r), 4),
        'pearson_p': round(float(p_p), 6),
        'spearman_r': round(float(spearman_r), 4),
        'spearman_p': round(float(p_s), 6),
    }


def run_esa(datasets: Dict[str, LoadedDataset],
            model_names: List[str] = None,
            n_samples: int = config.ESA_N_SAMPLES) -> dict:
    """
    ESA for every model × dataset.
    Returns {model: {dataset: {'esa_gold': {...}, 'esa_query': {...}, 'n': int}}}
    checkpointed as esa_analysis.
    """
    if checkpoint_exists('esa_analysis'):
        return load_checkpoint('esa_analysis')

    nli = NLIScorer()
    configs = [c for c in config.EMBEDDING_MODELS
               if not model_names or c['name'] in model_names]

    # NLI scores don't depend on the embedding model — compute once per dataset
    nli_cache = {}
    for ds_name, loaded in datasets.items():
        pairs = _gold_chunks(loaded, n_samples)
        print(f'  NLI signals [{ds_name}] ({len(pairs)} gold chunks)...')
        nli_cache[ds_name] = {
            'pairs': pairs,
            # Eq. 4: does the gold chunk entail the gold answer?
            'nli_gold': nli.entailment_probs(
                [(chunk, s.answer) for s, chunk in pairs]),
            # Point-6 variant: does the gold chunk entail the query?
            'nli_query': nli.entailment_probs(
                [(chunk, s.question) for s, chunk in pairs]),
        }

    results = {}
    for cfg in configs:
        wrapper = EmbeddingModelWrapper(cfg)
        results[cfg['name']] = {}
        for ds_name in datasets:
            cache = nli_cache[ds_name]
            pairs = cache['pairs']
            if len(pairs) < 3:
                continue
            q_embs = wrapper.encode([s.question for s, _ in pairs],
                                    is_query=True)
            d_embs = wrapper.encode([chunk for _, chunk in pairs],
                                    is_query=False)
            cos_scores = np.sum(q_embs * d_embs, axis=1).tolist()

            results[cfg['name']][ds_name] = {
                'esa_gold': _correlations(cos_scores, cache['nli_gold']),
                'esa_query': _correlations(cos_scores, cache['nli_query']),
                'n': len(pairs),
            }
            r = results[cfg['name']][ds_name]
            print(f"  [{ds_name}] {cfg['name']}: "
                  f"ESA_gold ρ={r['esa_gold']['spearman_r']} | "
                  f"ESA_query ρ={r['esa_query']['spearman_r']}")
        del wrapper
        free_memory()

    save_checkpoint('esa_analysis', results)
    print('[ESA] complete')
    return results
