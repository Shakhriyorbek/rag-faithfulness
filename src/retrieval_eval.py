"""
retrieval_eval.py — Phase B: retrieval quality (NDCG@5, Recall@5, MRR@5).

Audit fix B3: the notebook keyed qrels on a re-chunked gold string and
matched it against retrieved chunk strings — two lossy chunking passes that
essentially never produce equal strings, collapsing every retrieval metric
toward zero. Here relevance is decided by chunk ID provenance:

  - a chunk is relevant to query q iff its source document is gold for q;
  - on HotpotQA, additionally the chunk must contain one of q's gold
    supporting sentences (whitespace-normalized substring; falls back to
    doc-level relevance if a gold sentence straddles every chunk boundary).

Metrics are computed by a small internal implementation (binary relevance)
so the pipeline has no hard dependency on ranx; when ranx is installed the
numbers are cross-checked against it.
"""
from collections import defaultdict
from math import log2
from typing import Dict, List

import config
from datasets_loader import LoadedDataset
from embed_index import build_chunks
from utils import load_checkpoint, save_checkpoint, checkpoint_exists


def _norm(s: str) -> str:
    return ' '.join(s.split())


def build_qrels(loaded: LoadedDataset) -> Dict[str, Dict[str, int]]:
    """{query_id: {chunk_id: 1}} from document provenance. Cached."""
    from datasets_loader import CORPUS_VERSION
    ck = f'qrels_{loaded.name}_{len(loaded.samples)}_{CORPUS_VERSION}'
    cached = load_checkpoint(ck)
    if cached:
        return cached

    chunks = build_chunks(loaded)
    docs = {d.doc_id: d for d in loaded.documents}
    chunks_by_doc = defaultdict(list)
    for c in chunks:
        chunks_by_doc[c.doc_id].append(c)

    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for doc in loaded.documents:
        if not doc.gold_for:
            continue
        doc_chunks = chunks_by_doc.get(doc.doc_id, [])
        for qid in doc.gold_for:
            gold_sents = doc.gold_sentences.get(qid)
            if gold_sents:
                # HotpotQA: only chunks containing a gold sentence
                matched = [
                    c for c in doc_chunks
                    if any(_norm(s) in _norm(c.text) for s in gold_sents)
                ]
                for c in matched or doc_chunks:  # fallback: doc-level
                    qrels[qid][c.chunk_id] = 1
            else:
                for c in doc_chunks:
                    qrels[qid][c.chunk_id] = 1

    qrels = dict(qrels)
    n_empty = sum(1 for s in loaded.samples if qid_missing(qrels, s.query_id))
    if n_empty:
        print(f'  WARNING: {n_empty} queries with empty qrels in {loaded.name}')
    save_checkpoint(ck, qrels)
    return qrels


def qid_missing(qrels, qid) -> bool:
    return not qrels.get(qid)


# ── Internal binary-relevance metrics ─────────────────────────────
def ndcg_at_k(ranked_ids: List[str], relevant: set, k: int) -> float:
    dcg = sum(1.0 / log2(i + 2)
              for i, cid in enumerate(ranked_ids[:k]) if cid in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return hits / len(relevant)


def mrr_at_k(ranked_ids: List[str], relevant: set, k: int) -> float:
    for i, cid in enumerate(ranked_ids[:k]):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_run(qrels: Dict[str, Dict[str, int]],
                 run: Dict[str, List[str]], k: int = config.TOP_K) -> dict:
    """Mean NDCG@k / Recall@k / MRR@k over queries present in both."""
    ndcgs, recalls, mrrs = [], [], []
    for qid, ranked in run.items():
        relevant = set(qrels.get(qid, {}))
        if not relevant:
            continue
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
        recalls.append(recall_at_k(ranked, relevant, k))
        mrrs.append(mrr_at_k(ranked, relevant, k))
    n = len(ndcgs)
    if n == 0:
        return {'NDCG@5': 0.0, 'Recall@5': 0.0, 'MRR@5': 0.0, 'n': 0}
    return {
        'NDCG@5': round(sum(ndcgs) / n, 4),
        'Recall@5': round(sum(recalls) / n, 4),
        'MRR@5': round(sum(mrrs) / n, 4),
        'n': n,
    }


def _crosscheck_with_ranx(qrels, run_scores, ours):
    """If ranx is installed, verify our numbers against it."""
    try:
        from ranx import Qrels, Run, evaluate
    except ImportError:
        return
    r = evaluate(Qrels(qrels), Run(run_scores),
                 ['ndcg@5', 'recall@5', 'mrr@5'])
    for ours_key, ranx_key in [('NDCG@5', 'ndcg@5'), ('Recall@5', 'recall@5'),
                               ('MRR@5', 'mrr@5')]:
        if abs(ours[ours_key] - r[ranx_key]) > 0.01:
            print(f'  WARNING: {ours_key} differs from ranx: '
                  f'{ours[ours_key]} vs {r[ranx_key]:.4f}')


# ── Phase B ───────────────────────────────────────────────────────
def run_phase_b(datasets: Dict[str, LoadedDataset],
                model_names: List[str] = None) -> dict:
    """{model: {dataset: {metric: value}}} -> checkpoint retrieval_quality_all."""
    if checkpoint_exists('retrieval_quality_all'):
        return load_checkpoint('retrieval_quality_all')

    model_list = [c['name'] for c in config.EMBEDDING_MODELS
                  if not model_names or c['name'] in model_names]
    results = {}
    for model in model_list:
        results[model] = {}
        for ds_name, loaded in datasets.items():
            retrievals = load_checkpoint(f'retrieval_{model}_{ds_name}')
            if not retrievals:
                print(f'  missing retrieval_{model}_{ds_name} — run Phase A')
                continue
            qrels = build_qrels(loaded)
            run = {r['query_id']: r['retrieved_ids'] for r in retrievals}
            metrics = evaluate_run(qrels, run)
            run_scores = {
                r['query_id']: dict(zip(r['retrieved_ids'],
                                        r['retrieval_scores']))
                for r in retrievals
            }
            _crosscheck_with_ranx(qrels, run_scores, metrics)
            results[model][ds_name] = metrics
            print(f'  [{ds_name}] {model}: {metrics}')

    save_checkpoint('retrieval_quality_all', results)
    print('[phase B] complete')
    return results
