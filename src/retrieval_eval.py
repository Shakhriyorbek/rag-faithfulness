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
import textnorm
from datasets_loader import LoadedDataset
from embed_index import build_chunks
from utils import load_checkpoint, save_checkpoint, checkpoint_exists


# NOTE: the whitespace-only `_norm` that used to live here is gone on
# purpose. It decided chunk relevance and matched nothing on 187/1000 NQ
# queries, sending them all to the doc-level fallback. Use textnorm.
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
    n_fallback = 0
    for doc in loaded.documents:
        if not doc.gold_for:
            continue
        doc_chunks = chunks_by_doc.get(doc.doc_id, [])
        for qid in doc.gold_for:
            gold_sents = doc.gold_sentences.get(qid)
            if gold_sents:
                # Answer-bearing relevance: a chunk of a gold document counts
                # only if it actually carries the gold span. textnorm, not a
                # whitespace-only test — the strict version matched nothing on
                # 187/1000 NQ queries and sent every one of them down the
                # doc-level fallback below, which is precisely the B6 bug it
                # was supposed to have fixed.
                matched = [c for c in doc_chunks
                           if textnorm.contains_any(c.text, gold_sents)]
                if not matched:
                    # A gold sentence can genuinely straddle a chunk boundary
                    # (HotpotQA). Falling back to doc-level keeps the query
                    # evaluable, but it weakens `hit` from "was shown the
                    # answer" to "got the right document", so it is counted
                    # and reported rather than applied silently.
                    n_fallback += 1
                for c in matched or doc_chunks:
                    qrels[qid][c.chunk_id] = 1
            else:
                for c in doc_chunks:
                    qrels[qid][c.chunk_id] = 1

    qrels = dict(qrels)
    if n_fallback:
        pct = n_fallback / max(1, len(loaded.samples))
        print(f'  [{loaded.name}] {n_fallback} gold docs fell back to '
              f'doc-level relevance ({pct:.1%} of queries) — no chunk '
              f'contained the gold span')
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


def per_query_metrics(qrels: Dict[str, Dict[str, int]],
                      run: Dict[str, List[str]],
                      k: int = config.TOP_K) -> Dict[str, dict]:
    """{query_id: {NDCG@5, Recall@5, MRR@5}} — the unaggregated scores.

    evaluate_run returns means, which cannot support a PAIRED test. The
    equivalence test in results.py needs per-query values so that two
    embedders are compared on the same queries.
    """
    out = {}
    for qid, ranked in run.items():
        relevant = set(qrels.get(qid, {}))
        if not relevant:
            continue
        out[qid] = {
            'NDCG@5': ndcg_at_k(ranked, relevant, k),
            'Recall@5': recall_at_k(ranked, relevant, k),
            'MRR@5': mrr_at_k(ranked, relevant, k),
        }
    return out


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
    """{model: {dataset: {metric: value}}} -> checkpoint retrieval_quality_all.

    Recomputed every call. Phase B is seconds of arithmetic over checkpoints
    that already exist, and the previous blanket
    `if checkpoint_exists('retrieval_quality_all'): return` meant a later run
    that ADDED models returned the earlier run's aggregate and reported the
    new models as missing. Per-model results are merged into the existing
    aggregate so a 3-model run cannot silently truncate a 7-model one.
    """
    model_list = [c['name'] for c in config.EMBEDDING_MODELS
                  if not model_names or c['name'] in model_names]
    results = load_checkpoint('retrieval_quality_all') or {}
    for model in model_list:
        results.setdefault(model, {})
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
            # Per-query scores feed the paired equivalence test (results.py).
            # Saved here because it is the only place qrels and the run are
            # both in hand.
            save_checkpoint(f'per_query_rq_{model}_{ds_name}',
                            per_query_metrics(qrels, run))
            print(f'  [{ds_name}] {model}: {metrics}')

    save_checkpoint('retrieval_quality_all', results)
    print('[phase B] complete')
    return results
