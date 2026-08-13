"""
conditional.py — the analysis Berend's 2026-08-11 letter actually asks for.

WHY THIS EXISTS
    correctness.py and conditions.py produce the ingredients — a correctness
    label per answer, a no-retrieval floor, an oracle ceiling, an eval filter —
    but nothing consumed them. results.py still assembled one pooled
    faithfulness mean per model x dataset, which is the exact quantity Berend
    questioned:

        "Being faithful the way you defined has a somewhat limited relevance
         in the sense that that kind of faithfulness only matters when the
         final answer is correct."

    This module joins everything at QUERY level and produces the three tables
    that answer him.

THE REFRAME
    His letter restates the paper's target claim as: good retrieval quality is
    NECESSARY AND SUFFICIENT for a high-quality response. Both halves are
    falsifiable per query, and both live in one 2x2:

                             answer correct     answer incorrect
        retrieval hit          (as expected)    NOT SUFFICIENT   <- branch A
        retrieval miss        NOT NECESSARY     (as expected)
                                   ^ branch B

    Cell (hit, incorrect) is where faithfulness matters most: an answer that is
    grounded in correctly retrieved text and still wrong. If mean faithfulness
    is HIGH in that cell, the paper has direct evidence that faithfulness and
    answer quality come apart — which is a stronger, more defensible claim than
    the pooled RFG gap, and it survives the objection above.

    Cell (miss, correct) is branch B and is measured independently by C1: a
    query the model answers correctly with NO retrieval never needed retrieval
    at all. That is what --emit-filter removes.

WHAT IT PRODUCES
    build_query_frame()          one row per query x model x dataset x generator
    necessity_sufficiency()      the 2x2 above, counts + mean faithfulness
    conditional_faithfulness()   faithfulness split by correct / incorrect
    anchor_table()               C1 floor -> embedders -> C2 ceiling

    All four are free: they read existing checkpoints, no API calls, no GPU.

USAGE
    python src/conditional.py                  # all three tables
    python src/conditional.py --filtered       # restrict to the eval filter
    python src/conditional.py --datasets NQ
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config
from utils import load_checkpoint, save_checkpoint

GENERATORS = ('claude', 'llama3')
# Sentinel model names for the embedder-independent conditions, so they can sit
# in the same frame as the real models without pretending to be one.
NO_RETRIEVAL = '(C1 no-retrieval)'
ORACLE = '(C2 oracle)'


# ── per-query retrieval quality ──────────────────────────────────────────────
def per_query_retrieval(loaded, model: str, ds_name: str,
                        k: int = None) -> Dict[str, dict]:
    """
    {query_id: {ndcg, recall, hit}} for one model x dataset.

    Phase B reports these averaged over queries; the conditional analysis needs
    them per query, because "did retrieval work HERE" is what pairs with "was
    the answer correct HERE". Recomputed from the same qrels Phase B uses, so
    the mean of `ndcg` here reproduces the Phase B NDCG@5 exactly.
    """
    from retrieval_eval import build_qrels, mrr_at_k, ndcg_at_k, recall_at_k
    k = k or config.TOP_K
    retrievals = load_checkpoint(f'retrieval_{model}_{ds_name}')
    if not retrievals:
        return {}
    qrels = build_qrels(loaded)
    out = {}
    for r in retrievals:
        relevant = set(qrels.get(r['query_id'], {}))
        if not relevant:
            continue          # no gold -> retrieval is not scoreable here
        ranked = r['retrieved_ids']
        out[r['query_id']] = {
            'ndcg': ndcg_at_k(ranked, relevant, k),
            'recall': recall_at_k(ranked, relevant, k),
            'mrr': mrr_at_k(ranked, relevant, k),
            # Binary "retrieval succeeded": at least one relevant chunk made
            # the top-k the generator actually saw.
            'hit': any(cid in relevant for cid in ranked[:k]),
        }
    return out


def _scored(name: str) -> Optional[List[dict]]:
    """Prefer the correctness-scored checkpoint, fall back to the raw one."""
    return load_checkpoint(f'{name}_scored') or load_checkpoint(name)


def _nli_by_qid(name: str) -> Dict[str, float]:
    scores = load_checkpoint(f'nli_scores_{name}') or []
    return {s['query_id']: s['nli_max'] for s in scores}


def load_eval_filter() -> Dict[str, set]:
    """{dataset: {query_id}} from conditions.py --emit-filter, or {}."""
    raw = load_checkpoint('eval_filter') or {}
    return {ds: set(ids) for ds, ids in raw.items()}


# ── the frame everything else reads ──────────────────────────────────────────
def build_query_frame(datasets: Dict, model_names: List[str] = None,
                      filtered: bool = False) -> pd.DataFrame:
    """
    One row per query x model x dataset x generator, plus the C1/C2 conditions.

    Columns
        query_id dataset model paradigm generator condition
        ndcg recall hit          retrieval quality for THIS query (NaN for C1/C2)
        correct correct_f1 abstained
        faithfulness             nli_max, NaN where not scored
    """
    eval_filter = load_eval_filter() if filtered else {}
    if filtered and not eval_filter:
        print('[conditional] --filtered requested but no eval_filter checkpoint '
              '— run `python src/conditions.py --condition c1 --yes` then '
              '`--emit-filter`. Falling back to the unfiltered set.')

    def keep(ds_name, qid):
        ids = eval_filter.get(ds_name)
        return True if not ids else qid in ids

    paradigm = {c['name']: c['paradigm'] for c in config.EMBEDDING_MODELS}
    models = [c['name'] for c in config.EMBEDDING_MODELS
              if not model_names or c['name'] in model_names]
    rows = []

    # ── the RAG grid ──
    for ds_name, loaded in datasets.items():
        for model in models:
            rq = per_query_retrieval(loaded, model, ds_name)
            if not rq:
                continue
            for gen in GENERATORS:
                base = f'generated_{gen}_{model}_{ds_name}'
                records = _scored(base)
                if not records:
                    continue
                nli = _nli_by_qid(f'{gen}_{model}_{ds_name}')
                for r in records:
                    qid = r['query_id']
                    if not keep(ds_name, qid):
                        continue
                    q = rq.get(qid)
                    if q is None:
                        continue
                    rows.append({
                        'query_id': qid, 'dataset': ds_name, 'model': model,
                        'paradigm': paradigm.get(model, '?'), 'generator': gen,
                        'condition': 'rag',
                        'ndcg': q['ndcg'], 'recall': q['recall'], 'hit': q['hit'],
                        'correct': r.get('correct'),
                        'correct_f1': r.get('correct_f1'),
                        'abstained': r.get('abstained'),
                        'faithfulness': nli.get(qid, np.nan),
                    })

    # ── the anchors ──
    for ds_name in datasets:
        for ck, model_label, cond in ((f'norag_{ds_name}', NO_RETRIEVAL, 'c1_norag'),
                                      (f'oracle_{ds_name}', ORACLE, 'c2_oracle')):
            records = _scored(ck)
            if not records:
                continue
            nli = _nli_by_qid(ck)
            for r in records:
                qid = r['query_id']
                if not keep(ds_name, qid):
                    continue
                rows.append({
                    'query_id': qid, 'dataset': ds_name, 'model': model_label,
                    'paradigm': 'condition', 'generator': 'claude',
                    'condition': cond,
                    # Retrieval quality is not DEFINED for these: C1 retrieves
                    # nothing and C2 bypasses the retriever. NaN, never 0.0 —
                    # a 0 here would be read as "retrieval failed".
                    'ndcg': np.nan, 'recall': np.nan, 'hit': np.nan,
                    'correct': r.get('correct'),
                    'correct_f1': r.get('correct_f1'),
                    'abstained': r.get('abstained'),
                    # C1 has no context, so faithfulness is undefined rather
                    # than zero — you cannot be unfaithful to nothing.
                    'faithfulness': (np.nan if cond == 'c1_norag'
                                     else nli.get(qid, np.nan)),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            'No query-level rows. Need at least Phase A/B/C plus '
            '`python src/correctness.py`.')
    save_checkpoint('query_frame', df)
    return df


# ── Berend's necessary/sufficient 2x2 ────────────────────────────────────────
def necessity_sufficiency(df: pd.DataFrame,
                          generator: str = 'claude') -> pd.DataFrame:
    """
    The central table. Counts and mean faithfulness in each cell of
    retrieval-hit x answer-correct, pooled over models.

    Read it as:
      hit & incorrect  -> good retrieval was NOT SUFFICIENT (branch A).
                          High faithfulness here is the paper's sharpest
                          finding: grounded, and wrong.
      miss & correct   -> good retrieval was NOT NECESSARY (branch B).
                          Compare its size against the C1 floor: if C1 already
                          answers these, retrieval was never in play.
    """
    sub = df[(df['condition'] == 'rag') & (df['generator'] == generator)]
    sub = sub[sub['correct'].notna() & sub['hit'].notna()]
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for hit in (True, False):
        for correct in (True, False):
            cell = sub[(sub['hit'] == hit) & (sub['correct'] == correct)]
            if hit and not correct:
                verdict = 'retrieval NOT SUFFICIENT'
            elif not hit and correct:
                verdict = 'retrieval NOT NECESSARY'
            else:
                verdict = 'as the premise expects'
            rows.append({
                'retrieval': 'hit' if hit else 'miss',
                'answer': 'correct' if correct else 'incorrect',
                'verdict': verdict,
                'n': len(cell),
                'share': round(len(cell) / len(sub), 4),
                'mean_faithfulness': (round(float(cell['faithfulness'].mean()), 4)
                                      if cell['faithfulness'].notna().any()
                                      else np.nan),
            })
    out = pd.DataFrame(rows)
    save_checkpoint(f'necessity_sufficiency_{generator}', out)
    return out


def conditional_faithfulness(df: pd.DataFrame,
                             generator: str = 'claude') -> pd.DataFrame:
    """
    Per model: faithfulness pooled, then split by whether the answer is right.

    `faith_gap` = faithfulness(incorrect) - faithfulness(correct). A positive
    gap means the model is MORE grounded when it is wrong, which is the
    strongest possible statement of Berend's objection to pooled faithfulness.
    """
    sub = df[(df['condition'] == 'rag') & (df['generator'] == generator)]
    sub = sub[sub['correct'].notna()]
    rows = []
    for (model, paradigm), g in sub.groupby(['model', 'paradigm']):
        corr = g[g['correct']]['faithfulness']
        inco = g[~g['correct']]['faithfulness']
        f_c = float(corr.mean()) if corr.notna().any() else np.nan
        f_i = float(inco.mean()) if inco.notna().any() else np.nan
        rows.append({
            'model': model, 'paradigm': paradigm,
            'n': len(g),
            'accuracy': round(float(g['correct'].mean()), 4),
            'faith_pooled': round(float(g['faithfulness'].mean()), 4),
            'faith_correct': round(f_c, 4) if f_c == f_c else np.nan,
            'faith_incorrect': round(f_i, 4) if f_i == f_i else np.nan,
            'faith_gap': (round(f_i - f_c, 4)
                          if f_c == f_c and f_i == f_i else np.nan),
        })
    out = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
    save_checkpoint(f'conditional_faithfulness_{generator}', out)
    return out


def anchor_table(df: pd.DataFrame, generator: str = 'claude') -> pd.DataFrame:
    """
    Floor -> embedders -> ceiling, per dataset.

    `pct_of_oracle` places each embedder on the C1..C2 scale:
        (accuracy - floor) / (ceiling - floor)
    0% means the embedder adds nothing over answering with no context at all;
    100% means it matches perfect retrieval. Values above 100% are the case
    Berend flagged as interesting — retrieval beating the oracle — and are
    reported, not clipped.
    """
    sub = df[df['generator'] == generator]
    sub = sub[sub['correct'].notna()]
    rows = []
    for ds_name, g in sub.groupby('dataset'):
        acc = g.groupby('model')['correct'].mean()
        floor = acc.get(NO_RETRIEVAL, np.nan)
        ceiling = acc.get(ORACLE, np.nan)
        span = ceiling - floor if (floor == floor and ceiling == ceiling) else np.nan
        for model, a in acc.items():
            gm = g[g['model'] == model]
            rows.append({
                'dataset': ds_name, 'model': model,
                'n': len(gm),
                'accuracy': round(float(a), 4),
                'faithfulness': (round(float(gm['faithfulness'].mean()), 4)
                                 if gm['faithfulness'].notna().any() else np.nan),
                'pct_of_oracle': (round(float((a - floor) / span), 4)
                                  if span == span and abs(span) > 1e-9 else np.nan),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        # floor first, ceiling last, embedders in between by accuracy
        order = {NO_RETRIEVAL: 0, ORACLE: 2}
        out['_o'] = out['model'].map(lambda m: order.get(m, 1))
        out = out.sort_values(['dataset', '_o', 'accuracy'],
                              ascending=[True, True, False]).drop(columns='_o')
    save_checkpoint(f'anchor_table_{generator}', out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', default=None, help='comma-separated subset')
    ap.add_argument('--models', default=None, help='comma-separated subset')
    ap.add_argument('--n-queries', type=int, default=None)
    ap.add_argument('--generator', default='claude', choices=list(GENERATORS))
    ap.add_argument('--filtered', action='store_true',
                    help='restrict to queries the model gets wrong without '
                         'retrieval (needs conditions.py --emit-filter)')
    args = ap.parse_args()

    from datasets_loader import load_all
    from utils import set_seed
    set_seed()
    ds_names = args.datasets.split(',') if args.datasets else config.DATASETS
    models = args.models.split(',') if args.models else None
    datasets = load_all(args.n_queries or config.N_QUERIES, ds_names)

    df = build_query_frame(datasets, models, filtered=args.filtered)
    scope = 'FILTERED (retrieval-necessary queries only)' if args.filtered else 'all queries'
    print(f'\n=== query frame: {len(df):,} rows | {scope} ===')

    print(f'\n=== necessary/sufficient grid [{args.generator}] ===')
    ns = necessity_sufficiency(df, args.generator)
    print(ns.to_string(index=False) if not ns.empty else '  (no gradable rows)')

    print(f'\n=== faithfulness conditioned on correctness [{args.generator}] ===')
    cf = conditional_faithfulness(df, args.generator)
    print(cf.to_string(index=False) if not cf.empty else '  (no gradable rows)')
    if not cf.empty and (cf['faith_gap'] > 0).any():
        worse = cf[cf['faith_gap'] > 0]['model'].tolist()
        print(f'  [!] more faithful when WRONG: {", ".join(worse)} — report '
              f'this rather than the pooled column.')

    print(f'\n=== anchors: floor -> embedders -> ceiling [{args.generator}] ===')
    at = anchor_table(df, args.generator)
    print(at.to_string(index=False) if not at.empty else '  (run conditions.py)')
    if not at.empty and (at['pct_of_oracle'] > 1.0).any():
        beat = at[at['pct_of_oracle'] > 1.0]
        print(f'  [!] {len(beat)} embedder x dataset cells BEAT the oracle — '
              f'the case Berend asked to watch for. Do not clip it; check '
              f'context length and ordering with context_ablation.py.')


if __name__ == '__main__':
    main()
