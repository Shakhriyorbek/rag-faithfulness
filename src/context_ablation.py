"""
context_ablation.py — Berend 2026-08-11, third suggestion.

    "Maybe it would also worth some investigation if you only used the relevant
     snippets (i.e. you rely on the oracle) and the only thing you change is the
     subset and/or the order of the relevant documents, and your task would be
     not to retrieve relevant documents, but to select the best subset of
     relevant document / order them in a way that they are the most useful for
     the investigated LLM."

WHAT THIS ADDS OVER C2
    C2 asks "what if retrieval were perfect". This asks the sharper question:
    given perfect relevance, does the ARRANGEMENT still change the answer? If
    it does, then relevance — which is all NDCG@5 measures — cannot be the
    whole story, and that is the paper's thesis argued from the oracle side
    instead of the embedder side.

    Every condition holds the generator, the prompt and (where marked) the slot
    count fixed. Only the contents and ordering of the context move.

THE CONDITIONS
    gold_all          every relevant chunk, corpus order.        (C2 baseline)
    gold_reversed     same chunks, reversed. Pure ordering test: any delta is
                      position sensitivity, since the information is identical.
    gold_shuffled     same chunks, seeded shuffle.
    gold_top1         the single most relevant chunk, nothing else. Tests
                      whether extra relevant context helps or distracts.
    gold1_first       1 gold + (k-1) hard negatives, gold in slot 1.
    gold1_middle      the same mix, gold buried in the middle.
    gold1_last        the same mix, gold in the final slot.
                      -> the three together are a lost-in-the-middle probe at
                         constant relevance and constant context length.
    noise_only        k hard negatives, zero gold.

    noise_only earns its cost twice over. It is a floor measured through the
    REAL RAG prompt with a REAL context, so unlike C1 it shares the prompt with
    the main run. C1 has to use a closed-book prompt (see conditions.py), which
    leaves the floor-to-RAG comparison carrying a prompt confound. noise_only
    has none: any query answered correctly here was answered from parametric
    knowledge with the RAG prompt held fixed, which is branch B of Berend's
    framing with the confound removed.

HARD NEGATIVES, NOT RANDOM TEXT
    Distractors are drawn from the reference embedder's own top-20 pool, minus
    anything the qrels mark relevant. They are what a real retriever actually
    returns and mistakes for relevant — a far more informative distractor than
    a random chunk from another document.

COST
    1 request per query per condition. All 8 conditions on 300 queries of one
    dataset is 2,400 requests, about $4.40 at the measured $0.00186/query.
    Nothing is spent without --yes.

USAGE
    python src/context_ablation.py --dry-run
    python src/context_ablation.py --conditions gold_all,gold_reversed,noise_only \\
                                   --datasets NQ --limit 300 --yes
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import config
from conditions import (EST_INPUT_TOKENS_CTX, _abort_on_repeated_errors,
                        _confirm, _limited, _resume)
from utils import checkpoint_exists, cost_tracker, load_checkpoint, save_checkpoint

# Embedder whose top-20 pool supplies the hard negatives. Any of the seven
# works; fixing one keeps the distractor distribution constant across
# conditions, which is what makes them comparable.
DEFAULT_DISTRACTOR_SOURCE = 'all-mpnet-base-v2'

CONDITIONS = ('gold_all', 'gold_reversed', 'gold_shuffled', 'gold_top1',
              'gold1_first', 'gold1_middle', 'gold1_last', 'noise_only')

# Conditions that hold the slot count at exactly TOP_K, so faithfulness and
# accuracy differences between them cannot be attributed to context length.
FIXED_SLOT_CONDITIONS = frozenset(
    {'gold1_first', 'gold1_middle', 'gold1_last', 'noise_only'})


class ContextPool:
    """Gold and hard-negative chunks for every query in one dataset."""

    def __init__(self, loaded, distractor_source: str = DEFAULT_DISTRACTOR_SOURCE):
        from embed_index import build_chunks
        from retrieval_eval import build_qrels
        self.name = loaded.name
        self.qrels = build_qrels(loaded)
        self.chunk_text = {c.chunk_id: c.text for c in build_chunks(loaded)}
        self.all_ids = list(self.chunk_text)
        retr = load_checkpoint(f'retrieval_{distractor_source}_{loaded.name}')
        if not retr:
            raise RuntimeError(
                f'need retrieval_{distractor_source}_{loaded.name} for hard '
                f'negatives — run Phase A first, or pass --distractor-source '
                f'for a model that has been run.')
        self.pool = {r['query_id']: r['retrieved_ids'] for r in retr}

    def gold_ids(self, qid: str) -> List[str]:
        return [cid for cid in self.qrels.get(qid, {}) if cid in self.chunk_text]

    def distractor_ids(self, qid: str, n: int, rng: random.Random) -> List[str]:
        """Top-ranked non-relevant chunks; topped up randomly if the pool is thin."""
        relevant = set(self.qrels.get(qid, {}))
        out = [cid for cid in self.pool.get(qid, [])
               if cid not in relevant and cid in self.chunk_text][:n]
        if len(out) < n:
            taken = set(out) | relevant
            spare = [cid for cid in self.all_ids if cid not in taken]
            rng.shuffle(spare)
            out.extend(spare[:n - len(out)])
        return out

    def texts(self, ids: List[str]) -> List[str]:
        return [self.chunk_text[cid] for cid in ids if cid in self.chunk_text]


def build_context(pool: ContextPool, qid: str, condition: str,
                  rng: random.Random, top_k: int = None) -> Optional[List[str]]:
    """
    Chunk texts for one query under one condition, or None when the query
    cannot support it (no gold evidence).
    """
    top_k = top_k or config.TOP_K
    gold = pool.gold_ids(qid)

    if condition == 'noise_only':
        # The only condition that does not need gold, but it is still skipped
        # for ungraded queries so every condition covers the same query set.
        if not gold:
            return None
        return pool.texts(pool.distractor_ids(qid, top_k, rng))

    if not gold:
        return None

    if condition == 'gold_all':
        return pool.texts(gold[:top_k])
    if condition == 'gold_reversed':
        return pool.texts(gold[:top_k])[::-1]
    if condition == 'gold_shuffled':
        ids = gold[:top_k].copy()
        rng.shuffle(ids)
        return pool.texts(ids)
    if condition == 'gold_top1':
        return pool.texts(gold[:1])

    if condition in ('gold1_first', 'gold1_middle', 'gold1_last'):
        one = pool.texts(gold[:1])
        if not one:
            return None
        noise = pool.texts(pool.distractor_ids(qid, top_k - 1, rng))
        if condition == 'gold1_first':
            return one + noise
        if condition == 'gold1_last':
            return noise + one
        mid = len(noise) // 2
        return noise[:mid] + one + noise[mid:]

    raise ValueError(f'unknown condition {condition!r}')


def run_ablation(datasets: Dict, conditions: List[str], dry_run: bool = False,
                 yes: bool = False, limit: Optional[int] = None,
                 distractor_source: str = DEFAULT_DISTRACTOR_SOURCE) -> None:
    from generate import ClaudeGenerator

    n_per_ds = {name: len(_limited(ds.samples, limit))
                for name, ds in datasets.items()}
    total = sum(n_per_ds.values()) * len(conditions)
    print(f'=== context ablation: {len(conditions)} conditions x '
          f'{sum(n_per_ds.values()):,} queries = {total:,} requests ===')
    for c in conditions:
        fixed = ' [fixed slots]' if c in FIXED_SLOT_CONDITIONS else ''
        print(f'    {c}{fixed}')
    if not _confirm(total, EST_INPUT_TOKENS_CTX, yes) or dry_run:
        return

    gen = ClaudeGenerator()
    for ds_name, ds in datasets.items():
        pool = ContextPool(ds, distractor_source)
        samples = _limited(ds.samples, limit)
        for condition in conditions:
            ck = f'ctx_{condition}_{ds_name}'
            if checkpoint_exists(ck):
                print(f'  [skip] {ck}')
                continue
            records, done = _resume(ck, len(samples))
            consecutive, first_error, n_skip = 0, None, 0
            print(f'  [{ds_name}] {condition} ({done}/{len(samples)} done)...')
            # Seeded per (dataset, condition) so shuffles and distractor
            # top-ups reproduce exactly on a resume or a rerun.
            rng = random.Random(f'{config.SEED}:{ds_name}:{condition}')
            for i, s in enumerate(samples[done:], start=done):
                contexts = build_context(pool, s.query_id, condition, rng)
                if not contexts:
                    n_skip += 1
                    records.append({
                        'query_id': s.query_id, 'question': s.question,
                        'answer': s.answer, 'generated_answer': None,
                        'generator': 'claude', 'generator_model': gen.model,
                        'condition': f'ctx_{condition}', 'dataset': ds_name,
                        'retrieved_texts': [], 'n_context_chunks': 0,
                        'context_missing': True,
                    })
                    continue
                answer = gen.generate(s.question, contexts)
                consecutive, first_error = _abort_on_repeated_errors(
                    answer, consecutive, first_error)
                records.append({
                    'query_id': s.query_id, 'question': s.question,
                    'answer': s.answer, 'generated_answer': answer,
                    'generator': 'claude', 'generator_model': gen.model,
                    'condition': f'ctx_{condition}', 'dataset': ds_name,
                    'retrieved_texts': contexts,
                    'n_context_chunks': len(contexts),
                    'context_missing': False,
                })
                if (i + 1) % 100 == 0:
                    save_checkpoint(ck + '_partial', records)
                    print(f'    {i + 1}/{len(samples)}  '
                          f'running cost ${cost_tracker.cost:.4f}')
            save_checkpoint(ck, records)
            print(f'  {ds_name}/{condition}: {len(records) - n_skip} generated'
                  + (f', {n_skip} skipped (no gold)' if n_skip else ''))

    print('\nNext: `python src/correctness.py` then '
          '`python src/context_ablation.py --report`')


def report(datasets: Dict) -> None:
    """
    Accuracy and faithfulness per condition. Free — reads checkpoints only.

    Reads gold_all as the reference. The two comparisons that matter:
      gold_reversed vs gold_all   information identical, order changed
      gold1_first/middle/last     relevance and length identical, position changed
    A non-zero delta in either is evidence that relevance alone does not
    determine response quality.
    """
    import numpy as np
    import pandas as pd

    rows = []
    for ds_name in datasets:
        for condition in CONDITIONS:
            base = f'ctx_{condition}_{ds_name}'
            recs = load_checkpoint(f'{base}_scored') or load_checkpoint(base)
            if not recs:
                continue
            nli = {s['query_id']: s['nli_max']
                   for s in (load_checkpoint(f'nli_scores_{base}') or [])}
            graded = [r for r in recs if r.get('correct') is not None]
            if not graded:
                print(f'  [{base}] not scored — run `python src/correctness.py`')
                continue
            faith = [nli[r['query_id']] for r in graded if r['query_id'] in nli]
            rows.append({
                'dataset': ds_name,
                'condition': condition,
                'fixed_slots': condition in FIXED_SLOT_CONDITIONS,
                'n': len(graded),
                'mean_chunks': round(float(np.mean(
                    [r.get('n_context_chunks', 0) for r in graded])), 2),
                'accuracy': round(float(np.mean(
                    [r['correct'] for r in graded])), 4),
                'abstention': round(float(np.mean(
                    [bool(r.get('abstained')) for r in graded])), 4),
                'faithfulness': round(float(np.mean(faith)), 4) if faith else np.nan,
            })
    if not rows:
        print('No ablation checkpoints found — run the conditions first.')
        return

    df = pd.DataFrame(rows)
    save_checkpoint('context_ablation_report', df)
    print(df.to_string(index=False))

    for ds_name, g in df.groupby('dataset'):
        by = g.set_index('condition')['accuracy']
        print(f'\n  [{ds_name}]')
        if {'gold_all', 'gold_reversed'} <= set(by.index):
            d = by['gold_reversed'] - by['gold_all']
            print(f'    order effect (reversed - all):     {d:+.4f}'
                  f'   <- identical information, different order')
        if {'gold_all', 'gold_top1'} <= set(by.index):
            d = by['gold_top1'] - by['gold_all']
            print(f'    subset effect (top1 - all):        {d:+.4f}'
                  f'   <- fewer relevant chunks, {"better" if d > 0 else "worse"}')
        pos = [c for c in ('gold1_first', 'gold1_middle', 'gold1_last')
               if c in by.index]
        if len(pos) == 3:
            spread = by[pos].max() - by[pos].min()
            print(f'    position spread (1 gold + noise):  {spread:.4f}'
                  f'   <- relevance and length held constant')
        if 'noise_only' in by.index:
            print(f'    noise_only accuracy:               {by["noise_only"]:.4f}'
                  f'   <- answered with ZERO relevant context, RAG prompt intact')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--conditions', default=','.join(CONDITIONS),
                    help=f'comma-separated subset of {list(CONDITIONS)}')
    ap.add_argument('--datasets', default=None)
    ap.add_argument('--n-queries', type=int, default=None)
    ap.add_argument('--limit', type=int, default=300,
                    help='queries per dataset per condition (default 300)')
    ap.add_argument('--distractor-source', default=DEFAULT_DISTRACTOR_SOURCE,
                    help='embedder whose top-20 pool supplies hard negatives')
    ap.add_argument('--report', action='store_true',
                    help='summarize existing checkpoints; generates nothing')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--yes', action='store_true', help='authorize spending')
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(',') if c.strip()]
    unknown = set(conds) - set(CONDITIONS)
    if unknown:
        ap.error(f'unknown conditions: {sorted(unknown)}')

    from datasets_loader import load_all
    from utils import set_seed
    set_seed()
    ds_names = args.datasets.split(',') if args.datasets else config.DATASETS
    datasets = load_all(args.n_queries or config.N_QUERIES, ds_names)

    if args.report:
        report(datasets)
        return
    run_ablation(datasets, conds, args.dry_run, args.yes, args.limit,
                 args.distractor_source)
    if not args.dry_run:
        print(f'\n=== cost: ${cost_tracker.cost:.4f} over '
              f'{cost_tracker.requests:,} requests '
              f'({cost_tracker.errors} errors) ===')


if __name__ == '__main__':
    main()
