"""
doc_utility.py — per-document utility (Berend 2026-08-11, points 4 and 5).

TWO OF HIS POINTS ARE THE SAME POINT
    He linked https://dl.acm.org/doi/10.1145/3626772.3657957 and, separately,
    suggested a Shapley-inspired score for retrieved documents. Those connect:
    that paper is Salemi & Zamani, "Evaluating Retrieval Quality in
    Retrieval-Augmented Generation" (SIGIR 2024, arXiv:2404.13781), which
    proposes eRAG — run the LLM on each retrieved document INDIVIDUALLY and use
    the downstream score as that document's relevance label. eRAG is exactly the
    singleton term of a Shapley value. Shapley is its principled generalization:
    it also prices what a document contributes in the presence of the others.

    Read that paper before writing the related-work paragraph. Its headline
    result — annotation-based query-document relevance correlates only weakly
    with downstream RAG performance — is a published version of this paper's
    premise. Positioning has to account for it.

WHAT IS COMPUTED
    For the top-k chunks the retriever actually gave the generator, with
    v(S) = the quality of the answer produced from subset S:

      erag       v({i}) for each document alone           k requests/query
      loo        v(N) - v(N\\{i}), the marginal loss       k+1 requests/query
      shapley    the exact Shapley value                  2^k requests/query

                 phi_i = sum over S subset of N\\{i} of
                         |S|!(n-|S|-1)!/n! * [v(S + i) - v(S)]

    ENUMERATING ALL 2^k SUBSETS ONCE YIELDS ALL THREE. At k=5 that is 32
    generations per query, and eRAG (singletons) and LOO (N and N-minus-one)
    are already inside that set. Running --mode shapley therefore gives you
    eRAG and LOO for free; running them separately would only waste money.

    Two value functions are derived from the same generations:
      v_correct       1 if the answer matches gold (EM or token-F1 >= 0.6)
      v_faithful      nli_max against that subset's own context

    v_faithful is measured against S, so a document cannot be credited for
    grounding an answer in text that was not shown.

WHY IT MATTERS FOR THE ARGUMENT
    The report correlates each document's Shapley utility against its qrels
    relevance label and its retrieval rank. Weak correlation IS the claim
    "good retrieval quality is not sufficient", stated at document granularity
    instead of system granularity, and it is a stronger form of evidence than
    the aggregate RFG gap because it is per document and needs no metric of
    our own invention.

COST — read before running
    2^k grows fast, so this is a SUBSAMPLE experiment, not a grid one.
    At k=5, 32 requests/query. Mean subset size is k/2, so mean input is about
    half a full context: roughly $0.001/request.

        100 queries  ->  3,200 requests  ~ $3.4
        300 queries  ->  9,600 requests  ~ $10.2

    Default is 100 queries on one dataset. Nothing is spent without --yes.

USAGE
    python src/doc_utility.py --mode shapley --datasets NQ --dry-run
    python src/doc_utility.py --mode shapley --datasets NQ --limit 100 --yes
    python src/correctness.py                       # score, free
    python src/doc_utility.py --report --datasets NQ
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from math import factorial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import config
from conditions import _abort_on_repeated_errors, _limited, _resume
from utils import checkpoint_exists, cost_tracker, load_checkpoint, save_checkpoint

MODES = ('erag', 'loo', 'shapley')

# Mean input tokens for a subset generation. Subsets average k/2 chunks, so
# roughly half a full retrieved context plus the fixed prompt.
EST_INPUT_TOKENS_SUBSET = 800
COST_PER_1K_INPUT = 1.0 / 1000.0
COST_PER_1K_OUTPUT = 5.0 / 1000.0
EST_OUTPUT_TOKENS = 78


def subsets_for_mode(mode: str, k: int) -> List[Tuple[int, ...]]:
    """
    The subsets that must be generated, as sorted index tuples.

    Always includes the empty set: v({}) is the no-context baseline every
    marginal contribution is measured against, and it is the same quantity as
    the C1 floor measured through the RAG prompt.
    """
    full = tuple(range(k))
    if mode == 'erag':
        return [()] + [(i,) for i in range(k)]
    if mode == 'loo':
        return ([(), full]
                + [tuple(j for j in range(k) if j != i) for i in range(k)])
    if mode == 'shapley':
        out = []
        for size in range(k + 1):
            out.extend(combinations(range(k), size))
        return out
    raise ValueError(f'unknown mode {mode!r}')


def shapley_values(values: Dict[Tuple[int, ...], float], k: int) -> List[float]:
    """
    Exact Shapley values from a complete subset->value map.

    phi_i = sum over S subset of N\\{i} of
            |S|!(n-|S|-1)!/n! * [v(S + i) - v(S)]

    Requires all 2^k entries; raises rather than silently treating a missing
    subset as zero, which would bias every player's value toward zero without
    any indication that it happened.
    """
    n = k
    phi = [0.0] * k
    for i in range(k):
        others = [j for j in range(k) if j != i]
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for S in combinations(others, size):
                with_i = tuple(sorted(S + (i,)))
                if S not in values or with_i not in values:
                    raise KeyError(
                        f'incomplete subset coverage for player {i}: missing '
                        f'{S if S not in values else with_i}. Shapley needs '
                        f'all 2^{k} subsets; run --mode shapley to completion.')
                phi[i] += weight * (values[with_i] - values[S])
    return phi


def _mask(subset: Tuple[int, ...]) -> str:
    return ''.join('1' if i in subset else '0' for i in range(config.TOP_K))


def _project(n_requests: int) -> float:
    return n_requests * (EST_INPUT_TOKENS_SUBSET / 1000.0 * COST_PER_1K_INPUT
                         + EST_OUTPUT_TOKENS / 1000.0 * COST_PER_1K_OUTPUT)


def run_utility(datasets: Dict, mode: str, model: str,
                dry_run: bool = False, yes: bool = False,
                limit: int = 100, k: int = None) -> None:
    from generate import ClaudeGenerator

    k = k or config.TOP_K
    subsets = subsets_for_mode(mode, k)
    n_queries = sum(len(_limited(ds.samples, limit)) for ds in datasets.values())
    total = n_queries * len(subsets)
    est = _project(total)

    print(f'=== document utility [{mode}] over {model} retrievals ===')
    print(f'  {len(subsets)} subsets/query x {n_queries:,} queries '
          f'= {total:,} requests -> ~${est:.2f}')
    if mode == 'shapley':
        print(f'  (this run also yields eRAG and leave-one-out at no extra '
              f'cost — both are subsets of these {len(subsets)})')
    if not yes:
        print('  refusing to spend without --yes (--dry-run to inspect)')
        return
    if dry_run:
        return

    gen = ClaudeGenerator()
    for ds_name, ds in datasets.items():
        ck = f'util_{mode}_{model}_{ds_name}'
        if checkpoint_exists(ck):
            print(f'  [skip] {ck}')
            continue
        retr = load_checkpoint(f'retrieval_{model}_{ds_name}')
        if not retr:
            print(f'  missing retrieval_{model}_{ds_name} — run Phase A')
            continue
        by_qid = {r['query_id']: r for r in retr}
        samples = _limited(ds.samples, limit)

        records, done_rows = _resume(ck, len(samples) * len(subsets))
        done_q = done_rows // len(subsets)
        consecutive, first_error = 0, None
        print(f'  [{ds_name}] {done_q}/{len(samples)} queries done...')
        for qi, s in enumerate(samples[done_q:], start=done_q):
            r = by_qid.get(s.query_id)
            if not r:
                continue
            chunks = r['retrieved_texts'][:k]
            ids = r['retrieved_ids'][:k]
            for subset in subsets:
                # Subsets are presented in retrieval-rank order, so ordering is
                # held fixed and the only variable is membership. Ordering
                # effects are context_ablation.py's job, not this module's.
                ctx = [chunks[i] for i in subset if i < len(chunks)]
                if subset and not ctx:
                    continue
                answer = gen.generate(s.question, ctx)
                consecutive, first_error = _abort_on_repeated_errors(
                    answer, consecutive, first_error)
                records.append({
                    # Unique per row so correctness/NLI joins do not collide:
                    # one query contributes up to 2^k rows.
                    'query_id': f'{s.query_id}#{_mask(subset)}',
                    'base_query_id': s.query_id,
                    'subset': subset,
                    'subset_size': len(subset),
                    'chunk_ids': [ids[i] for i in subset if i < len(ids)],
                    'question': s.question, 'answer': s.answer,
                    'generated_answer': answer,
                    'generator': 'claude', 'generator_model': gen.model,
                    'condition': f'util_{mode}', 'dataset': ds_name,
                    'embedder': model,
                    'retrieved_texts': ctx, 'n_context_chunks': len(ctx),
                })
            if (qi + 1) % 20 == 0:
                save_checkpoint(ck + '_partial', records)
                print(f'    {qi + 1}/{len(samples)} queries  '
                      f'running cost ${cost_tracker.cost:.4f}')
        save_checkpoint(ck, records)
        print(f'  {ds_name}: {len(records):,} subset generations')

    print('\nNext: `python src/correctness.py`, then '
          f'`python src/doc_utility.py --report --mode {mode}`')


def report(datasets: Dict, mode: str, model: str, k: int = None) -> None:
    """
    Per-document utility, and how well it agrees with the retrieval signals.

    Free: reads scored checkpoints only.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    from retrieval_eval import build_qrels

    k = k or config.TOP_K
    all_rows = []
    for ds_name, loaded in datasets.items():
        ck = f'util_{mode}_{model}_{ds_name}'
        recs = load_checkpoint(f'{ck}_scored') or load_checkpoint(ck)
        if not recs:
            print(f'  [{ck}] nothing to report')
            continue
        if recs[0].get('correct') is None and 'correct' not in recs[0]:
            print(f'  [{ck}] not scored — run `python src/correctness.py`')
            continue
        nli = {s['query_id']: s['nli_max']
               for s in (load_checkpoint(f'nli_scores_{ck}') or [])}
        qrels = build_qrels(loaded)

        by_q: Dict[str, Dict[Tuple[int, ...], dict]] = {}
        for r in recs:
            by_q.setdefault(r['base_query_id'], {})[tuple(r['subset'])] = r

        n_skipped = 0
        for qid, subs in by_q.items():
            # Every subset must be gradable. A single ungradable row (an API
            # error, or a missing gold answer) makes the whole query's Shapley
            # decomposition ill-defined, and substituting 0.0 would bias every
            # player toward zero with nothing on screen to say so.
            if any(rec.get('correct') is None for s, rec in subs.items() if s):
                n_skipped += 1
                continue
            v_corr = {s: (1.0 if rec.get('correct') else 0.0)
                      for s, rec in subs.items()}
            v_corr[()] = 1.0 if subs.get((), {}).get('correct') else 0.0
            v_faith = {s: nli.get(rec['query_id'], 0.0)
                       for s, rec in subs.items()}
            v_faith[()] = 0.0     # no context -> nothing to be faithful to

            relevant = set(qrels.get(qid, {}))
            full = tuple(range(k))
            try:
                if mode == 'shapley':
                    phi_c = shapley_values(v_corr, k)
                    phi_f = shapley_values(v_faith, k)
                elif mode == 'loo':
                    phi_c = [v_corr.get(full, 0.0)
                             - v_corr.get(tuple(j for j in range(k) if j != i), 0.0)
                             for i in range(k)]
                    phi_f = [v_faith.get(full, 0.0)
                             - v_faith.get(tuple(j for j in range(k) if j != i), 0.0)
                             for i in range(k)]
                else:  # erag: the document's standalone downstream score
                    phi_c = [v_corr.get((i,), 0.0) for i in range(k)]
                    phi_f = [v_faith.get((i,), 0.0) for i in range(k)]
            except KeyError as e:
                print(f'  [{qid}] skipped: {e}')
                continue

            for i in range(k):
                single = subs.get((i,))
                cid = (single['chunk_ids'][0]
                       if single and single.get('chunk_ids') else None)
                all_rows.append({
                    'dataset': ds_name, 'query_id': qid, 'rank': i + 1,
                    'chunk_id': cid,
                    'qrel_relevant': int(cid in relevant) if cid else None,
                    'utility_correct': round(float(phi_c[i]), 4),
                    'utility_faithful': round(float(phi_f[i]), 4),
                })
        if n_skipped:
            print(f'  [{ds_name}] {n_skipped} queries skipped '
                  f'(at least one ungradable subset)')

    if not all_rows:
        print('No utility rows produced.')
        return

    df = pd.DataFrame(all_rows)
    save_checkpoint(f'doc_utility_{mode}_{model}', df)

    print(f'\n=== per-document utility [{mode}] — {len(df):,} documents ===')
    print(df.groupby(['dataset', 'rank'])[
        ['utility_correct', 'utility_faithful']].mean().round(4).to_string())

    graded = df[df['qrel_relevant'].notna()]
    if len(graded) > 10:
        print('\n=== does the retrieval label predict the utility? ===')
        for col in ('utility_correct', 'utility_faithful'):
            rho, p = spearmanr(graded['qrel_relevant'], graded[col])
            print(f'  qrel relevance vs {col:18s}: rho={rho:+.3f}  p={p:.4g}')
        rho_r, p_r = spearmanr(graded['rank'], graded['utility_correct'])
        print(f'  retrieval rank  vs utility_correct  : rho={rho_r:+.3f}  p={p_r:.4g}')
        print('\n  A weak or null correlation on the first line is the '
              'document-level form of "good retrieval is not sufficient", and\n'
              '  is the same effect Salemi & Zamani report (SIGIR 2024, '
              'arXiv:2404.13781). Cite them when writing it up.')

        harmful = graded[graded['utility_correct'] < 0]
        if len(harmful):
            share = len(harmful) / len(graded)
            rel_harm = harmful['qrel_relevant'].sum()
            print(f'\n  [!] {len(harmful):,} documents ({share:.1%}) have '
                  f'NEGATIVE utility — including them makes the answer worse.')
            print(f'      {rel_harm:,} of those are marked RELEVANT by the '
                  f'qrels. That subset is the paper\'s sharpest single '
                  f'observation:\n      correctly retrieved, genuinely '
                  f'relevant, and actively harmful.')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=list(MODES), default='shapley')
    ap.add_argument('--model', default='all-mpnet-base-v2',
                    help='which embedder\'s retrievals to price')
    ap.add_argument('--datasets', default='NQ')
    ap.add_argument('--n-queries', type=int, default=None)
    ap.add_argument('--limit', type=int, default=100,
                    help='queries per dataset (default 100 — 2^k is expensive)')
    ap.add_argument('--report', action='store_true',
                    help='summarize existing checkpoints; generates nothing')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--yes', action='store_true', help='authorize spending')
    args = ap.parse_args()

    from datasets_loader import load_all
    from utils import set_seed
    set_seed()
    ds_names = args.datasets.split(',')
    datasets = load_all(args.n_queries or config.N_QUERIES, ds_names)

    if args.report:
        report(datasets, args.mode, args.model)
        return
    run_utility(datasets, args.mode, args.model, args.dry_run, args.yes,
                args.limit)
    if args.yes and not args.dry_run:
        print(f'\n=== cost: ${cost_tracker.cost:.4f} over '
              f'{cost_tracker.requests:,} requests '
              f'({cost_tracker.errors} errors) ===')


if __name__ == '__main__':
    main()
