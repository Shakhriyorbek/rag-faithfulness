"""
minicheck_faithfulness.py — Bespoke-MiniCheck-7B over the matched grid, so it
can enter the system comparison as a fourth evaluator (third review: "I would
raise my score if the evaluator-dependence result were demonstrated beyond
NLI-derived scorers").

Until now MiniCheck ran on the falsification probe (D3, D4) and on RAGTruth
(D5) only, so Table 7 had no MiniCheck rows. This scores every answer of the
32 matched cells (4 generators x 4 systems x 2 datasets), refusals included so
the refusal-versus-attempt row can be filled too, and writes one checkpoint per
cell, ``minicheck_scores_{gen}_{model}_{ds}``, with both aggregates from one
pass, exactly as the probe does:

- ``minicheck``: the whole answer as one claim (the paper's MiniCheck column);
- ``minicheck_sent``: the minimum over nltk sentences (MiniCheck's own
  claim-level aggregation).

Inputs are the probe's, byte for byte: the same ``MiniCheckScorer.score``, the
same first TOP_K retrieved chunks, and the answer after ``strip_markdown``
(decision D1: every evaluator reads the stripped hypothesis).

``--validate N`` re-scores the untouched answers of the first N probe cases in
each named cell and compares them with the probe's stored ``orig`` scores. The
two runs batch differently, which moved shared scores by at most 0.004 in D4,
so anything above ``--tol`` (0.01) means the inputs differ and the run stops.

Resumable: a cell's progress is saved every 100 answers to
``..._partial`` and picked up on restart; the finished checkpoint replaces it.
Defaults name the matched grid explicitly (NQ and HotpotQA, never QASPER; the
four matched systems, never jina), and the selection is printed at startup,
because a default that silently drops or adds an arm has cost this project
eight runs (CLAUDE.md B30).
"""
import argparse
import sys
import time

import numpy as np

import config
import textnorm
from utils import (checkpoint_exists, checkpoint_path, load_checkpoint,
                   save_checkpoint)

DATASETS = ['NQ', 'HotpotQA']
MODELS = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
          'text-embedding-3-small']
GENERATORS = list(config.ACTIVE_GENERATORS)
SAVE_EVERY = 100


def _chunks(g):
    return (g.get('retrieved_texts') or [])[:config.TOP_K]


def _answer(g):
    """The stripped answer, or None for a row no evaluator scores."""
    ans = g.get('generated_answer')
    if not ans or ans.startswith('[ERROR'):
        return None
    return textnorm.strip_markdown(ans)


def _score(mc, g):
    s = mc.score(_chunks(g), _answer(g))
    if not (np.isfinite(s['minicheck']) and np.isfinite(s['minicheck_sent'])):
        # The rotary-buffer defect (B35) produced all-NaN logits; never store
        # a NaN as a score.
        raise RuntimeError(f'non-finite MiniCheck score on {g["query_id"]}: {s}')
    return s


def validate(mc, cells, n, tol):
    worst = 0.0
    for gen, model, ds in cells:
        gens = {g['query_id']: g for g in load_checkpoint(
            f'generated_{gen}_{model}_{ds}')}
        whole = load_checkpoint(f'perturb_minicheck_{gen}_{model}_{ds}')[:n]
        sent = {r['query_id']: r['orig'] for r in load_checkpoint(
            f'perturb_minicheck_sent_{gen}_{model}_{ds}')}
        for r in whole:
            s = _score(mc, gens[r['query_id']])
            d = max(abs(s['minicheck'] - r['orig']),
                    abs(s['minicheck_sent'] - sent[r['query_id']]))
            worst = max(worst, d)
        print(f'  [validate] {gen}/{model}/{ds}: {len(whole)} probe answers, '
              f'max |diff| so far {worst:.2e}')
    if worst > tol:
        raise SystemExit(f'VALIDATION FAILED: max |diff| {worst:.3e} > {tol}; '
                         'the grid inputs differ from the probe. Not running.')
    print(f'  [validate] PASSED: max |diff| {worst:.2e} <= {tol}')


def run_cell(mc, gen, model, ds, limit=None):
    out = f'minicheck_scores_{gen}_{model}_{ds}'
    if checkpoint_exists(out):
        print(f'  [skip] {out}')
        return
    ck = f'generated_{gen}_{model}_{ds}'
    if not checkpoint_exists(ck):
        raise SystemExit(f'missing {ck} under {config.CHECKPOINT_DIR}')
    gens = load_checkpoint(ck)[:limit] if limit else load_checkpoint(ck)
    part = out + '_partial'
    done = ({r['query_id']: r for r in load_checkpoint(part)}
            if checkpoint_exists(part) else {})
    todo = [g for g in gens if g['query_id'] not in done
            and _answer(g) is not None]
    print(f'  {out}: {len(gens)} answers, {len(done)} resumed, '
          f'{len(todo)} to score', flush=True)
    t0 = time.time()
    for i, g in enumerate(todo, 1):
        s = _score(mc, g)
        done[g['query_id']] = {'query_id': g['query_id'],
                               'minicheck': float(s['minicheck']),
                               'minicheck_sent': float(s['minicheck_sent'])}
        if i % SAVE_EVERY == 0:
            save_checkpoint(part, list(done.values()))
            rate = i / (time.time() - t0) * 60
            print(f'    {i}/{len(todo)}  {rate:.0f} answers/min', flush=True)
    rows = [done[g['query_id']] for g in gens if g['query_id'] in done]
    save_checkpoint(out, rows)
    p = checkpoint_path(part)
    if p.exists():
        p.unlink()
    mins = (time.time() - t0) / 60
    print(f'  {out}: {len(rows)} scored in {mins:.1f} min; mean whole '
          f'{np.mean([r["minicheck"] for r in rows]):.4f}, sentence-min '
          f'{np.mean([r["minicheck_sent"] for r in rows]):.4f}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--generators', default=','.join(GENERATORS))
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--limit', type=int, default=None,
                    help='answers per cell (default: all)')
    ap.add_argument('--validate', type=int, default=0,
                    help='first N probe cases per cell to re-score and compare')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--tol', type=float, default=0.01)
    ap.add_argument('--batch-size', type=int, default=4)
    args = ap.parse_args()

    split = lambda s: [x.strip() for x in s.split(',') if x.strip()]  # noqa: E731
    datasets, gens, models = (split(args.datasets), split(args.generators),
                              split(args.models))
    print(f'  [scope] checkpoints -> {config.CHECKPOINT_DIR}')
    print(f'  [datasets] {datasets}\n  [generators] {gens}\n  [models] {models}')
    cells = [(g, m, d) for d in datasets for g in gens for m in models]
    print(f'  [cells] {len(cells)}', flush=True)

    from minicheck import MiniCheckScorer
    mc = MiniCheckScorer(batch_size=args.batch_size)
    if args.validate:
        validate(mc, cells, args.validate, args.tol)
        if args.validate_only:
            return 0
    for gen, model, ds in cells:
        run_cell(mc, gen, model, ds, limit=args.limit)
    print('[minicheck grid] complete')
    return 0


if __name__ == '__main__':
    sys.exit(main())
