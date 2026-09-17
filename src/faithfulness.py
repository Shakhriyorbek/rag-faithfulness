"""
faithfulness.py — Phases D and E: faithfulness scoring.

DESIGN INVARIANT (paper §3.1, supervisor point 5): faithfulness is measured
between the retrieved context and the GENERATED answer — never the gold
answer. Gold answers appear only in retrieval qrels and in the ESA analysis.

Phase D: DeBERTa-v3-large NLI via src/nli.py. Per-chunk scoring with
         max/mean/concat aggregates (audit B5: the notebook truncated the
         concatenated context at 512 tokens, silently ignoring chunks 3-5).
Phase E: AlignScore (optional dependency; install from the AlignScore
         GitHub repo — it is not on PyPI).

Both phases score whichever generator checkpoints exist
(generated_claude_* and generated_llama3_*).
"""
from pathlib import Path
from typing import Dict, List

import numpy as np

import textnorm
import config
from nli import NLIScorer
from utils import checkpoint_exists, load_checkpoint, save_checkpoint

GENERATORS = ['claude', 'gpt4omini', 'llama3', config.OPEN_MODEL_LABEL]


def _generation_checkpoints(model: str, ds_name: str, generators=None):
    for gen in (generators or GENERATORS):
        ck = f'generated_{gen}_{model}_{ds_name}'
        if checkpoint_exists(ck):
            yield gen, ck


def run_phase_d(datasets: Dict, model_names: List[str] = None,
                generators=None):
    """NLI faithfulness for every generated answer."""
    nli = NLIScorer()
    model_list = [c['name'] for c in config.EMBEDDING_MODELS
                  if not model_names or c['name'] in model_names]

    for model in model_list:
        for ds_name in datasets:
            for gen, ck_gen in _generation_checkpoints(model, ds_name,
                                                       generators):
                ck_out = f'nli_scores_{gen}_{model}_{ds_name}'
                if checkpoint_exists(ck_out):
                    print(f'  [skip] {ck_out}')
                    continue
                generations = load_checkpoint(ck_gen)
                print(f'  NLI scoring [{gen}] [{model}] [{ds_name}] '
                      f'({len(generations)} answers)...')
                scores = []
                for g in generations:
                    answer = g['generated_answer']
                    if answer.startswith('[ERROR'):
                        continue
                    chunk_scores = nli.score_chunks(
                        g['retrieved_texts'][:config.TOP_K], answer)
                    scores.append({
                        'query_id': g['query_id'],
                        'nli_max': chunk_scores['nli_max'],
                        'nli_mean': chunk_scores['nli_mean'],
                        'nli_concat': chunk_scores['nli_concat'],
                    })
                save_checkpoint(ck_out, scores)
                mean_max = np.mean([s['nli_max'] for s in scores])
                print(f'  {model}/{ds_name}/{gen}: mean nli_max = {mean_max:.4f}')
    print('[phase D] complete')


def run_phase_d_conditions(patterns=('oracle_*', 'ctx_*', 'util_*')):
    """
    NLI-score the controlled conditions, not just the embedder grid.

    The oracle ceiling and the context ablations produce answers against a
    context, so faithfulness is defined for them and the comparison against the
    RAG rows is only meaningful if it is measured the same way. Phase D proper
    globs `generated_*` and would skip all of these.

    C1 is deliberately absent: it has no context, so faithfulness is undefined
    there rather than zero.
    """
    from pathlib import Path
    nli = NLIScorer()
    names = []
    for pattern in patterns:
        for p in sorted(Path(config.CHECKPOINT_DIR).glob(f'{pattern}.pkl')):
            if p.stem.endswith('_partial') or p.stem.endswith('_scored'):
                continue
            names.append(p.stem)

    for name in sorted(set(names)):
        ck_out = f'nli_scores_{name}'
        if checkpoint_exists(ck_out):
            print(f'  [skip] {ck_out}')
            continue
        records = load_checkpoint(name)
        if not records:
            continue
        scores = []
        for r in records:
            answer = r.get('generated_answer')
            contexts = r.get('retrieved_texts') or []
            # None answers (oracle_missing), API errors and empty contexts are
            # skipped rather than scored as 0.0 — same reasoning as the
            # ungradable rule in correctness.py.
            if not answer or answer.startswith('[ERROR') or not contexts:
                continue
            cs = nli.score_chunks(contexts[:config.TOP_K], answer)
            scores.append({
                'query_id': r['query_id'],
                'nli_max': cs['nli_max'],
                'nli_mean': cs['nli_mean'],
                'nli_concat': cs['nli_concat'],
            })
        save_checkpoint(ck_out, scores)
        mean_max = np.mean([s['nli_max'] for s in scores]) if scores else float('nan')
        print(f'  {name}: {len(scores)}/{len(records)} scored, '
              f'mean nli_max = {mean_max:.4f}')
    print('[phase D-conditions] complete')


def run_phase_e(datasets: Dict, model_names: List[str] = None,
                batch_size: int = 32, generators=None):
    """AlignScore faithfulness (GPU strongly recommended)."""
    try:
        from alignscore import AlignScore
    except ImportError:
        print('[phase E] alignscore not installed — install from '
              'https://github.com/yuh-zha/AlignScore, skipping')
        return

    # ckpt_path is REQUIRED by AlignScore's constructor and was missing here;
    # the omission could not surface until the package was actually installed.
    ckpt = Path(config.ALIGNSCORE_CKPT)
    if not ckpt.exists():
        print(f'[phase E] AlignScore checkpoint not found at {ckpt} — '
              f'download AlignScore-large.ckpt from huggingface.co/yzha/'
              f'AlignScore, or set ALIGNSCORE_CKPT. Skipping.')
        return

    scorer = AlignScore(model='roberta-large', batch_size=batch_size,
                        device=config.DEVICE, ckpt_path=str(ckpt),
                        evaluation_mode='nli_sp')
    model_list = [c['name'] for c in config.EMBEDDING_MODELS
                  if not model_names or c['name'] in model_names]

    for model in model_list:
        for ds_name in datasets:
            for gen, ck_gen in _generation_checkpoints(model, ds_name,
                                                       generators):
                ck_out = f'align_scores_{gen}_{model}_{ds_name}'
                if checkpoint_exists(ck_out):
                    print(f'  [skip] {ck_out}')
                    continue
                generations = load_checkpoint(ck_gen)
                pairs = [(g, ' '.join(g['retrieved_texts'][:config.TOP_K]))
                         for g in generations
                         if not g['generated_answer'].startswith('[ERROR')]
                print(f'  AlignScore [{gen}] [{model}] [{ds_name}]...')
                # Same hypothesis as the NLI arms — see textnorm.strip_markdown.
                # The paper states all three evaluators score identical inputs;
                # before 2026-09-03 that was only true of the answer text as
                # the generator wrote it, not as each evaluator read it.
                raw = scorer.score(
                    contexts=[ctx for _, ctx in pairs],
                    claims=[textnorm.strip_markdown(g['generated_answer'])
                            for g, _ in pairs])
                scores = [
                    {'query_id': g['query_id'], 'align_score': float(s)}
                    for (g, _), s in zip(pairs, raw)
                ]
                save_checkpoint(ck_out, scores)
                print(f'  {model}/{ds_name}/{gen}: '
                      f'mean AlignScore = {np.mean(raw):.4f}')
    print('[phase E] complete')


def main():
    """
    Entry point for phases D and E.

    Phase E needs AlignScore, which lives in an isolated install because it
    pins transformers 4.26 against the 5.x everything else uses:

        PYTHONPATH=$HOME/align_env python3 src/faithfulness.py --phase e

    See RUNBOOK for the install recipe.
    """
    import argparse
    ap = argparse.ArgumentParser(description='NLI / AlignScore faithfulness')
    ap.add_argument('--phase', default='e', choices=['d', 'e'],
                    help='d = DeBERTa NLI, e = AlignScore')
    ap.add_argument('--datasets', default=','.join(config.DATASETS))
    ap.add_argument('--generators', default=','.join(GENERATORS),
                    help='comma-separated generator labels. Default is the main grid. A non-default arm (e.g. a prompt-variant control) has its own label and is SKIPPED unless named here.')
    ap.add_argument('--models', default=None,
                    help='comma-separated; default = every configured model')
    ap.add_argument('--scope-n', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()

    if args.scope_n:
        from utils import set_scope
        set_scope(args.scope_n)
    print(f'  [scope] checkpoints -> {config.CHECKPOINT_DIR}')

    datasets = {d.strip(): None for d in args.datasets.split(',')}
    models = [m.strip() for m in args.models.split(',')] if args.models else None
    gens = [g.strip() for g in args.generators.split(',') if g.strip()]
    print(f'  [generators] {gens}')

    if args.phase == 'd':
        run_phase_d(datasets, models, generators=gens)
    else:
        run_phase_e(datasets, models, batch_size=args.batch_size,
                    generators=gens)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
