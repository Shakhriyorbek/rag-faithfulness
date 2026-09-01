"""
run_pipeline.py — experiment orchestrator.

    python src/run_pipeline.py --smoke-test          # N=50, 3 models, NQ only
    python src/run_pipeline.py --full                # N=1000, everything
    python src/run_pipeline.py --full --phases a,b   # selected phases
    python src/run_pipeline.py --smoke-test --models all-mpnet-base-v2

Phases (every phase checkpoints and resumes):
    a        chunk + embed + index + retrieve       (embed_index)
    b        retrieval quality NDCG/Recall/MRR      (retrieval_eval)
    c        Claude generation                      (generate)      [$]
    cgpt     GPT-4o-mini generation                 (generate)      [$]
    llama    Llama-3 validation subset [GPU]        (generate)
    d        NLI faithfulness                       (faithfulness)
    e        AlignScore faithfulness [GPU]          (faithfulness)
    esa      entailment-similarity alignment        (esa_analysis)
    rerank   Eq. 5 ablation on the worst model      (rerank)         [$]
    report   assemble df, robustness, hypotheses, figures

Berend 2026-08-11 additions:
    correct  answer correctness (EM / token-F1)     (correctness)    free
    c1       no-retrieval parametric floor          (conditions)     [$]
    c2       oracle ceiling                         (conditions)     [$]
    dcond    NLI-score the controlled conditions    (faithfulness)
    cond     necessary/sufficient grid, anchors,
             faithfulness conditioned on correctness (conditional)   free

Phases marked [$] spend money and are skipped unless --yes is passed.

Two further experiments stay as their own CLIs, because both are subsample
studies whose scope needs choosing per run rather than inheriting the grid:
    python src/context_ablation.py --help     subset/order of gold snippets
    python src/doc_utility.py --help          eRAG / leave-one-out / Shapley
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from utils import set_scope, set_seed

SMOKE_MODELS = ['all-mpnet-base-v2', 'E5-large-instruct', 'BGE-M3']
ALL_PHASES = ['a', 'b', 'c', 'cgpt', 'llama', 'd', 'e', 'esa', 'rerank',
              'report', 'correct', 'c1', 'c2', 'dcond', 'cond']
# Phases that call a paid API. Gated behind --yes so a phase list typed from
# memory cannot start spending.
PAID_PHASES = {'c', 'cgpt', 'c1', 'c2', 'rerank'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke-test', action='store_true',
                      help='N=50, 3 core models, NQ only')
    mode.add_argument('--full', action='store_true',
                      help='N=1000, all 7 models, all 3 datasets')
    ap.add_argument('--phases', default='a,b,c,d,report',
                    help=f'comma-separated subset of {ALL_PHASES} '
                         '(default: a,b,c,d,report — the CPU/API path)')
    ap.add_argument('--models', default=None,
                    help='comma-separated model names (overrides mode default)')
    ap.add_argument('--datasets', default=None,
                    help='comma-separated dataset names (overrides mode default)')
    ap.add_argument('--n-queries', type=int, default=None,
                    help='override query count')
    ap.add_argument('--filtered', action='store_true',
                    help='phase `cond`: restrict to retrieval-necessary '
                         'queries (needs conditions.py --emit-filter)')
    ap.add_argument('--correct-source', default='judge',
                    help='phase `cond`: where `correct` comes from — judge, '
                         'contains, em or f1. "judge" falls back to '
                         'containment per row and prints the coverage')
    ap.add_argument('--legacy-rfg', action='store_true',
                    help='also compute the RETIRED gap metric (RFG/nRFG, '
                         'paper Section III-C; src/legacy/rfg.py). Off by '
                         'default: the tables are produced without it.')
    ap.add_argument('--yes', action='store_true',
                    help=f'authorize the paid phases {sorted(PAID_PHASES)}')
    args = ap.parse_args()

    phases = [p.strip() for p in args.phases.split(',') if p.strip()]
    unknown = set(phases) - set(ALL_PHASES)
    if unknown:
        ap.error(f'unknown phases: {sorted(unknown)}')

    paid = sorted(set(phases) & PAID_PHASES)
    if paid and not args.yes:
        ap.error(f'phases {paid} spend money — re-run with --yes, or drop them')

    if args.smoke_test:
        n = args.n_queries or 50
        models = (args.models.split(',') if args.models else SMOKE_MODELS)
        ds_names = (args.datasets.split(',') if args.datasets else ['NQ'])
    else:
        n = args.n_queries or config.N_QUERIES
        models = args.models.split(',') if args.models else None  # None = all
        ds_names = (args.datasets.split(',') if args.datasets
                    else config.DATASETS)

    set_seed()
    # Scope the checkpoint directory to (N, CORPUS_VERSION) BEFORE any phase
    # runs, so a smoke test and a full run cannot reuse each other's work.
    set_scope(n)
    print(f'=== pipeline: N={n} | models={models or "ALL"} | '
          f'datasets={ds_names} | phases={phases} ===')

    from datasets_loader import load_all
    datasets = load_all(n, ds_names)

    if 'a' in phases:
        from embed_index import run_phase_a
        run_phase_a(datasets, models)
    if 'b' in phases:
        from retrieval_eval import run_phase_b
        results = run_phase_b(datasets, models)
        # Regression guard for audit B3: with provenance-based qrels a
        # working retriever must land above zero.
        for model, per_ds in results.items():
            for ds_name, m in per_ds.items():
                assert m['NDCG@5'] > 0, (
                    f'NDCG@5 == 0 for {model}/{ds_name} — qrels/ID mismatch '
                    f'(this is the failure mode audit item B3 guards against)')
    if 'c' in phases:
        from generate import run_phase_c
        # On a smoke test, extrapolate measured cost to the full grid
        # (7 models x 3 datasets x N_QUERIES) before committing to it.
        project_to = (len(config.EMBEDDING_MODELS) * len(config.DATASETS)
                      * config.N_QUERIES) if args.smoke_test else None
        run_phase_c(datasets, models, project_to=project_to)
    if 'cgpt' in phases:
        from generate import run_phase_c_openai
        project_to = (len(config.EMBEDDING_MODELS) * len(config.DATASETS)
                      * config.N_QUERIES) if args.smoke_test else None
        run_phase_c_openai(datasets, models, project_to=project_to)
    if 'llama' in phases:
        from generate import run_phase_llama
        run_phase_llama(datasets)
    if 'd' in phases:
        from faithfulness import run_phase_d
        run_phase_d(datasets, models)
    if 'e' in phases:
        from faithfulness import run_phase_e
        run_phase_e(datasets, models)
    # ── Berend 2026-08-11 additions ──
    if 'c1' in phases:
        from conditions import run_condition_c1
        run_condition_c1(datasets, yes=args.yes)
    if 'c2' in phases:
        from conditions import run_condition_c2
        run_condition_c2(datasets, yes=args.yes)
    if 'dcond' in phases:
        from faithfulness import run_phase_d_conditions
        run_phase_d_conditions()
    if 'correct' in phases:
        from correctness import run_phase_correctness
        print('=== phase: correctness scoring (no API cost) ===')
        run_phase_correctness()
    if 'cond' in phases:
        from conditional import build_query_frame, report_generator
        qdf = build_query_frame(datasets, models, filtered=args.filtered,
                                correct_source=args.correct_source)
        print(f'\n=== query frame: {len(qdf):,} rows ===')
        for gen in sorted(qdf['generator'].unique()):
            report_generator(qdf, gen)
    if 'esa' in phases:
        from esa_analysis import run_esa
        run_esa(datasets, models)
    if 'rerank' in phases:
        from rerank import run_rerank_experiment
        from results import assemble_results
        df = assemble_results(legacy_rfg=True)   # picks the model to re-rank
        claude_rows = df[df['generator'] == 'claude']
        worst = claude_rows.groupby('model')['nRFG'].mean().idxmax()
        print(f'worst model by nRFG: {worst}')
        run_rerank_experiment(worst, datasets)
    if 'report' in phases:
        from figures import export_all
        from results import (assemble_results, hypothesis_summary,
                             robustness_analysis)
        df = assemble_results(force=True, legacy_rfg=args.legacy_rfg)
        print('\n=== per-model summary (claude) ===')
        claude_rows = df[df['generator'] == 'claude']
        cols = [c for c in ('NDCG@5', 'faithfulness', 'nli_max', 'align_score',
                            'RFG', 'nRFG') if c in claude_rows.columns]
        print(claude_rows.groupby(['model', 'paradigm'])[cols].mean().round(3))
        robustness_analysis(df)
        print('\n=== hypotheses ===')
        print(hypothesis_summary(df).to_string(index=False))
        export_all(df)

    print('\n=== pipeline done ===')


if __name__ == '__main__':
    main()
