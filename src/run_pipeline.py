"""
run_pipeline.py — experiment orchestrator.

    python src/run_pipeline.py --smoke-test          # N=50, 3 models, NQ only
    python src/run_pipeline.py --full                # N=1000, everything
    python src/run_pipeline.py --full --phases a,b   # selected phases
    python src/run_pipeline.py --smoke-test --models all-mpnet-base-v2

Phases (every phase checkpoints and resumes):
    a       chunk + embed + index + retrieve       (embed_index)
    b       retrieval quality NDCG/Recall/MRR      (retrieval_eval)
    c       GPT-4o-mini generation                 (generate)
    llama   Llama-3 validation subset [GPU]        (generate)
    d       NLI faithfulness                       (faithfulness)
    e       AlignScore faithfulness [GPU]          (faithfulness)
    esa     entailment-similarity alignment        (esa_analysis)
    rerank  Eq. 5 ablation on the worst model      (rerank)
    report  assemble df, robustness, hypotheses, figures
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from utils import set_seed

SMOKE_MODELS = ['all-mpnet-base-v2', 'E5-large-instruct', 'BGE-M3']
ALL_PHASES = ['a', 'b', 'c', 'llama', 'd', 'e', 'esa', 'rerank', 'report']


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
    args = ap.parse_args()

    phases = [p.strip() for p in args.phases.split(',') if p.strip()]
    unknown = set(phases) - set(ALL_PHASES)
    if unknown:
        ap.error(f'unknown phases: {sorted(unknown)}')

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
        run_phase_c(datasets, models)
    if 'llama' in phases:
        from generate import run_phase_llama
        run_phase_llama(datasets)
    if 'd' in phases:
        from faithfulness import run_phase_d
        run_phase_d(datasets, models)
    if 'e' in phases:
        from faithfulness import run_phase_e
        run_phase_e(datasets, models)
    if 'esa' in phases:
        from esa_analysis import run_esa
        run_esa(datasets, models)
    if 'rerank' in phases:
        from rerank import run_rerank_experiment
        from results import assemble_results
        df = assemble_results()
        gpt = df[df['generator'] == 'gpt4o']
        worst = gpt.groupby('model')['nRFG'].mean().idxmax()
        print(f'worst model by nRFG: {worst}')
        run_rerank_experiment(worst, datasets)
    if 'report' in phases:
        from figures import export_all
        from results import (assemble_results, hypothesis_summary,
                             robustness_analysis)
        df = assemble_results(force=True)
        print('\n=== per-model summary (gpt4o) ===')
        gpt = df[df['generator'] == 'gpt4o']
        print(gpt.groupby(['model', 'paradigm'])[
            ['NDCG@5', 'faithfulness', 'RFG', 'nRFG']].mean().round(3))
        robustness_analysis(df)
        print('\n=== hypotheses ===')
        print(hypothesis_summary(df).to_string(index=False))
        export_all(df)

    print('\n=== pipeline done ===')


if __name__ == '__main__':
    main()
