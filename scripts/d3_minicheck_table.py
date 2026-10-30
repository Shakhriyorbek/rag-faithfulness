"""
d3_minicheck_table.py — the falsification probe under Bespoke-MiniCheck-7B
(third review, D3), beside the three published evaluators.

Per dataset x generator, pooled over the four matched systems exactly as
tab:2full is: mean drop, detection at the fixed 0.5 gate, and detection at
the floor-anchored gate (p95 of the random-context score, pooled over the
four systems), each detection rate with a Wilson interval.

The published scorers are printed first so the reader can confirm they
reproduce tab:2full before reading the new rows. Query identifiers are
checked to match across scorers in every cell: the cases must be the same.

    RAG_CHECKPOINT_DIR=~/rag-backup/checkpoints/n1000_v3 \
        python3 scripts/d3_minicheck_table.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import perturbation_check as pc                                # noqa: E402
from utils import checkpoint_exists, load_checkpoint           # noqa: E402

DATASETS = ['NQ', 'HotpotQA']
GENS = ['claude', 'qwen', 'mistral', 'gpt4omini', 'qwen_verbose']
MODELS = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
          'text-embedding-3-small']
SCORERS = ['nli', 'claim', 'align', 'minicheck', 'minicheck_sent']


def _ck(scorer, gen, model, ds):
    return f"perturb{'' if scorer == 'nli' else '_' + scorer}_{gen}_{model}_{ds}"


def cell(scorer, gen, ds):
    rows, ids = [], []
    for m in MODELS:
        name = _ck(scorer, gen, m, ds)
        if not checkpoint_exists(name):
            return None, None
        r = load_checkpoint(name)
        rows += r
        ids.append([x['query_id'] for x in r])
    return rows, ids


def stats(rows):
    o = np.array([r['orig'] for r in rows], float)
    n = np.array([r['number'] for r in rows], float)
    rnd = np.array([r['random'] for r in rows if r['random'] is not None], float)
    floor = float(np.percentile(rnd, 95))
    d05, k05, n05 = pc._detection(o, n, pc.GATE)
    df, kf, nf = pc._detection(o, n, floor)
    ent = [(r['orig'], r['entity']) for r in rows if r['entity'] is not None]
    eo, ee = (np.array(x, float) for x in zip(*ent))
    de, _, _ = pc._detection(eo, ee, pc.GATE)
    return {'n': len(rows), 'orig': o.mean(), 'drop': (o - n).mean(),
            'det05': d05, 'ci05': pc._wilson_ci(k05, n05),
            'floor': floor, 'detf': df, 'cif': pc._wilson_ci(kf, nf),
            'random': rnd.mean(), 'ent_det05': de}


def main():
    out = {}
    for ds in DATASETS:
        for gen in GENS:
            ref_ids = None
            for sc in SCORERS:
                rows, ids = cell(sc, gen, ds)
                if rows is None:
                    continue
                if ref_ids is None:
                    ref_ids = ids
                elif ids != ref_ids:
                    raise SystemExit(f'case mismatch: {ds}/{gen}/{sc}')
                out[(ds, gen, sc)] = s = stats(rows)
                print(f'{ds:<9}{gen:<13}{sc:<15}n={s["n"]:>4}  '
                      f'orig {s["orig"]:.3f}  drop {s["drop"]:.3f}  '
                      f'@0.5 {s["det05"]:6.1%} [{s["ci05"][0]:.1%}-{s["ci05"][1]:.1%}]  '
                      f'floor {s["floor"]:.3f} rnd {s["random"]:.3f}  '
                      f'@floor {s["detf"]:6.1%} [{s["cif"][0]:.1%}-{s["cif"][1]:.1%}]  '
                      f'ent@0.5 {s["ent_det05"]:6.1%}')
            print()
    return out

if __name__ == '__main__':
    main()
