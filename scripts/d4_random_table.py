"""
d4_random_table.py — the falsification probe on a seeded random 200 of the
eligible answers instead of the first 200 (third review, D4).

Measures, fixed before any random-sample score was read:

1. per dataset x generator x scorer, pooled over the four matched systems as
   tab:2full is: mean drop, detection at the 0.5 gate and at the floor-anchored
   gate (p95 of the random-context score, pooled over the four systems), each
   with a Wilson interval, for the first-200 sample and the random sample;
2. the same detection rates on the cases the random sample adds (random minus
   first), the cleanest contrast with the prefix since the two samples share
   most of their cases;
3. on the shared cases, the largest absolute difference between the two runs'
   scores (a reproducibility check: the cases are byte-identical);
4. whether the paper's ordering statements survive: Claude least and
   GPT-4o-mini most detected at 0.5 in every dataset x scorer; AlignScore the
   largest mean drop and claim-min above NLI-max in detection in every cell.

    RAG_CHECKPOINT_DIR=~/rag-backup/checkpoints/n1000_v3 \
        python3 scripts/d4_random_table.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import perturbation_check as pc                                # noqa: E402
from utils import checkpoint_exists, load_checkpoint           # noqa: E402

DATASETS = ['NQ', 'HotpotQA']
GENS = ['claude', 'qwen', 'mistral', 'gpt4omini']
MODELS = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
          'text-embedding-3-small']
SCORERS = ['nli', 'claim', 'align', 'minicheck', 'minicheck_sent']
TAIL = '_rand200'


def _ck(scorer, gen, model, ds, tail=''):
    return (f"perturb{'' if scorer == 'nli' else '_' + scorer}"
            f"_{gen}_{model}_{ds}{tail}")


def load(scorer, gen, ds, tail):
    """Rows per system, tagged with the system so query ids stay unique."""
    rows = []
    for m in MODELS:
        name = _ck(scorer, gen, m, ds, tail)
        if not checkpoint_exists(name):
            return None
        rows += [dict(r, system=m) for r in load_checkpoint(name)]
    return rows


def det(rows, gate):
    o = np.array([r['orig'] for r in rows], float)
    n = np.array([r['number'] for r in rows], float)
    d, k, m = pc._detection(o, n, gate)
    return d, pc._wilson_ci(k, m), m


def stats(rows, floor=None):
    rnd = np.array([r['random'] for r in rows if r['random'] is not None], float)
    floor = float(np.percentile(rnd, 95)) if floor is None else floor
    o = np.array([r['orig'] for r in rows], float)
    n = np.array([r['number'] for r in rows], float)
    return {'n': len(rows), 'drop': float((o - n).mean()),
            'det05': det(rows, pc.GATE), 'detf': det(rows, floor),
            'floor': floor}


def _f(d):
    v, (lo, hi), _ = d
    return f'{v:6.1%} [{lo:.1%}-{hi:.1%}]'


def main():
    out = {}
    worst = 0.0
    for ds in DATASETS:
        for gen in GENS:
            for sc in SCORERS:
                first, rand = load(sc, gen, ds, ''), load(sc, gen, ds, TAIL)
                if first is None or rand is None:
                    print(f'{ds}/{gen}/{sc}: missing checkpoints'); continue
                key = lambda r: (r['system'], r['query_id'])        # noqa: E731
                fk = {key(r): r for r in first}
                shared = [r for r in rand if key(r) in fk]
                added = [r for r in rand if key(r) not in fk]
                diff = max((abs(r[c] - fk[key(r)][c]) for r in shared
                            for c in ('orig', 'number')), default=0.0)
                worst = max(worst, diff)
                sf, sr = stats(first), stats(rand)
                sa = stats(added, floor=sr['floor']) if added else None
                out[(ds, gen, sc)] = {'first': sf, 'rand': sr, 'added': sa,
                                      'n_shared': len(shared), 'maxdiff': diff}
                print(f'{ds:<9}{gen:<10}{sc:<15}'
                      f'first n={sf["n"]:>3} drop {sf["drop"]:.3f} '
                      f'@0.5 {_f(sf["det05"])} @floor {_f(sf["detf"])} | '
                      f'rand drop {sr["drop"]:.3f} @0.5 {_f(sr["det05"])} '
                      f'@floor {_f(sr["detf"])} | added n={len(added):>3}'
                      + (f' @0.5 {_f(sa["det05"])}' if sa else '')
                      + f' | maxdiff {diff:.1e}')
        print()
    print(f'largest score difference on shared cases: {worst:.2e}')
    if not out:
        raise SystemExit('no checkpoints found')

    def get(ds, g, sc, which, q):
        return out[(ds, g, sc)][which][q]
    for which in ('first', 'rand'):
        ext = sum(
            min(GENS, key=lambda g: get(ds, g, sc, which, 'det05')[0]) == 'claude'
            and max(GENS, key=lambda g: get(ds, g, sc, which, 'det05')[0]) == 'gpt4omini'
            for ds in DATASETS for sc in SCORERS)
        al = sum(max(['nli', 'claim', 'align'],
                     key=lambda sc: get(ds, g, sc, which, 'drop')) == 'align'
                 for ds in DATASETS for g in GENS)
        cn = sum(get(ds, g, 'claim', which, 'det05')[0]
                 > get(ds, g, 'nli', which, 'det05')[0]
                 for ds in DATASETS for g in GENS)
        print(f'{which:<6} extremes Claude/GPT {ext}/{len(DATASETS) * len(SCORERS)}  '
              f'AlignScore largest drop {al}/8  claim>NLI det {cn}/8')
    moves = [abs(v['rand']['det05'][0] - v['first']['det05'][0])
             for v in out.values()]
    inside = sum(v['first']['det05'][1][0] <= v['rand']['det05'][0]
                 <= v['first']['det05'][1][1] for v in out.values())
    print(f'|det@0.5 random - first|: max {max(moves):.1%}, median '
          f'{np.median(moves):.1%}; random inside first interval '
          f'{inside}/{len(out)}')
    return out


if __name__ == '__main__':
    main()
