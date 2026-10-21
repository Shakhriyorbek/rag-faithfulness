#!/usr/bin/env python3
"""Does the attribution preamble cause the low falsification response?

Section 5.4 found, observationally, that answers opening with "Based on /
According to the provided context," respond far less to a falsified value
(single-assertion NQ, NLI-max: Qwen +0.457 without it, +0.129 with it). This
is the direct test: rebuild the probe's own cases (same SEED, same limit, so
byte-identical to the perturb_* checkpoints), take every case whose answer
opens with the clause, and score the original and the falsified answer twice,
as written and with the clause stripped from both.

The as-written scores must reproduce the checkpoint's `orig` / `number` to
within batch-padding noise; the script reports the largest difference and
stops if it is not small, because then the cases are not the same cases.

Read-only over generations; writes preamble_strip_{scorer}.pkl. Run from the
repo root (or with src/ importable):
    python3 scripts/preamble_strip_rescore.py --scorer nli
    PYTHONPATH=~/align_env python3 scripts/preamble_strip_rescore.py --scorer align
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from utils import set_scope  # noqa: E402

set_scope(1000)
import config  # noqa: E402
import perturbation_check as P  # noqa: E402
from abstention import _PREAMBLE_RE  # noqa: E402
from claim_faithfulness import split_claims  # noqa: E402
from textnorm import strip_markdown  # noqa: E402
from utils import load_checkpoint, save_checkpoint  # noqa: E402

M4 = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct', 'text-embedding-3-small']
GENS = ['claude', 'qwen', 'mistral', 'gpt4omini', 'qwen_verbose']
PREFIX = {'nli': '', 'claim': 'claim_', 'align': 'align_'}
LIMIT = 200          # the probe's --limit; must match or the cases differ


def strip(a):
    return _PREAMBLE_RE.sub('', a, count=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scorer', choices=sorted(PREFIX), required=True)
    ap.add_argument('--datasets', default='NQ,HotpotQA')
    args = ap.parse_args()
    print(f'scorer={args.scorer}  generators={GENS}  systems={M4}')
    scorer = P._Scorer(args.scorer)
    rows, worst = [], 0.0
    for ds in args.datasets.split(','):
        for g in GENS:
            for m in M4:
                gens = load_checkpoint(f'generated_{g}_{m}_{ds}')
                ref = load_checkpoint(f'perturb_{PREFIX[args.scorer]}{g}_{m}_{ds}')
                if not gens or not ref:
                    print(f'  [skip] {g}/{m}/{ds}: missing checkpoint')
                    continue
                ref = {r['query_id']: r for r in ref}
                cases = [c for c in P.build_cases(gens, LIMIT)
                         if _PREAMBLE_RE.match(c['answer'])]
                for c in cases:
                    r = ref.get(c['query_id'])
                    if r is None or r.get('number') is None:
                        continue
                    o = scorer.score(c['chunks'], c['answer'])
                    n = scorer.score(c['chunks'], c['num_answer'])
                    worst = max(worst, abs(o - float(r['orig'])),
                                abs(n - float(r['number'])))
                    so = scorer.score(c['chunks'], strip(c['answer']))
                    sn = scorer.score(c['chunks'], strip(c['num_answer']))
                    rows.append(dict(
                        dataset=ds, generator=g, model=m, query_id=c['query_id'],
                        n_claims=len(split_claims(strip_markdown(c['answer']))),
                        orig=o, number=n, orig_stripped=so, number_stripped=sn))
                print(f'  {g:13s} {m:24s} {ds:9s} {len(cases):4d} preamble cases  '
                      f'max |as-written - checkpoint| so far {worst:.2e}')
    if worst > 1e-3:
        raise SystemExit(f'as-written scores do not reproduce the checkpoint '
                         f'(max diff {worst:.4f}): these are not the probe cases')
    save_checkpoint(f'preamble_strip_{args.scorer}', rows)
    print(f'\n{len(rows)} cases, reproduction max diff {worst:.2e}')


if __name__ == '__main__':
    main()
