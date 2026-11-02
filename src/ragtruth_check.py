"""
ragtruth_check.py — the probe's evaluators on natural hallucinations
(third review, D5).

Every detection rate in the paper comes from a constructed error: one value
replaced in an otherwise untouched answer. The review asked whether the
pattern holds for errors a generator makes on its own. RAGTruth (Niu et al.
2024, github.com/ParticleMedia/RAGTruth, MIT licence) supplies those: 5,934
answers to MS MARCO questions from six generators, each grounded on three
passages, with hallucinated spans annotated by people.

Design, fixed before any scoring:

- population: the QA task, quality "good", attempted answers only. A refusal
  is our canonical rule (abstention.is_abstention) or an answer opening with
  RAGTruth's own refusal sentence, "Unable to answer based on given
  passages", after the same preamble strip. Refusals are excluded for the
  reason they are excluded from the probe: a refusal asserts nothing to check.
- label: hallucinated = at least one annotated span, of any type.
- premises: the three passages are the chunks, exactly as the five retrieved
  chunks are in the main grid (NLI-max scores each and their concatenation).
- scorers: the five the probe uses, through perturbation_check._Scorer, so the
  markdown strip and every model call are the ones behind the paper's tables.

The checkpoint holds one row per response with its score(s); the measures are
computed by scripts/ragtruth_table.py.

    RAGTRUTH_DIR=~/rag_faithfulness/data/ragtruth \
    RAG_CHECKPOINT_DIR=~/rag_faithfulness/checkpoints/n1000_v3 \
        python3 src/ragtruth_check.py --scorer nli
"""
import argparse
import json
import os
import re
from pathlib import Path

import abstention
import config
import textnorm
from utils import checkpoint_exists, load_checkpoint, save_checkpoint

# sha256 of the two files as downloaded on 2026-11-02 from the repository's
# main branch (last push 2024-12-02). A different download is refused, so the
# population cannot change silently.
SHA256 = {
    'response.jsonl':
        'e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073',
    'source_info.jsonl':
        '0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b',
}
RAGTRUTH_REFUSAL = 'Unable to answer based on given passages'


def _dir() -> Path:
    return Path(os.getenv('RAGTRUTH_DIR',
                          str(config.BASE_DIR / 'data' / 'ragtruth')))


def _verify(d: Path):
    import hashlib
    for name, want in SHA256.items():
        got = hashlib.sha256((d / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f'{d / name}: sha256 {got} != pinned {want}')


def is_refusal(text: str) -> bool:
    if abstention.is_abstention(text):
        return True
    stripped = abstention._PREAMBLE_RE.sub('', text or '')
    return textnorm.squad_normalize(stripped).startswith(
        textnorm.squad_normalize(RAGTRUTH_REFUSAL))


def split_passages(passages: str):
    parts = re.split(r'passage \d+:', passages)
    return [p.strip() for p in parts if p.strip()]


def load_rows():
    """The fixed population, with stats on what was dropped and why."""
    d = _dir()
    _verify(d)
    src = {s['source_id']: s
           for s in map(json.loads, open(d / 'source_info.jsonl'))
           if s['task_type'] == 'QA'}
    rows, dropped = [], {'quality': 0, 'refusal': 0}
    for r in map(json.loads, open(d / 'response.jsonl')):
        s = src.get(r['source_id'])
        if s is None:
            continue
        if r['quality'] != 'good':
            dropped['quality'] += 1
            continue
        if is_refusal(r['response']):
            dropped['refusal'] += 1
            continue
        chunks = split_passages(s['source_info']['passages'])
        assert len(chunks) == 3, r['id']
        spans = [(l['start'], l['end'], l['label_type']) for l in r['labels']]
        rows.append({
            'id': r['id'], 'source_id': r['source_id'], 'model': r['model'],
            'split': r['split'], 'response': r['response'], 'chunks': chunks,
            'hallucinated': bool(spans), 'spans': spans,
        })
    print(f'  [ragtruth] {len(rows)} attempted QA responses '
          f'(dropped: {dropped["quality"]} quality, {dropped["refusal"]} refusals); '
          f'{sum(r["hallucinated"] for r in rows)} hallucinated')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--scorer', required=True,
                    choices=['nli', 'claim', 'align', 'minicheck'])
    ap.add_argument('--limit', type=int, default=None,
                    help='score only the first N responses (smoke tests)')
    args = ap.parse_args()
    print(f'  [scope] checkpoints -> {config.CHECKPOINT_DIR}')

    from perturbation_check import _Scorer
    rows = load_rows()[:args.limit]
    kinds = (['minicheck', 'minicheck_sent'] if args.scorer == 'minicheck'
             else [args.scorer])
    suffix = '' if args.limit is None else f'_smoke{args.limit}'
    names = {k: f'ragtruth_{k}{suffix}' for k in kinds}
    done = {k: (load_checkpoint(n) if checkpoint_exists(n) else {})
            for k, n in names.items()}
    todo = [r for r in rows if any(r['id'] not in done[k] for k in kinds)]
    print(f'  [ragtruth/{args.scorer}] {len(rows) - len(todo)} done, '
          f'{len(todo)} to score')
    if not todo:
        return 0

    scorer = _Scorer(args.scorer)
    for i, r in enumerate(todo, 1):
        v = scorer.score(r['chunks'], r['response'])
        v = v if isinstance(v, dict) else {args.scorer: v}
        for k in kinds:
            done[k][r['id']] = float(v[k])
        if i % 250 == 0 or i == len(todo):
            for k in kinds:
                save_checkpoint(names[k], done[k])
            print(f'    {i}/{len(todo)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
