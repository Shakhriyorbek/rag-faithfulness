"""
d6_human_sample.py — draw the blinded sample for checking the LLM judge
against human annotation (third review, D6).

Design, fixed before any human label is collected:

- population: the 32 judged cells (4 generators x 4 matched systems x 2
  datasets), rows the judge graded (ungradable rows excluded, as they are from
  every correctness figure);
- sample: 25 rows per dataset x generator cell, 200 in all, drawn with
  random.Random(42) after pooling the four systems, so the sample is balanced
  across the eight cells the paper reports and not dominated by any one;
- the annotator sees the question, the reference answer(s) and the answer,
  exactly the three fields the judge's prompt contains, in a shuffled order and
  under an opaque id. Not shown: the judge's verdict, the generator, the
  system, the retrieved context and the containment heuristic;
- labels are the judge's three: CORRECT, INCORRECT, REFUSAL, under the same
  definitions (reports/d6_annotation_guide.md quotes the judge prompt).

Writes the annotation sheet to reports/d6/annotation_sheet.csv and the
unblinding key to the checkpoint directory as d6_key, where the annotator does
not need to look. scripts/d6_human_agreement.py reads both back.

    RAG_CHECKPOINT_DIR=~/rag-backup/checkpoints/n1000_v3 \
        python3 scripts/d6_human_sample.py
"""
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import correctness                                             # noqa: E402
from utils import checkpoint_exists, load_checkpoint, save_checkpoint  # noqa: E402

DATASETS = ['NQ', 'HotpotQA']
GENS = ['claude', 'qwen', 'mistral', 'gpt4omini']
MODELS = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
          'text-embedding-3-small']
PER_CELL = 25
SEED = 42
OUT = Path(__file__).resolve().parents[1] / 'reports' / 'd6'


def main():
    rng = random.Random(SEED)
    picked = []
    for ds in DATASETS:
        for gen in GENS:
            pool = []
            for m in MODELS:
                name = f'generated_{gen}_{m}_{ds}'
                jname = f'{name}_judged_claude'
                if not (checkpoint_exists(name) and checkpoint_exists(jname)):
                    raise SystemExit(f'missing {name} or {jname}')
                J = {j['query_id']: j for j in load_checkpoint(jname)}
                for r in load_checkpoint(name):
                    j = J.get(r['query_id'])
                    if j is None or j['correct'] is None or correctness.is_ungradable(r):
                        continue
                    verdict = ('REFUSAL' if j['abstained']
                               else 'CORRECT' if j['correct'] else 'INCORRECT')
                    pool.append({'dataset': ds, 'generator': gen, 'system': m,
                                 'query_id': r['query_id'],
                                 'question': r['question'],
                                 'reference': ' | '.join(
                                     g for g in correctness._gold_answers(r) if g),
                                 'answer': (r['generated_answer'] or '').strip(),
                                 'judge': verdict})
            pool.sort(key=lambda x: (x['system'], x['query_id']))
            take = rng.sample(pool, PER_CELL)
            print(f'{ds:<9}{gen:<10} pool {len(pool):>5}  sampled {len(take)}  '
                  f'judge: ' + ', '.join(
                      f'{v} {sum(t["judge"] == v for t in take)}'
                      for v in ('CORRECT', 'INCORRECT', 'REFUSAL')))
            picked += take
    rng.shuffle(picked)
    for i, p in enumerate(picked, 1):
        p['item'] = f'D6-{i:03d}'

    OUT.mkdir(parents=True, exist_ok=True)
    sheet = OUT / 'annotation_sheet.csv'
    if sheet.exists():
        raise SystemExit(f'{sheet} exists; refusing to overwrite annotations')
    with open(sheet, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['item', 'question', 'reference', 'answer', 'label', 'note'])
        for p in picked:
            w.writerow([p['item'], p['question'], p['reference'], p['answer'],
                        '', ''])
    save_checkpoint('d6_key', {p['item']: {k: p[k] for k in
                    ('dataset', 'generator', 'system', 'query_id', 'judge')}
                    for p in picked})
    print(f'wrote {sheet} ({len(picked)} items) and checkpoint d6_key')


if __name__ == '__main__':
    main()
