"""
d6_human_agreement.py — agreement between the LLM judge and human labels
(third review, D6).

Measures, fixed before any label was read:

1. raw agreement and Cohen's kappa over the three labels, with a Wilson
   interval on agreement;
2. the same for the binary correct / not-correct decision, which is what every
   accuracy figure in the paper uses, and for the refusal flag, which is what
   faith_gap and the 2x2 take from the judge;
3. the 3x3 confusion matrix (rows human, columns judge);
4. agreement per generator and per dataset, 25 and 100 items respectively, so
   a generator the judge grades worse is visible;
5. with a second annotator's sheet (--second), inter-annotator agreement and
   kappa, which bounds how well any judge could agree with one person.

Every disagreement is printed with its item id so it can be read.

    RAG_CHECKPOINT_DIR=~/rag-backup/checkpoints/n1000_v3 \
        python3 scripts/d6_human_agreement.py [--second path/to/sheet2.csv]
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from perturbation_check import _wilson_ci                       # noqa: E402
from utils import load_checkpoint                               # noqa: E402

LABELS = ['CORRECT', 'INCORRECT', 'REFUSAL']
SHEET = Path(__file__).resolve().parents[1] / 'reports' / 'd6' / 'annotation_sheet.csv'


def read_sheet(path):
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            lab = (row['label'] or '').strip().upper()
            if lab:
                if lab not in LABELS:
                    raise SystemExit(f'{path}: {row["item"]} has label {lab!r}')
                out[row['item']] = lab
    return out


def kappa(a, b, labels):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / n ** 2
    return po, (po - pe) / (1 - pe) if pe < 1 else float('nan')


def report(name, a, b, labels):
    k = sum(x == y for x, y in zip(a, b))
    po, kap = kappa(a, b, labels)
    lo, hi = _wilson_ci(k, len(a))
    print(f'  {name:<28} n={len(a):>3}  agree {po:6.1%} [{lo:.1%}-{hi:.1%}]  '
          f'kappa {kap:+.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sheet', default=str(SHEET))
    ap.add_argument('--second', default=None)
    args = ap.parse_args()
    key = load_checkpoint('d6_key')
    human = read_sheet(args.sheet)
    items = sorted(i for i in human if i in key)
    print(f'labelled {len(items)} of {len(key)} items')
    if not items:
        return
    h = [human[i] for i in items]
    j = [key[i]['judge'] for i in items]
    report('three labels', h, j, LABELS)
    report('correct vs not', [x == 'CORRECT' for x in h],
           [x == 'CORRECT' for x in j], [True, False])
    report('refusal vs not', [x == 'REFUSAL' for x in h],
           [x == 'REFUSAL' for x in j], [True, False])

    print('\n  confusion (rows human, columns judge)')
    print('  ' + ' ' * 11 + ''.join(f'{l:>11}' for l in LABELS))
    for lh in LABELS:
        print(f'  {lh:<11}' + ''.join(
            f'{sum(1 for x, y in zip(h, j) if x == lh and y == lj):>11}'
            for lj in LABELS))
    print(f'  human correct rate {h.count("CORRECT") / len(h):.1%}, '
          f'judge {j.count("CORRECT") / len(j):.1%}')

    print()
    for field in ('generator', 'dataset'):
        for v in sorted({key[i][field] for i in items}):
            sub = [i for i in items if key[i][field] == v]
            report(f'{field} {v}', [human[i] for i in sub],
                   [key[i]['judge'] for i in sub], LABELS)

    print('\n  disagreements')
    for i in items:
        if human[i] != key[i]['judge']:
            k = key[i]
            print(f'    {i} {k["dataset"]}/{k["generator"]}/{k["query_id"]}: '
                  f'human {human[i]}, judge {k["judge"]}')

    if args.second:
        other = read_sheet(args.second)
        both = [i for i in items if i in other]
        print(f'\n  second annotator: {len(both)} items in common')
        if both:
            report('annotator vs annotator', [human[i] for i in both],
                   [other[i] for i in both], LABELS)
            report('judge vs annotator 2', [key[i]['judge'] for i in both],
                   [other[i] for i in both], LABELS)


if __name__ == '__main__':
    main()
