"""
perturbation_check.py — Is NLI entailment sensitive to numeric infidelity?

MOTIVATION (document-QA / invoice case)
    A grounded RAG answer says "the Project Alpha budget is $1,000" because
    $1,000 is in the retrieved invoice. The failure a document-QA product
    fears most is the answer that says $1,500 instead — right shape, right
    entity, wrong value. src/faithfulness.py scores such an answer with
    DeBERTa NLI entailment, and NLI models are widely reported to be weak at
    numeric contradiction. If NLI cannot separate $1,000 from $1,500, then
    `nli_max` does not measure the property the product needs, and a literal
    value-grounding check is required alongside it.

DESIGN
    Paired, on answers the pipeline already generated. For each eligible
    answer four scores are computed against the SAME retrieved chunks:

      orig    the untouched answer                      -> baseline
      number  one grounded number replaced              -> the treatment
      entity  one grounded entity replaced              -> comparison substitution
      random  orig answer vs ANOTHER query's chunks     -> floor of the scale

    `entity` matters: without it a null on `number` is uninterpretable, since
    it could mean NLI is insensitive to every substitution, or that these
    answers are scored high for reasons unrelated to their content. `random`
    anchors what "not entailed" looks like on this data.

ELIGIBILITY (both directions are load-bearing)
    - the original number MUST appear in the retrieved context, or the answer
      was never grounded in it and perturbing it tests nothing;
    - the replacement MUST NOT appear in the retrieved context, or the
      "wrong" value is accidentally supported and a low delta is correct
      behaviour rather than blindness.

READ-ONLY on existing checkpoints. Costs no API spend; NLI runs locally.
"""
import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np

import config
from utils import checkpoint_exists, load_checkpoint, save_checkpoint

SEED = config.RANDOM_SEED if hasattr(config, 'RANDOM_SEED') else 42
GENERATORS = ['claude', 'gpt4omini', 'llama3']

# Gate used for the operational "would this be caught?" rate. A product that
# suppresses answers below a grounding threshold needs the DETECTION rate, not
# the mean delta — a 0.02 average drop catches nothing in practice.
GATE = 0.5

# Numbers, with thousands separators and decimals. The lookarounds keep the
# match off version strings and IDs (28.0.0.137) and off digits glued to
# letters (B12), where "the number" is not a quantity at all.
NUM_RE = re.compile(
    r'(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\w|\.\d|,\d)')

# Capitalised spans, as a cheap NER substitute (no spaCy dependency on gpu1).
ENT_RE = re.compile(r'\b[A-Z][a-z]{2,}(?:\s+(?:of|the|de|van)?\s*[A-Z][a-z]{2,})*\b')

# Sentence-initial and boilerplate capitals are not entities.
ENT_STOP = {
    'The', 'This', 'That', 'These', 'Those', 'There', 'Their', 'They',
    'Based', 'According', 'However', 'Although', 'While', 'Also', 'Both',
    'After', 'Before', 'When', 'Where', 'What', 'Which', 'Who', 'How',
    'Yes', 'No', 'Not', 'None', 'Context', 'Answer', 'Question',
    'Provided', 'Given', 'Cannot', 'Unfortunately', 'Note', 'Some', 'One',
}

ABSTAIN_MARKERS = (
    'cannot answer', 'can not answer', 'do not know', "don't know",
    'not provided in the context', 'no information', 'unable to answer',
    'does not contain', 'not mentioned in the context',
)


def is_abstention(answer: str) -> bool:
    a = (answer or '').lower()
    return any(m in a for m in ABSTAIN_MARKERS)


def _num_variants(s: str):
    """Surface forms the same quantity may take in a document."""
    out = {s}
    bare = s.replace(',', '')
    out.add(bare)
    try:
        v = float(bare)
        if v.is_integer():
            out.add(str(int(v)))
            out.add(f'{int(v):,}')
    except ValueError:
        pass
    return out


def num_in_text(numstr: str, text: str) -> bool:
    return any(v in text for v in _num_variants(numstr))


def perturb_number(numstr: str):
    """
    A plausible wrong value in the SAME surface format.

    Returns (new_string, kind) or None. Format is preserved deliberately:
    turning "$1,000" into "1500" would let NLI react to the formatting rather
    than the quantity, which answers a different question.

    `kind` is returned because the two perturbations are not equally hard.
    A year shifted by 7 is a small edit; a quantity scaled by 1.5 is a large
    one. Pooling them would hide a metric that catches one and not the other,
    and it is the quantity case the invoice scenario turns on.
    """
    bare = numstr.replace(',', '')
    has_comma = ',' in numstr
    if '.' in bare:
        dec = len(bare.split('.')[1])
        try:
            v = float(bare)
        except ValueError:
            return None
        new = round(v * 1.5 + 0.7, dec)
        s = f'{new:,.{dec}f}' if has_comma else f'{new:.{dec}f}'
        return (s, 'decimal') if s != numstr else None
    try:
        v = int(bare)
    except ValueError:
        return None
    # 1800-2100 only: 1000 is far more often a quantity than a year, and
    # sending it down the year branch would perturb it by 0.7% instead of 50%.
    if 1800 <= v <= 2100 and len(bare) == 4 and not has_comma:
        new, kind = (v - 7 if v > 1850 else v + 7), 'year'
    elif v == 0:
        new, kind = 7, 'magnitude'
    else:
        new, kind = int(v * 1.5) + 1, 'magnitude'
    s = f'{new:,}' if has_comma else str(new)
    return (s, kind) if s != numstr else None


def _entity_candidates(text: str):
    for m in ENT_RE.finditer(text):
        span = m.group(0)
        if span.split()[0] in ENT_STOP:
            continue
        if len(span) < 4:
            continue
        yield m.start(), m.end(), span


def build_number_case(answer: str, context: str):
    """Pick one grounded number in `answer` and return the perturbed answer."""
    for m in NUM_RE.finditer(answer):
        numstr = m.group(1)
        if not num_in_text(numstr, context):
            continue                                # not grounded -> uninformative
        pert = perturb_number(numstr)
        if pert is None:
            continue
        new, kind = pert
        if num_in_text(new, context):
            continue                                # replacement is itself supported
        return (answer[:m.start(1)] + new + answer[m.end(1):],
                numstr, new, kind)
    return None


def build_entity_case(answer: str, context: str, donor_pool):
    """Replace one grounded entity with a donor entity absent from the context."""
    for start, end, span in _entity_candidates(answer):
        if span not in context:
            continue
        for cand in donor_pool:
            if cand == span or cand in context or cand in answer:
                continue
            return (answer[:start] + cand + answer[end:], span, cand)
    return None


def _collect_donor_entities(generations, limit=400):
    pool = []
    seen = set()
    for g in generations:
        ans = g.get('generated_answer') or ''
        for _, _, span in _entity_candidates(ans):
            if span not in seen:
                seen.add(span)
                pool.append(span)
                if len(pool) >= limit:
                    return pool
    return pool


def _bootstrap_ci(deltas, n_boot=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=float)
    if arr.size < 2:
        return (float('nan'), float('nan'))
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def build_cases(generations, limit):
    """Eligible perturbation cases from one generation checkpoint."""
    donors = _collect_donor_entities(generations)
    rng = random.Random(SEED)
    pool = [g for g in generations if g.get('retrieved_texts')]
    cases = []
    for g in generations:
        answer = g.get('generated_answer') or ''
        chunks = (g.get('retrieved_texts') or [])[:config.TOP_K]
        if not chunks or answer.startswith('[ERROR') or is_abstention(answer):
            continue
        context = ' '.join(chunks)
        num_case = build_number_case(answer, context)
        if num_case is None:
            continue
        ent_case = build_entity_case(answer, context, donors)
        other = rng.choice(pool) if pool else None
        tries = 0
        while (other is not None and len(pool) > 1 and tries < 10
               and other.get('query_id') == g.get('query_id')):
            other = rng.choice(pool)
            tries += 1
        cases.append({
            'query_id': g.get('query_id'),
            'answer': answer,
            'chunks': chunks,
            'num_answer': num_case[0],
            'num_from': num_case[1],
            'num_to': num_case[2],
            'num_kind': num_case[3],
            'ent_answer': ent_case[0] if ent_case else None,
            'ent_from': ent_case[1] if ent_case else None,
            'ent_to': ent_case[2] if ent_case else None,
            'random_chunks': ((other.get('retrieved_texts') or [])[:config.TOP_K]
                              if other else []),
        })
        if len(cases) >= limit:
            break
    return cases


class _Scorer:
    """
    One falsification set, three evaluators.

    The cases are rebuilt deterministically (fixed SEED, same generation
    checkpoints), so every scorer sees byte-identical answers and contexts.
    That is the point: it separates "this metric is blind to falsified
    values" from "DeBERTa specifically is", which a second evaluator run on
    a different sample could not do.
    """

    def __init__(self, kind):
        self.kind = kind
        if kind == 'align':
            from alignscore import AlignScore
            import os
            ckpt = os.getenv('ALIGNSCORE_CKPT',
                             os.path.expanduser(
                                 '~/rag_faithfulness/alignscore/AlignScore-large.ckpt'))
            print(f'Loading AlignScore from {ckpt}...')
            self.m = AlignScore(model='roberta-large', batch_size=32,
                                device=config.DEVICE, ckpt_path=ckpt,
                                evaluation_mode='nli_sp')
        else:
            from nli import NLIScorer
            self.m = NLIScorer()
            if kind == 'claim':
                import claim_faithfulness
                self.cf = claim_faithfulness

    def score(self, chunks, answer):
        if self.kind == 'nli':
            return self.m.score_chunks(chunks, answer)['nli_max']
        if self.kind == 'claim':
            return self.cf.score_claims(chunks, answer, self.m)['claim_min']
        # AlignScore takes the whole context as one premise.
        return float(self.m.score(contexts=[' '.join(chunks)],
                                  claims=[answer])[0])


def run_check(model, ds_name, gen, limit, nli):
    ck_gen = f'generated_{gen}_{model}_{ds_name}'
    if not checkpoint_exists(ck_gen):
        return None
    generations = load_checkpoint(ck_gen)
    if not generations:
        return None

    suffix = '' if nli.kind == 'nli' else f'_{nli.kind}'
    ck_out = f'perturb{suffix}_{gen}_{model}_{ds_name}'
    if checkpoint_exists(ck_out):
        print(f'  [skip] {ck_out}')
        return load_checkpoint(ck_out)

    cases = build_cases(generations, limit)
    if not cases:
        print(f'  [{gen}/{model}/{ds_name}] no eligible cases '
              f'(of {len(generations)} answers)')
        return None

    print(f'  [{gen}/{model}/{ds_name}] {len(cases)} eligible cases '
          f'of {len(generations)} answers -> scoring 4 conditions')
    rows = []
    for i, c in enumerate(cases, 1):
        if i % 50 == 0:
            print(f'    {i}/{len(cases)}')
        orig = nli.score(c['chunks'], c['answer'])
        num = nli.score(c['chunks'], c['num_answer'])
        ent = (nli.score(c['chunks'], c['ent_answer'])
               if c['ent_answer'] else None)
        rnd = (nli.score(c['random_chunks'], c['answer'])
               if c['random_chunks'] else None)
        rows.append({
            'query_id': c['query_id'],
            'num_from': c['num_from'], 'num_to': c['num_to'],
            'num_kind': c['num_kind'],
            'ent_from': c['ent_from'], 'ent_to': c['ent_to'],
            'orig': orig, 'number': num, 'entity': ent, 'random': rnd,
        })
    save_checkpoint(ck_out, rows)
    return rows


def summarise(rows, label):
    orig = np.array([r['orig'] for r in rows], dtype=float)
    num = np.array([r['number'] for r in rows], dtype=float)
    has_ent = [r for r in rows if r['entity'] is not None]
    ent = np.array([r['entity'] for r in has_ent], dtype=float)
    ent_o = np.array([r['orig'] for r in has_ent], dtype=float)
    rnd = np.array([r['random'] for r in rows if r['random'] is not None],
                   dtype=float)

    d_num = orig - num
    ci_num = _bootstrap_ci(d_num)
    # Detection = the untouched answer passed the gate and the falsified one
    # does not. That is the operational question a grounding gate answers; a
    # small mean delta can still catch nothing at any usable threshold.
    above = orig >= GATE
    det_num = float((above & (num < GATE)).sum() / max(1, above.sum()))

    print(f'\n--- {label} (n={len(rows)}) ---')
    print(f'  orig                       {orig.mean():.4f}')
    print(f'  number-perturbed           {num.mean():.4f}   '
          f'delta {d_num.mean():+.4f}  '
          f'95% CI [{ci_num[0]:+.4f}, {ci_num[1]:+.4f}]')
    if ent.size:
        d_ent = ent_o - ent
        ci_ent = _bootstrap_ci(d_ent)
        above_e = ent_o >= GATE
        det_ent = float((above_e & (ent < GATE)).sum() / max(1, above_e.sum()))
        print(f'  entity-perturbed (n={ent.size})    {ent.mean():.4f}   '
              f'delta {d_ent.mean():+.4f}  '
              f'95% CI [{ci_ent[0]:+.4f}, {ci_ent[1]:+.4f}]')
        mean_d_ent = float(d_ent.mean())
    else:
        det_ent = float('nan')
        mean_d_ent = float('nan')
        print('  entity-perturbed           (no eligible cases)')
    if rnd.size:
        print(f'  random context             {rnd.mean():.4f}   <- floor')
    print(f'  detection @ gate {GATE}:  number {det_num:.1%}   '
          f'entity {det_ent:.1%}')

    # Broken out because a year shifted by 7 and a quantity scaled by 1.5 are
    # not the same test. The invoice case is the magnitude row.
    by_kind = {}
    for kind in sorted({r.get('num_kind') for r in rows if r.get('num_kind')}):
        sub = [r for r in rows if r.get('num_kind') == kind]
        o = np.array([r['orig'] for r in sub], dtype=float)
        nu = np.array([r['number'] for r in sub], dtype=float)
        ab = o >= GATE
        det = float((ab & (nu < GATE)).sum() / max(1, ab.sum()))
        by_kind[kind] = {'n': len(sub), 'orig': float(o.mean()),
                         'number': float(nu.mean()),
                         'delta': float((o - nu).mean()), 'det': det}
        print(f'    [{kind:<9} n={len(sub):>4}]  {o.mean():.4f} -> '
              f'{nu.mean():.4f}   delta {(o - nu).mean():+.4f}   det {det:.1%}')

    return {
        'label': label, 'n': len(rows),
        'orig': float(orig.mean()), 'number': float(num.mean()),
        'delta_number': float(d_num.mean()), 'ci_number': ci_num,
        'entity': float(ent.mean()) if ent.size else float('nan'),
        'delta_entity': mean_d_ent,
        'random': float(rnd.mean()) if rnd.size else float('nan'),
        'det_number': det_num, 'det_entity': det_ent,
        'by_kind': by_kind,
    }


def main():
    ap = argparse.ArgumentParser(
        description='Numeric-infidelity sensitivity of the NLI faithfulness metric')
    ap.add_argument('--datasets', default='NQ,HotpotQA')
    ap.add_argument('--models', default=None,
                    help='comma-separated; default = every configured model')
    ap.add_argument('--generators', default='claude,gpt4omini')
    ap.add_argument('--limit', type=int, default=300,
                    help='max cases per (model, dataset, generator)')
    ap.add_argument('--scope-n', type=int, default=None,
                    help='checkpoint scope N (default: config default)')
    ap.add_argument('--scorer', default='nli', choices=['nli', 'claim', 'align'],
                    help='nli = whole-answer max over chunks (the current '
                         'metric); claim = min over claims of max over chunks; '
                         'align = AlignScore (needs PYTHONPATH=~/align_env)')
    args = ap.parse_args()

    if args.scope_n:
        from utils import set_scope
        set_scope(args.scope_n)
    print(f'  [scope] checkpoints -> {config.CHECKPOINT_DIR}')

    models = ([m.strip() for m in args.models.split(',')] if args.models
              else [c['name'] for c in config.EMBEDDING_MODELS])
    datasets = [d.strip() for d in args.datasets.split(',')]
    gens = [g.strip() for g in args.generators.split(',')]

    # Constructed here, not at module scope: the perturbation logic must be
    # importable and testable on a machine without torch.
    nli = _Scorer(args.scorer)
    summaries = []
    for ds in datasets:
        for gen in gens:
            for model in models:
                rows = run_check(model, ds, gen, args.limit, nli)
                if rows:
                    summaries.append(summarise(rows, f'{ds}/{gen}/{model}'))

    if not summaries:
        print('\nNo eligible cases anywhere — check the scope and checkpoints.')
        return 1

    print('\n' + '=' * 82)
    print('SUMMARY — delta = fall in nli_max when a grounded value is falsified')
    print('=' * 82)
    print(f'{"condition":<40}{"n":>5}{"orig":>8}{"num":>8}'
          f'{"d_num":>9}{"d_ent":>9}{"det":>6}')
    for s in summaries:
        print(f'{s["label"]:<40}{s["n"]:>5}{s["orig"]:>8.3f}{s["number"]:>8.3f}'
              f'{s["delta_number"]:>+9.4f}{s["delta_entity"]:>+9.4f}'
              f'{s["det_number"]:>6.0%}')
    d = np.array([s['delta_number'] for s in summaries])
    de = np.array([s['delta_entity'] for s in summaries])
    de = de[~np.isnan(de)]
    rn = np.array([s['random'] for s in summaries])
    rn = rn[~np.isnan(rn)]
    print('-' * 82)
    print(f'mean delta  number {d.mean():+.4f}'
          + (f'   entity {de.mean():+.4f}' if de.size else '')
          + (f'   random-context floor {rn.mean():.3f}' if rn.size else ''))
    save_checkpoint(
        'perturb_summary' if args.scorer == 'nli'
        else f'perturb_summary_{args.scorer}', summaries)
    return 0


if __name__ == '__main__':
    sys.exit(main())
