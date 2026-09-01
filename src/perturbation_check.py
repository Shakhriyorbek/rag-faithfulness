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

import abstention
import config
from utils import checkpoint_exists, load_checkpoint, save_checkpoint

SEED = config.RANDOM_SEED if hasattr(config, 'RANDOM_SEED') else 42
GENERATORS = ['claude', 'gpt4omini', 'llama3', config.OPEN_MODEL_LABEL]

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

# Refusals are excluded from the falsification sample: an abstention has no
# grounded claim to falsify. The rule is now the SHARED one in abstention.py.
#
# It used to be a local list of 9 phrases matched as a substring ANYWHERE in
# the answer, which excluded 365 substantive answers over the 16,000-row grid
# — answers that carry a mid-text hedge ("...However, the context does not
# contain their ages.") while asserting plenty. Those rows counted as
# `answered` in the conditional analysis, so Section V and Section VI were
# describing different populations with the same word.
is_abstention = abstention.is_abstention


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


# One quantity, matched at numeric/word boundaries. A raw `in` test made
# num_in_text('1000', 'the total was 10000 USD') True, and both eligibility
# rules run through this function: false positives on rule (a) admitted answers
# whose value was never grounded (perturbing them cannot lower entailment, so
# they entered the sample with a near-zero delta and dragged the mean toward
# the reported conclusion), and false positives on rule (b) silently discarded
# eligible cases.
#
# The boundary mirrors NUM_RE rather than \b: `\b` treats ',' and '.' as
# boundaries, which reintroduces the bug on '1,000' and '3.55'. A digit glued
# to letters is an identifier, not a quantity ('model B7' does not ground 7),
# so the left side excludes \w as NUM_RE does.
_BOUNDARY_CACHE = {}


def _boundary_re(v: str):
    rx = _BOUNDARY_CACHE.get(v)
    if rx is None:
        rx = re.compile(rf'(?<![\w.,]){re.escape(v)}(?![\w.,]?\d)')
        _BOUNDARY_CACHE[v] = rx
    return rx


def num_in_text(numstr: str, text: str) -> bool:
    """True if `numstr` (in any of its surface forms) occurs in `text` as a
    standalone quantity rather than as a digit substring of a larger one."""
    if not numstr or not text:
        return False
    return any(_boundary_re(v).search(text) for v in _num_variants(numstr))


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


def answer_numbers(answer: str):
    """Every quantity NUM_RE finds in an answer, in order of appearance."""
    return [m.group(1) for m in NUM_RE.finditer(answer or '')]


def _donor_supports(other, numbers) -> bool:
    """True if a candidate random context carries any of the answer's values."""
    ctx = ' '.join((other.get('retrieved_texts') or [])[:config.TOP_K])
    return any(num_in_text(n, ctx) for n in numbers)


def build_cases(generations, limit):
    """Eligible perturbation cases from one generation checkpoint."""
    donors = _collect_donor_entities(generations)
    rng = random.Random(SEED)
    pool = [g for g in generations if g.get('retrieved_texts')]
    cases = []
    n_donor_fallback = 0
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
        # The `random` condition is the floor of the scale, so a donor context
        # that happens to support the answer inflates that floor and makes the
        # floor-anchored gate too permissive. Rejecting on the answer's
        # quantities is the cheap half of "unrelated context": same 10-try
        # loop, falling back to the old behaviour (any non-self donor) when
        # nothing passes, with the fallbacks counted rather than hidden.
        numbers = answer_numbers(answer)
        other = rng.choice(pool) if pool else None
        tries = 0
        while (other is not None and len(pool) > 1 and tries < 10
               and (other.get('query_id') == g.get('query_id')
                    or _donor_supports(other, numbers))):
            other = rng.choice(pool)
            tries += 1
        if (other is not None and len(pool) > 1
                and (other.get('query_id') == g.get('query_id')
                     or _donor_supports(other, numbers))):
            n_donor_fallback += 1
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
    if n_donor_fallback:
        print(f'    [donor] {n_donor_fallback}/{len(cases)} random contexts '
              f'kept despite supporting a value in the answer '
              f'(no clean donor in 10 tries)')
    return cases


def numeric_grounding_check(answer: str, chunks) -> bool:
    """
    Deterministic value grounding: does EVERY quantity in the answer occur in
    the retrieved context?

    Section IX of the paper recommends this check and notes it had not been
    evaluated. It is the complement of the NLI evaluators rather than a rival:
    it says nothing about whether the answer is entailed, only whether its
    literal values are present, which is exactly what the invoice scenario
    asks and exactly what a verbose answer hides from an entailment model.

    False by construction on the falsified variant (the replacement is
    rejected at case-construction time if it appears in the context), so the
    informative number is not its recall but its FALSE-POSITIVE rate on the
    untouched answers — a check that flags a third of correct answers is not
    deployable however well it catches fabrications.
    """
    ctx = ' '.join(c for c in (chunks or []) if c)
    if not ctx:
        return False
    return all(num_in_text(n, ctx) for n in answer_numbers(answer))


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson interval for a proportion. Detection rates in Table II sit
    on n≈200 with rates near 4%, where the normal approximation is useless
    (it produces negative lower bounds) and the interval is ±2-3pp, which is
    the same size as several of the differences being read off the table."""
    if n <= 0:
        return (float('nan'), float('nan'))
    ph = k / n
    d = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _detection(orig, pert, gate):
    """(rate, n_caught, n_eligible) at a gate: the untouched answer passes and
    the falsified one does not."""
    above = orig >= gate
    caught = int((above & (pert < gate)).sum())
    return (caught / max(1, int(above.sum())), caught, int(above.sum()))


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
        if kind == 'numeric':
            return                              # deterministic, no model
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
        if self.kind == 'nli_concat':
            # F5 diagnostic: the SAME NLI model on the SAME answer, but with
            # AlignScore's premise construction — one concatenated context,
            # no per-chunk maximum. NLI-max scores 6 premises (5 chunks + the
            # concatenation) and takes the best; AlignScore receives one
            # premise and splits it itself. Part of the NLI/AlignScore
            # difference could therefore be premise granularity rather than
            # evaluator sensitivity, and this bounds how much.
            return self.m.entailment_prob(' '.join(chunks), answer)
        if self.kind == 'claim':
            return self.cf.score_claims(chunks, answer, self.m)['claim_min']
        if self.kind == 'numeric':
            return 1.0 if numeric_grounding_check(answer, chunks) else 0.0
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


def floor_gate(rows, pct: float = 95.0):
    """
    A gate anchored to the random-context floor of THIS cell and evaluator.

    GATE = 0.5 is a module constant while the floor is measured per cell: on
    NQ/Claude the mean random-context nli_max is 0.522, so an answer scored
    against an unrelated context already clears the fixed gate. A detection
    rate read at 0.5 there partly reflects a compressed scale rather than
    blindness to falsification, and rates are not comparable across evaluators
    whose floors differ. The 95th percentile of the random distribution is the
    score an unrelated context beats only 5% of the time, so "above the gate"
    means "better supported than an unrelated document", which is the same
    statement on every scale.
    """
    rnd = np.array([r['random'] for r in rows if r.get('random') is not None],
                   dtype=float)
    if rnd.size < 20:
        return float('nan')
    return float(np.percentile(rnd, pct))


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
    det_num, k_num, n_num = _detection(orig, num, GATE)
    ci_det = _wilson_ci(k_num, n_num)
    gate_f = floor_gate(rows)
    if gate_f == gate_f:
        det_num_f, k_f, n_f = _detection(orig, num, gate_f)
        ci_det_f = _wilson_ci(k_f, n_f)
    else:
        det_num_f, k_f, n_f, ci_det_f = (float('nan'),) * 3 + ((float('nan'),) * 2,)

    print(f'\n--- {label} (n={len(rows)}) ---')
    print(f'  orig                       {orig.mean():.4f}')
    print(f'  number-perturbed           {num.mean():.4f}   '
          f'delta {d_num.mean():+.4f}  '
          f'95% CI [{ci_num[0]:+.4f}, {ci_num[1]:+.4f}]')
    if ent.size:
        d_ent = ent_o - ent
        ci_ent = _bootstrap_ci(d_ent)
        det_ent, k_e, n_e = _detection(ent_o, ent, GATE)
        print(f'  entity-perturbed (n={ent.size})    {ent.mean():.4f}   '
              f'delta {d_ent.mean():+.4f}  '
              f'95% CI [{ci_ent[0]:+.4f}, {ci_ent[1]:+.4f}]')
        mean_d_ent = float(d_ent.mean())
    else:
        det_ent = float('nan')
        mean_d_ent = float('nan')
        print('  entity-perturbed           (no eligible cases)')
    if rnd.size:
        print(f'  random context             {rnd.mean():.4f}   <- floor '
              f'(p95 {gate_f:.4f})')
    # Both gates, both with a binomial CI: at n=200 a 4% rate carries roughly
    # +/-2.7pp, which several of the differences read off Table II do not clear.
    print(f'  detection @ fixed gate {GATE}:      number {det_num:.1%} '
          f'[{ci_det[0]:.1%}, {ci_det[1]:.1%}]  ({k_num}/{n_num})   '
          f'entity {det_ent:.1%}')
    if gate_f == gate_f:
        print(f'  detection @ floor gate {gate_f:.3f}:    number {det_num_f:.1%} '
              f'[{ci_det_f[0]:.1%}, {ci_det_f[1]:.1%}]  ({k_f}/{n_f})')

    # Broken out because a year shifted by 7 and a quantity scaled by 1.5 are
    # not the same test. The invoice case is the magnitude row.
    by_kind = {}
    for kind in sorted({r.get('num_kind') for r in rows if r.get('num_kind')}):
        sub = [r for r in rows if r.get('num_kind') == kind]
        o = np.array([r['orig'] for r in sub], dtype=float)
        nu = np.array([r['number'] for r in sub], dtype=float)
        det, k, nn = _detection(o, nu, GATE)
        lo, hi = _wilson_ci(k, nn)
        by_kind[kind] = {'n': len(sub), 'orig': float(o.mean()),
                         'number': float(nu.mean()),
                         'delta': float((o - nu).mean()), 'det': det,
                         'det_ci': (lo, hi), 'det_k': k, 'det_n': nn}
        print(f'    [{kind:<9} n={len(sub):>4}]  {o.mean():.4f} -> '
              f'{nu.mean():.4f}   delta {(o - nu).mean():+.4f}   '
              f'det {det:.1%} [{lo:.1%}, {hi:.1%}]')

    return {
        'label': label, 'n': len(rows),
        'orig': float(orig.mean()), 'number': float(num.mean()),
        'delta_number': float(d_num.mean()), 'ci_number': ci_num,
        'entity': float(ent.mean()) if ent.size else float('nan'),
        'delta_entity': mean_d_ent,
        'random': float(rnd.mean()) if rnd.size else float('nan'),
        'det_number': det_num, 'det_number_ci': ci_det,
        'det_number_k': k_num, 'det_number_n': n_num,
        'gate_floor': gate_f, 'det_number_floor': det_num_f,
        'det_number_floor_ci': ci_det_f,
        'det_entity': det_ent,
        'by_kind': by_kind,
    }


def summarise_numeric(rows, label):
    """
    Deterministic value-grounding check, scored as a falsification DETECTOR.

    Each case contributes two instances: the untouched answer (negative — a
    flag here is a false positive) and the falsified one (positive). Recall is
    near-perfect by construction, so the deployable-or-not number is the
    false-positive rate on `orig`, reported first.
    """
    o = np.array([r['orig'] for r in rows], dtype=float)
    n = np.array([r['number'] for r in rows], dtype=float)
    tp = int((n < 0.5).sum())
    fn = int((n >= 0.5).sum())
    fp = int((o < 0.5).sum())
    tn = int((o >= 0.5).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    fpr = fp / max(1, fp + tn)
    ci_fpr = _wilson_ci(fp, fp + tn)
    ci_rec = _wilson_ci(tp, tp + fn)
    print(f'\n--- {label} (n={len(rows)}) — numeric grounding check ---')
    print(f'  false-positive rate on untouched answers  {fpr:.1%} '
          f'[{ci_fpr[0]:.1%}, {ci_fpr[1]:.1%}]  ({fp}/{fp + tn})')
    print(f'  recall on falsified answers               {rec:.1%} '
          f'[{ci_rec[0]:.1%}, {ci_rec[1]:.1%}]  ({tp}/{tp + fn})')
    print(f'  precision {prec:.3f}   F1 {f1:.3f}')
    by_kind = {}
    for kind in sorted({r.get('num_kind') for r in rows if r.get('num_kind')}):
        sub = [r for r in rows if r.get('num_kind') == kind]
        so = np.array([r['orig'] for r in sub], dtype=float)
        sn = np.array([r['number'] for r in sub], dtype=float)
        k_tp = int((sn < 0.5).sum())
        k_fp = int((so < 0.5).sum())
        k_rec = k_tp / max(1, len(sub))
        k_fpr = k_fp / max(1, len(sub))
        by_kind[kind] = {'n': len(sub), 'recall': k_rec, 'fpr': k_fpr,
                         'recall_ci': _wilson_ci(k_tp, len(sub)),
                         'fpr_ci': _wilson_ci(k_fp, len(sub))}
        print(f'    [{kind:<9} n={len(sub):>4}]  recall {k_rec:.1%}   '
              f'false-positive {k_fpr:.1%}')
    return {'label': label, 'n': len(rows), 'scorer': 'numeric',
            'precision': prec, 'recall': rec, 'f1': f1, 'fpr': fpr,
            'fpr_ci': ci_fpr, 'recall_ci': ci_rec,
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'orig': float(o.mean()), 'number': float(n.mean()),
            'delta_number': float((o - n).mean()),
            'delta_entity': float('nan'),
            'random': float(np.mean([r['random'] for r in rows
                                     if r.get('random') is not None]))
                       if any(r.get('random') is not None for r in rows)
                       else float('nan'),
            'det_number': rec, 'det_entity': float('nan'),
            'by_kind': by_kind}


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
    ap.add_argument('--scorer', default='nli',
                    choices=['nli', 'nli_concat', 'claim', 'align', 'numeric'],
                    help='nli = whole-answer max over chunks (the current '
                         'metric); nli_concat = the same model on the '
                         'concatenated premise only, no per-chunk max (the F5 '
                         'premise-granularity diagnostic); claim = min over '
                         'claims of max over chunks; align = AlignScore (needs '
                         'PYTHONPATH=~/align_env); numeric = deterministic '
                         'value-presence check, no model required')
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
                    fn = (summarise_numeric if args.scorer == 'numeric'
                          else summarise)
                    summaries.append(fn(rows, f'{ds}/{gen}/{model}'))

    if not summaries:
        print('\nNo eligible cases anywhere — check the scope and checkpoints.')
        return 1

    print('\n' + '=' * 82)
    print('SUMMARY — delta = fall in nli_max when a grounded value is falsified')
    print('=' * 82)
    if args.scorer == 'numeric':
        print(f'{"condition":<40}{"n":>5}{"recall":>9}{"fpr":>9}'
              f'{"prec":>8}{"F1":>8}')
        for s in summaries:
            print(f'{s["label"]:<40}{s["n"]:>5}{s["recall"]:>9.1%}'
                  f'{s["fpr"]:>9.1%}{s["precision"]:>8.3f}{s["f1"]:>8.3f}')
    else:
        print(f'{"condition":<40}{"n":>5}{"orig":>8}{"num":>8}'
              f'{"d_num":>9}{"d_ent":>9}{"det":>6}{"det 95% CI":>16}'
              f'{"gate_f":>8}{"det_f":>7}')
        for s in summaries:
            lo, hi = s.get('det_number_ci', (float('nan'), float('nan')))
            print(f'{s["label"]:<40}{s["n"]:>5}{s["orig"]:>8.3f}'
                  f'{s["number"]:>8.3f}{s["delta_number"]:>+9.4f}'
                  f'{s["delta_entity"]:>+9.4f}{s["det_number"]:>6.0%}'
                  f'{("[%.1f%%, %.1f%%]" % (lo * 100, hi * 100)):>16}'
                  f'{s.get("gate_floor", float("nan")):>8.3f}'
                  f'{s.get("det_number_floor", float("nan")):>7.0%}')
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
