"""
claim_faithfulness.py — claim-level faithfulness, replacing max-over-chunks.

THE DEFECT THIS FIXES
    faithfulness.py scores the WHOLE answer against each chunk and takes the
    maximum. For a single-claim answer that is fine. For

        "Roentgen of Germany received the prize in 1901. He received
         150,782 SEK for it."

    it asks only whether some chunk entails the whole string, so one
    fabricated field can be carried by the correct material around it. The
    2026-08-25 perturbation check measured the consequence: falsifying a
    grounded value moved nli_max by 0.033-0.094 on Claude's answers and was
    caught 2-15% of the time, against 0.357-0.480 and 55-76% on GPT-4o-mini's
    much shorter ones.

THE FORMULA

        F(a, R) = min over claims c in a  [ max over chunks d in R  NLI(d |= c) ]

    Max over chunks is correct — any retrieved document may support a claim.
    Max over CLAIMS is the bug: faithfulness is a weakest-link property, so
    one unsupported claim must not be hidden by three supported ones.

    Reported alongside the minimum:
      claim_mean   softer aggregate, for sensitivity analysis
      weakest      the claim that scored lowest, and the chunk that best
                   supported it — the audit trail a document-QA product needs
      whole_max    the old metric on the same answer, so the two are
                   comparable on identical inputs

CLAIM DECOMPOSITION
    Sentence-level, deliberately: it is deterministic, free, and reproducible
    without an LLM in the measurement loop. An LLM decomposer would produce
    finer claims but would put a second generator inside the metric, which is
    the kind of dependency this paper is arguing against.
"""
import argparse
import re
import sys
from typing import Dict, List

import numpy as np

import config
import textnorm
from utils import checkpoint_exists, load_checkpoint, save_checkpoint

GENERATORS = ['claude', 'gpt4omini', 'llama3', config.OPEN_MODEL_LABEL]

# Abbreviations that end in a period without ending a sentence.
_ABBREV = (r'(?:Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Co|Corp|Jr|Sr|St|vs|etc|al|Fig|No'
           r'|Vol|pp|Ph\.D|U\.S|U\.K|e\.g|i\.e|approx|Gen|Sen|Rep|Mt)')

_WORD = re.compile(r'\w+', re.UNICODE)

# Minimum content words for a span to be a claim. Kept deliberately low: a
# dropped claim is invisible to the minimum, so an aggressive threshold would
# hide exactly the short fabrications the metric exists to catch
# ("Sarah approved it." is three words and entirely checkable). Only headings
# and bare fragments ("Summary", "Yes.") fall below this.
MIN_CLAIM_WORDS = 3


# Canonical since 2026-09-03: textnorm owns it, so nli.py can strip the same
# way without importing this module (which imports nli).
strip_markdown = textnorm.strip_markdown


def split_claims(answer: str) -> List[str]:
    """
    Split an answer into sentence-level claims.

    Blank lines and list items are hard boundaries. Within a paragraph a
    period only ends a sentence when the preceding token is not an
    abbreviation or a single initial ("W. C. Roentgen" stays one claim).
    """
    text = strip_markdown(answer)
    if not text:
        return []

    claims = []
    for block in re.split(r'\n+', text):
        block = block.strip()
        if not block:
            continue
        start = 0
        for m in re.finditer(r'[.!?]["\')\]]*\s+', block):
            left = block[:m.start()]
            tail = re.search(r'(\S+)$', left)
            tok = tail.group(1) if tail else ''
            if re.fullmatch(_ABBREV, tok, re.I) or re.fullmatch(r'[A-Z]', tok):
                continue                      # abbreviation or initial
            nxt = block[m.end():m.end() + 1]
            # `.isdigit()` is required: a sentence that OPENS with a number
            # ("1,500 was requested by the team.") failed .isupper() and was
            # silently merged into the previous claim. That undercount landed
            # selectively on numeric-heavy answers — exactly the ones the
            # falsification probe operates on — and inflated n_claims' own
            # denominator. The character class also had a straight quote twice
            # and no opening curly quote.
            if nxt and not (nxt.isupper() or nxt.isdigit()
                            or nxt in '"\'(“‘'):
                continue                      # not a sentence start
            claims.append(block[start:m.end()].strip())
            start = m.end()
        if start < len(block):
            claims.append(block[start:].strip())

    # A span ending in ':' is a lead-in ("The context provides:"), not an
    # assertion. Length alone cannot separate those from real short claims,
    # and keeping them would drag the minimum down on nothing.
    out = [c for c in claims
           if len(_WORD.findall(c)) >= MIN_CLAIM_WORDS
           and not c.rstrip().endswith(':')]
    # An answer of only short fragments still needs one hypothesis to score,
    # or it would silently vanish from the analysis.
    return out or ([text] if _WORD.findall(text) else [])


def score_claims(chunks: List[str], answer: str, nli) -> dict:
    """
    Claim-level faithfulness for one answer.

    All (chunk, claim) pairs go through the NLI model in one batched call:
    looping score_chunks() per claim would re-tokenise every chunk once per
    claim and roughly double the runtime on long answers.
    """
    claims = split_claims(answer)
    ctx = [c for c in chunks if c]
    if not claims or not ctx:
        return {'claim_min': float('nan'), 'claim_mean': float('nan'),
                'n_claims': len(claims), 'per_claim': [], 'weakest': None,
                'weakest_score': float('nan'), 'whole_max': float('nan')}

    concat = ' '.join(ctx)                    # tokenizer truncates at 512
    premises = ctx + [concat]

    pairs = [(p, c) for c in claims for p in premises]
    pairs.append((concat, answer))            # old metric, same inputs
    for p in ctx:
        pairs.append((p, answer))

    probs = nli.entailment_probs(pairs)

    n_p = len(premises)
    per_claim = []
    for i, c in enumerate(claims):
        block = probs[i * n_p:(i + 1) * n_p]
        best = int(np.argmax(block))
        per_claim.append({
            'claim': c,
            'score': float(block[best]),
            # len(ctx) is the concatenation, not a single chunk
            'best_chunk': None if best == len(ctx) else best,
        })

    whole = probs[len(claims) * n_p:]
    scores = [p['score'] for p in per_claim]
    worst = int(np.argmin(scores))
    return {
        'claim_min': float(min(scores)),
        'claim_mean': float(np.mean(scores)),
        'n_claims': len(claims),
        'per_claim': per_claim,
        'weakest': per_claim[worst]['claim'],
        'weakest_score': float(scores[worst]),
        'whole_max': float(max(whole)),
    }


def _generation_checkpoints(model: str, ds_name: str):
    for gen in GENERATORS:
        ck = f'generated_{gen}_{model}_{ds_name}'
        if checkpoint_exists(ck):
            yield gen, ck


def run_phase_claim(datasets: Dict, model_names: List[str] = None,
                    limit: int = None):
    """Re-score existing generations with claim-level faithfulness."""
    from nli import NLIScorer
    nli = NLIScorer()
    model_list = [c['name'] for c in config.EMBEDDING_MODELS
                  if not model_names or c['name'] in model_names]

    for model in model_list:
        for ds_name in datasets:
            for gen, ck_gen in _generation_checkpoints(model, ds_name):
                ck_out = f'claim_scores_{gen}_{model}_{ds_name}'
                if checkpoint_exists(ck_out):
                    print(f'  [skip] {ck_out}')
                    continue
                generations = load_checkpoint(ck_gen)
                if limit:
                    generations = generations[:limit]
                print(f'  claim-level [{gen}] [{model}] [{ds_name}] '
                      f'({len(generations)} answers)...')
                scores = []
                for i, g in enumerate(generations, 1):
                    if i % 200 == 0:
                        print(f'    {i}/{len(generations)}')
                    ans = g.get('generated_answer') or ''
                    if ans.startswith('[ERROR'):
                        continue
                    s = score_claims(
                        (g.get('retrieved_texts') or [])[:config.TOP_K],
                        ans, nli)
                    scores.append({
                        'query_id': g['query_id'],
                        'claim_min': s['claim_min'],
                        'claim_mean': s['claim_mean'],
                        'n_claims': s['n_claims'],
                        'weakest_score': s['weakest_score'],
                        'whole_max': s['whole_max'],
                    })
                save_checkpoint(ck_out, scores)
                cmin = np.nanmean([s['claim_min'] for s in scores])
                wmax = np.nanmean([s['whole_max'] for s in scores])
                ncl = np.nanmean([s['n_claims'] for s in scores])
                print(f'  {model}/{ds_name}/{gen}: claim_min {cmin:.4f}  '
                      f'whole_max {wmax:.4f}  mean claims {ncl:.1f}')
    print('[phase claim] complete')


def main():
    ap = argparse.ArgumentParser(description='Claim-level faithfulness scoring')
    ap.add_argument('--datasets', default=','.join(config.DATASETS))
    ap.add_argument('--models', default=None)
    ap.add_argument('--limit', type=int, default=None,
                    help='answers per checkpoint (default: all)')
    ap.add_argument('--scope-n', type=int, default=None)
    args = ap.parse_args()

    if args.scope_n:
        from utils import set_scope
        set_scope(args.scope_n)
    print(f'  [scope] checkpoints -> {config.CHECKPOINT_DIR}')

    datasets = {d.strip(): None for d in args.datasets.split(',')}
    models = [m.strip() for m in args.models.split(',')] if args.models else None
    run_phase_claim(datasets, models, limit=args.limit)
    return 0


if __name__ == '__main__':
    sys.exit(main())
