"""
llm_judge.py — LLM grading of answer correctness, replacing EM/containment.

WHY THIS EXISTS
    EM is 0.000 on all 4,000 Claude/NQ rows. The generators never emit a bare
    answer span — "Based on the provided context, **Wilhelm Conrad Roentgen**
    of Germany received..." against gold "Wilhelm Conrad Roentgen, of Germany"
    is correct, and scores EM 0 and token-F1 0.28 because precision dies on
    every extra word. Containment is the upper bound (a verbose answer can
    mention the gold string incidentally), so the honest interval today is
    [0.000, 0.738], which is useless.

    Every accuracy figure and the whole necessary/sufficient 2x2 rests on that
    interval. This module replaces it with a graded judgement.

DESIGN

    Judged against the GOLD ANSWER, never the retrieved context. Correctness
    and faithfulness are different properties and the paper's argument depends
    on measuring them separately — a wrong answer can be perfectly grounded,
    and a right answer can be ungrounded. The judge therefore never sees the
    context, which also makes it impossible for it to quietly grade grounding.

    PROMPT PARITY, as with the generators. Both judges receive the
    byte-identical string from build_judge_prompt(). Agreement between them is
    only interpretable if they were asked the same question, and this paper's
    central finding is that evaluator choice changes conclusions — measuring
    judge agreement is on-topic, not incidental.

    A one-word verdict rather than a structured-output schema, for the same
    reason: the constraint mechanisms differ between vendors, and a difference
    there would confound the agreement measurement.

    UNGRADABLE IS NOT INCORRECT. Rows that correctness.is_ungradable() rejects
    ([ERROR: ...] rows, None answers from the C2 oracle) are never sent, and a
    reply that does not parse is recorded as None, not as a wrong answer.
    Grading an API failure as a wrong answer deflates whichever condition hit
    rate limits, invisibly.

COST
    ~16,000 rows. Run --limit first: it prints a MEASURED projection from real
    usage before you commit to the full grid, and a few hundred rows is also
    what you need to check the judge agrees with itself and with a human.
    Paid runs refuse to start without --yes.
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import config
import correctness
from utils import (checkpoint_exists, cost_tracker, load_checkpoint,
                   save_checkpoint)

JUDGE_MODEL_CLAUDE = 'claude-opus-5'
JUDGE_MODEL_OPENAI = 'gpt-4o-mini'

# Save every N rows so a killed run resumes near where it stopped.
SAVE_EVERY = 100

# Abort rules, mirroring generate.py: a systematic failure should stop the run
# rather than quietly fill the checkpoint with ungradable rows.
MAX_CONSECUTIVE_ERRORS = 5
MAX_ERROR_RATE = 0.20


JUDGE_PROMPT_TEMPLATE = """You are grading whether an answer to a question is correct.

Question: {question}

Reference answer(s): {gold}

Answer to grade: {answer}

Grade the answer against the reference. The answer is CORRECT if it conveys \
the reference answer, even if it is phrased differently, adds extra correct \
detail, or is embedded in a longer sentence. The answer is INCORRECT if it \
contradicts the reference, states a different value, or omits the reference \
answer entirely. If the answer declines to answer or says the information is \
not available, that is REFUSAL.

Reply with exactly one word on the first line: CORRECT, INCORRECT, or REFUSAL.
Add nothing else."""


def build_judge_prompt(question: str, golds: List[str], answer: str) -> str:
    """The one string both judges receive. Do not diverge them."""
    gold = ' | '.join(g for g in golds if g) or '(none provided)'
    return JUDGE_PROMPT_TEMPLATE.format(
        question=(question or '').strip(),
        gold=gold.strip(),
        answer=(answer or '').strip(),
    )


_VERDICT_RE = re.compile(r'\b(CORRECT|INCORRECT|REFUSAL)\b', re.I)


def parse_verdict(text: str):
    """
    (correct, abstained) or (None, None) when the reply does not parse.

    Unparseable is deliberately not False: a judge that returned something
    unexpected has told us nothing, and recording that as a wrong answer would
    bias accuracy downward exactly where the judge is least reliable.
    """
    if not text:
        return None, None
    m = _VERDICT_RE.search(text.strip().split('\n')[0]) or _VERDICT_RE.search(text)
    if not m:
        return None, None
    v = m.group(1).upper()
    if v == 'REFUSAL':
        return False, True          # a refusal is not correct, but is not a guess
    return v == 'CORRECT', False


class ClaudeJudge:
    """
    Anthropic judge.

    Model notes for claude-opus-5, which differ from the haiku-4-5 generator:
      - `temperature` is REMOVED and returns 400. The generators run at
        temperature 0; the judge cannot, so determinism here comes from a
        constrained one-word output rather than from sampling settings.
      - thinking is ON by default. It is left on deliberately: disabling it on
        Opus 5 can leak reasoning into the visible response, which would break
        verdict parsing. `effort: low` keeps the cost of that in check for
        what is a short classification.
    """

    name = 'claude'

    def __init__(self, model: str = None, max_retries: int = 5):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=max_retries)
        self.model = model or JUDGE_MODEL_CLAUDE

    def grade(self, prompt: str) -> str:
        import anthropic
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                output_config={'effort': 'low'},
                messages=[{'role': 'user', 'content': prompt}],
            )
        except anthropic.APIStatusError as e:
            cost_tracker.log_error()
            return f'[ERROR: {type(e).__name__} {e.status_code}]'
        except anthropic.APIConnectionError as e:
            cost_tracker.log_error()
            return f'[ERROR: APIConnectionError: {e}]'

        cost_tracker.log_judge(resp.usage, self.model)
        if resp.stop_reason == 'refusal':
            cost_tracker.log_error()
            return '[ERROR: refusal]'
        return ''.join(b.text for b in resp.content if b.type == 'text')


class OpenAIJudge:
    """Second judge, for inter-judge agreement. Same prompt, byte for byte."""

    name = 'gpt4omini'

    def __init__(self, model: str = None, max_retries: int = 5):
        import openai
        self.client = openai.OpenAI(max_retries=max_retries)
        self.model = model or JUDGE_MODEL_OPENAI

    def grade(self, prompt: str) -> str:
        import openai
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=16,
                temperature=0,
                messages=[{'role': 'user', 'content': prompt}],
            )
        except openai.APIStatusError as e:
            cost_tracker.log_error()
            return f'[ERROR: {type(e).__name__} {e.status_code}]'
        except openai.APIConnectionError as e:
            cost_tracker.log_error()
            return f'[ERROR: APIConnectionError: {e}]'

        cost_tracker.log_judge(resp.usage, self.model)
        choice = resp.choices[0]
        if choice.finish_reason == 'content_filter' or choice.message.content is None:
            cost_tracker.log_error()
            return '[ERROR: content_filter]'
        return choice.message.content


def judge_checkpoint(name: str, judge, limit: int = None,
                     budget_limit: float = 25.0) -> Dict:
    """Grade one generation checkpoint; write `<name>_judged_<judge>`."""
    records = load_checkpoint(name)
    if not records:
        return {}

    ck_out = f'{name}_judged_{judge.name}'

    # Two different reasons a stored row can carry correct=None, and they must
    # not be treated alike on resume:
    #   source_ungradable  the ROW has no usable prediction ([ERROR] generation,
    #                      or a None answer from the C2 oracle). Permanent.
    #   otherwise          the JUDGE failed on it — an API error or a reply that
    #                      did not parse. Transient, and must be retried, or a
    #                      rate-limit blip silently freezes those queries as
    #                      ungradable for every future run.
    done = {}
    for r in (load_checkpoint(ck_out) or []):
        if r.get('correct') is None and not r.get('source_ungradable'):
            continue                      # judge-side failure -> retry it
        done[r['query_id']] = r

    todo = []
    for r in records:
        if r['query_id'] in done:
            continue
        if correctness.is_ungradable(r):
            # Recorded, never sent. Correctness is undefined here, not False.
            done[r['query_id']] = {'query_id': r['query_id'], 'correct': None,
                                   'abstained': None, 'judge': judge.name,
                                   'source_ungradable': True}
            continue
        todo.append(r)
    if limit:
        todo = todo[:limit]

    if not todo:
        print(f'  [skip] {ck_out} — nothing to grade')
        save_checkpoint(ck_out, list(done.values()))
        return {'n': len(done), 'graded': 0}

    print(f'  judging [{judge.name}] {name}: {len(todo)} rows '
          f'({len(done)} already done)')

    consecutive = 0
    n_err = 0
    for i, r in enumerate(todo, 1):
        golds = correctness._gold_answers(r)
        prompt = build_judge_prompt(r.get('question', ''), golds,
                                    r.get('generated_answer', ''))
        reply = judge.grade(prompt)

        if reply.startswith('[ERROR'):
            consecutive += 1
            n_err += 1
            if consecutive >= MAX_CONSECUTIVE_ERRORS:
                print(f'  ABORT: {consecutive} consecutive API errors')
                break
            if i >= 20 and n_err / i > MAX_ERROR_RATE:
                print(f'  ABORT: error rate {n_err / i:.0%} over {i} rows')
                break
            correct, abstained = None, None
        else:
            consecutive = 0
            correct, abstained = parse_verdict(reply)

        done[r['query_id']] = {
            'query_id': r['query_id'], 'correct': correct,
            'abstained': abstained, 'judge': judge.name,
            'source_ungradable': False,   # judged; a None here is retryable
            'raw': reply[:80] if correct is None else None,
        }

        if i % SAVE_EVERY == 0:
            save_checkpoint(ck_out, list(done.values()))
            print(f'    {i}/{len(todo)}  ${cost_tracker.cost:.2f}')
        if cost_tracker.cost > budget_limit:
            print(f'  ABORT: budget ${budget_limit:.2f} reached')
            break

    save_checkpoint(ck_out, list(done.values()))
    graded = [v for v in done.values() if v['correct'] is not None]
    n_g = len(graded)
    return {
        'n': len(done),
        'graded': n_g,
        'ungradable': len(done) - n_g,
        'correct_rate': (sum(1 for v in graded if v['correct']) / n_g) if n_g else None,
        'abstention_rate': (sum(1 for v in graded if v['abstained']) / n_g) if n_g else None,
    }


def compare_to_heuristics(name: str, judge_name: str) -> Dict:
    """
    Agreement between the judge and the EM / containment bounds.

    This is the number that says whether the judge was worth paying for. If it
    tracks containment exactly it has added nothing; the interesting rows are
    the ones where containment says correct and the judge disagrees, which is
    where the upper bound was loose.
    """
    scored = load_checkpoint(f'{name}_scored')
    judged = load_checkpoint(f'{name}_judged_{judge_name}')
    if not scored or not judged:
        return {}
    J = {r['query_id']: r for r in judged}
    rows = [(s, J[s['query_id']]) for s in scored
            if s['query_id'] in J and J[s['query_id']]['correct'] is not None
            and s.get('correct') is not None]
    if not rows:
        return {}
    n = len(rows)
    agree_c = sum(1 for s, j in rows if s['correct_contains'] == j['correct'])
    agree_em = sum(1 for s, j in rows if s['correct_em'] == j['correct'])
    return {
        'n': n,
        'judge_correct_rate': sum(1 for _, j in rows if j['correct']) / n,
        'contains_rate': sum(1 for s, _ in rows if s['correct_contains']) / n,
        'em_rate': sum(1 for s, _ in rows if s['correct_em']) / n,
        'agreement_contains': agree_c / n,
        'agreement_em': agree_em / n,
        # containment says yes, judge says no -> the upper bound was loose
        'contains_false_positive': sum(
            1 for s, j in rows if s['correct_contains'] and not j['correct']) / n,
        # containment says no, judge says yes -> containment missed a paraphrase
        'contains_false_negative': sum(
            1 for s, j in rows if not s['correct_contains'] and j['correct']) / n,
    }


def _generation_checkpoint_names() -> List[str]:
    names = []
    for pattern in correctness.GENERATION_GLOBS:
        for p in sorted(Path(config.CHECKPOINT_DIR).glob(pattern)):
            if p.stem.endswith(('_partial', '_scored')) or '_judged_' in p.stem:
                continue
            names.append(p.stem)
    return sorted(set(names))


def main():
    ap = argparse.ArgumentParser(description='LLM-judge correctness grading')
    ap.add_argument('--judge', default='claude', choices=['claude', 'openai'])
    ap.add_argument('--model', default=None, help='override the judge model')
    ap.add_argument('--limit', type=int, default=None,
                    help='rows per checkpoint — use this for a calibration run')
    ap.add_argument('--budget', type=float, default=25.0,
                    help='hard spend cap in USD')
    ap.add_argument('--checkpoints', default=None,
                    help='comma-separated names; default = all generation checkpoints')
    ap.add_argument('--scope-n', type=int, default=1000)
    ap.add_argument('--yes', action='store_true',
                    help='required for a paid run')
    ap.add_argument('--compare-only', action='store_true',
                    help='no API calls; just report agreement with EM/containment')
    args = ap.parse_args()

    from utils import set_scope
    set_scope(args.scope_n)
    print(f'  [scope] checkpoints -> {config.CHECKPOINT_DIR}')

    names = ([n.strip() for n in args.checkpoints.split(',')]
             if args.checkpoints else _generation_checkpoint_names())
    if not names:
        print('No generation checkpoints found.')
        return 1

    judge_name = 'claude' if args.judge == 'claude' else 'gpt4omini'

    if args.compare_only:
        print(f'\nAgreement with the heuristics ({judge_name}):')
        for name in names:
            c = compare_to_heuristics(name, judge_name)
            if not c:
                continue
            print(f'\n  {name}  (n={c["n"]})')
            print(f'    judge correct     {c["judge_correct_rate"]:.4f}')
            print(f'    containment       {c["contains_rate"]:.4f}   '
                  f'agreement {c["agreement_contains"]:.4f}')
            print(f'    exact match       {c["em_rate"]:.4f}   '
                  f'agreement {c["agreement_em"]:.4f}')
            print(f'    containment says correct, judge disagrees: '
                  f'{c["contains_false_positive"]:.4f}')
            print(f'    containment says wrong, judge disagrees:   '
                  f'{c["contains_false_negative"]:.4f}')
        return 0

    total_rows = sum(len(load_checkpoint(n) or []) for n in names)
    scope = f'{len(names)} checkpoints, ~{total_rows:,} rows'
    if args.limit:
        scope = f'{len(names)} checkpoints, {args.limit} rows each ' \
                f'(~{args.limit * len(names):,} total)'
    print(f'\nJudge: {args.model or (JUDGE_MODEL_CLAUDE if args.judge == "claude" else JUDGE_MODEL_OPENAI)}')
    print(f'Scope: {scope}')
    print(f'Budget cap: ${args.budget:.2f}')

    if not args.yes:
        print('\nThis phase spends money. Re-run with --yes to proceed.')
        print('Start with --limit 200 for a calibration run: it prints a')
        print('measured projection and shows whether the judge is worth paying')
        print('for before you commit to the full grid.')
        return 1

    judge = (ClaudeJudge(args.model) if args.judge == 'claude'
             else OpenAIJudge(args.model))

    summaries = {}
    for name in names:
        summaries[name] = judge_checkpoint(name, judge, limit=args.limit,
                                           budget_limit=args.budget)
        if cost_tracker.cost > args.budget:
            print('Budget reached — stopping.')
            break

    print('\n' + '=' * 74)
    hdr = '%-44s %7s %9s %9s' % ('checkpoint', 'graded', 'correct', 'abstain')
    print(hdr); print('-' * len(hdr))
    for name, s in summaries.items():
        if not s or not s.get('graded'):
            continue
        print('%-44s %7d %9.4f %9.4f'
              % (name[:44], s['graded'],
                 s['correct_rate'] or 0.0, s['abstention_rate'] or 0.0))
    print('\n' + cost_tracker.summary())

    graded_rows = sum(s.get('graded', 0) for s in summaries.values() if s)
    if graded_rows and args.limit:
        per_row = cost_tracker.cost / graded_rows
        print(f'\nMEASURED ${per_row:.5f}/row -> '
              f'${per_row * total_rows:.2f} projected for all {total_rows:,} rows.')

    print('\nNext: --compare-only to see whether the judge disagrees with '
          'containment.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
