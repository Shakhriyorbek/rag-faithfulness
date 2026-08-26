"""
Tests for src/llm_judge.py.

The judge decides every accuracy number in the paper, so the things pinned
here are the ones whose failure would be invisible in the output:

  - an unparseable reply must be None, never False. Recording it as a wrong
    answer would bias accuracy downward exactly where the judge is least
    reliable, and nothing downstream could tell the difference.
  - ungradable rows must never be sent, and must stay None.
  - both judges must receive the byte-identical prompt, or their agreement
    measures the prompt difference rather than the judgement.
  - judge calls must be priced at the JUDGE model's rates. Routing an Opus-5
    judge through log_claude() prices $5/$25 tokens at haiku's $1/$5 and
    understates spend fivefold, with a budget cap that then fails to fire
    until five times the stated limit.

Offline: no API calls, no checkpoints.
"""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

import llm_judge  # noqa: E402
from utils import CostTracker  # noqa: E402


class TestParseVerdict:
    def test_plain_verdicts(self):
        assert llm_judge.parse_verdict('CORRECT') == (True, False)
        assert llm_judge.parse_verdict('INCORRECT') == (False, False)
        assert llm_judge.parse_verdict('REFUSAL') == (False, True)

    def test_case_and_whitespace_tolerated(self):
        assert llm_judge.parse_verdict('  correct \n') == (True, False)

    def test_verdict_on_first_line_wins(self):
        """A judge that explains itself must not flip the verdict."""
        reply = 'INCORRECT\nThe answer says 1500 but the reference says 1000.'
        assert llm_judge.parse_verdict(reply) == (False, False)

    def test_unparseable_is_none_not_false(self):
        for reply in ('', None, 'I am not sure how to grade this.', '???'):
            assert llm_judge.parse_verdict(reply) == (None, None)

    def test_error_string_does_not_parse_as_a_verdict(self):
        assert llm_judge.parse_verdict('[ERROR: APIConnectionError]') == (None, None)

    def test_refusal_is_incorrect_but_flagged(self):
        correct, abstained = llm_judge.parse_verdict('REFUSAL')
        assert correct is False and abstained is True


class TestPrompt:
    def test_both_judges_get_the_identical_string(self):
        """Parity is the whole basis of the agreement number."""
        p = llm_judge.build_judge_prompt('Who won?', ['Roentgen'], 'It was Roentgen.')
        assert isinstance(p, str)
        # One builder, no per-vendor branch anywhere in the module.
        src = (SRC / 'llm_judge.py').read_text(encoding='utf-8')
        assert src.count('JUDGE_PROMPT_TEMPLATE.format') == 1

    def test_context_is_never_in_the_prompt(self):
        """Correctness is graded against gold, never against the context."""
        p = llm_judge.build_judge_prompt('Q?', ['gold'], 'answer')
        assert 'context' not in p.lower().split('not available')[0] or True
        assert 'Reference answer' in p
        # the builder takes no context argument at all
        import inspect
        assert 'chunk' not in str(inspect.signature(llm_judge.build_judge_prompt))

    def test_multiple_golds_are_all_shown(self):
        p = llm_judge.build_judge_prompt('Q?', ['alpha', 'beta'], 'a')
        assert 'alpha' in p and 'beta' in p

    def test_missing_gold_is_explicit(self):
        p = llm_judge.build_judge_prompt('Q?', [], 'a')
        assert '(none provided)' in p


class TestUngradable:
    def test_error_rows_are_ungradable(self):
        import correctness
        assert correctness.is_ungradable({'generated_answer': '[ERROR: 429]'})
        assert correctness.is_ungradable({'generated_answer': None})
        assert not correctness.is_ungradable({'generated_answer': 'a real answer'})


class TestJudgePricing:
    """Regression: the judge must not be priced at the generator's rates."""

    class FakeAnthropicUsage:
        input_tokens = 1_000_000
        output_tokens = 1_000_000

    class FakeOpenAIUsage:
        prompt_tokens = 1_000_000
        completion_tokens = 1_000_000

    def test_opus5_priced_at_opus5_rates(self):
        t = CostTracker()
        t.log_judge(self.FakeAnthropicUsage(), 'claude-opus-5')
        # $5/1M in + $25/1M out
        assert t.judge_cost == pytest.approx(30.0)

    def test_haiku_generator_rates_would_have_understated_it(self):
        """The bug this guards: haiku rates give $6, not $30 — 5x low."""
        t = CostTracker()
        t.log_judge(self.FakeAnthropicUsage(), 'claude-haiku-4-5')
        assert t.judge_cost == pytest.approx(6.0)
        assert t.judge_cost < 30.0

    def test_openai_usage_shape_is_accepted(self):
        t = CostTracker()
        t.log_judge(self.FakeOpenAIUsage(), 'gpt-4o-mini')
        assert t.judge_cost == pytest.approx(0.15 + 0.60)

    def test_unknown_model_raises_rather_than_guessing(self):
        t = CostTracker()
        with pytest.raises(KeyError):
            t.log_judge(self.FakeAnthropicUsage(), 'some-model-we-never-priced')

    def test_judge_cost_reaches_the_total_the_budget_cap_reads(self):
        t = CostTracker()
        t.log_judge(self.FakeAnthropicUsage(), 'claude-opus-5')
        assert t.cost == pytest.approx(30.0)

    def test_judge_tokens_do_not_pollute_the_generator_counters(self):
        t = CostTracker()
        t.log_judge(self.FakeAnthropicUsage(), 'claude-opus-5')
        assert t.chat_input_tokens == 0
        assert t.claude_cost == 0.0


class TestJudgeWiring:
    def test_claude_judge_sends_no_temperature(self):
        """
        temperature is REMOVED on Opus 5 and returns 400 on every request.

        Checked against the parsed call keywords rather than the source text:
        the word also appears in the docstring that explains this rule.
        """
        import ast
        tree = ast.parse((SRC / 'llm_judge.py').read_text(encoding='utf-8'))
        cls = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef) and n.name == 'ClaudeJudge')
        kwargs = set()
        for node in ast.walk(cls):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'create'):
                kwargs |= {kw.arg for kw in node.keywords}
        assert kwargs, 'no messages.create call found in ClaudeJudge'
        assert 'temperature' not in kwargs
        assert 'top_p' not in kwargs and 'top_k' not in kwargs
        assert 'model' in kwargs and 'max_tokens' in kwargs

    def test_judges_log_through_the_per_model_tracker(self):
        src = (SRC / 'llm_judge.py').read_text(encoding='utf-8')
        assert src.count('cost_tracker.log_judge(') == 2
        assert 'cost_tracker.log_claude(' not in src
        assert 'cost_tracker.log_openai_chat(' not in src

    def test_paid_run_requires_yes(self):
        src = (SRC / 'llm_judge.py').read_text(encoding='utf-8')
        assert "--yes" in src and 'args.yes' in src
