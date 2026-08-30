"""
Tests for the restored GPT-4o-mini generator (2026-08-14).

No network: the OpenAI client is replaced with a fake. What is worth testing
here is not the API call but the invariants that would quietly invalidate H3
if they broke — prompt parity across generators, correct checkpoint labelling,
vendor-separated cost accounting, and ungradable-vs-wrong error handling.
"""
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

# OpenAIGenerator.generate() imports `openai` for its exception classes, and
# faithfulness imports torch. Those skip where the dependency is absent (the
# Windows laptop) and run where it matters (gpu1). The skip is applied per
# test rather than at module level so the wiring checks — which need neither —
# always run.


class _Usage:
    def __init__(self, pi=1000, co=50):
        self.prompt_tokens = pi
        self.completion_tokens = co


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason='stop'):
        self.message = _Msg(content)
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, content, finish_reason='stop', usage=None):
        self.choices = [_Choice(content, finish_reason)]
        self.usage = usage if usage is not None else _Usage()


class _FakeCompletions:
    def __init__(self, resp):
        self._resp = resp
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


def _make_gen(resp):
    """An OpenAIGenerator whose client is a fake, bypassing __init__."""
    pytest.importorskip('openai', reason='openai SDK not installed')
    import generate
    g = generate.OpenAIGenerator.__new__(generate.OpenAIGenerator)
    g.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_FakeCompletions(resp)))
    g.model = 'gpt-4o-mini'
    return g


@pytest.fixture(autouse=True)
def _reset_cost():
    from utils import cost_tracker
    for attr in ('oai_chat_input_tokens', 'oai_chat_output_tokens',
                 'chat_input_tokens', 'chat_output_tokens', 'requests',
                 'errors'):
        setattr(cost_tracker, attr, 0)
    yield


class TestPromptParity:
    def test_identical_prompt_to_claude(self):
        """H3 compares faithfulness RANKINGS across generators. Any prompt
        difference confounds that comparison, so both arms must send the
        byte-identical string produced by build_prompt()."""
        import generate
        g = _make_gen(_Resp('Paris.'))
        g.generate('What is the capital?', ['[ctx] France info'])
        sent = g.client.chat.completions.last_kwargs
        expected = generate.build_prompt('What is the capital?',
                                         ['[ctx] France info'])
        assert sent['messages'] == [{'role': 'user', 'content': expected}]

    def test_no_system_message(self):
        """The instruction stays in the user turn on purpose — hoisting it
        into a system message would be idiomatic but asymmetric."""
        g = _make_gen(_Resp('x'))
        g.generate('q', ['c'])
        roles = [m['role'] for m in g.client.chat.completions.last_kwargs['messages']]
        assert roles == ['user']

    def test_temperature_zero(self):
        g = _make_gen(_Resp('x'))
        g.generate('q', ['c'])
        assert g.client.chat.completions.last_kwargs['temperature'] == 0

    def test_only_top_k_chunks_reach_the_prompt(self):
        """generate() is handed the slice; the driver does the slicing.
        This pins that the generator itself does not re-slice or reorder."""
        import generate
        g = _make_gen(_Resp('x'))
        chunks = [f'chunk{i}' for i in range(5)]
        g.generate('q', chunks)
        content = g.client.chat.completions.last_kwargs['messages'][0]['content']
        for i in range(5):
            assert f'[Chunk {i + 1}]: chunk{i}' in content


class TestErrorHandling:
    def test_none_content_is_ungradable_not_wrong(self):
        """A refusal/empty completion must be an '[ERROR' row so that
        correctness.py marks it ungradable. Scoring it as a wrong answer
        would deflate whichever condition hit the filter."""
        from correctness import is_ungradable
        g = _make_gen(_Resp(None, finish_reason='length'))
        out = g.generate('q', ['c'])
        assert out.startswith('[ERROR')
        assert is_ungradable({'generated_answer': out})

    def test_content_filter_is_error(self):
        g = _make_gen(_Resp('partial', finish_reason='content_filter'))
        assert g.generate('q', ['c']).startswith('[ERROR')

    def test_api_status_error_is_caught(self):
        openai = pytest.importorskip('openai')
        err = openai.APIStatusError(
            'bad key', response=types.SimpleNamespace(status_code=401,
                                                      headers={}, request=None),
            body=None)
        g = _make_gen(err)
        out = g.generate('q', ['c'])
        assert out.startswith('[ERROR')

    def test_missing_usage_does_not_crash(self):
        """Some proxies omit usage. Cost tracking must degrade, not raise."""
        g = _make_gen(_Resp('answer', usage=None))
        # _Resp substitutes a default when usage is None, so force it off:
        g.client.chat.completions._resp.usage = None
        assert g.generate('q', ['c']) == 'answer'


class TestCostAccounting:
    def test_openai_tokens_are_priced_separately_from_claude(self):
        """The two vendors have different prices. Sharing counters would
        bill GPT tokens at Anthropic rates (~7x too high on input)."""
        from utils import cost_tracker
        g = _make_gen(_Resp('answer', usage=_Usage(pi=1_000_000, co=0)))
        g.generate('q', ['c'])
        assert cost_tracker.oai_chat_input_tokens == 1_000_000
        assert cost_tracker.chat_input_tokens == 0
        assert cost_tracker.openai_chat_cost == pytest.approx(0.15)
        assert cost_tracker.claude_cost == 0.0
        assert cost_tracker.generation_cost == pytest.approx(0.15)

    def test_errors_are_counted_not_billed(self):
        from utils import cost_tracker
        g = _make_gen(_Resp(None, finish_reason='length'))
        g.generate('q', ['c'])
        assert cost_tracker.errors == 1


class TestWiring:
    def test_results_lists_the_generator(self):
        """A checkpoint written as generated_gpt4omini_* is invisible to the
        results assembly unless the label is listed."""
        import results
        assert 'gpt4omini' in results.GENERATORS

    def test_faithfulness_lists_the_generator(self):
        """Same for phase D — an unlisted generator is silently never scored."""
        pytest.importorskip('torch', reason='faithfulness imports nli/torch')
        import faithfulness
        assert 'gpt4omini' in faithfulness.GENERATORS

    def test_cgpt_is_a_paid_phase(self):
        import run_pipeline
        assert 'cgpt' in run_pipeline.ALL_PHASES
        assert 'cgpt' in run_pipeline.PAID_PHASES, \
            'an unlisted paid phase could start spending without --yes'

    def test_correctness_globs_pick_up_gpt_checkpoints(self):
        import fnmatch

        from correctness import GENERATION_GLOBS
        name = 'generated_gpt4omini_BGE-M3_NQ.pkl'
        assert any(fnmatch.fnmatch(name, g) for g in GENERATION_GLOBS)


# ── the open-weight arm is no longer pinned to Llama-3 ───────────────────────
class TestOpenWeightGenerator:

    def test_llama_subclass_keeps_its_own_id(self):
        import config
        from generate import Llama3Generator, LocalHFGenerator
        assert issubclass(Llama3Generator, LocalHFGenerator)
        assert Llama3Generator.MODEL_ID == 'meta-llama/Meta-Llama-3-8B-Instruct'
        assert LocalHFGenerator.MODEL_ID is None
        assert config.OPEN_MODEL_ID and config.OPEN_MODEL_LABEL

    def test_label_scopes_the_checkpoint_key(self, monkeypatch, tmp_path):
        """A Qwen run and a Llama run must not write to the same checkpoint."""
        import config as cfg
        import generate
        monkeypatch.setenv('RAG_CHECKPOINT_DIR', str(tmp_path))
        monkeypatch.setattr(cfg, 'CHECKPOINT_DIR', tmp_path)

        seen = []

        class FakeGen:
            def __init__(self, model_id=None):
                seen.append(model_id)

            def generate(self, question, chunks):
                return 'answer'

        monkeypatch.setattr(generate, 'LocalHFGenerator', FakeGen)
        from utils import load_checkpoint, save_checkpoint
        save_checkpoint('retrieval_m1_NQ', [
            {'query_id': 'q1', 'question': 'q?', 'retrieved_texts': ['ctx']}])

        generate.run_phase_llama({'NQ': object()}, model_names=['m1'], subset=1)
        assert seen == [cfg.OPEN_MODEL_ID]
        rows = load_checkpoint(f'generated_{cfg.OPEN_MODEL_LABEL}_m1_NQ')
        assert rows and rows[0]['generator'] == cfg.OPEN_MODEL_LABEL
        assert load_checkpoint('generated_llama3_m1_NQ') is None

    def test_open_arm_is_visible_to_every_analysis(self):
        """Every module that fans out over generators must know the label,
        or the arm gets generated and then silently never scored."""
        import importlib

        import config
        label = config.OPEN_MODEL_LABEL
        checked = 0
        for mod in ('conditional', 'results', 'faithfulness',
                    'perturbation_check', 'claim_faithfulness'):
            try:
                m = importlib.import_module(mod)
            except ImportError:
                continue          # torch/transformers absent off the GPU box
            assert label in m.GENERATORS, mod
            checked += 1
        assert checked >= 2
