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
            def __init__(self, model_id=None, prompt_style='standard'):
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


class TestLocalHFGenerateCall:
    """
    Drives LocalHFGenerator.generate with stubs instead of a 15 GB model.

    This is the only arm that had never been run, and it did not work: in
    transformers 4.x `apply_chat_template(return_tensors='pt')` returned a
    bare tensor, in 5.x it returns a BatchEncoding, and the original code
    passed that straight into model.generate() as the first positional
    argument, where it died on `inputs_tensor.shape[0]`. A stub is enough to
    catch that, and cheap enough to keep running everywhere.
    """

    def _gen(self, monkeypatch, n_prompt=5, n_new=3):
        torch = pytest.importorskip('torch')
        from generate import LocalHFGenerator

        state = {}

        class StubTok:
            eos_token_id = 7

            def apply_chat_template(self, conv, **kw):
                state['prompt'] = conv[0]['content']
                state['add_generation_prompt'] = kw.get('add_generation_prompt')
                # The regression: the caller must ask for a dict explicitly
                # rather than depend on a default that changed between majors.
                assert kw.get('return_dict') is True
                ids = torch.arange(n_prompt).unsqueeze(0)
                return {'input_ids': ids,
                        'attention_mask': torch.ones_like(ids)}

            def decode(self, ids, skip_special_tokens=True):
                return '  answer ' + ','.join(str(int(i)) for i in ids) + ' '

        class StubModel:
            device = 'cpu'

            def generate(self, **kw):
                state['kw'] = kw
                # input_ids must arrive as a keyword tensor, not positionally
                assert 'input_ids' in kw
                total = kw['input_ids'].shape[1] + n_new
                return torch.arange(total).unsqueeze(0)

        g = object.__new__(LocalHFGenerator)
        g.tokenizer, g.model, g.torch = StubTok(), StubModel(), torch
        return g, state

    def test_returns_only_the_new_tokens(self, monkeypatch):
        """Slicing is by token count; a character slice would leak the prompt."""
        g, _ = self._gen(monkeypatch, n_prompt=5, n_new=3)
        out = g.generate('q?', ['c1', 'c2'])
        assert out == 'answer 5,6,7'          # the 3 new ids, stripped

    def test_prompt_parity_with_the_api_generators(self, monkeypatch):
        """H3 compares generators, so the prompt must be build_prompt() exactly."""
        from generate import build_prompt
        g, state = self._gen(monkeypatch)
        g.generate('who?', ['ctx a', 'ctx b'])
        assert state['prompt'] == build_prompt('who?', ['ctx a', 'ctx b'])
        assert state['add_generation_prompt'] is True

    def test_decoding_is_greedy_and_unsampled(self, monkeypatch):
        g, state = self._gen(monkeypatch)
        g.generate('q?', ['c'])
        assert state['kw']['do_sample'] is False
        assert 'temperature' not in state['kw']
        assert state['kw']['max_new_tokens'] == 256

    def test_attention_mask_is_passed_through(self, monkeypatch):
        g, state = self._gen(monkeypatch)
        g.generate('q?', ['c'])
        assert 'attention_mask' in state['kw']


class TestGeneratorCliDefaults:
    """
    B10 at the CLI layer.

    The module-level GENERATORS lists have known the open label for a while,
    but every analysis CLI still defaulted --generators to 'claude,gpt4omini'.
    That is the same failure that left 8,000 paid GPT-4o-mini rows out of the
    conditional tables: the arm is generated, the scoring modules can see it,
    and the report you actually run silently omits it.
    """

    def test_open_arm_defaults_to_every_configured_model(self, monkeypatch,
                                                         tmp_path):
        """
        None must mean ALL models, as in every other phase.

        It used to mean a hardcoded three, so run_pipeline passing None (the
        no --models case) produced 3 of the 4 embedders. The arm then
        contributes an embedder spread taken over 3 systems to a table whose
        other rows are over 4 — incomparable, not merely smaller.
        """
        import config as cfg
        import generate
        monkeypatch.setenv('RAG_CHECKPOINT_DIR', str(tmp_path))
        monkeypatch.setattr(cfg, 'CHECKPOINT_DIR', tmp_path)
        monkeypatch.setattr(generate, 'LocalHFGenerator',
                            lambda model_id=None, prompt_style='standard': type(
                                'G', (), {'generate': lambda s, q, c: 'a'})())
        from utils import load_checkpoint, save_checkpoint

        names = [c['name'] for c in cfg.EMBEDDING_MODELS]
        for m in names:
            save_checkpoint(f'retrieval_{m}_NQ', [
                {'query_id': 'q1', 'question': 'q?',
                 'retrieved_texts': ['ctx']}])

        generate.run_phase_llama({'NQ': object()}, subset=1)
        for m in names:
            assert load_checkpoint(
                f'generated_{cfg.OPEN_MODEL_LABEL}_{m}_NQ'), m

    def test_active_generators_covers_the_arms_with_checkpoints(self):
        import config
        assert config.OPEN_MODEL_LABEL in config.ACTIVE_GENERATORS
        assert 'claude' in config.ACTIVE_GENERATORS
        assert 'gpt4omini' in config.ACTIVE_GENERATORS
        # legacy label, no checkpoints — must not be queried by default
        assert 'llama3' not in config.ACTIVE_GENERATORS

    def test_every_analysis_cli_defaults_to_all_active_arms(self):
        """Parse each CLI's default rather than trusting the source string."""
        import importlib
        import config

        checked = 0
        for mod_name in ('compare_evaluators', 'perturbation_check',
                         'perturb_report', 'copying_check'):
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue          # torch/transformers absent off the GPU box
            src = __import__('inspect').getsource(mod.main)
            assert 'ACTIVE_GENERATORS' in src, mod_name
            checked += 1
        assert checked >= 1
        assert len(config.ACTIVE_GENERATORS) >= 3
