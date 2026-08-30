"""
Offline tests for the modules added in response to Berend's 2026-08-11 letter.

No network, no API, no GPU — every test runs on the laptop.

Run:  python -m pytest tests/test_berend_conditions.py -q
"""
import math
import random
import sys
from itertools import combinations
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from context_ablation import CONDITIONS, FIXED_SLOT_CONDITIONS, build_context
from correctness import is_abstention, is_ungradable, score_records
from doc_utility import shapley_values, subsets_for_mode


# ── correctness: the ungradable rule ─────────────────────────────────────────
def test_api_error_is_ungradable_not_wrong():
    """
    A 429 is not the model being wrong. Grading it as incorrect would let a
    rate-limit episode look like reduced accuracy for whichever condition
    happened to hit it.
    """
    out = score_records([{'generated_answer': '[ERROR: APIStatusError 429: x]',
                          'answer': 'Paris'}])[0]
    assert out['correct'] is None
    assert out['correct_em'] is None and out['correct_f1'] is None


def test_missing_oracle_answer_is_ungradable_not_wrong():
    """C2 records queries with no gold evidence as None; grading those as
    incorrect would understate the oracle ceiling."""
    out = score_records([{'generated_answer': None, 'answer': 'Paris',
                          'oracle_missing': True}])[0]
    assert out['correct'] is None


def test_ungradable_predicate_matches_scoring():
    assert is_ungradable({'generated_answer': None})
    assert is_ungradable({'generated_answer': '[ERROR: x]'})
    assert not is_ungradable({'generated_answer': 'Paris'})
    assert not is_ungradable({'generated_answer': ''})   # empty is a real miss


@pytest.mark.parametrize('text,expected', [
    ('I do not know.', True),
    ('i dont know', True),
    ('I cannot answer based on the provided context.', True),
    ('Paris', False),
    ('', False),
])
def test_abstention_detection(text, expected):
    assert is_abstention(text) is expected


def test_abstention_is_incorrect_but_flagged():
    """An abstention is wrong for grading, but must stay distinguishable from
    a confident wrong answer — the C1 floor means different things either way."""
    out = score_records([{'generated_answer': 'I do not know.',
                          'answer': 'Paris'}])[0]
    assert out['correct'] is False and out['abstained'] is True


def test_confident_wrong_answer_is_not_an_abstention():
    out = score_records([{'generated_answer': 'Berlin', 'answer': 'Paris'}])[0]
    assert out['correct'] is False and out['abstained'] is False


# ── metrics: the zero-denominator hole patch 0001 left open ──────────────────
def test_nrfg_zero_retrieval_quality_is_nan_not_huge():
    """
    The old `rq = max(1e-9, retrieval_quality)` guard turned RQ=0 into a
    division by 1e-9, so nrfg(0.0, 0.78) came out as -7.8e8 and poisoned any
    mean containing it. RQ=0 is common per query — every query whose top-5
    misses all gold chunks — and the conditional analysis works per query.
    """
    import metrics
    assert math.isnan(metrics.nrfg(0.0, 0.78))
    assert math.isnan(metrics.nrfg(0.0, 0.0))


def test_nrfg_still_separates_berends_example():
    import metrics
    assert metrics.nrfg(0.9, 0.85) == pytest.approx(0.0556, abs=1e-4)
    assert metrics.nrfg(0.3, 0.25) == pytest.approx(0.1667, abs=1e-4)
    assert metrics.rfg(0.9, 0.85) == metrics.rfg(0.3, 0.25)   # RFG cannot


# ── Shapley ──────────────────────────────────────────────────────────────────
def _full_game(k, fn):
    return {S: fn(S) for size in range(k + 1)
            for S in combinations(range(k), size)}


def test_shapley_efficiency_axiom():
    """sum(phi) == v(N) - v({}) for any value function."""
    k = 5
    rng = random.Random(0)
    for _ in range(50):
        vals = _full_game(k, lambda S: rng.random())
        phi = shapley_values(vals, k)
        assert sum(phi) == pytest.approx(vals[tuple(range(k))] - vals[()])


def test_shapley_dummy_player_gets_zero():
    k = 5
    vals = _full_game(k, lambda S: 1.0 if 2 in S else 0.0)
    phi = shapley_values(vals, k)
    assert phi[2] == pytest.approx(1.0)
    assert all(p == pytest.approx(0.0) for i, p in enumerate(phi) if i != 2)


def test_shapley_symmetric_players_split_credit():
    """Two interchangeable documents, either alone sufficient -> 0.5 each."""
    k = 5
    vals = _full_game(k, lambda S: 1.0 if (0 in S or 1 in S) else 0.0)
    phi = shapley_values(vals, k)
    assert phi[0] == pytest.approx(0.5) and phi[1] == pytest.approx(0.5)


def test_shapley_prices_a_harmful_document_negatively():
    """
    A document that breaks an otherwise-correct answer must get NEGATIVE
    utility. This is the case the report calls out: relevant by the qrels,
    retrieved correctly, and actively harmful.
    """
    k = 3
    vals = _full_game(k, lambda S: 0.0 if 1 in S else (1.0 if 0 in S else 0.0))
    phi = shapley_values(vals, k)
    assert phi[1] < 0


def test_shapley_raises_on_incomplete_coverage():
    """Missing subsets must fail loudly, not silently bias every value to 0."""
    k = 3
    vals = _full_game(k, lambda S: len(S) / k)
    del vals[(0, 1)]
    with pytest.raises(KeyError):
        shapley_values(vals, k)


def test_subset_counts_per_mode():
    k = 5
    assert len(subsets_for_mode('erag', k)) == k + 1        # singletons + empty
    assert len(subsets_for_mode('loo', k)) == k + 2         # + full set
    assert len(subsets_for_mode('shapley', k)) == 2 ** k
    # eRAG and LOO are strict subsets of the Shapley enumeration, which is why
    # --mode shapley yields all three for the price of one.
    full = set(subsets_for_mode('shapley', k))
    assert set(subsets_for_mode('erag', k)) <= full
    assert set(subsets_for_mode('loo', k)) <= full


def test_empty_subset_always_present():
    """v({}) is the baseline every marginal contribution is measured against."""
    for mode in ('erag', 'loo', 'shapley'):
        assert () in subsets_for_mode(mode, 5)


# ── context ablation ─────────────────────────────────────────────────────────
class _FakePool:
    """Five gold chunks, five hard negatives, no I/O."""

    def __init__(self, n_gold=5):
        self._gold = [f'g{i}' for i in range(n_gold)]
        self._noise = [f'n{i}' for i in range(5)]

    def gold_ids(self, qid):
        return list(self._gold)

    def distractor_ids(self, qid, n, rng):
        return self._noise[:n]

    def texts(self, ids):
        return [f'text::{i}' for i in ids]


def _ctx(condition, n_gold=5, seed=0):
    return build_context(_FakePool(n_gold), 'q1', condition,
                         random.Random(seed), top_k=5)


def test_reversed_holds_content_and_changes_only_order():
    """The ordering test is only valid if the information is identical."""
    a, b = _ctx('gold_all'), _ctx('gold_reversed')
    assert a != b
    assert sorted(a) == sorted(b)


def test_shuffle_is_deterministic_for_a_given_seed():
    """Resuming a run must not silently change the context it already used."""
    assert _ctx('gold_shuffled', seed=7) == _ctx('gold_shuffled', seed=7)


@pytest.mark.parametrize('condition', sorted(FIXED_SLOT_CONDITIONS))
def test_fixed_slot_conditions_all_fill_top_k(condition):
    """
    Position and noise effects are only interpretable if context LENGTH is
    held constant — NLI entailment is sensitive to premise length.
    """
    assert len(_ctx(condition)) == 5


def test_gold_position_moves_across_the_three_position_conditions():
    first, middle, last = (_ctx('gold1_first'), _ctx('gold1_middle'),
                           _ctx('gold1_last'))
    gold = 'text::g0'
    assert first.index(gold) == 0
    assert last.index(gold) == len(last) - 1
    assert 0 < middle.index(gold) < len(middle) - 1
    # same multiset in all three: only position differs
    assert sorted(first) == sorted(middle) == sorted(last)


def test_noise_only_contains_no_gold():
    ctx = _ctx('noise_only')
    assert ctx and not any(c.startswith('text::g') for c in ctx)


def test_conditions_needing_gold_return_none_without_it():
    """A query with no gold evidence must be recorded as skipped, not
    generated under a label claiming gold was present."""
    pool = _FakePool(n_gold=0)
    for condition in CONDITIONS:
        assert build_context(pool, 'q1', condition, random.Random(0)) is None


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        _ctx('gold_sideways')


# ── the query frame: which generators and which correctness source ───────────
# Added 2026-08-30. GENERATORS omitted 'gpt4omini' until then, so the 2x2 and
# the conditional table silently described the Claude arm only and ignored
# 8,000 paid GPT-4o-mini generations.
class TestQueryFrame:

    @staticmethod
    def _setup(tmp_path, monkeypatch, judged=None):
        import config as cfg
        import conditional
        import utils
        from correctness import score_records
        monkeypatch.setenv('RAG_CHECKPOINT_DIR', str(tmp_path))
        monkeypatch.setattr(cfg, 'CHECKPOINT_DIR', tmp_path)
        monkeypatch.setattr(cfg, 'EMBEDDING_MODELS',
                            [{'name': 'm1', 'paradigm': 'contrastive'}])

        raw = [{'query_id': 'q1', 'answer': 'Paris',
                'generated_answer': 'The capital is Paris.'},
               {'query_id': 'q2', 'answer': 'Paris',
                'generated_answer': 'The capital is Berlin.'}]
        for gen in ('claude', 'gpt4omini'):
            utils.save_checkpoint(f'generated_{gen}_m1_NQ', raw)
            utils.save_checkpoint(f'generated_{gen}_m1_NQ_scored',
                                  score_records(raw))
            utils.save_checkpoint(f'nli_scores_{gen}_m1_NQ',
                                  [{'query_id': 'q1', 'nli_max': 0.9},
                                   {'query_id': 'q2', 'nli_max': 0.8}])
        if judged is not None:
            utils.save_checkpoint('generated_claude_m1_NQ_judged_claude', judged)

        monkeypatch.setattr(conditional, 'per_query_retrieval',
                            lambda loaded, model, ds, k=None: {
                                'q1': {'ndcg': 1.0, 'recall': 1.0, 'mrr': 1.0,
                                       'hit': True},
                                'q2': {'ndcg': 0.0, 'recall': 0.0, 'mrr': 0.0,
                                       'hit': False}})
        return conditional

    def test_both_paid_generator_arms_reach_the_frame(self, tmp_path, monkeypatch):
        conditional = self._setup(tmp_path, monkeypatch)
        assert 'gpt4omini' in conditional.GENERATORS
        df = conditional.build_query_frame({'NQ': object()})
        assert set(df['generator']) == {'claude', 'gpt4omini'}
        assert len(df) == 4

    def test_frame_records_where_each_label_came_from(self, tmp_path, monkeypatch):
        conditional = self._setup(tmp_path, monkeypatch, judged=[
            {'query_id': 'q1', 'correct': False, 'abstained': False,
             'source_ungradable': False}])
        df = conditional.build_query_frame({'NQ': object()}, correct_source='judge')
        judged = df[df['correct_source'] == 'judge:claude']
        assert len(judged) == 1                       # only the claude arm
        assert judged.iloc[0]['correct'] is False or judged.iloc[0]['correct'] == False
        # the gpt arm has no judged checkpoint and falls back, visibly
        assert (df['correct_source'] == 'contains').sum() == 3

    def test_source_changes_the_2x2(self, tmp_path, monkeypatch):
        """The whole point: switching the source must move the reported cells."""
        conditional = self._setup(tmp_path, monkeypatch, judged=[
            {'query_id': 'q1', 'correct': False, 'abstained': False,
             'source_ungradable': False}])
        con = conditional.build_query_frame({'NQ': object()},
                                            correct_source='contains')
        jud = conditional.build_query_frame({'NQ': object()},
                                            correct_source='judge')
        c = con[(con['generator'] == 'claude') & (con['query_id'] == 'q1')]
        j = jud[(jud['generator'] == 'claude') & (jud['query_id'] == 'q1')]
        assert bool(c.iloc[0]['correct']) is True
        assert bool(j.iloc[0]['correct']) is False

    def test_bad_source_raises_before_any_work(self, tmp_path, monkeypatch):
        conditional = self._setup(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            conditional.build_query_frame({'NQ': object()}, correct_source='vibes')


# ── the tables must not pool datasets ────────────────────────────────────────
# Added 2026-08-30. Pooling NQ and HotpotQA made faith_gap look like a
# generator effect (Claude negative, GPT-4o-mini positive). Split by dataset,
# NQ is flat for both and HotpotQA carries all of it in opposite directions.
class TestDatasetsAreNotPooled:

    @staticmethod
    def _frame(tmp_path, monkeypatch):
        import config as cfg
        import pandas as pd
        monkeypatch.setenv('RAG_CHECKPOINT_DIR', str(tmp_path))
        monkeypatch.setattr(cfg, 'CHECKPOINT_DIR', tmp_path)

        rows = []
        # ds A: wrong answers are LESS faithful. ds B: MORE. Pooled ~ 0.
        for ds, f_correct, f_wrong in (('A', 0.90, 0.60), ('B', 0.60, 0.90)):
            for i in range(10):
                for correct, faith in ((True, f_correct), (False, f_wrong)):
                    rows.append({
                        'query_id': f'{ds}{i}{correct}', 'dataset': ds,
                        'model': 'm1', 'paradigm': 'contrastive',
                        'generator': 'claude', 'condition': 'rag',
                        'ndcg': 1.0, 'recall': 1.0, 'hit': True,
                        'correct': correct, 'correct_f1': 0.0,
                        'abstained': False, 'correct_source': 'contains',
                        'faithfulness': faith})
        return pd.DataFrame(rows)

    def test_faith_gap_keeps_the_opposite_signs(self, tmp_path, monkeypatch):
        from conditional import conditional_faithfulness
        cf = conditional_faithfulness(self._frame(tmp_path, monkeypatch))
        assert set(cf['dataset']) == {'A', 'B'}
        gaps = dict(zip(cf['dataset'], cf['faith_gap']))
        assert gaps['A'] < 0 and gaps['B'] > 0
        # the pooled average would have been ~0 and reported nothing
        assert abs(gaps['A'] + gaps['B']) < 1e-9

    def test_2x2_is_emitted_per_dataset_plus_all(self, tmp_path, monkeypatch):
        from conditional import necessity_sufficiency
        ns = necessity_sufficiency(self._frame(tmp_path, monkeypatch))
        assert set(ns['dataset']) == {'A', 'B', 'ALL'}
        # `share` is within-dataset, so each block sums to 1
        for ds in ('A', 'B', 'ALL'):
            assert abs(ns[ns['dataset'] == ds]['share'].sum() - 1.0) < 1e-6

    def test_answered_share_excludes_refusals(self, tmp_path, monkeypatch):
        """The hit x incorrect cell is mostly abstention; its size alone is
        not evidence that retrieval was insufficient."""
        import pandas as pd
        from conditional import necessity_sufficiency
        df = self._frame(tmp_path, monkeypatch)
        df.loc[(df['dataset'] == 'A') & (~df['correct']), 'abstained'] = True
        ns = necessity_sufficiency(df)
        cell = ns[(ns['dataset'] == 'A') & (ns['retrieval'] == 'hit')
                  & (ns['answer'] == 'incorrect')].iloc[0]
        assert cell['share'] == 0.5 and cell['share_answered'] == 0.0
        assert cell['n_abstained'] == 10
