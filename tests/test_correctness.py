"""
Unit tests for correctness scoring and the metric guards from the 2026-08-11 audit.

Run:  python -m pytest tests/test_correctness.py -q
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from correctness import (CORRECT_F1_THRESHOLD, best_f1, exact_match,
                         normalize_answer, score_records, token_f1)


# ── normalization ────────────────────────────────────────────────────────────
@pytest.mark.parametrize('raw,want', [
    ('The Beatles', 'beatles'),
    ('  a   CAT.  ', 'cat'),
    ('An Apple, Inc.', 'apple inc'),
    ('', ''),
])
def test_normalize(raw, want):
    assert normalize_answer(raw) == want


def test_normalize_handles_none():
    assert normalize_answer(None) == ''


# ── exact match ──────────────────────────────────────────────────────────────
def test_em_ignores_articles_case_punctuation():
    assert exact_match('The  Beatles!', ['beatles'])


def test_em_matches_any_gold_alternative():
    assert exact_match('JFK', ['John F. Kennedy', 'JFK'])
    assert not exact_match('Nixon', ['John F. Kennedy', 'JFK'])


# ── token F1 ─────────────────────────────────────────────────────────────────
def test_f1_identical_is_one():
    assert token_f1('the quick brown fox', 'quick brown fox') == 1.0


def test_f1_disjoint_is_zero():
    assert token_f1('cats', 'dogs') == 0.0


def test_f1_partial_overlap():
    # pred 3 tokens, gold 2 tokens, 2 shared -> P=2/3, R=1, F1=0.8
    assert token_f1('paris in france', 'paris france') == pytest.approx(0.8)


def test_f1_both_empty_is_one_not_nan():
    """Guards the audit's D1 class of bug: no silent nan into downstream means."""
    v = token_f1('', '')
    assert v == 1.0 and not math.isnan(v)


def test_f1_one_empty_is_zero():
    assert token_f1('', 'something') == 0.0
    assert token_f1('something', '') == 0.0


def test_best_f1_no_golds_is_zero_not_nan():
    v = best_f1('anything', [])
    assert v == 0.0 and not math.isnan(v)


# ── record scoring ───────────────────────────────────────────────────────────
def _rec(pred, gold, **kw):
    return dict(generated_answer=pred, answer=gold, **kw)


def test_score_marks_exact_match_correct():
    out = score_records([_rec('Paris', 'Paris')])[0]
    assert out['correct_em'] is True and out['correct'] is True


def test_score_graded_match_above_threshold():
    out = score_records([_rec('Paris, France', 'Paris France')])[0]
    assert out['correct_em'] is True or out['correct_f1'] >= CORRECT_F1_THRESHOLD
    assert out['correct'] is True


def test_score_wrong_answer_is_incorrect():
    out = score_records([_rec('Berlin', 'Paris')])[0]
    assert out['correct'] is False and out['correct_f1'] == 0.0


def test_missing_gold_is_none_not_false():
    """
    Critical: absent gold must not be silently counted as 'incorrect'.
    That is exactly the failure mode of the nan-clamp bug in metrics.rfg —
    a missing value quietly becoming a real one and shifting the mean.
    """
    out = score_records([_rec('Paris', None)])[0]
    assert out['correct'] is None
    assert out['correct_em'] is None and out['correct_f1'] is None


def test_absent_prediction_is_ungradable_not_incorrect():
    """
    REVISED 2026-08-13. This test previously asserted correct is False.

    Nothing in the pipeline writes generated_answer=None to mean "the model
    answered badly" — generation always returns a string, '' at worst. None is
    written by exactly one place: conditions.py, for a C2 query with no gold
    evidence, i.e. a question that was never asked. Grading those as incorrect
    pushed them into the 'incorrect' cell and understated the oracle CEILING,
    which is the one number C2 exists to produce.
    """
    out = score_records([_rec(None, 'Paris')])[0]
    assert out['correct'] is None


def test_empty_string_prediction_is_incorrect():
    """An empty answer IS a real miss — the model was asked and said nothing."""
    out = score_records([_rec('', 'Paris')])[0]
    assert out['correct'] is False and out['correct_f1'] == 0.0


def test_list_valued_gold_answers():
    out = score_records([_rec('JFK', ['John F. Kennedy', 'JFK'])])[0]
    assert out['correct'] is True


def test_score_records_does_not_mutate_input():
    src = [_rec('Paris', 'Paris')]
    score_records(src)
    assert 'correct' not in src[0]


def test_alternative_gold_key_is_found():
    out = score_records([dict(generated_answer='Paris', gold_answers=['Paris'])])[0]
    assert out['correct'] is True


# ── the conditional the whole module exists to enable ────────────────────────
def test_conditional_faithfulness_split_is_computable():
    recs = score_records([
        _rec('Paris', 'Paris', faithfulness=0.9),
        _rec('Berlin', 'Paris', faithfulness=0.95),   # faithful to wrong context
        _rec('Rome', 'Rome', faithfulness=0.4),
    ])
    corr = [r['faithfulness'] for r in recs if r['correct']]
    inco = [r['faithfulness'] for r in recs if r['correct'] is False]
    assert len(corr) == 2 and len(inco) == 1
    # The pooled mean hides that the single most 'faithful' answer is wrong.
    pooled = sum(r['faithfulness'] for r in recs) / len(recs)
    assert max(inco) > pooled


# ── audit S1-1: nan must propagate through rfg/nrfg ──────────────────────────
def test_rfg_propagates_nan():
    from legacy import rfg as metrics
    assert math.isnan(metrics.rfg(0.9, float('nan')))
    assert math.isnan(metrics.rfg(float('nan'), 0.9))


def test_nrfg_propagates_nan_and_none():
    from legacy import rfg as metrics
    assert math.isnan(metrics.nrfg(0.9, float('nan')))
    assert math.isnan(metrics.nrfg(0.9, None))


def test_rfg_normal_values_unchanged():
    from legacy import rfg as metrics
    assert metrics.rfg(0.9, 0.78) == pytest.approx(0.12)
    # Berend's discriminating example must still separate
    assert metrics.nrfg(0.9, 0.85) == pytest.approx(0.0556, abs=1e-4)
    assert metrics.nrfg(0.3, 0.25) == pytest.approx(0.1667, abs=1e-4)


# ── load_correctness: which source supplies `correct` ────────────────────────
# Added 2026-08-30. Until then llm_judge.py wrote `*_judged_*` and nothing read
# it: conditional.py and results.py went straight to `*_scored`, so grading the
# full grid would have cost ~$32 and moved no number in the paper.
class TestLoadCorrectness:

    @staticmethod
    def _fixture(tmp_path, monkeypatch, judged=None):
        import config as cfg
        import utils
        from correctness import score_records
        monkeypatch.setenv('RAG_CHECKPOINT_DIR', str(tmp_path))
        monkeypatch.setattr(cfg, 'CHECKPOINT_DIR', tmp_path)
        raw = [
            # containment right, EM wrong — the B7 case
            {'query_id': 'q1', 'answer': 'Paris',
             'generated_answer': 'Based on the context, the capital is Paris.'},
            # plainly wrong
            {'query_id': 'q2', 'answer': 'Paris',
             'generated_answer': 'The capital is Berlin.'},
            # abstention
            {'query_id': 'q3', 'answer': 'Paris',
             'generated_answer': 'I cannot answer based on the provided context.'},
            # unusable source row -> correctness undefined
            {'query_id': 'q4', 'answer': 'Paris',
             'generated_answer': '[ERROR: RateLimit]'},
        ]
        utils.save_checkpoint('gen', raw)
        utils.save_checkpoint('gen_scored', score_records(raw))
        if judged is not None:
            utils.save_checkpoint('gen_judged_claude', judged)
        return raw

    def test_sources_disagree_as_expected(self, tmp_path, monkeypatch):
        from correctness import load_correctness
        self._fixture(tmp_path, monkeypatch)
        con, _ = load_correctness('gen', source='contains')
        em, _ = load_correctness('gen', source='em')
        assert con['q1']['correct'] is True     # containment finds "Paris"
        assert em['q1']['correct'] is False     # EM cannot, because of B7
        assert con['q2']['correct'] is False

    def test_ungradable_source_row_is_none_never_false(self, tmp_path, monkeypatch):
        from correctness import load_correctness
        self._fixture(tmp_path, monkeypatch)
        for src in ('contains', 'em', 'f1', 'judge'):
            by, cov = load_correctness('gen', source=src)
            assert by['q4']['correct'] is None, src
            assert cov['ungradable'] == 1, src

    def test_judge_overrides_heuristic_and_is_counted(self, tmp_path, monkeypatch):
        from correctness import load_correctness
        self._fixture(tmp_path, monkeypatch, judged=[
            {'query_id': 'q1', 'correct': False, 'abstained': False,
             'source_ungradable': False},
        ])
        by, cov = load_correctness('gen', source='judge')
        # containment said True; the judge is authoritative
        assert by['q1']['correct'] is False
        assert by['q1']['correct_source'] == 'judge:claude'
        assert cov['judge'] == 1
        # every other row falls back, and the fallback is visible
        assert by['q2']['correct_source'] == 'contains'
        assert cov['heuristic'] == 2 and cov['ungradable'] == 1

    def test_judge_side_failure_falls_back_rather_than_dropping(self, tmp_path,
                                                               monkeypatch):
        """A rate-limited judge call is retryable; it must not delete a query."""
        from correctness import load_correctness
        self._fixture(tmp_path, monkeypatch, judged=[
            {'query_id': 'q1', 'correct': None, 'abstained': None,
             'source_ungradable': False, 'raw': '[ERROR: 429]'},
        ])
        by, cov = load_correctness('gen', source='judge')
        assert by['q1']['correct'] is True          # containment, not None
        assert by['q1']['correct_source'] == 'contains'
        assert cov['judge'] == 0

    def test_legacy_ungradable_field_still_understood(self, tmp_path, monkeypatch):
        """Checkpoints written before the field rename used `ungradable`."""
        from correctness import load_correctness
        self._fixture(tmp_path, monkeypatch, judged=[
            {'query_id': 'q2', 'correct': None, 'ungradable': True,
             'raw': '[ERROR: AuthenticationError 401]'},
        ])
        by, _ = load_correctness('gen', source='judge')
        assert by['q2']['correct'] is None

    def test_abstention_survives_every_source(self, tmp_path, monkeypatch):
        """faith_gap flips sign if abstentions leak into 'incorrect'."""
        from correctness import load_correctness
        self._fixture(tmp_path, monkeypatch, judged=[
            {'query_id': 'q3', 'correct': False, 'abstained': True,
             'source_ungradable': False},
        ])
        for src in ('contains', 'em', 'f1', 'judge'):
            by, _ = load_correctness('gen', source=src)
            assert by['q3']['abstained'] is True, src
            assert by['q3']['correct'] is False, src

    def test_unknown_source_raises(self, tmp_path, monkeypatch):
        from correctness import load_correctness
        self._fixture(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            load_correctness('gen', source='vibes')

    def test_missing_checkpoint_is_empty_not_an_error(self, tmp_path, monkeypatch):
        from correctness import load_correctness
        self._fixture(tmp_path, monkeypatch)
        by, cov = load_correctness('nope', source='judge')
        assert by == {} and not cov

    def test_judged_checkpoints_are_not_rescored(self, tmp_path, monkeypatch):
        """`generated_*_judged_*` matches the generation glob but holds verdicts."""
        import config as cfg
        import utils
        from correctness import _generation_checkpoint_names
        monkeypatch.setenv('RAG_CHECKPOINT_DIR', str(tmp_path))
        monkeypatch.setattr(cfg, 'CHECKPOINT_DIR', tmp_path)
        utils.save_checkpoint('generated_claude_m_NQ', [{'query_id': 'q'}])
        utils.save_checkpoint('generated_claude_m_NQ_judged_claude', [{'query_id': 'q'}])
        assert _generation_checkpoint_names() == ['generated_claude_m_NQ']
