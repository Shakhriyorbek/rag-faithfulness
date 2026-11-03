"""
Tests for src/minicheck_faithfulness.py, the MiniCheck pass over the grid.

Pinned here, offline and without the model:

  - the defaults name the matched grid explicitly. A default that admitted
    QASPER or jina, or dropped a generator, would change which cells enter the
    system comparison without any line of output saying so (CLAUDE.md B30);
  - the answer every evaluator reads is markdown-stripped (decision D1), and
    error rows are skipped rather than scored;
  - a cell resumes from its partial checkpoint without re-scoring, writes rows
    in generation order, and removes the partial file;
  - a non-finite score stops the run instead of being stored;
  - results.FAITH_SOURCES maps both aggregates to the one checkpoint, so the
    comparison code reads them like any other evaluator.
"""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

import config                                                  # noqa: E402
import minicheck_faithfulness as mf                             # noqa: E402
from results import FAITH_SOURCES                               # noqa: E402
from utils import checkpoint_exists, load_checkpoint, save_checkpoint  # noqa: E402


class FakeScorer:
    def __init__(self, value=0.5):
        self.value = value
        self.calls = []

    def score(self, chunks, answer):
        self.calls.append(answer)
        return {'minicheck': self.value, 'minicheck_sent': self.value / 2}


@pytest.fixture
def ckdir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'CHECKPOINT_DIR', tmp_path)
    return tmp_path


def _gens(n):
    return [{'query_id': f'q{i}', 'generated_answer': f'**Answer** {i}.',
             'retrieved_texts': [f'chunk {j}' for j in range(7)]}
            for i in range(n)]


def test_defaults_are_the_matched_grid():
    assert mf.DATASETS == ['NQ', 'HotpotQA']
    assert mf.MODELS == ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct',
                         'text-embedding-3-small']
    assert mf.GENERATORS == list(config.ACTIVE_GENERATORS)
    assert len(mf.GENERATORS) == 4


def test_faith_sources_read_both_aggregates_from_one_file():
    assert FAITH_SOURCES['minicheck'] == ('minicheck_scores', 'minicheck')
    assert FAITH_SOURCES['minicheck_sent'] == ('minicheck_scores',
                                               'minicheck_sent')
    for m in ('nli', 'align', 'claim'):
        assert m in FAITH_SOURCES


def test_answer_is_stripped_and_errors_are_skipped():
    assert mf._answer({'generated_answer': '**Bob** Seger'}) == 'Bob Seger'
    assert mf._answer({'generated_answer': '[ERROR: 429]'}) is None
    assert mf._answer({'generated_answer': None}) is None
    assert mf._chunks({'retrieved_texts': list('abcdefg')}) == list('abcde')


def test_run_cell_resumes_orders_and_cleans_up(ckdir):
    gens = _gens(5)
    gens[2]['generated_answer'] = '[ERROR: timeout]'
    save_checkpoint('generated_claude_BGE-M3_NQ', gens)
    save_checkpoint('minicheck_scores_claude_BGE-M3_NQ_partial',
                    [{'query_id': 'q3', 'minicheck': 0.9, 'minicheck_sent': 0.8}])
    fake = FakeScorer(0.5)
    mf.run_cell(fake, 'claude', 'BGE-M3', 'NQ')
    rows = load_checkpoint('minicheck_scores_claude_BGE-M3_NQ')
    assert [r['query_id'] for r in rows] == ['q0', 'q1', 'q3', 'q4']
    assert rows[2]['minicheck'] == 0.9          # resumed, not re-scored
    assert fake.calls == ['Answer 0.', 'Answer 1.', 'Answer 4.']
    assert not checkpoint_exists('minicheck_scores_claude_BGE-M3_NQ_partial')
    mf.run_cell(FakeScorer(0.1), 'claude', 'BGE-M3', 'NQ')   # finished: skip
    assert load_checkpoint('minicheck_scores_claude_BGE-M3_NQ') == rows


def test_non_finite_score_stops_the_run(ckdir):
    save_checkpoint('generated_qwen_BGE-M3_NQ', _gens(2))
    with pytest.raises(RuntimeError, match='non-finite'):
        mf.run_cell(FakeScorer(float('nan')), 'qwen', 'BGE-M3', 'NQ')
    assert not checkpoint_exists('minicheck_scores_qwen_BGE-M3_NQ')
