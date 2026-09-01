"""
Tests for the 2026-08-14 fixes:
  - textnorm: the containment rule shared by loader, qrels and correctness
  - checkpoint scoping: N=50 and N=1000 must not collide
  - TOST: paired equivalence testing for "matched retrieval quality"

All offline: no model downloads, no API calls.
"""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

import textnorm  # noqa: E402


# ── textnorm ──────────────────────────────────────────────────────
class TestTextNorm:
    def test_token_joined_possessive_matches(self):
        """The 187-row NQ artifact: `Röntgen 's` vs `Röntgen's`."""
        ctx = "Wilhelm Conrad Röntgen 's discovery of X-rays in 1895"
        assert textnorm.contains(ctx, "Wilhelm Conrad Röntgen's")

    def test_unicode_is_preserved(self):
        """An ASCII-only character class would eat the ö and break this."""
        assert textnorm.squash('Röntgen') == 'röntgen'
        assert textnorm.contains('the Röntgen prize', 'Röntgen')

    def test_number_spacing_matches(self):
        assert textnorm.contains('weighs 1,020 - 1,080 kg total',
                                 '1,020–1,080 kg')

    def test_punctuation_becomes_space_not_deletion(self):
        """Deleting punctuation glues tokens and invents matches.

        `28.0.0.137` must not be findable inside a context that merely
        contains the digit run `2800137`.
        """
        assert textnorm.squash('28.0.0.137') == '28 0 0 137'
        assert not textnorm.contains('build 2800137 released', '28.0.0.137')

    def test_empty_needle_is_never_contained(self):
        assert not textnorm.contains('anything at all', '')
        assert not textnorm.contains('anything at all', None)
        assert not textnorm.contains('anything', '   ...  ')

    def test_empty_haystack(self):
        assert not textnorm.contains('', 'something')
        assert not textnorm.contains(None, 'something')

    def test_case_insensitive(self):
        assert textnorm.contains('The CAPITAL of France', 'capital')

    def test_contains_any(self):
        assert textnorm.contains_any('a b c', ['zz', 'b'])
        assert not textnorm.contains_any('a b c', ['zz', 'yy'])
        assert not textnorm.contains_any('a b c', [])

    def test_match_is_on_tokens_not_characters(self):
        """
        squash() normalises correctly, but the test used to be a raw substring
        test on the result, so a gold answer matched inside a longer word or a
        longer number. Every false positive is load-bearing: it retains an NQ
        query whose answer is not in its own window, marks a chunk relevant
        that does not carry the gold span (the B6 bug), and grades an answer
        correct on a coincidence. Short answers — years, counts, single
        tokens — are the ones that matched spuriously.
        """
        assert not textnorm.contains('the budget was 10000 dollars', '1000')
        assert not textnorm.contains('he was born in 19051 census district',
                                     '1905')
        assert not textnorm.contains('Alice went to Paris', 'Ali')
        assert not textnorm.contains('the state artifact', 'art')

    def test_token_boundary_does_not_break_the_nq_tokenisation_case(self):
        """The behaviour the module exists for must survive the fix."""
        assert textnorm.contains('wilhelm conrad röntgen s', 'Röntgen')

    def test_contains_any_matches_on_tokens_too(self):
        assert not textnorm.contains_any('the budget was 10000 dollars',
                                         ['1000'])
        assert textnorm.contains_any('the budget was 10000 dollars', ['10000'])

    def test_squad_normalize_is_shared_with_correctness(self):
        """One SQuAD normalizer, so abstention.py can use it without
        importing correctness."""
        import correctness
        assert correctness.normalize_answer is textnorm.squad_normalize
        assert textnorm.squad_normalize("The Röntgen's!") == 'röntgens'
        assert not textnorm.contains_any('a b c', None)

    def test_substring_of_a_word_is_not_a_match(self):
        """Changed 2026-09-01. This test used to assert the opposite.

        The old rule was character-level containment on the squashed string,
        justified as "gold spans are frequently sub-token". They are not:
        squash() splits on punctuation, so the sub-token cases that actually
        occur ("Ren" in "Kylo Ren", "Röntgen" in "Röntgen 's") are separate
        TOKENS and still match. What character containment additionally bought
        was `nation` inside `international` and `1000` inside `10000`, which
        is a false positive everywhere the function is used.
        """
        assert not textnorm.contains('international', 'nation')
        assert textnorm.contains('Kylo Ren fought back', 'Ren')


# ── correctness uses the same rule ────────────────────────────────
class TestCorrectnessContainment:
    def test_verbose_answer_with_possessive_is_correct(self):
        """Regression: normalize_answer deleted punctuation and glued
        `Röntgen's` into `röntgens`, which then failed against the
        generated `Röntgen 's` -> `röntgen s`."""
        from correctness import contains_answer
        pred = ("Based on the provided context, Wilhelm Conrad Röntgen 's "
                "work earned the first Nobel Prize in Physics.")
        assert contains_answer(pred, ["Wilhelm Conrad Röntgen's"])

    def test_empty_prediction_is_not_correct(self):
        from correctness import contains_answer
        assert not contains_answer('', ['anything'])
        assert not contains_answer(None, ['anything'])

    def test_none_golds_are_skipped(self):
        from correctness import contains_answer
        assert not contains_answer('some answer', [None])


# ── checkpoint scoping ────────────────────────────────────────────
class TestCheckpointScope:
    def test_scope_separates_n(self, tmp_path, monkeypatch):
        """The 2026-08-14 bug: N=50 and N=1000 shared every key, so the
        full run skipped all work and exited 0 on pilot data."""
        import config
        import utils

        monkeypatch.delenv('RAG_CHECKPOINT_DIR', raising=False)
        monkeypatch.setattr(config, 'CHECKPOINT_ROOT', tmp_path)

        utils.set_scope(50)
        utils.save_checkpoint('retrieval_BGE-M3_NQ', ['pilot'])
        assert utils.checkpoint_exists('retrieval_BGE-M3_NQ')

        utils.set_scope(1000)
        assert not utils.checkpoint_exists('retrieval_BGE-M3_NQ'), \
            'full run must not see the pilot checkpoint'

        utils.save_checkpoint('retrieval_BGE-M3_NQ', ['full'])
        assert utils.load_checkpoint('retrieval_BGE-M3_NQ') == ['full']

        utils.set_scope(50)
        assert utils.load_checkpoint('retrieval_BGE-M3_NQ') == ['pilot'], \
            'pilot checkpoint must survive the full run untouched'

    def test_explicit_dir_override_wins(self, tmp_path, monkeypatch):
        import config
        import utils

        monkeypatch.setenv('RAG_CHECKPOINT_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'CHECKPOINT_DIR', tmp_path)
        utils.set_scope(1000)
        assert config.CHECKPOINT_DIR == tmp_path


# ── TOST ──────────────────────────────────────────────────────────
class TestTOST:
    def test_identical_scores_are_equivalent(self):
        from results import tost_equivalence
        a = [0.9, 0.8, 0.7, 0.95, 0.6] * 20
        r = tost_equivalence(a, list(a), margin=0.02)
        assert r['equivalent']
        assert r['mean_diff'] == 0.0

    def test_large_true_difference_is_not_equivalent(self):
        from results import tost_equivalence
        a = [0.90] * 200
        b = [0.50] * 200
        r = tost_equivalence(a, b, margin=0.02)
        assert not r['equivalent']
        assert r['p_tost'] == pytest.approx(1.0, abs=1e-6)

    def test_tiny_difference_with_large_n_is_equivalent(self):
        """The case the paper needs: a real but negligible gap."""
        import numpy as np
        from results import tost_equivalence
        rng = np.random.default_rng(0)
        base = rng.normal(0.8, 0.05, 1000)
        a = base
        b = base + 0.001  # 0.1 NDCG points apart
        r = tost_equivalence(a, b, margin=0.02)
        assert r['equivalent'], r

    def test_small_n_cannot_claim_equivalence(self):
        """Underpowered NOISY data must FAIL to establish equivalence rather
        than pass it by default — the whole point of inverting the null."""
        import numpy as np
        from results import tost_equivalence
        rng = np.random.default_rng(1)
        a = rng.normal(0.8, 0.15, 5)
        b = a + rng.normal(0.0, 0.15, 5)
        r = tost_equivalence(a, b, margin=0.02)
        assert not r['equivalent'], r

    def test_constant_offset_is_equivalent_at_any_n(self):
        """Counterpart to the test above, and the reason it needed noise:
        if every query differs by the SAME 0.001, the models really are
        equivalent — there is no sampling uncertainty left to worry about."""
        from results import tost_equivalence
        a = [0.8, 0.6, 0.95, 0.72, 0.5]
        b = [x + 0.001 for x in a]
        assert tost_equivalence(a, b, margin=0.02)['equivalent']

    def test_single_query_never_claims_equivalence(self):
        """n=1 has no variance to estimate; it must not pass by default."""
        from results import tost_equivalence
        r = tost_equivalence([0.8], [0.8], margin=0.02)
        assert not r['equivalent'], r

    def test_is_symmetric_up_to_sign(self):
        import numpy as np
        from results import tost_equivalence
        rng = np.random.default_rng(2)
        a = rng.normal(0.8, 0.05, 300)
        b = rng.normal(0.81, 0.05, 300)
        ab = tost_equivalence(a, b)
        ba = tost_equivalence(b, a)
        assert ab['equivalent'] == ba['equivalent']
        assert ab['mean_diff'] == pytest.approx(-ba['mean_diff'], abs=1e-9)
        assert ab['p_tost'] == pytest.approx(ba['p_tost'], abs=1e-9)

    def test_unpaired_lengths_rejected(self):
        from results import tost_equivalence
        with pytest.raises(AssertionError):
            tost_equivalence([0.1, 0.2, 0.3], [0.1, 0.2])

    def test_wider_margin_is_easier(self):
        import numpy as np
        from results import tost_equivalence
        rng = np.random.default_rng(3)
        a = rng.normal(0.80, 0.10, 400)
        b = a + 0.03
        assert not tost_equivalence(a, b, margin=0.02)['equivalent']
        assert tost_equivalence(a, b, margin=0.10)['equivalent']


# ── qrels no longer silently falls back ───────────────────────────
class TestQrelsAnswerBearing:
    def test_token_joined_gold_sentence_matches_its_chunk(self):
        """Before textnorm this returned no match and fell back to marking
        the whole document relevant — B6 all over again."""
        # build_qrels chunks documents, which needs a HF tokenizer.
        pytest.importorskip('transformers',
                            reason='build_qrels chunks via a HF tokenizer')
        from datasets_loader import Document, LoadedDataset, QASample
        import retrieval_eval

        ctx = ("Wilhelm Conrad Röntgen 's discovery of X-rays in 1895 "
               "earned him the first Nobel Prize in Physics .")
        sample = QASample(query_id='q1', question='who?',
                          answer="Wilhelm Conrad Röntgen's",
                          gold_context=ctx, dataset='NQ')
        doc = Document(doc_id='q1_doc', text=ctx, gold_for={'q1'},
                       gold_sentences={'q1': ["Wilhelm Conrad Röntgen's"]})
        loaded = LoadedDataset('SYNTXT', [sample], [doc])

        qrels = retrieval_eval.build_qrels(loaded)
        assert qrels['q1'], 'gold chunk should be marked relevant'
