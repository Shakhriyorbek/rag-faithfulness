"""
Tests for src/copying_check.py — the verbatim-quotation control.

The Section V-B claim ("the effect is not copying") is a NEGATIVE result, so
its credibility rests entirely on the overlap measure being able to detect
copying when copying is present. These pin that, plus the two choices that
decide the number: what happens to answers too short to have a 5-gram, and
which tokenisation is used.

All offline: pure string logic, no model, no checkpoints, no API.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

import copying_check as cc  # noqa: E402


class TestOverlapDetectsCopying:
    def test_verbatim_answer_scores_one(self):
        """The measure must saturate when the answer IS the context."""
        ctx = 'the treaty was signed in Paris in 1919 by the delegates'
        assert cc.ngram_overlap(ctx, ctx) == 1.0

    def test_disjoint_answer_scores_zero(self):
        a = 'alpha beta gamma delta epsilon zeta'
        c = 'nothing here matches any of that at all whatsoever'
        assert cc.ngram_overlap(a, c) == 0.0

    def test_partial_copying_is_between(self):
        ctx = 'the treaty was signed in Paris in 1919'
        ans = 'the treaty was signed in Lisbon by unrelated other people'
        ov = cc.ngram_overlap(ans, ctx)
        assert 0.0 < ov < 1.0


class TestShortAnswers:
    def test_answer_shorter_than_n_is_none_not_zero(self):
        """
        Excluded, never scored 0. GPT-4o-mini's median answer is 18 words
        against Claude's 49, so scoring short answers as "copied nothing"
        would manufacture a cross-generator overlap difference out of
        answer length — the exact confound this control exists to rule out.
        """
        assert cc.ngram_overlap('Paris in 1919', 'Paris in 1919 was') is None

    def test_exactly_n_tokens_is_scored(self):
        assert cc.ngram_overlap('a b c d e', 'x a b c d e y') == 1.0


class TestTokenisation:
    def test_squash_neutralises_markdown(self):
        """
        `squash` maps non-word characters to spaces, so a bolded answer and
        the plain context still match. This is why the primary variant is
        insensitive to D1's markdown stripping.
        """
        ctx = 'the prize went to Graduados in the final round'
        ans = 'the prize went to **Graduados** in the final round'
        assert cc.ngram_overlap(ans, ctx, tokeniser='squash') == 1.0

    def test_whitespace_tokenisation_is_broken_by_markdown(self):
        """The contrast case: raw .split() makes `**Graduados**` unmatchable."""
        ctx = 'the prize went to Graduados in the final round'
        ans = 'the prize went to **Graduados** in the final round'
        assert cc.ngram_overlap(ans, ctx, tokeniser='whitespace') < 1.0


class TestCorrelationHelpers:
    def test_spearman_averages_ties(self):
        """
        Ties are common — many answers share an overlap of exactly 0.0 — and
        an arbitrary tie order would bake noise into the reported statistic.
        """
        import numpy as np
        x = np.array([0.0, 0.0, 0.0, 1.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        rho = cc._spearman(x, y)
        assert 0.0 < rho <= 1.0

    def test_zero_variance_is_nan_not_a_crash(self):
        import numpy as np
        x = np.array([0.3, 0.3, 0.3])
        y = np.array([1.0, 2.0, 3.0])
        assert np.isnan(cc._pearson(x, y))
