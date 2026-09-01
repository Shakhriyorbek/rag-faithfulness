"""
Tests for src/claim_faithfulness.py — claim-level faithfulness.

The metric is min-over-claims of max-over-chunks, so a bad split changes the
score directly: an over-split invents unsupported fragments and drives the
minimum down, an under-split reproduces the whole-answer metric it replaces.
These pin the splitter's behaviour on the answer shapes both generators
actually produce.

All offline: pure string logic, no model, no checkpoints, no API.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

import pytest  # noqa: E402

import claim_faithfulness as cf  # noqa: E402


class TestSplitClaims:
    def test_multi_claim_answer_splits(self):
        a = ('Roentgen of Germany received the prize in 1901. '
             'He received 150,782 SEK for it.')
        assert len(cf.split_claims(a)) == 2

    def test_initials_do_not_split(self):
        """'W. C. Roentgen' is one name, not three sentences."""
        a = 'The prize went to W. C. Roentgen in 1901. He taught at Munich.'
        claims = cf.split_claims(a)
        assert len(claims) == 2
        assert 'W. C. Roentgen' in claims[0]

    def test_abbreviations_do_not_split(self):
        a = 'The study was led by Dr. Smith at Acme Inc. in Boston. It ran three years.'
        claims = cf.split_claims(a)
        assert len(claims) == 2
        assert 'Dr. Smith' in claims[0]

    def test_decimals_do_not_split(self):
        """A version string must not become several claims."""
        a = 'The version is 28.0.0.137 and it shipped on January 9, 2018.'
        assert len(cf.split_claims(a)) == 1

    def test_sentence_starting_with_a_number_splits(self):
        """
        The boundary test required the next character to be uppercase, and a
        digit fails .isupper(), so a sentence opening with a number was merged
        into the one before it. The undercount landed selectively on
        numeric-heavy answers — the ones the falsification probe operates on —
        and on the verbose generator, which is the same direction as the
        reported effect.
        """
        a = 'The budget is 1,000 USD. 1500 was requested by the team.'
        assert len(cf.split_claims(a)) == 2
        b = 'The prize was awarded in 1901. 150,782 SEK was the sum.'
        assert len(cf.split_claims(b)) == 2

    def test_opening_curly_quote_starts_a_sentence(self):
        a = 'He scored 3.5 points. \u201cGreat,\u201d said the coach afterwards.'
        assert len(cf.split_claims(a)) == 2

    def test_terse_single_claim_stays_one(self):
        a = 'The first Nobel Prize in Physics was awarded in 1901 to Roentgen.'
        assert len(cf.split_claims(a)) == 1

    def test_bullet_list_becomes_separate_claims(self):
        a = ('The context provides:\n- The budget is $1,000 for Project Alpha\n'
             '- The approval date was March 2024\n- The owner is the finance team')
        claims = cf.split_claims(a)
        assert len(claims) == 3
        assert not any(c.startswith('-') for c in claims)

    def test_markdown_is_stripped(self):
        claims = cf.split_claims('The budget is **$1,000** for Project Alpha.')
        assert '**' not in claims[0]

    def test_short_fragments_are_dropped(self):
        """A heading or bare 'Yes.' is not a scorable claim."""
        a = 'Summary\nThe budget is $1,000 for Project Alpha and is approved.'
        claims = cf.split_claims(a)
        assert all(len(c.split()) >= 3 for c in claims)
        assert not any(c.strip() == 'Summary' for c in claims)

    def test_answer_of_only_fragments_still_yields_one_claim(self):
        """Never return [] — the answer would vanish from the analysis."""
        assert len(cf.split_claims('Yes.')) == 1

    def test_empty_answer_yields_nothing(self):
        assert cf.split_claims('') == []
        assert cf.split_claims('   ') == []


class TestScoreClaimsAggregation:
    class FakeNLI:
        """Returns a score keyed on the hypothesis, so aggregation is checkable."""

        def __init__(self, mapping):
            self.mapping = mapping
            self.calls = []

        def entailment_probs(self, pairs, batch_size=16):
            self.calls.extend(pairs)
            return [self.mapping.get(h.strip(), 0.5) for _, h in pairs]

    def test_min_over_claims_not_max(self):
        """
        The defect being fixed: one unsupported claim must drag the answer
        down even when the others are strongly entailed.
        """
        answer = 'The budget is one thousand. The owner is finance. Sarah approved it.'
        mapping = {
            'The budget is one thousand.': 0.95,
            'The owner is finance.': 0.93,
            'Sarah approved it.': 0.04,          # fabricated
        }
        nli = self.FakeNLI(mapping)
        out = cf.score_claims(['ctx a', 'ctx b'], answer, nli)
        assert out['n_claims'] == 3
        assert out['claim_min'] == 0.04
        assert out['weakest'] == 'Sarah approved it.'
        assert out['claim_mean'] > out['claim_min']

    def test_whole_max_is_reported_for_comparison(self):
        answer = 'The budget is one thousand. Sarah approved it.'
        nli = self.FakeNLI({answer: 0.91,
                            'The budget is one thousand.': 0.95,
                            'Sarah approved it.': 0.04})
        out = cf.score_claims(['ctx a'], answer, nli)
        assert out['whole_max'] == 0.91
        assert out['claim_min'] == 0.04

    def test_empty_context_is_nan_not_zero(self):
        """No context means undefined, not perfectly unfaithful."""
        out = cf.score_claims([], 'Some claim about a budget.', self.FakeNLI({}))
        assert out['claim_min'] != out['claim_min']  # NaN

    class DeterministicNLI:
        """A score that depends on the exact (premise, hypothesis) pair, so
        an accidental difference in the hypothesis string shows up as a
        different number rather than being masked by a constant."""

        def entailment_probs(self, pairs, batch_size=16):
            return [((hash((p, h)) % 9973) / 9973.0) for p, h in pairs]

    def test_single_claim_answer_reduces_to_the_whole_answer_metric(self):
        """
        The identity the paper's 0.4922 figure asserts: with one claim and no
        markdown, claim_min IS whole_max. If it is not, strip_markdown is
        changing the hypothesis for one metric and not the other and the two
        are not being compared on identical inputs.
        """
        answer = 'The first Nobel Prize in Physics was awarded in 1901.'
        out = cf.score_claims(['ctx a', 'ctx b', 'ctx c'], answer,
                              self.DeterministicNLI())
        assert out['n_claims'] == 1
        assert out['claim_min'] == pytest.approx(out['whole_max'], abs=1e-9)

    def test_best_chunk_is_recorded_for_attribution(self):
        answer = 'The budget is one thousand dollars.'
        nli = self.FakeNLI({answer: 0.9})
        out = cf.score_claims(['a', 'b'], answer, nli)
        assert 'best_chunk' in out['per_claim'][0]
