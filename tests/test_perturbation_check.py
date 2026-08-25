"""
Tests for src/perturbation_check.py — the numeric-infidelity probe.

The experiment only means something if the case construction is right, so
these pin the two eligibility rules and the two regex bugs found in review:

  - a sentence-final number ("$1,000.") must still match; the original
    lookahead treated the full stop as part of a version string and silently
    excluded most answers;
  - 1000 is a quantity, not a year; the year branch would have perturbed it
    by 0.7% instead of 50% and understated the metric's blindness.

All offline: pure string logic, no model, no checkpoints, no API.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

import perturbation_check as pc  # noqa: E402


class TestNumberRegex:
    def test_sentence_final_number_matches(self):
        r"""The bug: `(?![\w.])` rejected a number before a full stop."""
        found = [m.group(1) for m in pc.NUM_RE.finditer('The budget is $1,000.')]
        assert found == ['1,000']

    def test_year_at_end_of_sentence_matches(self):
        found = [m.group(1) for m in pc.NUM_RE.finditer('He won it in 1901.')]
        assert found == ['1901']

    def test_version_string_is_not_a_quantity(self):
        assert not list(pc.NUM_RE.finditer('Version 28.0.0.137 shipped.'))

    def test_digits_glued_to_letters_are_skipped(self):
        found = [m.group(1) for m in pc.NUM_RE.finditer('Model B12 costs 45.5.')]
        assert found == ['45.5']

    def test_range_yields_both_endpoints(self):
        found = [m.group(1) for m in pc.NUM_RE.finditer('It weighs 1,020-1,080 kg.')]
        assert found == ['1,020', '1,080']

    def test_comma_after_number_is_not_a_thousands_separator(self):
        found = [m.group(1) for m in pc.NUM_RE.finditer('The total was 1,000, plus tax.')]
        assert found == ['1,000']


class TestPerturbNumber:
    def test_thousand_is_a_quantity_not_a_year(self):
        """1000 -> 1501 (magnitude), never 1007 (year shift)."""
        new, kind = pc.perturb_number('1000')
        assert kind == 'magnitude'
        assert new == '1501'

    def test_real_years_take_the_year_branch(self):
        for year in ('1901', '1980', '2011'):
            new, kind = pc.perturb_number(year)
            assert kind == 'year'
            assert new != year

    def test_thousands_separator_is_preserved(self):
        """Format must survive, or NLI reacts to formatting not quantity."""
        new, _ = pc.perturb_number('1,000')
        assert ',' in new and new != '1,000'

    def test_decimal_places_are_preserved(self):
        new, kind = pc.perturb_number('1500.50')
        assert kind == 'decimal'
        assert len(new.split('.')[1]) == 2

    def test_result_always_differs_from_input(self):
        for s in ('0', '1', '45', '999', '12,345', '3.5'):
            out = pc.perturb_number(s)
            assert out is None or out[0] != s


class TestEligibility:
    CTX = ('Invoice 2024-11. Project Alpha total budget is $1,000 '
           'payable on receipt.')

    def test_grounded_number_yields_a_case(self):
        case = pc.build_number_case('The Project Alpha budget is $1,000.', self.CTX)
        assert case is not None
        perturbed, old, new, kind = case
        assert old == '1,000'
        assert new not in self.CTX
        assert perturbed != 'The Project Alpha budget is $1,000.'
        assert new in perturbed

    def test_ungrounded_number_is_rejected(self):
        """Perturbing a value the context never supported tests nothing."""
        assert pc.build_number_case('The budget is $7,777.', self.CTX) is None

    def test_replacement_present_in_context_is_rejected(self):
        """If the 'wrong' value is also supported, a low delta is correct."""
        ctx = 'Budget is 100, revised to 151.'
        assert pc.build_number_case('It is 100.', ctx) is None

    def test_perturbation_changes_only_the_number(self):
        ans = 'Based on the context, the Project Alpha budget is $1,000.'
        perturbed, old, new, _ = pc.build_number_case(ans, self.CTX)
        assert perturbed.replace(new, old) == ans


class TestEntityCase:
    def test_leading_boilerplate_is_not_an_entity(self):
        ans = 'Based on the provided context, Marie Curie won the prize.'
        spans = [s for _, _, s in pc._entity_candidates(ans)]
        assert not any(s.startswith('Based') for s in spans)
        assert any('Curie' in s for s in spans)

    def test_donor_must_be_absent_from_context_and_answer(self):
        ctx = 'Marie Curie won the prize.'
        ans = 'Marie Curie won the prize.'
        case = pc.build_entity_case(ans, ctx, ['Marie Curie', 'Albert Einstein'])
        assert case is not None
        _, old, new = case
        assert new not in ctx and new != old


class TestAbstention:
    def test_abstentions_are_excluded(self):
        """An abstention has no grounded claim to falsify."""
        assert pc.is_abstention('I cannot answer based on the provided context.')
        assert pc.is_abstention('The context does not contain this information.')
        assert not pc.is_abstention('The budget is $1,000.')
