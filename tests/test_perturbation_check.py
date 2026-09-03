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


class TestNumInText:
    """
    Boundary matching. The original test was a raw `in`, so a number matched
    as a digit substring of a bigger one and both eligibility rules broke:
    rule (a) admitted answers whose value was never grounded (their delta is
    ~0 by construction, which drags the headline mean down), rule (b)
    discarded eligible cases.
    """

    def test_substring_of_a_longer_number_is_not_a_match(self):
        assert not pc.num_in_text('1000', 'the total was 10000 USD')
        assert not pc.num_in_text('150', 'the price is 1500 dollars')
        assert not pc.num_in_text('5', 'chapter 51 discusses this')

    def test_thousands_separator_variant_still_matches(self):
        assert pc.num_in_text('1000', 'the total was 1,000 USD')
        assert pc.num_in_text('1,020', 'weighs 1020 kg')

    def test_year_inside_a_longer_year_is_not_a_match(self):
        assert not pc.num_in_text('20', 'in 2019 the figure rose')

    def test_digits_glued_to_letters_are_not_a_quantity(self):
        assert not pc.num_in_text('7', 'model B7 was tested')

    def test_decimal_prefix_is_not_a_match(self):
        assert not pc.num_in_text('3.5', 'grew by 3.55 percent')
        assert not pc.num_in_text('5', 'the value is 5.5')

    def test_sentence_final_and_punctuated_forms_match(self):
        assert pc.num_in_text('1,000', 'the budget is $1,000.')
        assert pc.num_in_text('1901', 'awarded in 1901, in Munich')

    def test_empty_inputs_are_never_grounded(self):
        assert not pc.num_in_text('', 'anything')
        assert not pc.num_in_text('5', '')


class TestNumericGroundingCheck:
    """F9: the deterministic value-presence check the paper recommends."""

    def test_all_values_present_is_grounded(self):
        assert pc.numeric_grounding_check(
            'The budget is $1,000 and 12 people signed.',
            ['the total was 1,000 USD', 'a team of 12'])

    def test_one_absent_value_fails(self):
        assert not pc.numeric_grounding_check('The budget is $1,500.',
                                              ['the total was 1,000 USD'])

    def test_substring_grounding_does_not_count(self):
        """The whole point: 1,000 in the context does not ground 100."""
        assert not pc.numeric_grounding_check('It costs 100.',
                                              ['the total was 1,000 USD'])

    def test_answer_without_numbers_is_vacuously_grounded(self):
        assert pc.numeric_grounding_check('Roentgen won it.', ['some context'])

    def test_no_context_is_not_grounded(self):
        assert not pc.numeric_grounding_check('It is 5.', [])


class TestValueRole:
    """
    Not every number in an answer is a claim about the world. Both generators
    cite the prompt's chunk numbering and write ordered lists; counting those
    as ungrounded values took the numeric check's false-positive rate on
    untouched HotpotQA/Claude answers from 6.0% to 39.5%.
    """

    def test_chunk_citation_is_not_content(self):
        a = 'According to Chunk 4, the total is 1,000.'
        assert pc.content_numbers(a) == ['1,000']

    def test_ordered_list_marker_is_not_content(self):
        a = 'The members are:\n1. Royce da 5 (Bad)\n2. Eminem (Evil)'
        assert '1' not in pc.content_numbers(a)
        assert '2' not in pc.content_numbers(a)

    def test_grounding_check_ignores_citations_and_markers(self):
        a = 'According to Chunk 4, the total is 1,000.\n1. First\n2. Second'
        assert pc.numeric_grounding_check(a, ['the total was 1,000 USD'])

    def test_a_content_number_still_has_to_be_grounded(self):
        a = 'According to Chunk 4, the total is 1,500.'
        assert not pc.numeric_grounding_check(a, ['the total was 1,000 USD'])


class TestWilsonCI:
    def test_interval_brackets_the_estimate_and_stays_in_range(self):
        lo, hi = pc._wilson_ci(8, 200)
        assert 0.0 <= lo < 0.04 < hi <= 1.0

    def test_zero_events_gives_a_non_negative_lower_bound(self):
        lo, hi = pc._wilson_ci(0, 200)
        assert lo == 0.0 and hi > 0.0


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
    def test_the_detector_is_the_shared_one(self):
        """
        There used to be two: a substring-anywhere rule here and an anchored
        rule in correctness.py. They disagreed on 365 of 16,000 rows, so the
        falsification sample and the conditional analysis called different
        populations "answered".
        """
        import abstention
        import correctness
        assert pc.is_abstention is abstention.is_abstention
        assert correctness.is_abstention is abstention.is_abstention

    def test_anchored_refusals_are_excluded(self):
        """An abstention has no grounded claim to falsify."""
        assert pc.is_abstention('I cannot answer based on the provided context.')
        assert not pc.is_abstention('The budget is $1,000.')

    def test_a_mid_answer_hedge_is_still_an_answer(self):
        """
        The old substring rule dropped this row from the sample while the
        conditional analysis counted it as answered.
        """
        assert not pc.is_abstention(
            'The budget is $1,000, though the context does not contain the date.')

    def test_a_refusal_behind_a_preamble_is_a_refusal(self):
        """
        Adopted 2026-09-03. Both generators prepend "Based on the provided
        context," to everything, refusals included; the bare anchored rule
        therefore graded 107 genuine refusals as attempts (all Claude's).
        """
        assert pc.is_abstention(
            'Based on the provided context, I cannot answer this question. '
            'The context mentions Home Alone 2 but does not say where it is set.')
        assert pc.is_abstention(
            'According to the context, I cannot determine when Arsenal last won.')
        # ...but the preamble strip must not turn a hedge into a refusal.
        assert not pc.is_abstention(
            'Based on the provided context, the members were A and B. '
            'However, the context does not contain their ages.')

    def test_the_bare_rule_is_kept_and_still_disagrees(self):
        """The 107-row difference must stay measurable, not be erased."""
        import abstention
        preamble = 'Based on the provided context, I cannot answer this question.'
        assert abstention.is_abstention(preamble)
        assert not abstention.is_abstention_bare(preamble)


class TestCaseConstructionIsContentOnly:
    """
    Decided 2026-09-03. build_number_case took the first GROUNDED number,
    which on HotpotQA/Claude was a chunk citation or a list marker in 9.5% of
    cases — values carrying a NEGATIVE mean delta (-0.0625), since falsifying
    "Chunk 4" into "Chunk 7" is not a falsification.
    """

    def test_a_chunk_citation_is_not_chosen(self):
        answer = 'According to Chunk 4, the budget was 1,000 USD.'
        context = 'chunk 4 of 5 — the invoice total was 1,000 USD'
        out = pc.build_number_case(answer, context)
        assert out is not None
        assert out[1] == '1,000', f'falsified the citation, not the value: {out[1]}'

    def test_a_list_marker_is_not_chosen(self):
        answer = '1. The population was 5,000 people.'
        context = 'the town recorded 5,000 people in the census'
        out = pc.build_number_case(answer, context)
        assert out is not None
        assert out[1] == '5,000', f'falsified the list marker: {out[1]}'

    def test_a_plain_grounded_value_is_still_chosen(self):
        out = pc.build_number_case('The budget is 1,000 USD.',
                                   'the invoice total was 1,000 USD')
        assert out is not None and out[1] == '1,000'


class TestDonorSelection:
    def test_a_donor_supporting_the_answer_is_rejected(self):
        answer = 'The budget is 1,000 USD.'
        donor = {'retrieved_texts': ['the invoice total was 1,000 USD']}
        clean = {'retrieved_texts': ['unrelated text about penguins']}
        nums = pc.answer_numbers(answer)
        assert pc._donor_supports(donor, nums)
        assert not pc._donor_supports(clean, nums)
