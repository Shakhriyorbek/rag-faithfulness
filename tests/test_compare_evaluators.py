"""
Tests for src/compare_evaluators.py — the multiplicity-scope sensitivity.

The paper's headline is a COUNT ("k of 4 nulls break under a more sensitive
evaluator"), and k depends on which family Holm corrects over. These pin the
arithmetic of each scope and, more importantly, pin that a cell which changes
verdict between scopes is actually detected and printed — a silent swing is
the failure mode this block exists to prevent.

All offline: pure arithmetic, no model, no checkpoints, no API.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

import compare_evaluators as ce  # noqa: E402
from results import holm_bonferroni  # noqa: E402


def _rows(spec):
    return [dict(dataset=d, generator=g, metric=m, p_value=p,
                 correct_source='contains') for d, g, m, p in spec]


# The real 2026-09-03 table, post-D1. NQ/claude/align at p=0.0081 is the cell
# that lands either side of 0.05 depending on the family size.
LIVE = _rows([
    ('HotpotQA', 'claude', 'nli', 0.1586),
    ('HotpotQA', 'claude', 'align', 0.0633),
    ('HotpotQA', 'gpt4omini', 'nli', 0.0652),
    ('HotpotQA', 'gpt4omini', 'align', 0.0005),
    ('NQ', 'claude', 'nli', 0.1551),
    ('NQ', 'claude', 'align', 0.0081),
    ('NQ', 'gpt4omini', 'nli', 0.0978),
    ('NQ', 'gpt4omini', 'align', 0.1173),
])


class TestHolmArithmetic:
    def test_table_wide_gives_one_break(self):
        adj = holm_bonferroni([r['p_value'] for r in LIVE])
        assert sum(a < 0.05 for a in adj) == 1

    def test_within_evaluator_gives_two_breaks(self):
        align = [r for r in LIVE if r['metric'] == 'align']
        adj = holm_bonferroni([r['p_value'] for r in align])
        assert sum(a < 0.05 for a in adj) == 2

    def test_holm_is_monotone_in_family_size(self):
        """The swing cell must get strictly easier as the family shrinks."""
        p = 0.0081
        big = holm_bonferroni([0.0005, p, 0.0633, 0.0652,
                               0.0978, 0.1173, 0.1551, 0.1586])[1]
        small = holm_bonferroni([0.0005, p, 0.0633, 0.1173])[1]
        assert big > 0.05 > small


class TestScopeSensitivityIsReported:
    def test_a_swinging_cell_is_named(self, capsys):
        ce._report_holm_scope_sensitivity(LIVE)
        out = capsys.readouterr().out
        assert 'cells whose verdict depends on the family' in out
        assert 'NQ/claude/align' in out
        # every scope must be named with its own adjusted p, so the reader can
        # see the size of the swing rather than just that one exists
        assert 'table-wide p_holm=0.0567' in out
        assert 'within-evaluator p_holm=0.0243' in out

    def test_counts_per_family_are_printed(self, capsys):
        ce._report_holm_scope_sensitivity(LIVE)
        out = capsys.readouterr().out
        assert 'table-wide' in out and 'within-evaluator' in out
        table = [l for l in out.splitlines() if l.startswith('table-wide')][0]
        assert table.split()[2] == '1'
        wev = [l for l in out.splitlines() if l.startswith('within-evaluator')][0]
        assert wev.split()[2] == '2'

    def test_stable_table_says_so(self, capsys):
        """No false alarm when every scope agrees."""
        stable = _rows([
            ('NQ', 'claude', 'nli', 0.9), ('NQ', 'claude', 'align', 0.8),
            ('NQ', 'gpt4omini', 'nli', 0.7), ('NQ', 'gpt4omini', 'align', 0.6),
        ])
        ce._report_holm_scope_sensitivity(stable)
        out = capsys.readouterr().out
        assert 'no cell changes verdict' in out
