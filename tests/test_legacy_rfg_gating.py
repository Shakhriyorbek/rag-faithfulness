"""
The retired gap metric must not be on the path that produces the tables.

Section III-C of the paper retires RFG/nRFG. The code that computed them is
kept in src/legacy/ for reproducibility of the superseded draft, and reaches
the pipeline only through an explicit flag. These tests pin that, and pin the
removal of the invalid H1 test.

All offline: no checkpoints, no model, no API.
"""
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

import results  # noqa: E402


def _frame(with_rfg: bool):
    rows = []
    for model, para, nd, f in (('m1', 'contrastive', 0.80, 0.70),
                               ('m2', 'instruction-tuned', 0.82, 0.74),
                               ('m3', 'multilingual', 0.79, 0.72)):
        row = {'model': model, 'paradigm': para, 'dataset': 'NQ',
               'generator': 'claude', 'NDCG@5': nd, 'Recall@5': nd,
               'MRR@5': nd, 'nli_max': f, 'nli_mean_agg': f,
               'align_score': f, 'faithfulness': f}
        if with_rfg:
            row['RFG'] = nd - f
            row['nRFG'] = (nd - f) / nd
        rows.append(row)
    return pd.DataFrame(rows)


class TestLegacyGating:
    def test_rfg_lives_in_legacy(self):
        from legacy import rfg as legacy_rfg
        assert legacy_rfg.rfg(0.9, 0.85) == 0.05
        with __import__('pytest').raises(ImportError):
            __import__('metrics')

    def test_hypothesis_summary_runs_without_the_legacy_columns(self):
        out = results.hypothesis_summary(_frame(with_rfg=False))
        assert isinstance(out, pd.DataFrame)
        assert not any('nRFG' in h for h in out.get('hypothesis', []))

    def test_the_invalid_h1_test_is_gone(self):
        """H1 fed two UNPAIRED groups into a paired bootstrap. Even dead, an
        invalid test must not ship."""
        for frame in (_frame(with_rfg=True), _frame(with_rfg=False)):
            out = results.hypothesis_summary(frame)
            assert not any(str(h).startswith('H1')
                           for h in out.get('hypothesis', []))

    def test_robustness_analysis_skips_instead_of_raising(self):
        msg = results.robustness_analysis(_frame(with_rfg=False))
        assert 'skipped' in msg
