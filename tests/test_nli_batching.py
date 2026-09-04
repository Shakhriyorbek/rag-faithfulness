"""
Tests for the adaptive NLI batching in src/nli.py.

gpu1 is shared, and a co-tenant holding 27 of its 32 GB leaves room for a
batch of 2 rather than 16. The scorer therefore shrinks its batch on OOM and
grows it back when memory returns.

The invariant that matters is NOT throughput, it is alignment: score_claims
slices the returned probability list by block
(`probs[i * n_p:(i + 1) * n_p]`), so a retry that dropped or reordered a
failing slice would silently attribute one claim's score to another claim and
every claim_min in the grid would be wrong in a way no summary statistic
would reveal. These pin that.

Offline: no model is loaded — the scorer is driven through a stub _forward.
"""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))


def _scorer(fail_above, free_mib=760, max_batch=16):
    """An NLIScorer that never loads a model and OOMs above `fail_above`."""
    pytest.importorskip('torch', reason='nli.py imports torch at module level')
    import nli

    s = object.__new__(nli.NLIScorer)          # bypass __init__ / model load
    s.device = 'cuda:0'
    s.entailment_idx = 1
    s.batch_size = max_batch
    s.calls = []

    def fake_forward(batch):
        if len(batch) > fail_above:
            raise nli._OOM[0]('CUDA out of memory (simulated)')
        s.calls.append(len(batch))
        # identity-ish payload so ordering is checkable
        return [float(p) for p, _ in batch]

    s._forward = fake_forward
    s._free_mib = lambda: free_mib
    return s, nli


class TestOOMBackoff:
    def test_every_pair_is_returned_in_order(self):
        """The whole point: a retried slice must not be dropped or reordered."""
        s, _ = _scorer(fail_above=2)
        pairs = [(i, 'h') for i in range(10)]
        out = s.entailment_probs(pairs)
        assert out == [float(i) for i in range(10)]

    def test_batch_shrinks_to_a_size_that_fits(self):
        s, _ = _scorer(fail_above=2)
        s.entailment_probs([(i, 'h') for i in range(10)])
        assert s.batch_size == 2
        assert max(s.calls) <= 2

    def test_shrink_halves_rather_than_dropping_to_one(self):
        s, _ = _scorer(fail_above=4)
        s.entailment_probs([(i, 'h') for i in range(8)])
        assert s.batch_size == 4

    def test_oom_at_batch_one_propagates(self):
        """Nothing left to halve — fail loudly instead of returning short."""
        s, nli = _scorer(fail_above=0)
        with pytest.raises(nli._OOM):
            s.entailment_probs([(1, 'h')])

    def test_explicit_batch_size_is_not_persisted(self):
        """A pinned call must not rewrite the scorer's adaptive state."""
        s, _ = _scorer(fail_above=2)
        s.batch_size = 8
        s.entailment_probs([(i, 'h') for i in range(6)], batch_size=2)
        assert s.batch_size == 8


class TestGrowBack:
    def test_batch_grows_when_memory_returns(self):
        """
        A co-tenant releasing the GPU should not leave the run crippled for
        hours at the batch size that fitted when it started.
        """
        s, nli = _scorer(fail_above=64, free_mib=8000)
        s.batch_size = 2
        n = (nli.NLIScorer._GROW_AFTER + 5) * 2
        s.entailment_probs([(i, 'h') for i in range(n)])
        assert s.batch_size > 2

    def test_no_growth_while_memory_is_tight(self):
        s, nli = _scorer(fail_above=64, free_mib=500)
        s.batch_size = 2
        n = (nli.NLIScorer._GROW_AFTER + 5) * 2
        s.entailment_probs([(i, 'h') for i in range(n)])
        assert s.batch_size == 2


class TestBatchSizeChoice:
    def test_cpu_uses_the_maximum(self):
        pytest.importorskip('torch')
        import config
        import nli
        s = object.__new__(nli.NLIScorer)
        s.device = 'cpu'
        assert s._pick_batch_size() == config.NLI_BATCH_MAX

    def test_tight_vram_picks_two(self):
        """The measured case on gpu1: ~760 MiB free after the model loads."""
        pytest.importorskip('torch')
        import nli
        s = object.__new__(nli.NLIScorer)
        s.device = 'cuda:0'
        s._free_mib = lambda: 760
        assert s._pick_batch_size() == 2

    def test_never_zero(self):
        pytest.importorskip('torch')
        import nli
        s = object.__new__(nli.NLIScorer)
        s.device = 'cuda:0'
        s._free_mib = lambda: 10
        assert s._pick_batch_size() == 1
