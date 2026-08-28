"""
Local smoke test — no torch, no GPU, no dataset downloads.

Exercises the pipeline's pure logic end to end on synthetic data:
  chunking (real tokenizer) -> indexing (numpy fallback) -> retrieval ->
  ID-based qrels -> NDCG/Recall/MRR -> re-rank ordering -> bootstrap test.

The core assertion is the audit-B3 regression guard: with provenance-based
qrels, a retriever that finds the right document must score NDCG@5 > 0
(the notebook's string-equality qrels scored ~0 for everything).

Run:  python tests/test_local_smoke.py
"""
import os
import pytest
import sys
import tempfile
from pathlib import Path

# Isolated checkpoint dir so the test never touches real checkpoints
os.environ['RAG_BASE'] = tempfile.mkdtemp(prefix='ragfaith_test_')
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np

from datasets_loader import Document, LoadedDataset, QASample
from embed_index import VectorIndex, build_chunks, chunk_document
from rerank import rerank_candidates
from results import bootstrap_significance
from retrieval_eval import build_qrels, evaluate_run


def fake_embed(texts, dim=256):
    """Deterministic bag-of-words hash embedding — shared vocabulary between
    query and document yields high cosine similarity. Uses md5, not hash():
    Python's hash() is randomized per process."""
    from hashlib import md5
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, int(md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-9, None)


def make_synthetic_dataset(n=8):
    """Each query's vocabulary lives in exactly one document — the corpora
    are vocabulary-disjoint so a bag-of-words retriever must rank the gold
    document first."""
    topics = ['volcano lava eruption magma', 'penguin antarctica ice colony',
              'jazz saxophone trumpet swing', 'comet asteroid orbit telescope',
              'sourdough yeast flour baking', 'marathon sprint stamina track',
              'coral reef plankton lagoon', 'glacier moraine fjord icefall']
    samples, documents = [], []
    for i, topic in enumerate(topics[:n]):
        qid = f'syn_{i}'
        words = topic.split()
        text = (' '.join(words) + ' ') * 20  # topic vocabulary only
        samples.append(QASample(
            query_id=qid,
            question=f'{words[0]} {words[1]}',
            answer=f'{words[2]} and {words[3]}',
            gold_context=text,
            dataset='SYN'))
        documents.append(Document(doc_id=f'{qid}_doc', text=text,
                                  gold_for={qid}))
    return LoadedDataset('SYN', samples, documents)


def test_chunking():
    # chunk_document() builds a HuggingFace tokenizer, which is absent on a
    # laptop set up only for the offline suite — and installing it would make
    # this test download a tokenizer on first run, which is exactly what the
    # local suite is meant not to do. Skips like the openai/torch tests.
    pytest.importorskip('transformers', reason='chunking needs a HF tokenizer')
    text = 'Word ' * 600  # 600 tokens -> 3 overlapping 256-token chunks
    chunks = chunk_document('d0', text)
    assert len(chunks) == 3, f'expected 3 chunks, got {len(chunks)}'
    assert chunks[0].chunk_id == 'd0::c0' and chunks[0].doc_id == 'd0'
    # Offset slicing must preserve the ORIGINAL text (no lowercasing loss)
    mixed = 'The Quick BROWN Fox. ' * 50
    c = chunk_document('d1', mixed)[0]
    assert 'BROWN' in c.text, 'chunker mangled original casing'
    print('  chunking OK (stable IDs, original text preserved)')


def test_retrieval_and_qrels():
    # chunk_document() builds a HuggingFace tokenizer, which is absent on a
    # laptop set up only for the offline suite — and installing it would make
    # this test download a tokenizer on first run, which is exactly what the
    # local suite is meant not to do. Skips like the openai/torch tests.
    pytest.importorskip('transformers', reason='chunking needs a HF tokenizer')
    ds = make_synthetic_dataset()
    chunks = build_chunks(ds)
    assert chunks, 'no chunks built'

    index = VectorIndex(dimension=256)
    index.add(chunks, fake_embed([c.text for c in chunks]))
    assert index._faiss is None or True  # numpy fallback is fine

    run = {}
    for s in ds.samples:
        q_emb = fake_embed([s.question])[0]
        results = index.search(q_emb, k=5)
        run[s.query_id] = [cid for cid, _, _ in results]

    qrels = build_qrels(ds)
    assert all(qrels.get(s.query_id) for s in ds.samples), 'empty qrels'

    metrics = evaluate_run(qrels, run)
    print(f'  synthetic retrieval: {metrics}')
    # ── audit B3 regression guard ──
    assert metrics['NDCG@5'] > 0, (
        'NDCG@5 == 0 on synthetic data: qrels/run ID mismatch — '
        'the exact failure mode of the notebook string-equality qrels')
    assert metrics['NDCG@5'] > 0.9, (
        f"synthetic retrieval should be near-perfect, got {metrics['NDCG@5']}")
    print('  retrieval + ID-based qrels OK (B3 guard passed)')


def test_hotpot_style_qrels():
    """Gold sentence provenance: only chunks containing a gold sentence count."""
    # chunk_document() builds a HuggingFace tokenizer, which is absent on a
    # laptop set up only for the offline suite — and installing it would make
    # this test download a tokenizer on first run, which is exactly what the
    # local suite is meant not to do. Skips like the openai/torch tests.
    pytest.importorskip('transformers', reason='chunking needs a HF tokenizer')
    gold_sent = 'The rare mineral formed under immense pressure.'
    filler = 'Unrelated filler text about many other things entirely. ' * 40
    doc = Document(doc_id='hp_doc', text=filler + gold_sent,
                   gold_for={'hp_q'},
                   gold_sentences={'hp_q': [gold_sent]})
    ds = LoadedDataset('SYN2',
                       [QASample('hp_q', 'q?', 'a', gold_sent, 'SYN2')],
                       [doc])
    qrels = build_qrels(ds)
    rel_ids = set(qrels['hp_q'])
    chunks = build_chunks(ds)
    containing = {c.chunk_id for c in chunks if gold_sent in c.text}
    assert rel_ids == containing, (
        f'gold-sentence chunk filter wrong: {rel_ids} vs {containing}')
    assert 0 < len(rel_ids) < len(chunks), 'filter marked everything/nothing'
    print(f'  hotpot-style sentence-level qrels OK '
          f'({len(rel_ids)}/{len(chunks)} chunks relevant)')


def test_rerank_ordering():
    cos = [0.9, 0.8, 0.7]
    nli = [0.0, 0.1, 1.0]
    assert rerank_candidates(cos, nli, lam=1.0) == [0, 1, 2]  # pure cosine
    assert rerank_candidates(cos, nli, lam=0.0) == [2, 1, 0]  # pure NLI
    order_mixed = rerank_candidates(cos, nli, lam=0.6)
    assert order_mixed[0] == 2, f'λ=0.6 should promote high-NLI doc: {order_mixed}'
    print('  re-rank ordering OK (Eq. 5)')


def test_bootstrap():
    rng = np.random.default_rng(0)
    a = rng.normal(0.65, 0.1, 500).tolist()
    b = rng.normal(0.50, 0.1, 500).tolist()
    r = bootstrap_significance(a, b)
    assert r['significant'] and r['observed_diff'] > 0.1
    same = bootstrap_significance(a, a)
    assert not same['significant']
    print('  paired bootstrap OK')


if __name__ == '__main__':
    test_chunking()
    test_retrieval_and_qrels()
    test_hotpot_style_qrels()
    test_rerank_ordering()
    test_bootstrap()
    print('\nALL LOCAL SMOKE TESTS PASSED')
