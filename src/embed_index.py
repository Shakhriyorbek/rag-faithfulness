"""
embed_index.py — Phase A: chunking, encoding, indexing, retrieval.

Audit fixes carried here:
  B3 — chunks get stable IDs (doc provenance) at creation; retrieval stores
       chunk_ids, so retrieval quality is evaluated on IDs, never on lossy
       chunk-string equality.
  --  chunking slices the ORIGINAL text via tokenizer offsets. The notebook
       rebuilt chunk text with convert_tokens_to_string on bert-base-uncased,
       which lowercases and strips accents — the generator and NLI would have
       read mangled text.
  C8 — jina-embeddings-v3 loads with trust_remote_code=True and uses its
       retrieval task adapters.
  C9 — Instructor models are called with [instruction, text] pairs, not
       string-prefixed queries.
  C12 — OpenAI embedding calls are cost-tracked.

Chunks are model-independent (one shared tokenizer), so they are built and
checkpointed once per dataset and reused across all seven models.
"""
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

import config
from datasets_loader import LoadedDataset
from utils import (checkpoint_exists, cost_tracker, free_memory,
                   load_checkpoint, save_checkpoint)

_CHUNK_TOKENIZER = None


def _chunk_tokenizer():
    global _CHUNK_TOKENIZER
    if _CHUNK_TOKENIZER is None:
        from transformers import AutoTokenizer
        _CHUNK_TOKENIZER = AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True)
    return _CHUNK_TOKENIZER


@dataclass
class Chunk:
    chunk_id: str   # f'{doc_id}::c{j}' — stable, carries doc provenance
    doc_id: str
    text: str       # exact slice of the original document text


def chunk_document(doc_id: str, text: str,
                   chunk_size: int = config.CHUNK_SIZE,
                   overlap: int = config.CHUNK_OVERLAP) -> List[Chunk]:
    """Token-window chunking that slices the original string via offsets."""
    tok = _chunk_tokenizer()
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True,
              truncation=False, verbose=False)
    offsets = enc['offset_mapping']
    if not offsets:
        return []
    chunks, start, j = [], 0, 0
    stride = chunk_size - overlap
    while start < len(offsets):
        end = min(start + chunk_size, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        piece = text[char_start:char_end]
        if piece.strip():
            chunks.append(Chunk(f'{doc_id}::c{j}', doc_id, piece))
            j += 1
        if end == len(offsets):
            break
        start += stride
    return chunks


def build_chunks(loaded: LoadedDataset) -> List[Chunk]:
    """Chunk every corpus document. Cached per dataset — model-independent."""
    from datasets_loader import CORPUS_VERSION
    ck = f'chunks_{loaded.name}_{len(loaded.samples)}_{CORPUS_VERSION}'
    cached = load_checkpoint(ck)
    if cached:
        return cached
    print(f'  chunking {len(loaded.documents)} docs [{loaded.name}]...')
    chunks = []
    for doc in loaded.documents:
        chunks.extend(chunk_document(doc.doc_id, doc.text))
    print(f'  -> {len(chunks)} chunks')
    save_checkpoint(ck, chunks)
    return chunks


# ── Vector index ──────────────────────────────────────────────────
class VectorIndex:
    """
    Exact inner-product search (cosine on normalized vectors).
    Uses FAISS IndexFlatIP when available, otherwise a numpy matmul —
    both are exact, so results are identical either way.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.chunk_ids: List[str] = []
        self.texts: List[str] = []
        try:
            import faiss
            self._faiss = faiss.IndexFlatIP(dimension)
            self._matrix = None
        except ImportError:
            self._faiss = None
            self._matrix = np.zeros((0, dimension), dtype=np.float32)

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        assert embeddings.shape[1] == self.dimension, (
            f'dim mismatch: expected {self.dimension}, got {embeddings.shape[1]}')
        self.chunk_ids.extend(c.chunk_id for c in chunks)
        self.texts.extend(c.text for c in chunks)
        emb = embeddings.astype(np.float32)
        if self._faiss is not None:
            self._faiss.add(emb)
        else:
            self._matrix = np.vstack([self._matrix, emb])

    def search(self, query_emb: np.ndarray, k: int):
        """Return list of (chunk_id, text, score), best first."""
        q = query_emb.astype(np.float32).reshape(1, -1)
        k = min(k, len(self.chunk_ids))
        if self._faiss is not None:
            scores, indices = self._faiss.search(q, k)
            idx, sc = indices[0], scores[0]
        else:
            sims = (self._matrix @ q.T).ravel()
            idx = np.argsort(-sims)[:k]
            sc = sims[idx]
        return [(self.chunk_ids[i], self.texts[i], float(s))
                for i, s in zip(idx, sc) if i >= 0]

    def __len__(self):
        return len(self.chunk_ids)


# ── Embedding model wrapper ───────────────────────────────────────
class EmbeddingModelWrapper:
    """
    Unified encoder over four call conventions:
      - plain SentenceTransformer models
      - instruction-prefix models (E5-instruct: prefix on queries only)
      - Instructor family ([instruction, text] pairs — C9)
      - jina-v3 (trust_remote_code + task adapters — C8)
      - OpenAI API (cost-tracked — C12)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg['name']
        self.is_openai = cfg['hf_id'] == 'openai'
        self.is_instructor = 'instructor' in cfg['hf_id'].lower()
        self.is_jina = 'jina' in cfg['hf_id'].lower()
        self.model = None
        if self.is_openai:
            return
        print(f'Loading {self.name}...')
        if self.is_instructor:
            # C9: Instructor needs its own class for instruction pooling
            from InstructorEmbedding import INSTRUCTOR
            self.model = INSTRUCTOR(cfg['hf_id'], device=config.DEVICE)
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                cfg['hf_id'], device=config.DEVICE,
                trust_remote_code=self.is_jina)  # C8
        print(f"  loaded ({cfg['dim']}d)")

    def encode(self, texts: List[str], is_query: bool = False,
               batch_size: int = 32) -> np.ndarray:
        if self.is_openai:
            return self._encode_openai(texts)
        if self.is_instructor:
            instruction = (self.cfg.get('instruction') if is_query
                           else self.cfg.get('doc_instruction')) or ''
            pairs = [[instruction, t] for t in texts]
            return self.model.encode(
                pairs, batch_size=batch_size,
                normalize_embeddings=True, show_progress_bar=False)
        if self.is_jina:
            task = 'retrieval.query' if is_query else 'retrieval.passage'
            try:
                return self.model.encode(
                    texts, task=task, batch_size=batch_size,
                    normalize_embeddings=True, show_progress_bar=False)
            except TypeError:
                pass  # older sentence-transformers: no task kwarg
        if self.cfg.get('instruction') and is_query:
            texts = [self.cfg['instruction'] + t for t in texts]
        return self.model.encode(
            texts, batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=False)

    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from env
        embeddings = []
        for i in range(0, len(texts), 100):
            batch = [t[:8000] for t in texts[i:i + 100]]
            resp = client.embeddings.create(
                model='text-embedding-3-small', input=batch)
            cost_tracker.log_embedding(resp.usage)  # C12
            for item in resp.data:
                emb = np.array(item.embedding, dtype=np.float32)
                embeddings.append(emb / np.linalg.norm(emb))
        return np.array(embeddings)


def build_index(chunks: List[Chunk],
                wrapper: EmbeddingModelWrapper,
                batch_size: int = 128) -> VectorIndex:
    index = VectorIndex(dimension=wrapper.cfg['dim'])
    print(f'  encoding {len(chunks)} chunks with {wrapper.name}...')
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embs = wrapper.encode([c.text for c in batch], is_query=False)
        index.add(batch, embs)
    print(f'  index built: {len(index)} vectors')
    return index


# ── Phase A ───────────────────────────────────────────────────────
def run_phase_a(datasets: Dict[str, LoadedDataset],
                model_names: List[str] = None,
                pool_k: int = config.RERANK_POOL):
    """
    For every model × dataset: build the index, retrieve the top pool_k
    candidates per query (top-5 feed generation; the full pool feeds the
    re-ranking ablation), checkpoint as retrieval_{model}_{dataset}.
    Models load ONE AT A TIME and are freed before the next.
    """
    configs = [c for c in config.EMBEDDING_MODELS
               if not model_names or c['name'] in model_names]

    for cfg in configs:
        todo = [name for name in datasets
                if not checkpoint_exists(f"retrieval_{cfg['name']}_{name}")]
        if not todo:
            print(f"[phase A] {cfg['name']}: all datasets done, skipping")
            continue

        t0 = time.time()
        wrapper = EmbeddingModelWrapper(cfg)
        for ds_name in todo:
            loaded = datasets[ds_name]
            chunks = build_chunks(loaded)
            index = build_index(chunks, wrapper)

            print(f'  retrieving top-{pool_k} [{ds_name}]...')
            retrievals = []
            for sample in loaded.samples:
                q_emb = wrapper.encode([sample.question], is_query=True)
                results = index.search(q_emb[0], k=pool_k)
                retrievals.append({
                    'query_id': sample.query_id,
                    'question': sample.question,
                    'answer': sample.answer,
                    'retrieved_ids': [cid for cid, _, _ in results],
                    'retrieved_texts': [t for _, t, _ in results],
                    'retrieval_scores': [s for _, _, s in results],
                })
            save_checkpoint(f"retrieval_{cfg['name']}_{ds_name}", retrievals)
            print(f"  {cfg['name']}/{ds_name}: {len(retrievals)} queries")

        del wrapper
        free_memory()
        print(f"[phase A] {cfg['name']} done in {time.time() - t0:.0f}s")
    print('[phase A] complete')
