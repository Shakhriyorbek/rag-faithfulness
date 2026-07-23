"""
config.py — Central configuration for RAG faithfulness experiments.
Runs on SZTE gpu1 (NVIDIA V100).
"""
import os
from pathlib import Path

# ── Runtime ──
# torch is optional at import time so this file can be inspected before
# dependencies are installed (e.g. during local setup).
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    torch = None
    DEVICE = 'cpu'
    print('[config] Warning: torch not installed — DEVICE defaults to cpu.')

SEED = 42

# ── Experiment scale ──
N_QUERIES     = 1000     # Full run. Set to 50 for a quick smoke test.
TOP_K         = 5
RERANK_POOL   = 20       # Candidates retrieved per query; top-5 feed generation,
                         # all 20 form the re-ranking pool (Eq. 5 ablation).
CHUNK_SIZE    = 256
CHUNK_OVERLAP = 32
LAMBDA_RERANK = 0.6      # Re-ranking mix weight (Eq. 5)
RERANK_LAMBDAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # λ sensitivity sweep
LLAMA_SUBSET  = 300      # Llama-3 validation queries per dataset
ESA_N_SAMPLES = 200      # Queries per model×dataset for the ESA correlation

# ── NLI model (faithfulness, ESA, re-ranking) ──
# Label order is resolved from model config at load time (src/nli.py).
# NEVER hardcode a class index: for this checkpoint id2label is
# {0: contradiction, 1: entailment, 2: neutral} — index 2 is NOT entailment.
NLI_MODEL = 'cross-encoder/nli-deberta-v3-large'

# ── V100 memory handling ──
# 16GB V100: Llama-3-8B needs 8-bit. 32GB V100: fp16 is fine.
# Auto-detected at runtime in generate_llama.py
LLAMA_FORCE_8BIT = None  # None = auto-decide by VRAM; True/False to force

# ── Paths (all under home on gpu1) ──
BASE_DIR       = Path(os.getenv('RAG_BASE', str(Path.home() / 'rag_faithfulness')))
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
OUTPUT_DIR     = BASE_DIR / 'outputs'
HF_CACHE       = BASE_DIR / 'hf_cache'
for d in (CHECKPOINT_DIR, OUTPUT_DIR, HF_CACHE):
    d.mkdir(parents=True, exist_ok=True)

# Point HuggingFace at our cache (avoids filling home quota elsewhere)
os.environ.setdefault('HF_HOME', str(HF_CACHE))
os.environ.setdefault('TRANSFORMERS_CACHE', str(HF_CACHE))

# ── API keys (set as env vars, never hardcode) ──
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
HF_TOKEN       = os.getenv('HF_TOKEN', '')

# ── Datasets ──
DATASETS = ['NQ', 'HotpotQA', 'QASPER']

# ── Embedding models (7 total, incl. Jina per Berend's suggestion) ──
EMBEDDING_MODELS = [
    dict(name='all-mpnet-base-v2',      hf_id='sentence-transformers/all-mpnet-base-v2',
         paradigm='contrastive',        dim=768,  instruction=None),
    dict(name='GTE-large',              hf_id='thenlper/gte-large',
         paradigm='contrastive',        dim=1024, instruction=None),
    dict(name='BGE-M3',                 hf_id='BAAI/bge-m3',
         paradigm='multilingual',       dim=1024, instruction=None),
    # NOTE: 'intfloat/e5-large-instruct' does not exist on the Hub (verified
    # via HF API 2026-07-23). The instruction-tuned E5 checkpoint is
    # multilingual-e5-large-instruct, which uses the "Instruct: ...\nQuery: "
    # template on queries and raw text on passages.
    dict(name='E5-large-instruct',      hf_id='intfloat/multilingual-e5-large-instruct',
         paradigm='instruction-tuned',  dim=1024,
         instruction=('Instruct: Given a web search query, retrieve relevant '
                      'passages that answer the query\nQuery: ')),
    # Instructor models take [instruction, text] PAIRS, not prefixed strings —
    # handled by the instructor branch in embed_index.EmbeddingModelWrapper.
    dict(name='Instructor-XL',          hf_id='hkunlp/instructor-xl',
         paradigm='instruction-tuned',  dim=768,
         instruction='Represent the question for retrieving supporting documents: ',
         doc_instruction='Represent the document for retrieval: '),
    dict(name='text-embedding-3-small', hf_id='openai',
         paradigm='contrastive',        dim=1536, instruction=None),
    # ── Jina v3 (distillation-based multilingual, Berend point 2) ──
    # Requires trust_remote_code=True. Paper ref [8] currently cites a
    # non-existent "jina-embeddings-v5-text" — must be corrected to v3
    # (arXiv:2409.10173) in the next paper pass.
    dict(name='jina-embeddings-v3',     hf_id='jinaai/jina-embeddings-v3',
         paradigm='distilled',          dim=1024, instruction=None),
]

# ── Robustness analysis: metric combinations (Berend Point 3) ──
RETRIEVAL_METRICS    = ['ndcg@5', 'recall@5', 'mrr@5']
FAITHFULNESS_METRICS = ['alignscore', 'nli', 'mean']   # -> 3x3 = 9 RFG variants

if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Base dir: {BASE_DIR}")
    print(f"Models: {len(EMBEDDING_MODELS)} | Datasets: {len(DATASETS)}")
    print(f"RFG variants: {len(RETRIEVAL_METRICS)} x {len(FAITHFULNESS_METRICS)} = {len(RETRIEVAL_METRICS)*len(FAITHFULNESS_METRICS)}")
