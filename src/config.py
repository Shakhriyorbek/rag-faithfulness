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
LLAMA_SUBSET  = 300      # open-weight validation queries per dataset

# ── open-weight generator arm ──
# Berend asked for Qwen or Gemma rather than Llama-3. Qwen2.5-7B-Instruct is
# UNGATED, which removes the HF_TOKEN blocker that has held this arm up; it
# also fits fp16 on the 32GB V100. Under the v6 framing this arm does real
# work: the perturbation effect scales with how many assertions an answer
# contains, so a third generator tests that mechanism rather than only
# reproducing H3. The label goes into the checkpoint key, so changing the
# model without changing the label would silently mix two models' answers.
OPEN_MODEL_ID    = 'Qwen/Qwen2.5-7B-Instruct'
OPEN_MODEL_LABEL = 'qwen'
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

# ── Corpus semantics version ──
# Bump whenever the corpus, gold provenance or relevance definition changes,
# so a cache built under the old semantics can never be silently reused.
# (Lives here rather than in datasets_loader because CHECKPOINT_DIR below
# depends on it, and datasets_loader imports config.)
#   v1  original loaders
#   v2  B6: NQ context centred on the answer span; NQ relevance is
#       answer-bearing rather than document-level (2026-08-13)
#   v3  B8: NQ samples whose answer is genuinely absent from the retained
#       window are dropped and re-drawn, and containment is decided by
#       textnorm.contains rather than a whitespace-only test (2026-08-14)
CORPUS_VERSION = 'v3'

# ── Paths (all under home on gpu1) ──
BASE_DIR        = Path(os.getenv('RAG_BASE', str(Path.home() / 'rag_faithfulness')))
CHECKPOINT_ROOT = BASE_DIR / 'checkpoints'

# ⚠️ Checkpoints are scoped by (N, CORPUS_VERSION) — do not flatten this.
#
# Every checkpoint key in this project is named for its CONTENT
# (`retrieval_BGE-M3_NQ`, `generated_claude_BGE-M3_NQ`, ...) and not for the
# run that produced it. With one flat directory the N=50 pilot and the
# N=1000 full run collide on every single key, and because each phase skips
# work whose checkpoint already exists, the full run "succeeds" in seconds
# by reusing pilot data. That is exactly what happened on 2026-08-14:
#
#     [phase A] all-mpnet-base-v2: all datasets done, skipping
#     [ckpt] loaded retrieval_quality_all.pkl
#     === pipeline done ===  EXIT=0
#
# Phase A skipped all three models, Phase B returned the pilot's numbers,
# and the process exited 0. Had it reached the paid phases it would have
# skipped generation too and handed 50 stale rows to the analysis as if they
# were the full run. Putting the scope in the DIRECTORY rather than in every
# key keeps all existing key names working and makes collisions impossible.
#
# RAG_SCOPE_N       — pick a different N's scope (e.g. 50 for the pilot)
# RAG_CHECKPOINT_DIR — absolute override, e.g. to read the pre-scope pilot
#                      checkpoints that still sit flat in checkpoints/
_SCOPE_N       = int(os.getenv('RAG_SCOPE_N', N_QUERIES))
CHECKPOINT_DIR = Path(os.getenv(
    'RAG_CHECKPOINT_DIR',
    str(CHECKPOINT_ROOT / f'n{_SCOPE_N}_{CORPUS_VERSION}')))
OUTPUT_DIR     = BASE_DIR / 'outputs'
HF_CACHE       = BASE_DIR / 'hf_cache'
for d in (CHECKPOINT_DIR, OUTPUT_DIR, HF_CACHE):
    d.mkdir(parents=True, exist_ok=True)

# Point HuggingFace at our cache (avoids filling home quota elsewhere)
os.environ.setdefault('HF_HOME', str(HF_CACHE))
os.environ.setdefault('TRANSFORMERS_CACHE', str(HF_CACHE))

# ── API keys (set as env vars, never hardcode) ──
# ANTHROPIC_API_KEY — Claude, the closed-source generator (phase C).
# OPENAI_API_KEY    — still required: text-embedding-3-small is one of the 7
#                     embedding models under study. Anthropic has no
#                     embeddings API, so this is NOT interchangeable.
# HF_TOKEN          — Llama-3-8B is a gated model.
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
OPENAI_API_KEY    = os.getenv('OPENAI_API_KEY', '')
HF_TOKEN          = os.getenv('HF_TOKEN', '')

# ── Generator (paper §4.4) ──
# Claude replaces GPT-4o-mini as the closed-source generator (decided
# 2026-07-26). Haiku 4.5 is the closest analog to GPT-4o-mini in capability
# tier and cost, so the paper's design intent — a small, widely-deployed
# closed-source model — is preserved. Paper §4.4 and §5.2 must be updated.
#
# Model-specific API notes for claude-haiku-4-5:
#   - `temperature=0` IS accepted (sampling params are only removed on
#     Opus 4.7+ / Opus 5 / Sonnet 5 / Fable 5), so the paper's
#     temperature-0 protocol carries over unchanged.
#   - `output_config.effort` ERRORS on Haiku 4.5 — never pass it.
#   - Omitting `thinking` means no thinking, which is what we want: this
#     measures grounding in retrieved context, not reasoning depth.
CLAUDE_MODEL = 'claude-haiku-4-5'

# Second closed-source generator, restored 2026-08-14 at Berend's request.
# The paper's §4.4, §5.2 and H3 all still name GPT-4o-mini, and running BOTH
# is strictly better than having swapped one for the other: H3 asks whether
# the faithfulness ranking of embedders survives a change of generator, and
# two closed-source arms plus Llama-3 test that far better than one.
# Same protocol as Claude — temperature 0, max 256 tokens, and the
# byte-identical single-user-turn prompt from generate.build_prompt.
GPT_MODEL = 'gpt-4o-mini'

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

# AlignScore lives in an ISOLATED install (~/align_env) because it pins
# transformers 4.26 against the 5.x the rest of the pipeline uses. Run its
# phase with PYTHONPATH=~/align_env; see RUNBOOK. The checkpoint is a separate
# ~455 MB download and is NOT bundled with the pip package.
ALIGNSCORE_CKPT = os.getenv(
    'ALIGNSCORE_CKPT', str(BASE_DIR / 'alignscore' / 'AlignScore-large.ckpt'))

if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Base dir: {BASE_DIR}")
    print(f"Models: {len(EMBEDDING_MODELS)} | Datasets: {len(DATASETS)}")
    print(f"RFG variants: {len(RETRIEVAL_METRICS)} x {len(FAITHFULNESS_METRICS)} = {len(RETRIEVAL_METRICS)*len(FAITHFULNESS_METRICS)}")
