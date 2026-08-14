"""
utils.py — Shared infrastructure: seeding, checkpoints, API cost tracking.
Every phase checkpoints to config.CHECKPOINT_DIR as pickle and is resumable.
"""
import gc
import pickle
import random

import numpy as np

import config


def set_seed(seed: int = config.SEED):
    """Seed every RNG we use. Torch is optional at import time."""
    random.seed(seed)
    np.random.seed(seed)
    if config.torch is not None:
        config.torch.manual_seed(seed)
        if config.torch.cuda.is_available():
            config.torch.cuda.manual_seed_all(seed)


# ── Checkpoint scope ──────────────────────────────────────────────
def set_scope(n: int):
    """Point the checkpoint directory at the (n, CORPUS_VERSION) scope.

    Called by run_pipeline once N is known, so that `--smoke-test` (N=50)
    and `--full` (N=1000) cannot overwrite or silently reuse each other's
    checkpoints — see the comment on config.CHECKPOINT_DIR. An explicit
    RAG_CHECKPOINT_DIR always wins, so a deliberate override is never
    undone by a later phase.
    """
    import os
    if os.getenv('RAG_CHECKPOINT_DIR'):
        print(f'  [scope] RAG_CHECKPOINT_DIR set — keeping '
              f'{config.CHECKPOINT_DIR}')
        return config.CHECKPOINT_DIR
    config.CHECKPOINT_DIR = (config.CHECKPOINT_ROOT
                             / f'n{n}_{config.CORPUS_VERSION}')
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'  [scope] checkpoints -> {config.CHECKPOINT_DIR}')
    return config.CHECKPOINT_DIR


# ── Checkpoint helpers ────────────────────────────────────────────
def checkpoint_path(name: str):
    # Read config.CHECKPOINT_DIR at CALL time, never capture it at import —
    # set_scope() rebinds it after these modules are already imported.
    return config.CHECKPOINT_DIR / f'{name}.pkl'


def save_checkpoint(name: str, data):
    path = checkpoint_path(name)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f'  [ckpt] saved {path.name}')


def load_checkpoint(name: str):
    """Return checkpoint contents, or None if it does not exist."""
    path = checkpoint_path(name)
    if path.exists():
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f'  [ckpt] loaded {path.name}')
        return data
    return None


def checkpoint_exists(name: str) -> bool:
    return checkpoint_path(name).exists()


def free_memory():
    """Release model memory between phase-A models (one model at a time)."""
    gc.collect()
    if config.torch is not None and config.torch.cuda.is_available():
        config.torch.cuda.empty_cache()


# ── API cost tracking ─────────────────────────────────────────────
class CostTracker:
    """
    Running total across BOTH vendors:
      - Anthropic (Claude) for generation, phase C
      - OpenAI for text-embedding-3-small, one of the 7 embedding models

    The original notebook only tracked chat; embedding an entire corpus
    through the API was invisible to the budget (audit item C12).
    """
    # $/1M tokens — verify against current pricing before a full run.
    # claude-haiku-4-5
    CHAT_INPUT_PER_1M = 1.00
    CHAT_OUTPUT_PER_1M = 5.00
    # Cache writes cost 1.25x input, reads 0.1x. Each RAG query has a unique
    # retrieved context and the shared prefix is under the cache minimum, so
    # these should stay at zero — tracked to make it obvious if they don't.
    CACHE_WRITE_PER_1M = 1.25
    CACHE_READ_PER_1M = 0.10
    # text-embedding-3-small
    EMBED_PER_1M = 0.02

    def __init__(self):
        self.chat_input_tokens = 0
        self.chat_output_tokens = 0
        self.cache_write_tokens = 0
        self.cache_read_tokens = 0
        self.embed_tokens = 0
        self.requests = 0
        self.errors = 0

    def log_claude(self, usage):
        """Log an Anthropic Messages API `usage` object."""
        self.chat_input_tokens += usage.input_tokens
        self.chat_output_tokens += usage.output_tokens
        # These fields exist on the usage object but are often None/absent.
        self.cache_write_tokens += getattr(
            usage, 'cache_creation_input_tokens', 0) or 0
        self.cache_read_tokens += getattr(
            usage, 'cache_read_input_tokens', 0) or 0
        self.requests += 1

    def log_embedding(self, usage):
        """Log an OpenAI embeddings `usage` object."""
        self.embed_tokens += usage.total_tokens
        self.requests += 1

    def log_error(self):
        self.errors += 1

    @property
    def generation_cost(self) -> float:
        return (
            self.chat_input_tokens / 1e6 * self.CHAT_INPUT_PER_1M
            + self.chat_output_tokens / 1e6 * self.CHAT_OUTPUT_PER_1M
            + self.cache_write_tokens / 1e6 * self.CACHE_WRITE_PER_1M
            + self.cache_read_tokens / 1e6 * self.CACHE_READ_PER_1M
        )

    @property
    def embedding_cost(self) -> float:
        return self.embed_tokens / 1e6 * self.EMBED_PER_1M

    @property
    def cost(self) -> float:
        return self.generation_cost + self.embedding_cost

    def project(self, done: int, total: int) -> str:
        """Extrapolate the measured generation cost to a larger run."""
        if done <= 0:
            return 'no generations yet — nothing to project'
        per_query = self.generation_cost / done
        return (f'measured ${per_query:.6f}/query over {done:,} queries '
                f'-> ${per_query * total:,.2f} projected for {total:,}')

    def summary(self) -> str:
        return (
            f'API cost summary:\n'
            f'  Requests:            {self.requests:,}\n'
            f'  Claude in/out toks:  {self.chat_input_tokens:,} / '
            f'{self.chat_output_tokens:,}\n'
            f'  Claude cache w/r:    {self.cache_write_tokens:,} / '
            f'{self.cache_read_tokens:,}\n'
            f'  Embedding tokens:    {self.embed_tokens:,}\n'
            f'  Errors:              {self.errors}\n'
            f'  Generation cost:     ${self.generation_cost:.4f}\n'
            f'  Embedding cost:      ${self.embedding_cost:.4f}\n'
            f'  Total cost:          ${self.cost:.4f}'
        )


# Single shared instance, imported by embed_index.py and generate.py
cost_tracker = CostTracker()
