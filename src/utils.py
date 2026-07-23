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


# ── Checkpoint helpers ────────────────────────────────────────────
def checkpoint_path(name: str):
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
    Running total for ALL OpenAI usage — chat AND embeddings.
    The original notebook only tracked chat; embedding an entire corpus
    through the API was invisible to the budget (audit item C12).
    """
    # $/1M tokens (verify against current pricing before a full run)
    CHAT_INPUT_PER_1M = 0.15    # gpt-4o-mini input
    CHAT_OUTPUT_PER_1M = 0.60   # gpt-4o-mini output
    EMBED_PER_1M = 0.02         # text-embedding-3-small

    def __init__(self):
        self.chat_input_tokens = 0
        self.chat_output_tokens = 0
        self.embed_tokens = 0
        self.requests = 0
        self.errors = 0

    def log_chat(self, usage):
        self.chat_input_tokens += usage.prompt_tokens
        self.chat_output_tokens += usage.completion_tokens
        self.requests += 1

    def log_embedding(self, usage):
        self.embed_tokens += usage.total_tokens
        self.requests += 1

    def log_error(self):
        self.errors += 1

    @property
    def cost(self) -> float:
        return (
            self.chat_input_tokens / 1e6 * self.CHAT_INPUT_PER_1M
            + self.chat_output_tokens / 1e6 * self.CHAT_OUTPUT_PER_1M
            + self.embed_tokens / 1e6 * self.EMBED_PER_1M
        )

    def summary(self) -> str:
        return (
            f'API cost summary:\n'
            f'  Requests:          {self.requests:,}\n'
            f'  Chat in/out toks:  {self.chat_input_tokens:,} / {self.chat_output_tokens:,}\n'
            f'  Embedding tokens:  {self.embed_tokens:,}\n'
            f'  Errors:            {self.errors}\n'
            f'  Total cost:        ${self.cost:.4f}'
        )


# Single shared instance, imported by embed_index.py and generate.py
cost_tracker = CostTracker()
