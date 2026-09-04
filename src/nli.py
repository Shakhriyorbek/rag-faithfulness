"""
nli.py — Shared NLI entailment scorer for faithfulness, ESA, and re-ranking.

Fixes two audit findings:

B1 — The notebook hardcoded probs[2] as "entailment". For
     cross-encoder/nli-deberta-v3-large the label order is
     {0: contradiction, 1: entailment, 2: neutral}, so probs[2] is the
     NEUTRAL probability. The index is resolved from model config here,
     once, at load time.

B5 — The notebook concatenated all 5 retrieved chunks (~1,280 tokens) into
     one premise and truncated at 512, so chunks 3-5 never influenced the
     score. score_chunks() scores each chunk separately and aggregates.
"""
from typing import List, Tuple

import numpy as np
import textnorm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

# torch renamed this: torch.cuda.OutOfMemoryError in 2.0-2.3, torch.OutOfMemoryError
# after. Catch whichever this install has rather than pinning a version.
_OOM = tuple({e for e in (getattr(torch, 'OutOfMemoryError', None),
                          getattr(torch.cuda, 'OutOfMemoryError', None))
              if e is not None} or {RuntimeError})


class NLIScorer:
    """DeBERTa-v3-large NLI scorer with config-resolved entailment index."""

    def __init__(self, model_name: str = config.NLI_MODEL, device: str = None):
        self.device = device or config.DEVICE
        print(f'Loading NLI model {model_name}...')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device).eval()

        # ── B1 fix: resolve the entailment index from the model config ──
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        if 'entailment' not in label2id:
            raise ValueError(
                f'No "entailment" label in {model_name} config: '
                f'{self.model.config.id2label}'
            )
        self.entailment_idx = label2id['entailment']
        print(f'  labels: {self.model.config.id2label}')
        print(f'  entailment index: {self.entailment_idx}')

        self.batch_size = config.NLI_BATCH_SIZE or self._pick_batch_size()
        print(f'  batch size: {self.batch_size} '
              f'({self._free_mib()} MiB free)')

    # Grow the batch back only after this many clean batches, and only with
    # this much VRAM spare, so a run does not oscillate against a co-tenant
    # that is itself allocating.
    _GROW_AFTER = 200
    _GROW_FREE_MIB = 1500

    def _free_mib(self) -> int:
        if not str(self.device).startswith('cuda'):
            return 1 << 30
        return torch.cuda.mem_get_info()[0] // 2 ** 20

    def _pick_batch_size(self) -> int:
        """Choose a batch size from the VRAM actually free after model load.

        gpu1 is shared and a co-tenant routinely holds 27 of its 32 GB.
        Measured on this box with ~760 MiB free after the model loads: batch 2
        fits at a 1,917 MiB peak, batch 4 does not. The allocator-reported
        delta between batch 1 and 2 is only ~118 MiB, so the true cost of a
        pair is well above what the deltas suggest — fragmentation and
        non-PyTorch overhead dominate. Hence the deliberately pessimistic
        300 MiB per pair; being wrong here costs a crash hours into a run.
        """
        if not str(self.device).startswith('cuda'):
            return config.NLI_BATCH_MAX
        return max(1, min(config.NLI_BATCH_MAX, self._free_mib() // 300))

    @torch.no_grad()
    def _forward(self, batch: List[Tuple[str, str]]) -> List[float]:
        inputs = self.tokenizer(
            [p for p, _ in batch], [h for _, h in batch],
            return_tensors='pt', truncation=True,
            max_length=512, padding=True
        ).to(self.device)
        logits = self.model(**inputs).logits
        return torch.softmax(
            logits, dim=-1)[:, self.entailment_idx].cpu().tolist()

    def entailment_probs(self, pairs: List[Tuple[str, str]],
                         batch_size: int = None) -> List[float]:
        """P(entailment | premise, hypothesis) for a list of (premise, hypothesis).

        The batch size adapts unless one is passed explicitly. On out of
        memory the failing slice is retried at half the size rather than
        skipped — dropping it would silently shorten the returned list and
        misalign every downstream index in score_claims. After a long clean
        run the size is allowed to grow again, so a run does not stay
        crippled for hours because the GPU was briefly busy when it started.
        """
        pinned = batch_size is not None
        bs = batch_size or self.batch_size
        probs: List[float] = []
        i, streak = 0, 0
        while i < len(pairs):
            batch = pairs[i:i + bs]
            try:
                probs.extend(self._forward(batch))
            except _OOM:
                if str(self.device).startswith('cuda'):
                    torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                streak = 0
                if not pinned:
                    self.batch_size = bs
                    print(f'  [nli] out of memory — batch size -> {bs}')
                continue                      # retry the SAME slice, smaller
            i += len(batch)
            streak += 1
            if (not pinned and streak >= self._GROW_AFTER
                    and bs < config.NLI_BATCH_MAX
                    and self._free_mib() > self._GROW_FREE_MIB):
                bs = min(config.NLI_BATCH_MAX, bs * 2)
                self.batch_size = bs
                streak = 0
                print(f'  [nli] memory freed — batch size -> {bs}')
        return probs

    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        return self.entailment_probs([(premise, hypothesis)])[0]

    def score_chunks(self, chunks: List[str], hypothesis: str) -> dict:
        """
        B5 fix: score the hypothesis against EACH retrieved chunk (each pair
        fits in the 512-token window), plus against a truncated concatenation
        (helps multi-hop answers no single chunk entails).

        Returns per-chunk probabilities and three aggregates. Downstream
        default is 'nli_max' = max(per-chunk max, concat) — "is the answer
        entailed by any part of the retrieved context".
        """
        # Strip markdown so every evaluator reads the SAME hypothesis.
        # Before 2026-09-03 only the claim-level scorer stripped, so nli_max
        # and claim_min were not comparable on the 74.9% of Claude's answers
        # that carry formatting, and the single-assertion identity failed on
        # them by up to 0.729. See textnorm.strip_markdown.
        hypothesis = textnorm.strip_markdown(hypothesis)
        pairs = [(c, hypothesis) for c in chunks]
        concat = ' '.join(chunks)  # tokenizer truncates at 512
        pairs.append((concat, hypothesis))
        probs = self.entailment_probs(pairs)
        per_chunk, concat_prob = probs[:-1], probs[-1]
        return {
            'per_chunk': per_chunk,
            'nli_concat': concat_prob,
            'nli_mean': float(np.mean(per_chunk)) if per_chunk else 0.0,
            'nli_max': max([*per_chunk, concat_prob]) if probs else 0.0,
        }


if __name__ == '__main__':
    # Sanity check: requires transformers + torch + model download.
    scorer = NLIScorer()
    p_ent = scorer.entailment_prob(
        'The Eiffel Tower is located in Paris, France.',
        'The Eiffel Tower is in France.'
    )
    p_con = scorer.entailment_prob(
        'The Eiffel Tower is located in Paris, France.',
        'The Eiffel Tower is in Berlin.'
    )
    print(f'entail-case P(ent) = {p_ent:.4f} (expect high)')
    print(f'contra-case P(ent) = {p_con:.4f} (expect low)')
    assert p_ent > 0.5 > p_con, 'NLI scorer failed sanity check'
    print('OK')
