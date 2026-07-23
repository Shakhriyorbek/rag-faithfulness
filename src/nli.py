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
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


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

    @torch.no_grad()
    def entailment_probs(self, pairs: List[Tuple[str, str]],
                         batch_size: int = 16) -> List[float]:
        """P(entailment | premise, hypothesis) for a list of (premise, hypothesis)."""
        probs = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            inputs = self.tokenizer(
                [p for p, _ in batch], [h for _, h in batch],
                return_tensors='pt', truncation=True,
                max_length=512, padding=True
            ).to(self.device)
            logits = self.model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, self.entailment_idx]
            probs.extend(batch_probs.cpu().tolist())
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
