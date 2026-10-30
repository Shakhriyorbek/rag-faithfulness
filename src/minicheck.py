"""
minicheck.py — Bespoke-MiniCheck-7B as a fifth evaluator (third review, D3).

Every evaluator in the paper so far is either a DeBERTa entailment model under
some aggregation or AlignScore, a RoBERTa classifier. The review asked whether
the structural bias reaches a generative LLM checker, which reads the whole
document and the claim in one prompt and answers Yes or No. This module
reproduces the published MiniCheck inference (Tang et al. 2024,
github.com/Liyan06/MiniCheck, minicheck/inference.py) in plain transformers,
because vLLM is not usable on the V100:

- the system and user prompts are copied verbatim from minicheck/utils.py;
- the score is the probability mass on "yes" among the top-5 first-token
  candidates, as in get_support_prob;
- the document is one chunk: their chunk size is 32,468 tokens and our five
  retrieved chunks are about 1,500, so the chunk maximum is the identity.

Two aggregates come out of one pass, because MiniCheck as published is
claim-level: it splits the claim with nltk and takes the minimum over
sentences. ``whole`` scores the full answer as one claim, the single-number
setting the paper studies; ``sent`` is the published aggregation. The model
is open-weight (CC BY-NC 4.0) and runs locally, so the arm costs nothing.
"""
from typing import Dict, List

import torch
from huggingface_hub import hf_hub_download
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerFast)

import config

MODEL_ID = 'bespokelabs/Bespoke-MiniCheck-7B'
# The model ships its own InternLM2 code (trust_remote_code). Pinned to the
# revision whose four .py files were read before first use (2026-10-29), so a
# later upload cannot change what runs.
REVISION = '1ed7786bcda3fa1dc35f7c4ed9e3f36b785d33b8'

# Verbatim from minicheck/utils.py.
SYSTEM_PROMPT = """Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. Please assess the claim's consistency with the document by responding with either "Yes" or "No"."""
USER_PROMPT = """Document: [DOCUMENT]\nClaim: [CLAIM]"""


def split_sentences(text: str) -> List[str]:
    """MiniCheck's sent_tokenize_with_newlines: newline blocks, then nltk."""
    from nltk.tokenize import sent_tokenize
    blocks = [b for b in text.split('\n') if b.strip()]
    return [s for b in blocks for s in sent_tokenize(b)] or [text]


class MiniCheckScorer:
    def __init__(self, model_id: str = MODEL_ID, device: str = None,
                 batch_size: int = 4):
        self.device = device or config.DEVICE
        self.batch_size = batch_size
        print(f'Loading {model_id}...')
        # The shipped InternLM2 tokenizer class is used only for its chat
        # template. Under transformers 5 its encoder splits text into single
        # characters and drops spaces ("system" -> s y s t e m), so prompts
        # are encoded with the repository's own tokenizer.json instead.
        self.tok = AutoTokenizer.from_pretrained(
            model_id, revision=REVISION, trust_remote_code=True)
        self.enc = PreTrainedTokenizerFast(
            tokenizer_file=hf_hub_download(model_id, 'tokenizer.json',
                                           revision=REVISION))
        self.enc.padding_side = 'left'
        self.enc.pad_token = self.tok.pad_token or '</s>'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=REVISION, trust_remote_code=True,
            dtype=torch.float16,
        ).to(self.device).eval()
        # transformers 5 builds the model on the meta device and leaves
        # non-persistent buffers uninitialised; InternLM2's rotary inv_freq is
        # one, and with it garbage every logit is NaN. Recompute it exactly as
        # InternLM2RotaryEmbedding.__init__ does.
        n_fixed = 0
        for mod in self.model.modules():
            if hasattr(mod, 'inv_freq') and hasattr(mod, 'base'):
                mod.inv_freq = 1.0 / (mod.base ** (
                    torch.arange(0, mod.dim, 2, dtype=torch.int64).float()
                    .to(self.device) / mod.dim))
                n_fixed += 1
        print(f'  rotary inv_freq recomputed in {n_fixed} modules')
        # The same scores recur: the untouched answer's sentences are shared
        # with the falsified one except for the edited sentence.
        self._cache: Dict[tuple, float] = {}

    def _prompt(self, doc: str, claim: str) -> str:
        msg = [{'role': 'system', 'content': SYSTEM_PROMPT},
               {'role': 'user', 'content': USER_PROMPT.replace('[DOCUMENT]', doc)
                                                      .replace('[CLAIM]', claim)}]
        return self.tok.apply_chat_template(msg, add_generation_prompt=True,
                                            tokenize=False)

    @torch.no_grad()
    def _support(self, prompts: List[str]) -> List[float]:
        out = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self.enc(prompts[i:i + self.batch_size], return_tensors='pt',
                           padding=True, add_special_tokens=False).to(self.device)
            # use_cache=False: the shipped InternLM2 code calls
            # DynamicCache.from_legacy_cache, which transformers 5 removed, and
            # one forward pass needs no cache. Positions are counted from the
            # first real token so left padding does not shift them.
            pos = (enc['attention_mask'].cumsum(-1) - 1).clamp(min=0)
            logits = self.model(**enc, position_ids=pos,
                                use_cache=False).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, 5, dim=-1)
            for p_row, i_row in zip(top.values.tolist(), top.indices.tolist()):
                out.append(sum(p for p, t in zip(p_row, i_row)
                               if self.enc.decode([t]).lower() == 'yes'))
        return out

    def support(self, doc: str, claims: List[str]) -> List[float]:
        # Bound the cache before filling it: clearing after the fill would
        # drop the entries the return line is about to read.
        if len(self._cache) > 20000:
            self._cache.clear()
        todo = [c for c in dict.fromkeys(claims) if (doc, c) not in self._cache]
        if todo:
            for c, s in zip(todo, self._support([self._prompt(doc, c) for c in todo])):
                self._cache[(doc, c)] = s
        return [self._cache[(doc, c)] for c in claims]

    def score(self, chunks: List[str], answer: str) -> Dict[str, float]:
        doc = ' '.join(chunks)
        sents = split_sentences(answer)
        scores = self.support(doc, [answer] + sents)
        return {'minicheck': scores[0], 'minicheck_sent': min(scores[1:])}
