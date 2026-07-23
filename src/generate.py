"""
generate.py — Phase C: answer generation.

Generators:
  - GPT-4o-mini via the OpenAI API (all 7 models × 3 datasets), cost-tracked,
    incremental checkpoints every 100 queries, resumable mid-dataset.
  - Llama-3-8B-Instruct on the V100 (validation subset). VRAM is probed at
    load: <20 GB -> 8-bit quantization (config.LLAMA_FORCE_8BIT overrides).

Audit fixes:
  C13 — greedy decoding is do_sample=False with NO temperature argument
        (the notebook passed temperature=1.0 alongside do_sample=False);
        the completion is sliced by token count, not len(prompt) chars.
  --  Llama-3-Instruct gets its chat template applied instead of a raw
        completion prompt.
"""
import time
from typing import Dict, List

import config
from utils import (checkpoint_exists, cost_tracker, load_checkpoint,
                   save_checkpoint)

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the provided context.
Do not use any external knowledge. If the context does not contain enough information, say "I cannot answer based on the provided context."

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: List[str]) -> str:
    context = '\n\n'.join(f'[Chunk {i + 1}]: {c}'
                          for i, c in enumerate(chunks))
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


# ── GPT-4o-mini ───────────────────────────────────────────────────
class Gpt4oMiniGenerator:
    def __init__(self, model: str = 'gpt-4o-mini'):
        from openai import OpenAI
        self.client = OpenAI()  # OPENAI_API_KEY from env, never hardcoded
        self.model = model

    def generate(self, question: str, chunks: List[str],
                 max_retries: int = 3) -> str:
        prompt = build_prompt(question, chunks)
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0,
                    max_tokens=256,
                )
                cost_tracker.log_chat(resp.usage)
                return resp.choices[0].message.content.strip()
            except Exception as e:
                cost_tracker.log_error()
                if attempt == max_retries - 1:
                    return f'[ERROR: {e}]'
                time.sleep(2 ** attempt)


def run_phase_c(datasets: Dict, model_names: List[str] = None,
                budget_limit: float = 15.0):
    """GPT-4o-mini generation for every model × dataset. Resumable."""
    gen = Gpt4oMiniGenerator()
    model_list = [c['name'] for c in config.EMBEDDING_MODELS
                  if not model_names or c['name'] in model_names]

    for model in model_list:
        for ds_name in datasets:
            ck = f'generated_gpt4o_{model}_{ds_name}'
            if checkpoint_exists(ck):
                print(f'  [skip] {ck}')
                continue
            retrievals = load_checkpoint(f'retrieval_{model}_{ds_name}')
            if not retrievals:
                print(f'  missing retrieval_{model}_{ds_name} — run Phase A')
                continue

            # Resume from partial checkpoint if a previous run died mid-way
            generations = load_checkpoint(ck + '_partial') or []
            done = len(generations)
            print(f'  generating [{model}] [{ds_name}] '
                  f'({done}/{len(retrievals)} done)...')
            for i, r in enumerate(retrievals[done:], start=done):
                answer = gen.generate(
                    r['question'], r['retrieved_texts'][:config.TOP_K])
                generations.append({
                    **r,
                    'generated_answer': answer,
                    'generator': 'gpt4o-mini',
                })
                if (i + 1) % 100 == 0:
                    save_checkpoint(ck + '_partial', generations)
                    print(f'    {i + 1}/{len(retrievals)} | '
                          f'running cost ${cost_tracker.cost:.4f}')
                    if cost_tracker.cost > budget_limit:
                        raise RuntimeError(
                            f'Cost ${cost_tracker.cost:.2f} exceeded the '
                            f'${budget_limit} budget — stopping. Partial '
                            f'checkpoint saved; raise budget_limit to resume.')
            save_checkpoint(ck, generations)
            print(f'  {model}/{ds_name}: {len(generations)} answers')

    print(cost_tracker.summary())
    print('[phase C] complete')


# ── Llama-3-8B-Instruct ───────────────────────────────────────────
class Llama3Generator:
    MODEL_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        assert torch.cuda.is_available(), 'Llama-3 requires a GPU'

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        use_8bit = (config.LLAMA_FORCE_8BIT if config.LLAMA_FORCE_8BIT
                    is not None else vram_gb < 20)
        print(f'Loading Llama-3-8B-Instruct '
              f'(VRAM {vram_gb:.0f} GB -> {"8-bit" if use_8bit else "fp16"})...')

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, token=config.HF_TOKEN or None)
        kwargs = dict(device_map='auto', token=config.HF_TOKEN or None)
        if use_8bit:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_8bit=True)
        else:
            kwargs['torch_dtype'] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, **kwargs).eval()
        self.torch = torch
        print('  loaded')

    def generate(self, question: str, chunks: List[str]) -> str:
        prompt = build_prompt(question, chunks)
        # Instruct model -> chat template, not raw completion
        input_ids = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            add_generation_prompt=True, return_tensors='pt',
            truncation=True, max_length=4096,
        ).to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,  # C13: greedy — no temperature argument
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # C13: slice by token count, not character count
        new_tokens = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens,
                                     skip_special_tokens=True).strip()


def run_phase_llama(datasets: Dict, model_names: List[str] = None,
                    subset: int = config.LLAMA_SUBSET):
    """
    Llama-3 validation subset (H3: generator independence).
    Default: the 3 core models, `subset` queries per dataset.
    NLI scoring happens in Phase D against these checkpoints.
    """
    val_models = model_names or ['all-mpnet-base-v2', 'E5-large-instruct',
                                 'BGE-M3']
    llama = Llama3Generator()

    for model in val_models:
        for ds_name in datasets:
            ck = f'generated_llama3_{model}_{ds_name}'
            if checkpoint_exists(ck):
                print(f'  [skip] {ck}')
                continue
            retrievals = load_checkpoint(f'retrieval_{model}_{ds_name}')
            if not retrievals:
                print(f'  missing retrieval_{model}_{ds_name} — run Phase A')
                continue

            todo = retrievals[:subset]
            generations = load_checkpoint(ck + '_partial') or []
            done = len(generations)
            print(f'  Llama-3 [{model}] [{ds_name}] '
                  f'({done}/{len(todo)} done)...')
            for i, r in enumerate(todo[done:], start=done):
                answer = llama.generate(
                    r['question'], r['retrieved_texts'][:config.TOP_K])
                generations.append({
                    **r,
                    'generated_answer': answer,
                    'generator': 'llama3',
                })
                if (i + 1) % 50 == 0:
                    save_checkpoint(ck + '_partial', generations)
                    print(f'    {i + 1}/{len(todo)}')
            save_checkpoint(ck, generations)
            print(f'  {model}/{ds_name}: {len(generations)} answers')
    print('[phase Llama] complete')
