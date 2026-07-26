"""
generate.py — Phase C: answer generation.

Generators (paper §4.4):
  - Claude (config.CLAUDE_MODEL) via the Anthropic API — the closed-source
    generator, all 7 embedding models × 3 datasets. Cost-tracked,
    incremental checkpoints every 100 queries, resumable mid-dataset.
  - Llama-3-8B-Instruct on the V100 (validation subset) — the open-source
    generator. VRAM is probed at load: <20 GB -> 8-bit quantization
    (config.LLAMA_FORCE_8BIT overrides).

Claude replaced GPT-4o-mini on 2026-07-26; paper §4.4/§5.2 need updating.

Both generators receive the BYTE-IDENTICAL prompt — one user turn carrying
the full template. That is deliberate: H3 compares faithfulness rankings
across generators, so any prompt difference would confound the comparison.
It is why the instruction is not hoisted into Claude's `system` parameter,
which would otherwise be the idiomatic choice.

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


# ── Claude ────────────────────────────────────────────────────────
class ClaudeGenerator:
    """
    Anthropic Messages API generator.

    Model-specific notes for claude-haiku-4-5:
      - `temperature=0` is accepted (sampling params are only removed on
        Opus 4.7+ / Opus 5 / Sonnet 5 / Fable 5), so the paper's
        temperature-0 protocol carries over unchanged.
      - `output_config.effort` ERRORS on Haiku 4.5 — never pass it.
      - `thinking` is omitted, which on this model means no thinking. That
        is what we want: the experiment measures grounding in retrieved
        context, not reasoning depth, and it keeps parity with Llama-3.
    """

    def __init__(self, model: str = None, max_retries: int = 5):
        import anthropic
        # Credentials resolve from the environment (ANTHROPIC_API_KEY, or an
        # `ant auth login` profile). Never hardcoded.
        self.client = anthropic.Anthropic(max_retries=max_retries)
        self.model = model or config.CLAUDE_MODEL
        self._anthropic = anthropic

    def generate(self, question: str, chunks: List[str]) -> str:
        import anthropic
        prompt = build_prompt(question, chunks)
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0,
                messages=[{'role': 'user', 'content': prompt}],
            )
        except anthropic.APIStatusError as e:
            cost_tracker.log_error()
            return f'[ERROR: {type(e).__name__} {e.status_code}: {e.message}]'
        except anthropic.APIConnectionError as e:
            cost_tracker.log_error()
            return f'[ERROR: APIConnectionError: {e}]'

        cost_tracker.log_claude(resp.usage)

        # Claude 4+ can decline; content is then empty or partial. Treat it
        # as a scored-out row rather than letting content[0] raise.
        if resp.stop_reason == 'refusal':
            cost_tracker.log_error()
            return '[ERROR: refusal]'

        text = ''.join(b.text for b in resp.content if b.type == 'text')
        return text.strip()


def run_phase_c(datasets: Dict, model_names: List[str] = None,
                budget_limit: float = 60.0, project_to: int = None):
    """
    Claude generation for every model × dataset. Resumable.

    budget_limit — hard stop. Raises rather than draining the account;
                   partial checkpoints survive, so raising it resumes.
    project_to   — if set, extrapolate measured cost to this many queries
                   (the smoke-test cost probe before committing to a full run).
    """
    gen = ClaudeGenerator()
    model_list = [c['name'] for c in config.EMBEDDING_MODELS
                  if not model_names or c['name'] in model_names]
    n_generated = 0
    # Fail-fast state. A bad API key produces an error on EVERY call; without
    # this the phase "succeeds" with 100% errors, $0.00 cost and a saved
    # checkpoint full of '[ERROR: ...]' strings — which then makes a rerun
    # skip the work entirely. At full scale that is 21,000 silent failures.
    consecutive_errors = 0
    first_error = None
    MAX_CONSECUTIVE_ERRORS = 5
    MAX_ERROR_RATE = 0.20

    for model in model_list:
        for ds_name in datasets:
            ck = f'generated_claude_{model}_{ds_name}'
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
                    'generator': 'claude',
                    'generator_model': gen.model,
                })
                n_generated += 1

                if answer.startswith('[ERROR'):
                    consecutive_errors += 1
                    if first_error is None:
                        first_error = answer
                        print(f'    !! first API error: {answer[:300]}')
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        raise RuntimeError(
                            f'{consecutive_errors} consecutive API failures — '
                            f'aborting before burning the full run.\n'
                            f'First error: {first_error}\n'
                            f'Check ANTHROPIC_API_KEY (a valid key is ~100-110 '
                            f'chars; a 401 usually means it is wrong, truncated '
                            f'or has stray characters).')
                else:
                    consecutive_errors = 0

                if (i + 1) % 100 == 0:
                    save_checkpoint(ck + '_partial', generations)
                    print(f'    {i + 1}/{len(retrievals)} | '
                          f'running cost ${cost_tracker.cost:.4f}')
                    if cost_tracker.cost > budget_limit:
                        raise RuntimeError(
                            f'Cost ${cost_tracker.cost:.2f} exceeded the '
                            f'${budget_limit} budget — stopping. Partial '
                            f'checkpoint saved; raise budget_limit to resume.')
            # Never persist a checkpoint that is mostly errors: its existence
            # makes every future run skip this model/dataset silently.
            n_err = sum(1 for g in generations
                        if g['generated_answer'].startswith('[ERROR'))
            rate = n_err / max(1, len(generations))
            if rate > MAX_ERROR_RATE:
                raise RuntimeError(
                    f'{model}/{ds_name}: {n_err}/{len(generations)} generations '
                    f'failed ({rate:.0%}) — refusing to save a poisoned '
                    f'checkpoint.\nFirst error: {first_error}')
            save_checkpoint(ck, generations)
            print(f'  {model}/{ds_name}: {len(generations)} answers'
                  + (f' ({n_err} errors)' if n_err else ''))

    print(cost_tracker.summary())
    if project_to:
        print(f'  COST PROBE: {cost_tracker.project(n_generated, project_to)}')
    if cost_tracker.requests == 0 and n_generated > 0:
        raise RuntimeError(
            'Phase C made ZERO successful API requests. Nothing was generated; '
            'the cost probe above is meaningless. Check ANTHROPIC_API_KEY.')
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
