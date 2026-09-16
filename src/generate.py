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


# W2 control. The falsification gap in Section V tracks assertion count, but
# assertion count is confounded with model family: the three generators differ
# in provider and training as well as in verbosity. Varying the PROMPT on a
# single model holds the family fixed and moves only the answer structure,
# which a fourth model could not do.
#
# The grounding instruction and the refusal sentence are byte-identical to the
# standard template. Only an instruction to answer more fully is added, so the
# abstention rule still matches and the grounding constraint is unchanged.
RAG_PROMPT_TEMPLATE_VERBOSE = """You are a helpful assistant. Answer the question using ONLY the provided context.
Do not use any external knowledge. If the context does not contain enough information, say "I cannot answer based on the provided context."
Give a thorough answer: state every relevant detail the context provides, and explain how the context supports each part of your answer.

Context:
{context}

Question: {question}

Answer:"""

PROMPT_STYLES = {'standard': RAG_PROMPT_TEMPLATE,
                 'verbose': RAG_PROMPT_TEMPLATE_VERBOSE}


def build_prompt(question: str, chunks: List[str],
                 style: str = 'standard') -> str:
    """Build the RAG prompt.

    `style` defaults to 'standard', which every generator in the main grid
    uses and which tests/test_openai_generator.py::TestPromptParity pins as
    byte-identical across arms. Do not change that default: H3 compares
    rankings across generators, so a prompt difference between them would
    confound it. 'verbose' exists only for the single-model control in
    Section~\ref{sec:verbosity}, which is reported under its own label.
    """
    template = PROMPT_STYLES[style]
    context = '\n\n'.join(f'[Chunk {i + 1}]: {c}'
                          for i, c in enumerate(chunks))
    return template.format(context=context, question=question)


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


# ── GPT-4o-mini ───────────────────────────────────────────────────
class OpenAIGenerator:
    """
    OpenAI chat.completions generator (config.GPT_MODEL).

    Restored 2026-08-14 after being removed in the 2026-07-26 Claude
    migration. The paper's §4.4, §5.2 and H3 all still name GPT-4o-mini, and
    running both closed-source generators is strictly better than swapping
    one for the other: H3 asks whether the faithfulness RANKING of embedders
    survives a change of generator, and two closed-source arms plus Llama-3
    tests that far more convincingly than one.

    Prompt parity is the whole point — build_prompt() is shared verbatim with
    Claude and Llama-3, a single user turn with no system message. Do not
    "improve" this call with a system prompt, few-shot examples or a JSON
    schema: any of those would confound the cross-generator comparison.

    `max_tokens` (not `max_completion_tokens`) is used deliberately —
    gpt-4o-mini accepts it across every openai-python version likely to be
    installed here, whereas the newer name is not accepted by older clients.
    """

    def __init__(self, model: str = None, max_retries: int = 5):
        from openai import OpenAI
        # Reads OPENAI_API_KEY from the environment. Never hardcoded.
        self.client = OpenAI(max_retries=max_retries)
        self.model = model or config.GPT_MODEL

    def generate(self, question: str, chunks: List[str]) -> str:
        import openai
        prompt = build_prompt(question, chunks)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                temperature=0,
                messages=[{'role': 'user', 'content': prompt}],
            )
        except openai.APIStatusError as e:
            cost_tracker.log_error()
            return f'[ERROR: {type(e).__name__} {e.status_code}: {e}]'
        except openai.APIConnectionError as e:
            cost_tracker.log_error()
            return f'[ERROR: APIConnectionError: {e}]'

        if resp.usage is not None:
            cost_tracker.log_openai_chat(resp.usage)

        choice = resp.choices[0]
        # A content filter or refusal leaves content None. Mark it as a
        # scored-out row instead of letting .strip() raise — correctness.py
        # treats '[ERROR' rows as UNGRADABLE, never as wrong answers.
        if choice.finish_reason == 'content_filter':
            cost_tracker.log_error()
            return '[ERROR: content_filter]'
        text = choice.message.content
        if text is None:
            cost_tracker.log_error()
            return f'[ERROR: empty content, finish_reason={choice.finish_reason}]'
        return text.strip()


def run_phase_c(datasets: Dict, model_names: List[str] = None,
                budget_limit: float = 60.0, project_to: int = None):
    """Claude generation for every model × dataset. Resumable."""
    return _run_api_generation(
        ClaudeGenerator(), 'claude', datasets, model_names,
        budget_limit=budget_limit, project_to=project_to,
        key_env='ANTHROPIC_API_KEY')


def run_phase_c_openai(datasets: Dict, model_names: List[str] = None,
                       budget_limit: float = 60.0, project_to: int = None):
    """GPT-4o-mini generation for every model × dataset. Resumable."""
    return _run_api_generation(
        OpenAIGenerator(), 'gpt4omini', datasets, model_names,
        budget_limit=budget_limit, project_to=project_to,
        key_env='OPENAI_API_KEY')


def _run_api_generation(gen, label: str, datasets: Dict,
                        model_names: List[str] = None,
                        budget_limit: float = 60.0, project_to: int = None,
                        key_env: str = 'ANTHROPIC_API_KEY'):
    """
    Shared driver for every API generator. Resumable.

    One implementation on purpose: the fail-fast guards, the budget cap, the
    mid-dataset resume and the refusal to save a poisoned checkpoint are the
    parts that actually protect a paid run, and a copy-pasted second copy is
    exactly the kind of thing that drifts and then only protects one vendor.

    label        — goes into the checkpoint key and the 'generator' field
                   (`generated_{label}_{model}_{dataset}`). Must match the
                   GENERATORS lists in faithfulness.py and results.py.
    budget_limit — hard stop. Raises rather than draining the account;
                   partial checkpoints survive, so raising it resumes.
    project_to   — if set, extrapolate measured cost to this many queries
                   (the smoke-test cost probe before committing to a full run).
    """
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
            ck = f'generated_{label}_{model}_{ds_name}'
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
                    'generator': label,
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
                            f'Check {key_env} (a 401 usually means the key is '
                            f'wrong, truncated or has stray characters; an '
                            f'Anthropic key is ~100-110 chars).')
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
            f'Phase C ({label}) made ZERO successful API requests. Nothing was '
            f'generated; the cost probe above is meaningless. Check {key_env}.')
    print(f'[phase C: {label}] complete')


# ── open-weight generator (local HF model) ────────────────────────
class LocalHFGenerator:
    """
    Any instruct-tuned causal LM from the Hub, run locally on the GPU.

    Was Llama3Generator with the id hardcoded. Berend asked for Qwen or Gemma
    instead, and pinning the id in the class meant swapping models required an
    edit rather than an argument. The prompt still comes from build_prompt()
    unchanged — prompt parity across generators is what makes the cross-
    generator comparison mean anything.
    """
    MODEL_ID = None      # subclasses may pin one; otherwise config.OPEN_MODEL_ID

    def __init__(self, model_id: str = None, prompt_style: str = 'standard'):
        import torch
        self.prompt_style = prompt_style
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.MODEL_ID = model_id or self.MODEL_ID or config.OPEN_MODEL_ID
        assert torch.cuda.is_available(), \
            f'{self.MODEL_ID} requires a GPU'

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        use_8bit = (config.LLAMA_FORCE_8BIT if config.LLAMA_FORCE_8BIT
                    is not None else vram_gb < 20)
        print(f'Loading {self.MODEL_ID} '
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
        prompt = build_prompt(question, chunks, style=self.prompt_style)
        # Instruct model -> chat template, not raw completion.
        #
        # `return_dict=True` is passed explicitly rather than relying on the
        # default, which flipped: transformers 4.x returned a bare tensor here
        # and 5.x returns a BatchEncoding, so the old code reached
        # model.generate() with a dict as its first positional argument and
        # died on `inputs_tensor.shape[0]`. Being explicit works on both, and
        # it also gets us the attention mask instead of letting generate()
        # infer one.
        enc = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            add_generation_prompt=True, return_tensors='pt',
            truncation=True, max_length=4096, return_dict=True,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with self.torch.no_grad():
            output = self.model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=False,  # C13: greedy — no temperature argument
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # C13: slice by token count, not character count
        new_tokens = output[0][enc['input_ids'].shape[1]:]
        return self.tokenizer.decode(new_tokens,
                                     skip_special_tokens=True).strip()


class Llama3Generator(LocalHFGenerator):
    """Kept so existing callers and checkpoints keep working."""
    MODEL_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'


def run_phase_llama(datasets: Dict, model_names: List[str] = None,
                    subset: int = config.LLAMA_SUBSET,
                    model_id: str = None, label: str = None,
                    prompt_style: str = 'standard'):
    """
    The open-weight generator arm (H3: generator independence).

    `model_names=None` means EVERY configured model, as it does in every other
    phase. It used to mean a hardcoded three, which silently dropped
    text-embedding-3-small: run_pipeline passes None when --models is omitted,
    so a run asking for the full grid quietly produced three quarters of it.
    That is not a smaller run, it is an incomparable one — the embedder spread
    this arm contributes to Table VII would have been taken over 3 systems
    while the other generators' was taken over 4.

    Models without a retrieval checkpoint are reported and skipped, so naming
    all seven configured models costs nothing when only four have been run.

    `label` goes into the checkpoint key, so a Qwen run and a Llama run never
    collide. Passing a different `model_id` under the SAME label would mix two
    models' answers into one checkpoint, which resume would then treat as done.
    """
    model_id = model_id or config.OPEN_MODEL_ID
    label = label or config.OPEN_MODEL_LABEL
    val_models = model_names or [c['name'] for c in config.EMBEDDING_MODELS]
    llama = LocalHFGenerator(model_id, prompt_style=prompt_style)

    for model in val_models:
        for ds_name in datasets:
            ck = f'generated_{label}_{model}_{ds_name}'
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
            print(f'  {label} [{model}] [{ds_name}] '
                  f'({done}/{len(todo)} done)...')
            for i, r in enumerate(todo[done:], start=done):
                answer = llama.generate(
                    r['question'], r['retrieved_texts'][:config.TOP_K])
                generations.append({
                    **r,
                    'generated_answer': answer,
                    'generator': label,
                })
                if (i + 1) % 50 == 0:
                    save_checkpoint(ck + '_partial', generations)
                    print(f'    {i + 1}/{len(todo)}')
            save_checkpoint(ck, generations)
            print(f'  {model}/{ds_name}: {len(generations)} answers')
    print(f'[phase open-weight: {label}] complete')
