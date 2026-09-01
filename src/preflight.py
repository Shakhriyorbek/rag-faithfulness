"""
preflight.py — verify every assumption BEFORE spending money or GPU hours.

Run this first on gpu1. It is read-only and free: no API calls, no generation.
It fails loudly on the things that have actually bitten this project before
(V100 sm_70 kernels, NLI label order, qrels/ID mismatch, missing keys), plus
the integration points the new modules depend on.

    python src/preflight.py            # full check
    python src/preflight.py --quick    # skip model downloads

Exit code 0 = safe to launch. Non-zero = do not start the run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS, FAIL, WARN = '  [PASS]', '  [FAIL]', '  [WARN]'
_results = []


def check(name, fn, fatal=True, skip=False):
    if skip:
        print(f'  [SKIP] {name}')
        return None
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f'{type(e).__name__}: {e}'
    tag = PASS if ok else (FAIL if fatal else WARN)
    print(f'{tag} {name}' + (f' — {detail}' if detail else ''))
    _results.append((name, ok, fatal))
    return ok


# ── environment ──────────────────────────────────────────────────────────────
def c_python():
    v = sys.version_info
    return v >= (3, 8), f'{v.major}.{v.minor}.{v.micro}'


def c_anthropic_key():
    """Fatal: every generating phase needs it."""
    import config
    key = getattr(config, 'ANTHROPIC_API_KEY', '')
    if not key:
        return False, 'ANTHROPIC_API_KEY not set — no phase C, C1, C2 or ablation'
    # ~100-110 chars is normal. A wildly long value means the key was pasted
    # more than once into a silent `read -rs` prompt, which has happened here.
    return 90 <= len(key) <= 130, f'present, length {len(key)} (expect ~108)'


def c_optional_keys():
    """
    Non-fatal: which phases these gate depends on what is being run.

    OPENAI_API_KEY is needed for the text-embedding-3-small embedder AND,
    since 2026-08-14, for the GPT-4o-mini generator (phase `cgpt`); HF_TOKEN
    only for gated Llama-3. A three-model Claude-only NQ run needs neither, so
    blocking on them would stop a legitimate launch.
    """
    import config
    missing = [k for k in ('OPENAI_API_KEY', 'HF_TOKEN')
               if not getattr(config, k, '')]
    if not missing:
        return True, 'OPENAI_API_KEY + HF_TOKEN present'
    gated = {'OPENAI_API_KEY': 'text-embedding-3-small embedder + phase cgpt '
                               '(GPT-4o-mini generator)',
             'HF_TOKEN': 'Llama-3-8B (gated repo)'}
    return False, 'missing ' + '; '.join(f'{k} -> blocks {gated[k]}'
                                         for k in missing)


def c_disk():
    import shutil
    import config
    free_gb = shutil.disk_usage(config.BASE_DIR).free / 1e9
    return free_gb > 50, f'{free_gb:.0f} GB free at {config.BASE_DIR}'


# ── torch / GPU — the sm_70 trap ─────────────────────────────────────────────
def c_torch_cuda():
    """
    torch.cuda.is_available() is NOT sufficient on this V100 (Volta, sm_70):
    default PyPI wheels ship sm_75+ kernels, import fine, report cuda True,
    then die on the first real kernel launch. Verify with an actual matmul.
    """
    import torch
    if not torch.cuda.is_available():
        return False, 'CUDA not available (CPU-only is fine for phases a,b,c,d)'
    arch = torch.cuda.get_arch_list()
    dev = torch.cuda.get_device_name(0)
    x = torch.randn(64, 64, device='cuda')
    (x @ x).sum().item()          # the check that actually matters
    torch.cuda.synchronize()
    has70 = any('sm_70' in a for a in arch)
    return has70, (f'{dev} | matmul OK | arch={",".join(arch)}'
                   + ('' if has70 else ' | sm_70 MISSING — reinstall with --force-reinstall'))


def c_vram():
    import torch
    if not torch.cuda.is_available():
        return True, 'no GPU — skipping (phases e/llama will need one)'
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return gb > 20, f'{gb:.1f} GB — fp16 Llama-3-8B {"fits" if gb > 20 else "does NOT fit, needs 8-bit"}'


# ── NLI label order — audit item B1 ──────────────────────────────────────────
def c_nli_labels(quick=False):
    if quick:
        return True, 'skipped (--quick)'
    import nli
    idx = None
    for attr in ('ENTAILMENT_IDX', 'entailment_index', 'get_entailment_index'):
        v = getattr(nli, attr, None)
        if callable(v):
            idx = v()
            break
        if isinstance(v, int):
            idx = v
            break
    if idx is None:
        from transformers import AutoConfig
        import config as cfg
        c = AutoConfig.from_pretrained(cfg.NLI_MODEL)
        idx = c.label2id.get('entailment', c.label2id.get('ENTAILMENT'))
    return idx == 1, (f'entailment index = {idx} '
                      f'(must be 1 for nli-deberta-v3-large; 2 is NEUTRAL)')


# ── integration points the new modules rely on ───────────────────────────────
def c_generate_api():
    """
    conditions.py binds to generate.ClaudeGenerator and generate.build_prompt.
    Verify both exist and that build_prompt still takes (question, chunks).
    """
    import inspect

    import generate
    if not hasattr(generate, 'ClaudeGenerator'):
        return False, 'generate.ClaudeGenerator missing'
    if not hasattr(generate, 'build_prompt'):
        return False, 'generate.build_prompt missing — C2 loses prompt parity'
    params = list(inspect.signature(generate.build_prompt).parameters)
    if params[:2] != ['question', 'chunks']:
        return False, f'build_prompt signature changed: {params}'
    return True, 'ClaudeGenerator + build_prompt(question, chunks)'


def c_c1_prompt_is_closed_book():
    """
    C1 must NOT reuse RAG_PROMPT_TEMPLATE. That template says "answer using
    ONLY the provided context" and offers an explicit refusal string; with an
    empty context it would measure willingness to refuse, not parametric
    knowledge, and C1 would report a floor near zero for the wrong reason.
    """
    import conditions
    import generate
    t = conditions.CLOSED_BOOK_PROMPT
    if 'ONLY the provided context' in t or 'Context:' in t:
        return False, 'CLOSED_BOOK_PROMPT still carries RAG context language'
    if t == getattr(generate, 'RAG_PROMPT_TEMPLATE', None):
        return False, 'C1 is reusing the RAG template'
    return True, 'closed-book prompt is distinct from RAG template'


def c_checkpoint_api():
    from utils import checkpoint_path, load_checkpoint, save_checkpoint
    save_checkpoint('preflight_probe', {'ok': True})
    got = load_checkpoint('preflight_probe')
    checkpoint_path('preflight_probe').unlink(missing_ok=True)  # leave no litter
    return isinstance(got, dict) and got.get('ok') is True, 'save/load round-trip'


# The two checks below hit the network to fetch a handful of dataset rows.
# They are skipped under --quick along with the model downloads.
def c_sample_fields():
    """C1/C2 need question, gold answer, query_id and gold_context per sample."""
    import config
    from datasets_loader import load_all
    ds = load_all(5, [config.DATASETS[0]])
    name = list(ds)[0]
    s = ds[name].samples[0]
    missing = [f for f in ('query_id', 'question', 'answer', 'gold_context')
               if not getattr(s, f, None)]
    return not missing, (f'{name}: ' + ('all fields present' if not missing
                                        else f'MISSING {missing}'))


def c_oracle_contexts():
    """
    C2 is meaningless if gold evidence cannot be resolved per query.

    Checks the gold_context path only — the default 'qrels' path additionally
    needs Phase A chunks, which do not exist before the first run.
    """
    import config
    from conditions import build_oracle_contexts
    from datasets_loader import load_all
    ds = load_all(20, [config.DATASETS[0]])
    name = list(ds)[0]
    samples = ds[name].samples[:20]
    n_ok = sum(1 for s in samples if build_oracle_contexts(s))
    n_fallback = sum(1 for s in samples
                     if getattr(s, 'gold_is_fallback', False))
    detail = (f'{n_ok}/{len(samples)} queries yield gold chunks '
              f'(low => check QASample.gold_context in the loader)')
    if n_fallback:
        detail += (f'; {n_fallback} carry a STAND-IN gold context and are '
                   f'excluded from the C2 ceiling')
    return n_ok >= 15, detail


def c_correctness_module():
    from correctness import score_records
    r = score_records([{'generated_answer': 'Paris', 'answer': 'Paris'},
                       {'generated_answer': 'Berlin', 'answer': 'Paris'},
                       {'generated_answer': 'x', 'answer': None}])
    ok = r[0]['correct'] is True and r[1]['correct'] is False and r[2]['correct'] is None
    return ok, 'EM/F1/None-gold semantics correct'


def c_correctness_ungradable():
    """
    An API failure and a missing oracle context must be UNGRADABLE, not wrong.

    Grading '[ERROR: ...]' as an incorrect answer would let a rate-limit
    episode masquerade as the generator being less accurate, and grading a
    None oracle answer as incorrect would understate the C2 ceiling — the one
    number that condition exists to produce.
    """
    from correctness import score_records
    r = score_records([
        {'generated_answer': '[ERROR: APIStatusError 429: rate]', 'answer': 'Paris'},
        {'generated_answer': None, 'answer': 'Paris', 'oracle_missing': True},
        {'generated_answer': 'I do not know.', 'answer': 'Paris'},
    ])
    ok = (r[0]['correct'] is None and r[1]['correct'] is None
          and r[2]['correct'] is False and r[2]['abstained'] is True)
    return ok, ('errors/missing are ungradable, abstention flagged' if ok else
                f'got {[x["correct"] for x in r]} — ungradable rows are '
                f'leaking into the incorrect bucket')


def c_rfg_nan_guard():
    """
    Audit S1-1: a nan faithfulness must not be silently clamped to 0.0 by
    max(0.0, f) — that turns a missing measurement into a real one.
    """
    from legacy import rfg as metrics
    nan = float('nan')
    out = metrics.rfg(0.9, nan)
    import math
    ok = (out is None) or (isinstance(out, float) and math.isnan(out))
    return ok, ('nan propagates correctly' if ok else
                f'nan faithfulness produced {out!r} — S1-1 NOT fixed, '
                'results will be silently wrong')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='skip checks that hit the network (models + datasets)')
    args = ap.parse_args()

    print('=== preflight: environment ===')
    check('python >= 3.8', c_python)
    check('ANTHROPIC_API_KEY usable', c_anthropic_key)
    check('optional keys (OpenAI / HF)', c_optional_keys, fatal=False)
    check('disk space', c_disk, fatal=False)

    print('=== preflight: GPU (needed for phases e, llama) ===')
    check('torch CUDA + real matmul (sm_70)', c_torch_cuda, fatal=False)
    check('VRAM for Llama-3-8B fp16', c_vram, fatal=False)

    print('=== preflight: known-bug regression guards ===')
    check('NLI entailment index == 1 (B1)', lambda: c_nli_labels(args.quick),
          skip=args.quick)
    check('rfg() propagates nan (S1-1)', c_rfg_nan_guard, fatal=False)

    print('=== preflight: integration points for new modules ===')
    check('utils checkpoint round-trip', c_checkpoint_api)
    check('generate.py Claude API shape', c_generate_api)
    check('C1 prompt is closed-book, not RAG', c_c1_prompt_is_closed_book)
    check('correctness.py semantics', c_correctness_module)
    check('ungradable rows are None, not False', c_correctness_ungradable)
    check('sample carries question/answer/query_id', c_sample_fields,
          skip=args.quick)
    check('oracle gold contexts resolvable (C2)', c_oracle_contexts,
          fatal=False, skip=args.quick)

    fatal_fails = [n for n, ok, f in _results if not ok and f]
    warns = [n for n, ok, f in _results if not ok and not f]
    print('\n=== summary ===')
    print(f'  {sum(1 for _, ok, _ in _results if ok)}/{len(_results)} passed')
    if warns:
        print(f'  warnings (non-blocking): {", ".join(warns)}')
    if fatal_fails:
        print(f'  BLOCKING: {", ".join(fatal_fails)}')
        print('  Do NOT start the full run until these are fixed.')
        sys.exit(1)
    print('  preflight OK — safe to launch.')


if __name__ == '__main__':
    main()
