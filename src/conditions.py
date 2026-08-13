"""
conditions.py — controlled conditions C1 and C2 (Berend 2026-08-11).

WHY THIS EXISTS
    The 7-embedder spread has no anchors. Without knowing what the generator
    scores with NO retrieval and with PERFECT retrieval, a gap between two
    embedders cannot be attributed to retrieval quality at all.

    C1  no-RAG floor      question only, no context.
                          Measures parametric knowledge. Questions answered
                          correctly here never needed retrieval -> evidence that
                          good retrieval is not NECESSARY (Berend branch B).
                          Also yields the sampling filter (--emit-filter).

    C2  oracle ceiling    gold evidence only, no embedder.
                          The glass ceiling for any retriever. Makes the
                          embedder spread interpretable as "% of oracle
                          recovered" rather than a bare score.

    Watch for retrieval BEATING the oracle. Not an anomaly to discard: it would
    show gold-relevant snippets are not optimal CONTEXT (ordering, redundancy,
    distractor-induced hedging), which is a result in itself.

COST (measured basis: smoke test, 150 requests, $0.2782, 1465 in-tok/query)
    Both conditions are EMBEDDER-INDEPENDENT: computed once, not once per
    embedding model. At N=1000 x 3 datasets that is 3,000 requests each.
        C1  ~$1.42    C2  ~$5.56    together ~$7 on top of the $38.95 grid.
    Nothing is spent without --yes.

USAGE
    python src/conditions.py --condition c1 --dry-run
    python src/conditions.py --condition c1 --yes
    python src/conditions.py --condition c2 --yes
    python src/conditions.py --emit-filter          # after c1 + correctness.py

PROMPT PARITY — read before reporting C1
    CLAUDE.md fixes a byte-identical prompt across generators so H3 is not
    confounded. C2 reuses generate.build_prompt() unchanged, so it is exactly
    the main-run prompt with gold chunks substituted for retrieved ones.

    C1 CANNOT reuse it. RAG_PROMPT_TEMPLATE says "Answer using ONLY the
    provided context" and "if the context does not contain enough information,
    say 'I cannot answer based on the provided context.'" Feeding it an empty
    context would measure the model's willingness to REFUSE, not its parametric
    knowledge — C1 would report a floor near zero for entirely the wrong
    reason. C1 therefore uses CLOSED_BOOK_PROMPT below, and this difference is
    inherent to the condition rather than a design choice. State it in the paper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import config
from utils import checkpoint_exists, cost_tracker, load_checkpoint, save_checkpoint

# Measured on the 2026-07-26 smoke test.
COST_PER_1K_INPUT = 1.0 / 1000.0
COST_PER_1K_OUTPUT = 5.0 / 1000.0
EST_OUTPUT_TOKENS = 78
EST_INPUT_TOKENS_CTX = 1465
EST_INPUT_TOKENS_NOCTX = 85

MAX_CONSECUTIVE_ERRORS = 10

# Closed-book prompt for C1. Deliberately NOT RAG_PROMPT_TEMPLATE — see the
# module docstring. Kept minimal and answer-style-matched to the RAG prompt so
# the comparison is about knowledge, not verbosity.
CLOSED_BOOK_PROMPT = """You are a helpful assistant. Answer the question as concisely as possible.
If you do not know the answer, say "I do not know."

Question: {question}

Answer:"""


class ClosedBookGenerator:
    """
    C1 generator: question only, no context.

    Wraps generate.ClaudeGenerator to inherit its client configuration, retry
    policy, refusal handling and cost tracking, but substitutes the closed-book
    prompt. Mirrors ClaudeGenerator.generate()'s contract exactly: returns
    '[ERROR: ...]' strings rather than raising, so callers stay uniform.
    """

    def __init__(self, model: str = None, max_retries: int = 5):
        from generate import ClaudeGenerator
        self._inner = ClaudeGenerator(model=model, max_retries=max_retries)
        self.model = self._inner.model
        self.client = self._inner.client

    def generate(self, question: str, chunks: List[str] = None) -> str:
        import anthropic
        prompt = CLOSED_BOOK_PROMPT.format(question=question)
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
        if resp.stop_reason == 'refusal':
            cost_tracker.log_error()
            return '[ERROR: refusal]'
        return ''.join(b.text for b in resp.content if b.type == 'text').strip()


# ── oracle context (C2) ──────────────────────────────────────────────────────
#
# THE ORACLE MUST BE DEFINED THE SAME WAY THE QRELS ARE.
#
# There are two defensible notions of "gold evidence" in this codebase and they
# are NOT the same object:
#
#   qrels     retrieval_eval.build_qrels() marks a CHUNK relevant iff its source
#             document is gold for the query (on HotpotQA, additionally, iff the
#             chunk contains a supporting sentence). This is what NDCG@5 and
#             Recall@5 are computed against.
#
#   gold_context  QASample.gold_context — for HotpotQA just the supporting
#             sentences, stripped of their surrounding paragraph.
#
# An oracle built from gold_context is a STRICTLY EASIER condition than perfect
# retrieval: on HotpotQA it hands the generator the two supporting sentences
# with none of the paragraph around them, which no retriever could ever return
# under the qrels. It would be a ceiling on something the paper never measures.
#
# Default is therefore `qrels`: the top-k chunks a perfect retriever would rank
# first, taken from the same chunk pool Phase A searches. `gold_context` stays
# available as a tighter evidence-only variant, and running both is informative
# — the gap between them is the cost of retrieving paragraphs instead of
# sentences.
ORACLE_SOURCES = ('qrels', 'gold_context')


class OracleBuilder:
    """
    Builds oracle contexts for one dataset. Loads qrels and the chunk pool once
    (both already cached by Phase A/B) rather than per query.
    """

    def __init__(self, loaded, source: str = 'qrels', top_k: int = None):
        if source not in ORACLE_SOURCES:
            raise ValueError(f'oracle source must be one of {ORACLE_SOURCES}')
        self.source = source
        self.top_k = top_k or config.TOP_K
        self.loaded = loaded
        self._qrels = None
        self._chunk_text = None
        if source == 'qrels':
            from embed_index import build_chunks
            from retrieval_eval import build_qrels
            self._qrels = build_qrels(loaded)
            self._chunk_text = {c.chunk_id: c.text for c in build_chunks(loaded)}

    def contexts(self, sample) -> List[str]:
        """
        Oracle chunks for one query, no embedder involved. Returns [] when the
        query has no usable gold evidence; the caller records that rather than
        generating anyway.
        """
        if self.source == 'qrels':
            ids = list(self._qrels.get(sample.query_id, {}))
            return [self._chunk_text[cid] for cid in ids[:self.top_k]
                    if cid in self._chunk_text]
        gold = (getattr(sample, 'gold_context', '') or '').strip()
        if not gold:
            return []
        from embed_index import chunk_document
        chunks = chunk_document(f'{sample.query_id}_gold', gold)
        return [c.text for c in chunks][:self.top_k]

    def is_trustworthy(self, sample) -> bool:
        """
        False when this query's gold evidence is a loader stand-in rather than
        annotation. QASPER free-form answers sometimes cite a figure, and the
        loader substitutes the paper's first three paragraphs so the query keeps
        a retrieval target — fine for qrels, fatal for a ceiling claim.
        """
        return not getattr(sample, 'gold_is_fallback', False)


def build_oracle_contexts(sample, top_k: int = None,
                          loaded=None, source: str = 'gold_context') -> List[str]:
    """
    Single-query convenience wrapper, kept for preflight and tests.
    Batch callers should use OracleBuilder so qrels load once per dataset.
    """
    return OracleBuilder(loaded, source=source, top_k=top_k).contexts(sample)


# ── cost gate ────────────────────────────────────────────────────────────────
def _project_cost(n_requests: int, input_tokens: int) -> float:
    return n_requests * (input_tokens / 1000.0 * COST_PER_1K_INPUT
                         + EST_OUTPUT_TOKENS / 1000.0 * COST_PER_1K_OUTPUT)


def _confirm(n_requests: int, input_tokens: int, yes: bool) -> bool:
    est = _project_cost(n_requests, input_tokens)
    print(f'  projected: {n_requests:,} requests x ~{input_tokens} in-tok -> ~${est:.2f}')
    if yes:
        return True
    print('  refusing to spend without --yes (--dry-run to inspect)')
    return False


def _limited(samples, limit: Optional[int]):
    return samples[:limit] if limit else samples


def _resume(ck: str, n_samples: int):
    """
    Load a partial checkpoint and return (records, done).

    Guards a resume hazard: if the previous run used a larger --limit, the
    partial file holds more records than the current sample list, and
    `samples[done:]` would be empty — the run would "succeed" instantly and
    save a checkpoint describing a different query set than it claims.
    """
    records = load_checkpoint(ck + '_partial') or []
    if len(records) > n_samples:
        raise RuntimeError(
            f'{ck}_partial holds {len(records)} records but only {n_samples} '
            f'samples are in scope — this partial came from a run with a '
            f'different --limit/--n-queries. Delete it or match the earlier '
            f'setting; resuming would mislabel the query set.')
    return records, len(records)


def _abort_on_repeated_errors(answer: str, consecutive: int, first: Optional[str]):
    """Same guard as run_phase_c: stop early rather than burn the budget."""
    if answer.startswith('[ERROR'):
        consecutive += 1
        if first is None:
            first = answer
            print(f'    !! first API error: {answer[:300]}')
        if consecutive >= MAX_CONSECUTIVE_ERRORS:
            raise RuntimeError(
                f'{consecutive} consecutive API failures — aborting before '
                f'burning the run.\nFirst error: {first}\n'
                f'Check ANTHROPIC_API_KEY (a valid key is ~100-110 chars).')
    else:
        consecutive = 0
    return consecutive, first


# ── conditions ───────────────────────────────────────────────────────────────
def run_condition_c1(datasets: Dict, dry_run: bool = False,
                     yes: bool = False, limit: Optional[int] = None) -> None:
    """No-RAG parametric floor. Embedder-independent: one pass per dataset."""
    total = sum(min(len(ds.samples), limit or len(ds.samples))
                for ds in datasets.values())
    print(f'=== C1 no-RAG floor ({len(datasets)} datasets, {total:,} queries) ===')
    if not _confirm(total, EST_INPUT_TOKENS_NOCTX, yes) or dry_run:
        return

    gen = ClosedBookGenerator()
    for ds_name, ds in datasets.items():
        ck = f'norag_{ds_name}'
        if checkpoint_exists(ck):
            print(f'  [skip] {ck}')
            continue
        samples = _limited(ds.samples, limit)
        records, done = _resume(ck, len(samples))
        consecutive, first_error = 0, None
        print(f'  C1 [{ds_name}] ({done}/{len(samples)} done)...')
        for i, s in enumerate(samples[done:], start=done):
            answer = gen.generate(s.question)
            consecutive, first_error = _abort_on_repeated_errors(
                answer, consecutive, first_error)
            records.append({
                'query_id': s.query_id,
                'question': s.question,
                'answer': s.answer,
                'generated_answer': answer,
                'generator': 'claude',
                'generator_model': gen.model,
                'condition': 'c1_norag',
                'prompt_template': 'closed_book',
                'dataset': ds_name,
                'retrieved_texts': [],
                'n_context_chunks': 0,
            })
            if (i + 1) % 100 == 0:
                save_checkpoint(ck + '_partial', records)
                print(f'    {i + 1}/{len(samples)}  '
                      f'running cost ${cost_tracker.cost:.4f}')
        save_checkpoint(ck, records)
        n_err = sum(1 for r in records if r['generated_answer'].startswith('[ERROR'))
        print(f'  {ds_name}: {len(records)} answers'
              + (f'  [!] {n_err} errors' if n_err else ''))
        print(f'    -> score with `python src/correctness.py`; read '
              f'correct_rate as the parametric floor and abstention_rate '
              f'alongside it (a floor made of "I do not know" means something '
              f'different from a floor made of confident wrong answers).')


def run_condition_c2(datasets: Dict, dry_run: bool = False,
                     yes: bool = False, limit: Optional[int] = None,
                     source: str = 'qrels') -> None:
    """
    Oracle ceiling: relevant chunks only, no embedder. Reuses build_prompt(),
    so the prompt is byte-identical to the main run with gold chunks
    substituted for retrieved ones.

    `source` selects what counts as relevant — see the ORACLE_SOURCES comment.
    Default 'qrels' makes C2 the ceiling of the retrieval task the paper
    actually scores.
    """
    from generate import ClaudeGenerator
    total = sum(min(len(ds.samples), limit or len(ds.samples))
                for ds in datasets.values())
    print(f'=== C2 oracle ceiling [{source}] '
          f'({len(datasets)} datasets, {total:,} queries) ===')
    if not _confirm(total, EST_INPUT_TOKENS_CTX, yes) or dry_run:
        return

    gen = ClaudeGenerator()
    for ds_name, ds in datasets.items():
        ck = f'oracle_{ds_name}' if source == 'qrels' else f'oracle_{source}_{ds_name}'
        if checkpoint_exists(ck):
            print(f'  [skip] {ck}')
            continue
        samples = _limited(ds.samples, limit)
        oracle = OracleBuilder(ds, source=source)
        records, done = _resume(ck, len(samples))
        consecutive, first_error, n_empty, n_fallback = 0, None, 0, 0
        ctx_sizes = []
        print(f'  C2 [{ds_name}] ({done}/{len(samples)} done)...')
        for i, s in enumerate(samples[done:], start=done):
            contexts = oracle.contexts(s)
            trustworthy = oracle.is_trustworthy(s)
            if not trustworthy:
                n_fallback += 1
            if not contexts or not trustworthy:
                # No gold evidence, or only a loader stand-in. Record it; do
                # NOT silently generate under an 'oracle' label. correctness.py
                # treats generated_answer=None as ungradable, so these rows
                # cannot deflate the ceiling.
                n_empty += 1
                records.append({
                    'query_id': s.query_id, 'question': s.question,
                    'answer': s.answer, 'generated_answer': None,
                    'generator': 'claude', 'generator_model': gen.model,
                    'condition': 'c2_oracle', 'oracle_source': source,
                    'dataset': ds_name, 'retrieved_texts': [],
                    'oracle_missing': True,
                    'oracle_gold_is_fallback': not trustworthy,
                    'n_context_chunks': 0,
                })
                continue
            answer = gen.generate(s.question, contexts)   # uses build_prompt
            consecutive, first_error = _abort_on_repeated_errors(
                answer, consecutive, first_error)
            ctx_sizes.append(len(contexts))
            records.append({
                'query_id': s.query_id, 'question': s.question,
                'answer': s.answer, 'generated_answer': answer,
                'generator': 'claude', 'generator_model': gen.model,
                'condition': 'c2_oracle', 'oracle_source': source,
                'dataset': ds_name, 'retrieved_texts': contexts,
                'oracle_missing': False, 'oracle_gold_is_fallback': False,
                'n_context_chunks': len(contexts),
            })
            if (i + 1) % 100 == 0:
                save_checkpoint(ck + '_partial', records)
                print(f'    {i + 1}/{len(samples)}  '
                      f'running cost ${cost_tracker.cost:.4f}')
        save_checkpoint(ck, records)

        n = max(1, len(records))
        mean_ctx = (sum(ctx_sizes) / len(ctx_sizes)) if ctx_sizes else 0.0
        print(f'  {ds_name}: {len(records)} records, '
              f'{len(ctx_sizes)} generated, mean {mean_ctx:.1f} chunks'
              + (f'  [!] {n_empty} skipped' if n_empty else '')
              + (f' ({n_fallback} stand-in gold)' if n_fallback else ''))
        if n_empty > 0.2 * n:
            print(f'  [WARN] {n_empty / n:.0%} of {ds_name} queries have no '
                  f'trustworthy gold evidence — C2 is not a clean ceiling here, '
                  f'and the ceiling must be reported on the retained subset only.')
        # Context-length parity matters: the RAG condition always fills TOP_K
        # slots, so an oracle averaging far fewer chunks differs from it in
        # context LENGTH as well as relevance. NLI entailment is sensitive to
        # premise length, so this must be reported, not assumed away.
        if ctx_sizes and mean_ctx < 0.6 * config.TOP_K:
            print(f'  [WARN] oracle averages {mean_ctx:.1f} chunks vs '
                  f'{config.TOP_K} retrieved — report the comparison as '
                  f'relevance-AND-length, or use context_ablation.py to hold '
                  f'slot count fixed.')


def run_prompt_probe(datasets: Dict, n: int = 100, dry_run: bool = False,
                     yes: bool = False) -> None:
    """
    Quantify the C1 prompt confound instead of only documenting it.

    C1 must use a closed-book prompt (see the module docstring), so C1 and the
    RAG condition differ in TWO ways at once: presence of context, and prompt
    wording. A reviewer can reasonably ask how much of the floor-to-RAG gap is
    the wording. This runs the SAME queries through RAG_PROMPT_TEMPLATE with an
    empty context block; the difference against C1 on the same queries is the
    prompt effect, isolated.

    Small on purpose: n=100 per dataset is enough to bound the effect, and the
    whole probe costs about $0.03.
    """
    from generate import ClaudeGenerator, build_prompt
    total = n * len(datasets)
    print(f'=== C1b prompt-confound probe ({total:,} queries) ===')
    if not _confirm(total, EST_INPUT_TOKENS_NOCTX, yes) or dry_run:
        return

    gen = ClaudeGenerator()
    for ds_name, ds in datasets.items():
        ck = f'norag_promptprobe_{ds_name}'
        if checkpoint_exists(ck):
            print(f'  [skip] {ck}')
            continue
        samples = ds.samples[:n]
        records, done = _resume(ck, len(samples))
        consecutive, first_error = 0, None
        for i, s in enumerate(samples[done:], start=done):
            # RAG template, empty context — deliberately the thing C1 avoids.
            answer = gen.generate(s.question, [])
            consecutive, first_error = _abort_on_repeated_errors(
                answer, consecutive, first_error)
            records.append({
                'query_id': s.query_id, 'question': s.question,
                'answer': s.answer, 'generated_answer': answer,
                'generator': 'claude', 'generator_model': gen.model,
                'condition': 'c1b_norag_ragprompt',
                'prompt_template': 'rag_empty_context',
                'dataset': ds_name, 'retrieved_texts': [],
                'n_context_chunks': 0,
            })
        save_checkpoint(ck, records)
        print(f'  {ds_name}: {len(records)} answers — compare correct_rate and '
              f'abstention_rate against norag_{ds_name} on the same query ids')


def emit_eval_filter(datasets: Dict) -> Dict[str, List[str]]:
    """
    The principled sampling justification (Berend's last paragraph).

    Retain only queries the model answers INCORRECTLY without retrieval:
    instances already answered from parametric knowledge cannot inform a study
    of retrieval quality. Needs C1, scored by correctness.py.

    Writes checkpoint `eval_filter` -> {dataset: [query_id, ...]} and prints
    per-dataset yield, itself a finding about benchmark contamination.
    """
    from correctness import score_records
    keep: Dict[str, List[str]] = {}
    print('=== eval filter: queries where retrieval is NECESSARY ===')
    for ds_name in datasets:
        recs = (load_checkpoint(f'norag_{ds_name}_scored')
                or load_checkpoint(f'norag_{ds_name}'))
        if not recs:
            print(f'  [{ds_name}] no C1 checkpoint — run --condition c1 first')
            continue
        if 'correct' not in recs[0]:
            recs = score_records(recs)
            save_checkpoint(f'norag_{ds_name}_scored', recs)
        graded = [r for r in recs if r.get('correct') is not None]
        ids = [r['query_id'] for r in graded if r['correct'] is False]
        keep[ds_name] = ids
        n = len(graded)
        skipped = len(recs) - n
        n_abstain = sum(1 for r in graded
                        if r['correct'] is False and r.get('abstained'))
        print(f'  [{ds_name}] {len(ids)}/{n} retained ({len(ids)/max(1,n):.1%}) — '
              f'{n - len(ids)} answerable without retrieval'
              + (f', {skipped} ungradable' if skipped else ''))
        # A retained query where the model ABSTAINED is a clean "retrieval is
        # needed" case. A retained query where it answered confidently and
        # wrongly is a different animal: RAG there has to overturn a belief,
        # not fill a gap. Worth separating in the paper.
        if ids:
            print(f'              of the retained: {n_abstain} abstained, '
                  f'{len(ids) - n_abstain} answered confidently but wrong')
    if keep:
        save_checkpoint('eval_filter', keep)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--condition', choices=['c1', 'c2', 'both'])
    ap.add_argument('--oracle-source', choices=list(ORACLE_SOURCES),
                    default='qrels',
                    help="what C2 treats as relevant: 'qrels' (default — the "
                         "ceiling of the retrieval task actually scored) or "
                         "'gold_context' (tighter evidence-only variant)")
    ap.add_argument('--prompt-probe', type=int, nargs='?', const=100,
                    default=None, metavar='N',
                    help='isolate the C1 prompt confound on N queries/dataset')
    ap.add_argument('--emit-filter', action='store_true',
                    help='write the parametric-knowledge eval filter (needs C1)')
    ap.add_argument('--datasets', default=None, help='comma-separated subset')
    ap.add_argument('--n-queries', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None, help='cap per dataset (testing)')
    ap.add_argument('--dry-run', action='store_true', help='project cost only')
    ap.add_argument('--yes', action='store_true', help='authorize spending')
    args = ap.parse_args()

    if not any((args.condition, args.emit_filter, args.prompt_probe)):
        ap.error('pass --condition {c1,c2,both}, --prompt-probe and/or --emit-filter')

    from datasets_loader import load_all
    from utils import set_seed
    set_seed()
    ds_names = args.datasets.split(',') if args.datasets else config.DATASETS
    datasets = load_all(args.n_queries or config.N_QUERIES, ds_names)

    if args.condition in ('c1', 'both'):
        run_condition_c1(datasets, args.dry_run, args.yes, args.limit)
    if args.condition in ('c2', 'both'):
        run_condition_c2(datasets, args.dry_run, args.yes, args.limit,
                         source=args.oracle_source)
    if args.prompt_probe:
        run_prompt_probe(datasets, args.prompt_probe, args.dry_run, args.yes)
    if args.emit_filter:
        emit_eval_filter(datasets)
    if (args.condition or args.prompt_probe) and not args.dry_run:
        print(f'\n=== cost: ${cost_tracker.cost:.4f} over '
              f'{cost_tracker.requests:,} requests '
              f'({cost_tracker.errors} errors) ===')


if __name__ == '__main__':
    main()
