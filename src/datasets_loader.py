"""
datasets_loader.py — NQ, HotpotQA, QASPER loaders.

Ports the notebook loaders, keeping the HuggingFace API fixes documented in
CLAUDE.md §7 exactly:
  - NQ: short_answers fields are LISTS -> use sa['text'][0]; filter HTML
    tokens via tokens['is_html'].
  - HotpotQA: supporting_facts is parallel lists (title, sent_id); sent_id
    indexes context.sentences[title_idx].
  - QASPER: paper['qas'] is columnar (dict of lists); handle both
    free_form_answer and extractive_spans; trust_remote_code=True.

Structural change vs the notebook (audit B3/B4): the corpus is a list of
Document records carrying provenance — which query each document is gold
evidence for — so retrieval relevance is decided by ID, never by lossy
chunk-string equality. NQ now pools contexts across all sampled queries so
its index contains distractors (the notebook indexed only each query's own
gold context).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set

import config
import textnorm
from utils import load_checkpoint, save_checkpoint


# Canonical namespaced repo ids. Bare ids like 'natural_questions' only
# resolve via a 307 redirect, and huggingface_hub >= the 2026 releases
# validates the URI shape BEFORE following it:
#   HfUriError: Repository id must be 'namespace/name', got 'natural_questions'
# The bare id is kept as a fallback for older `datasets` installs.
HF_REPO_IDS = {
    'NQ': ['google-research-datasets/natural_questions', 'natural_questions'],
    'HotpotQA': ['hotpotqa/hotpot_qa', 'hotpot_qa'],
    'QASPER': ['allenai/qasper'],
}


# Corpus semantics version. Defined in config (CHECKPOINT_DIR depends on it)
# and re-exported here, where every reader already expects to find it.
CORPUS_VERSION = config.CORPUS_VERSION


def _load_dataset(*args, **kwargs):
    """Lazy import so the dataclasses stay usable without `datasets`."""
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


def _norm_contains(haystack: str, needle: str) -> bool:
    """Containment test. Delegates to the shared normalizer.

    This used to be a whitespace-only comparison, which failed on NQ's
    token-joined text (`Röntgen 's` vs `Röntgen's`) and over-reported
    missing answers by 3x — see textnorm.py for the measurement.
    """
    return textnorm.contains(haystack, needle)


def _load_first_available(candidates, *args, **kwargs):
    """Try each repo id in turn; raise the last error if all fail."""
    last = None
    for repo_id in candidates:
        try:
            return _load_dataset(repo_id, *args, **kwargs)
        except Exception as e:  # noqa: BLE001 - report the final failure
            print(f'  [{repo_id}] failed: {type(e).__name__}: {e}')
            last = e
    raise RuntimeError(
        f'none of {candidates} could be loaded; last error: {last}')


@dataclass
class Document:
    doc_id: str
    text: str
    # Query ids this document is gold evidence for (usually one).
    gold_for: Set[str] = field(default_factory=set)
    # HotpotQA only: query_id -> gold sentences inside this document,
    # used for chunk-level relevance (a chunk of a gold doc counts as
    # relevant only if it contains a gold sentence).
    gold_sentences: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class QASample:
    query_id: str
    question: str
    answer: str
    gold_context: str          # gold evidence text (qrels + ESA only, never F)
    dataset: str = ''
    # True when gold_context is a STAND-IN, not annotated evidence. QASPER
    # free-form answers sometimes cite a figure or table rather than a body
    # paragraph; the loader then falls back to the paper's first three
    # paragraphs so the query keeps a retrieval target. That fallback is
    # acceptable for qrels but NOT for the C2 oracle condition, which claims
    # to feed the generator ground-truth evidence. Read it with
    # getattr(s, 'gold_is_fallback', False) — older pickles predate the field.
    gold_is_fallback: bool = False


@dataclass
class LoadedDataset:
    name: str
    samples: List[QASample]
    documents: List[Document]
    # Provenance of the sample: how many candidates were examined and why any
    # were discarded. The paper has to state how the n queries were selected
    # (Berend point 8), and "we drew until we had n usable ones" is only an
    # honest sentence if the discard count is recorded. Read with
    # getattr(ds, 'stats', {}) — older pickles predate the field.
    stats: Dict[str, int] = field(default_factory=dict)


NQ_CONTEXT_TOKENS = 500


def load_nq(n: int) -> LoadedDataset:
    """
    Natural Questions — single-hop factoid QA.

    ⚠️ B6 fix (2026-08-13, measured on the pilot). This loader used to keep the
    FIRST 500 non-HTML tokens of the document. On the 50-query pilot the short
    answer fell outside that window for **17 of 50 queries (34%)** — the answer
    was not in the corpus at all, so no retriever could ever surface it. Yet
    the qrels still marked chunks of that document relevant, so NDCG@5 reported
    ~0.95 and the generator, correctly, answered "I cannot answer based on the
    provided context".

    That combination is poison for this paper specifically: it manufactures
    "retrieval succeeded but the answer was wrong" rows, which is the exact
    cell the central claim rests on. It would have looked like overwhelming
    evidence that good retrieval is not sufficient, and it would have been an
    artifact of the loader.

    The window is now CENTRED on the annotated answer span, so a sampled
    document contains its own answer by construction. NQ gives start_token and
    end_token indices into the FULL token list, so they have to be remapped
    through the HTML filter before use.
    """
    ck = f'dataset_nq_{n}_{CORPUS_VERSION}'
    cached = load_checkpoint(ck)
    if cached:
        return cached

    print('Loading Natural Questions...')
    ds = _load_first_available(HF_REPO_IDS['NQ'], split='validation',
                               streaming=True)
    samples, documents = [], []
    n_recentred = n_scanned = n_no_answer = n_absent = 0
    # Draw until n USABLE queries are collected rather than taking the first
    # n candidates (B8). The cap only exists so a change upstream cannot turn
    # this into an unbounded scan of the split.
    max_scanned = max(200, n * 30)
    for item in ds:
        if len(samples) >= n or n_scanned >= max_scanned:
            break
        n_scanned += 1
        # §7 fix: short_answers fields are lists, take text[0] directly
        answer_text, ans_start = None, None
        for sa in item['annotations']['short_answers']:
            if sa['text']:
                answer_text = sa['text'][0]
                ans_start = sa['start_token'][0] if sa.get('start_token') else None
                break
        if not answer_text:
            n_no_answer += 1
            continue

        # §7 fix: drop HTML tokens, but keep each survivor's ORIGINAL index so
        # the answer's start_token can be located in the filtered sequence.
        tokens = item['document']['tokens']
        kept = [(i, t) for i, (t, h)
                in enumerate(zip(tokens['token'], tokens['is_html'])) if not h]
        text_tokens = [t for _, t in kept]

        # B6: centre the window on the answer instead of taking the head.
        start = 0
        if ans_start is not None:
            pos = next((j for j, (orig, _) in enumerate(kept)
                        if orig >= ans_start), None)
            if pos is not None and pos >= NQ_CONTEXT_TOKENS:
                start = max(0, pos - NQ_CONTEXT_TOKENS // 2)
                n_recentred += 1
        context = ' '.join(text_tokens[start:start + NQ_CONTEXT_TOKENS])

        # B8: discard the query if its answer is STILL not in the retained
        # window. Re-centring fixes most of these, but not all: when NQ gives
        # no start_token there is nothing to centre on, and a few annotated
        # spans sit outside the document text altogether. Keeping them was
        # actively harmful — qrels marked the document relevant, so NDCG
        # counted a hit for a context that cannot possibly answer the
        # question, manufacturing the `hit x incorrect` rows the paper's
        # central claim rests on. A query whose answer is nowhere in the
        # corpus tests nothing about retrieval.
        if not textnorm.contains(context, answer_text):
            n_absent += 1
            continue

        qid = f'nq_{len(samples)}'
        samples.append(QASample(
            query_id=qid,
            question=item['question']['text'],
            answer=answer_text,
            gold_context=context,
            dataset='NQ',
        ))
        # B4 fix: every sampled context joins one shared corpus, so each
        # query retrieves against n-1 distractor documents, not zero.
        # B6: relevance is answer-bearing, matching how HotpotQA already works
        # (gold_sentences) — a chunk of the gold document counts as relevant
        # only if it actually carries the answer. Without this, `hit` in the
        # necessary/sufficient grid means "right document" rather than "the
        # model was shown the answer", and the grid measures the wrong thing.
        documents.append(Document(
            doc_id=f'{qid}_doc',
            text=context,
            gold_for={qid},
            gold_sentences={qid: [answer_text]},
        ))
    stats = {
        'requested': n,
        'kept': len(samples),
        'scanned': n_scanned,
        'skipped_no_short_answer': n_no_answer,
        'skipped_answer_absent': n_absent,
        'recentred_on_answer': n_recentred,
    }
    print(f'  loaded {len(samples)} NQ samples, {len(documents)} corpus docs')
    print(f'  scanned {n_scanned} candidates: '
          f'{n_no_answer} had no short answer, '
          f'{n_absent} had the answer outside the retained window (dropped), '
          f'{n_recentred} re-centred on the answer span (B6)')
    if len(samples) < n:
        print(f'  [!] only {len(samples)}/{n} usable NQ queries after '
              f'scanning {n_scanned} — the split ran out')
    # Invariant, not a warning: after B8 every retained sample contains its
    # own answer, so `hit` in the necessary/sufficient grid means "the model
    # was shown the answer".
    assert all(textnorm.contains(s.gold_context, s.answer) for s in samples), \
        'B8 invariant violated: a retained NQ sample lacks its own answer'
    result = LoadedDataset('NQ', samples, documents, stats=stats)
    save_checkpoint(ck, result)
    return result


def load_hotpotqa(n: int) -> LoadedDataset:
    """HotpotQA — multi-hop reasoning QA (distractor setting)."""
    ck = f'dataset_hotpot_{n}_{CORPUS_VERSION}'
    cached = load_checkpoint(ck)
    if cached:
        return cached

    print('Loading HotpotQA...')
    ds = _load_first_available(HF_REPO_IDS['HotpotQA'], 'distractor',
                               split='validation', streaming=True)
    samples, documents = [], []
    for item in ds:
        if len(samples) >= n:
            break
        sf = item['supporting_facts']
        ctx = item['context']

        # §7 fix: (title, sent_id) pairs index into context.sentences
        gold_by_title: Dict[str, List[str]] = {}
        for title, sid in zip(sf['title'], sf['sent_id']):
            if title in ctx['title']:
                idx = ctx['title'].index(title)
                if sid < len(ctx['sentences'][idx]):
                    gold_by_title.setdefault(title, []).append(
                        ctx['sentences'][idx][sid])
        if not gold_by_title:
            continue

        qid = f'hotpot_{len(samples)}'
        gold_sents_flat = [s for sents in gold_by_title.values() for s in sents]
        samples.append(QASample(
            query_id=qid,
            question=item['question'],
            answer=item['answer'],
            gold_context=' '.join(gold_sents_flat),
            dataset='HotpotQA',
        ))
        # One document per paragraph (title). The 8 distractor paragraphs of
        # each question join the corpus too, exactly as in the notebook.
        for p_idx, (title, sents) in enumerate(zip(ctx['title'],
                                                   ctx['sentences'])):
            doc = Document(
                doc_id=f'{qid}_p{p_idx}',
                text=' '.join(sents),
            )
            if title in gold_by_title:
                doc.gold_for.add(qid)
                doc.gold_sentences[qid] = gold_by_title[title]
            documents.append(doc)
    print(f'  loaded {len(samples)} HotpotQA samples, {len(documents)} corpus docs')
    result = LoadedDataset('HotpotQA', samples, documents)
    save_checkpoint(ck, result)
    return result


def load_qasper(n: int) -> LoadedDataset:
    """QASPER — scientific paper QA."""
    ck = f'dataset_qasper_{n}_{CORPUS_VERSION}'
    cached = load_checkpoint(ck)
    if cached:
        return cached

    print('Loading QASPER...')
    # `datasets` 5.0 removed dataset-script execution, and allenai/qasper ships
    # a loading script (qasper.py), so the plain call now raises
    #   RuntimeError: Dataset scripts are no longer supported
    # regardless of trust_remote_code. HuggingFace's own auto-conversion of the
    # same dataset lives on the refs/convert/parquet branch and carries the
    # IDENTICAL nested structure (verified 2026-09-14): full_text.paragraphs is
    # list-of-lists, `qas` is a dict of parallel lists (the §7 fix still
    # applies), and each answer keeps free_form_answer / extractive_spans /
    # evidence. Everything below this line is therefore unchanged.
    # The script path is kept as a fallback for older `datasets` installs.
    try:
        ds = _load_dataset('allenai/qasper', split='validation',
                           revision='refs/convert/parquet')
    except Exception:
        ds = _load_dataset('allenai/qasper', split='validation',
                           trust_remote_code=True)

    samples, documents = [], []
    for paper_idx, paper in enumerate(ds):
        if len(samples) >= n:
            break
        # Corpus documents: one per paragraph of the paper, shared by all
        # of this paper's questions.
        paragraphs = []
        for section_paragraphs in paper['full_text']['paragraphs']:
            paragraphs.extend(p for p in section_paragraphs if p.strip())
        para_docs = [
            Document(doc_id=f'qasper_paper{paper_idx}_p{j}', text=p)
            for j, p in enumerate(paragraphs)
        ]

        # §7 fix: qas is a dict of parallel lists, index columns together
        qas = paper['qas']
        for q_idx in range(len(qas['question'])):
            if len(samples) >= n:
                break
            question = qas['question'][q_idx]
            answers_data = qas['answers'][q_idx]
            answer, evidence = None, []
            for ans in answers_data['answer']:
                if ans.get('free_form_answer'):
                    answer = ans['free_form_answer']
                    evidence = list(ans.get('evidence', []))
                    break
                elif ans.get('extractive_spans'):
                    answer = ' '.join(ans['extractive_spans'])
                    evidence = list(ans.get('evidence', []))
                    break
            if not answer:
                continue

            qid = f'qasper_{len(samples)}'
            # Evidence strings come from full_text, so exact match against
            # paragraph text identifies the gold documents.
            evidence_set = {e.strip() for e in evidence if e.strip()}
            marked = False
            for doc in para_docs:
                if doc.text.strip() in evidence_set:
                    doc.gold_for.add(qid)
                    marked = True
            if not marked and para_docs:
                # Free-form answers sometimes carry figure/table evidence
                # that is not a body paragraph; fall back to the first
                # paragraphs so the query keeps a gold target.
                for doc in para_docs[:3]:
                    doc.gold_for.add(qid)
            # Flag the stand-in so the oracle condition can exclude it. Without
            # this, C2 would feed "the first three paragraphs of the paper" to
            # the generator and report the result as an oracle CEILING.
            is_fallback = (not evidence) or (not marked)
            gold_ctx = ' '.join(evidence) if evidence else ' '.join(paragraphs[:3])
            samples.append(QASample(
                query_id=qid,
                question=question,
                answer=answer,
                gold_context=gold_ctx,
                dataset='QASPER',
                gold_is_fallback=is_fallback,
            ))
        documents.extend(para_docs)
    n_fallback = sum(1 for s in samples if s.gold_is_fallback)
    print(f'  loaded {len(samples)} QASPER samples, {len(documents)} corpus docs')
    if n_fallback:
        print(f'  [!] {n_fallback}/{len(samples)} use a stand-in gold context '
              f'({n_fallback / len(samples):.0%}) — excluded from the strict '
              f'C2 oracle')
    result = LoadedDataset('QASPER', samples, documents)
    save_checkpoint(ck, result)
    return result


LOADERS = {
    'NQ': load_nq,
    'HotpotQA': load_hotpotqa,
    'QASPER': load_qasper,
}


def load_all(n: int, names: List[str] = None) -> Dict[str, LoadedDataset]:
    names = names or config.DATASETS
    out = {}
    for name in names:
        out[name] = LOADERS[name](n)
    return out


if __name__ == '__main__':
    # Smoke test: 10 samples per dataset, print a couple from each.
    for name, loaded in load_all(10).items():
        print(f'\n=== {name}: {len(loaded.samples)} samples, '
              f'{len(loaded.documents)} docs ===')
        n_gold = sum(1 for d in loaded.documents if d.gold_for)
        print(f'  gold-marked docs: {n_gold}')
        for s in loaded.samples[:2]:
            print(f'  [{s.query_id}] Q: {s.question[:80]}')
            print(f'             A: {s.answer[:80]}')
