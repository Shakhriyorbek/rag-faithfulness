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


def _load_dataset(*args, **kwargs):
    """Lazy import so the dataclasses stay usable without `datasets`."""
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


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


def load_nq(n: int) -> LoadedDataset:
    """Natural Questions — single-hop factoid QA."""
    cached = load_checkpoint(f'dataset_nq_{n}')
    if cached:
        return cached

    print('Loading Natural Questions...')
    ds = _load_first_available(HF_REPO_IDS['NQ'], split='validation',
                               streaming=True)
    samples, documents = [], []
    for item in ds:
        if len(samples) >= n:
            break
        # §7 fix: short_answers fields are lists, take text[0] directly
        answer_text = None
        for sa in item['annotations']['short_answers']:
            if sa['text']:
                answer_text = sa['text'][0]
                break
        if not answer_text:
            continue
        # §7 fix: drop HTML tokens
        tokens = item['document']['tokens']
        text_tokens = [t for t, h in zip(tokens['token'], tokens['is_html'])
                       if not h]
        context = ' '.join(text_tokens[:500])

        qid = f'nq_{len(samples)}'
        samples.append(QASample(
            query_id=qid,
            question=item['question']['text'],
            answer=answer_text,
            gold_context=context,
            dataset='NQ',
        ))
        # B4 fix: every sampled context joins one shared corpus, so each
        # query retrieves against 999 distractor documents, not zero.
        documents.append(Document(
            doc_id=f'{qid}_doc',
            text=context,
            gold_for={qid},
        ))
    print(f'  loaded {len(samples)} NQ samples, {len(documents)} corpus docs')
    result = LoadedDataset('NQ', samples, documents)
    save_checkpoint(f'dataset_nq_{n}', result)
    return result


def load_hotpotqa(n: int) -> LoadedDataset:
    """HotpotQA — multi-hop reasoning QA (distractor setting)."""
    cached = load_checkpoint(f'dataset_hotpot_{n}')
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
    save_checkpoint(f'dataset_hotpot_{n}', result)
    return result


def load_qasper(n: int) -> LoadedDataset:
    """QASPER — scientific paper QA."""
    cached = load_checkpoint(f'dataset_qasper_{n}')
    if cached:
        return cached

    print('Loading QASPER...')
    # §7 fix: needs trust_remote_code on newer `datasets`
    try:
        ds = _load_dataset('allenai/qasper', split='validation',
                          trust_remote_code=True)
    except Exception:
        ds = _load_dataset('allenai/qasper', split='validation')

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
    save_checkpoint(f'dataset_qasper_{n}', result)
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
