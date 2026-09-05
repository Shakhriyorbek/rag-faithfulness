// build_v6.js — regenerates RAG_Faithfulness_v6_evaluators.docx
//
// v6 rewrites v5 around the 2026-08-25/26 results. The v5 title
// ("Retrieval Quality Predicts Correctness, Not Faithfulness") is contradicted
// by its own data: the verdict flips with the evaluator in 1 of 4 cells, and
// the retrieval-faithfulness correlation is positive on HotpotQA. The spine is
// now measurement validity, with the embedder grid as the testbed.
//
// ⚠️ REFERENCES ARE UNVERIFIED. Five were wrong in earlier drafts (see
// CLAUDE.md §6). Check every entry against the real source before submitting.
//
//   NODE_PATH=$(npm root -g) node paper/build_v6.js

const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType,
} = require('docx');

const W = 9000; // content width in DXA (A4 with default margins)

const P = (text, opts = {}) => new Paragraph({
  alignment: opts.align,
  spacing: { after: opts.after === undefined ? 120 : opts.after,
             before: opts.before || 0, line: 276 },
  indent: opts.indent,
  children: [new TextRun({
    text, bold: opts.bold, italics: opts.italics,
    size: opts.size || 20, font: 'Times New Roman',
  })],
});

const PR = (parts, opts = {}) => new Paragraph({
  alignment: opts.align,
  spacing: { after: opts.after === undefined ? 120 : opts.after, line: 276 },
  children: parts.map((p) => {
    const [t, bold, italics] = Array.isArray(p) ? p : [p, false, false];
    return new TextRun({ text: t, bold, italics, size: opts.size || 20,
                         font: 'Times New Roman' });
  }),
});

const H = (text, level) => new Paragraph({
  heading: level,
  spacing: { before: 260, after: 120 },
  children: [new TextRun({ text, bold: true, font: 'Times New Roman',
                           size: level === HeadingLevel.HEADING_1 ? 26 : 22 })],
});

const BULLET = (text) => new Paragraph({
  bullet: { level: 0 },
  spacing: { after: 60, line: 276 },
  children: [new TextRun({ text, size: 20, font: 'Times New Roman' })],
});

const EQ = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 120 },
  children: [new TextRun({ text, size: 20, font: 'Times New Roman',
                           italics: true })],
});

function cell(text, { bold = false, fill = null, widths } = {}) {
  return new TableCell({
    width: { size: widths, type: WidthType.DXA },
    shading: fill ? { type: ShadingType.CLEAR, fill, color: 'auto' } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({
      spacing: { after: 0, line: 240 },
      children: [new TextRun({ text: String(text), bold, size: 18,
                               font: 'Times New Roman' })],
    })],
  });
}

function table(header, rows, cols) {
  const mk = (cells, isHead) => new TableRow({
    tableHeader: isHead,
    children: cells.map((c, i) =>
      cell(c, { bold: isHead, fill: isHead ? 'E8E8E8' : null, widths: cols[i] })),
  });
  return new Table({
    columnWidths: cols,
    width: { size: W, type: WidthType.DXA },
    rows: [mk(header, true), ...rows.map((r) => mk(r, false))],
  });
}

const CAP = (text) => new Paragraph({
  spacing: { before: 100, after: 180 },
  children: [new TextRun({ text, size: 17, italics: true,
                           font: 'Times New Roman' })],
});

const H1 = HeadingLevel.HEADING_1;
const H2 = HeadingLevel.HEADING_2;
const kids = [];

// ── Title block ────────────────────────────────────────────────────
kids.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 120 },
  children: [new TextRun({
    text: 'When the Evaluator Decides the Result: Measuring Faithfulness in Retrieval-Augmented Generation',
    bold: true, size: 32, font: 'Times New Roman',
  })],
}));
kids.push(P('Shakhriyorbek Boltabaev', { align: AlignmentType.CENTER, after: 40, size: 22 }));
kids.push(P('Institute of Informatics, University of Szeged', { align: AlignmentType.CENTER, after: 40, size: 20 }));
kids.push(P('Szeged, Hungary', { align: AlignmentType.CENTER, after: 300, size: 20 }));

// ── Abstract ───────────────────────────────────────────────────────
kids.push(H('Abstract', H1));
kids.push(P(
  'Faithfulness in retrieval-augmented generation is commonly measured by asking a natural '
  + 'language inference model whether the generated answer is entailed by the retrieved '
  + 'context, aggregated by taking the maximum over retrieved chunks. We hold a generation '
  + 'grid fixed — four embedding models, two question answering datasets, two generators, '
  + '16,000 generated answers — and vary only the faithfulness evaluator. Across four '
  + 'dataset-generator cells, the conclusion about whether embedding choice affects '
  + 'faithfulness changes in one of them depending on which evaluator is used. We trace this '
  + 'to measured properties of the evaluators rather than to the retrieval systems. Replacing '
  + 'a grounded value in an answer with one absent from the context — the failure a document '
  + 'question answering system most needs to detect — moves the entailment score by 0.054 on '
  + 'Natural Questions and 0.085 on HotpotQA for one generator, against 0.451 and 0.391 for the '
  + 'other, and is caught by a fixed threshold 5-14% of the time against 57-71%. The gap is '
  + 'largely accounted for by answer structure: the score falls as the number of assertions in '
  + 'an answer grows, and the two generators average 3.08 and 1.25 assertions per answer. It is '
  + 'not accounted for entirely — at a matched single assertion the two still differ, +0.102 '
  + 'against +0.485 — so answer structure is a partial account rather than a complete '
  + 'explanation. Claim-level aggregation and an alternative evaluator each recover '
  + 'part of the sensitivity but neither removes the effect. We also show that including '
  + 'refusals in the evaluation population produces an apparent relationship between retrieval '
  + 'quality and faithfulness that disappears when they are excluded. We report what these '
  + 'measurement choices do to a concrete equivalence claim, and give the conditions under '
  + 'which such a claim can be stated.',
  { after: 240 }));

// ── I. Introduction ────────────────────────────────────────────────
kids.push(H('I. Introduction', H1));
kids.push(P(
  'A document question answering system is given a set of uploaded files and must answer from '
  + 'them. If an invoice records a project budget of $1,000, an answer stating $1,500 is a '
  + 'failure of a specific kind: it is fluent, on topic, cites the right document, and is '
  + 'wrong in the one place that matters. Systems of this kind are typically guarded by a '
  + 'faithfulness score — a measure of whether the generated answer is supported by the '
  + 'retrieved text — with answers below a threshold suppressed or flagged.',
  { after: 120 }));
kids.push(P(
  'This paper asks what such a score actually measures. We do not propose a new evaluator. We '
  + 'take a fixed set of 16,000 generated answers and vary the measurement, and report where '
  + 'the measurement rather than the system determines the conclusion.',
  { after: 120 }));
kids.push(P('Our contributions are:', { after: 80 }));
kids.push(BULLET(
  'A controlled falsification experiment. Replacing one grounded value in an answer with a '
  + 'value absent from the retrieved context, holding everything else fixed, gives a direct '
  + 'measurement of evaluator sensitivity to the error type that matters (Section V).'));
kids.push(BULLET(
  'Evidence that the sensitivity depends strongly on answer structure, not only on the '
  + 'evaluator: entailment scores fall as the number of assertions per answer grows, for both '
  + 'generators tested (Section V-C).'));
kids.push(BULLET(
  'A demonstration that the choice of evaluation population — specifically, whether refusals '
  + 'are included — can produce an apparent relationship between retrieval quality and '
  + 'faithfulness where none survives their exclusion (Section VI).'));
kids.push(BULLET(
  'A statement of what an equivalence claim about faithfulness requires: naming the evaluator, '
  + 'reporting the margin as a curve rather than a point, and anchoring that margin to a '
  + 'measured effect size (Section VIII).'));
kids.push(P(
  'The embedding-model grid is a testbed, not the object of study. It supplies a set of '
  + 'retrieval systems spanning 4.4 and 11.9 NDCG@5 points on the two datasets, with generation '
  + 'and prompting held constant, which is what makes the measurement comparison controlled.',
  { after: 160 }));

// ── II. Related Work ───────────────────────────────────────────────
kids.push(H('II. Related Work', H1));
kids.push(H('A. Retrieval quality and downstream outcome', H2));
kids.push(P(
  'Salemi and Zamani [1] report that human document-level relevance labels, and LLM relevance '
  + 'judgments, correlate only weakly with downstream retrieval-augmented generation '
  + 'performance, and propose eRAG, which labels each retrieved document by the downstream '
  + 'result the system\'s own generator produces from it alone. Their strongest baseline is '
  + 'answer-containment labelling, the scheme our qrels use, which retains a moderate '
  + 'correlation (Kendall\'s tau 0.31-0.40 on the question-answering datasets); their reported '
  + 'correlations also decline as the number of retrieved documents grows, and are measured at '
  + 'k = 50 against ours at k = 5. Their downstream measure is correctness throughout, obtained '
  + 'from a fine-tuned 60M-parameter encoder-decoder reader. Our starting point is downstream '
  + 'of theirs: we take as given that retrieval metrics do not directly predict generation '
  + 'quality, and ask which downstream property a given measurement instrument can resolve at '
  + 'all — for faithfulness rather than correctness, and for prompted '
  + 'instruction-following generators, which can abstain.',
  { after: 120 }));
kids.push(H('B. Faithfulness measurement', H2));
kids.push(P(
  'Entailment-based faithfulness scoring is standard: a natural language inference model is '
  + 'asked whether the retrieved context entails the generated answer. RAGAS [2] packages this '
  + 'alongside context and answer relevance metrics. AlignScore [3] trains a unified alignment '
  + 'function over several factual-consistency tasks rather than relying on a general-purpose '
  + 'inference model. FaithJudge [4] uses a language model as the judge. Our work is '
  + 'orthogonal to proposing another such metric: we measure how three of them respond to a '
  + 'controlled, known error, and report the consequences for conclusions drawn with them.',
  { after: 120 }));
kids.push(H('C. Meta-evaluation of faithfulness detectors', H2));
kids.push(P(
  'A separate line of work evaluates the detectors themselves. TRUE [5] standardises factual '
  + 'consistency evaluation over eleven annotated datasets and finds inference-based and '
  + 'question-generation approaches to be strong and complementary. AGGREFACT [6] aggregates '
  + 'nine summarisation factuality datasets, stratifies them by the model that produced the '
  + 'errors, and reports that no single metric is best in all settings or for all error types. '
  + 'RAGTruth [7] supplies word-level hallucination annotations over roughly 18,000 '
  + 'retrieval-augmented responses. FaithBench [8] is assembled from summaries on which current '
  + 'detectors disagree, and reports detection accuracies near 50% on them. MiniCheck [9] '
  + 'unifies recent grounding datasets as LLM-AGGREFACT and trains compact fact-checkers '
  + 'against that benchmark.',
  { after: 120 }));
kids.push(P(
  'These works compare detectors against gold hallucination labels on a labelled corpus, and '
  + 'the quantity they report is detector accuracy. Our design contains no gold hallucination '
  + 'label. We hold the generation grid, the queries and the generated answers fixed, swap the '
  + 'evaluator, and ask whether the ranking of the systems under study survives. The two '
  + 'questions come apart: a detector that is more accurate on a labelled corpus need not be one '
  + 'that resolves differences between the systems a practitioner is actually choosing between, '
  + 'and it is the second property that decides whether a published comparison describes the '
  + 'systems or the instrument. Xiao et al. [10] argue on general grounds that generation '
  + 'metrics should be analysed as measurement instruments with stated reliability and validity. '
  + 'The falsification probe in Section V and the equivalence curve in Section VIII are two such '
  + 'analyses, carried out on the instruments a RAG faithfulness result is usually reported '
  + 'with.',
  { after: 120 }));
kids.push(H('D. Hallucination and grounding analysis', H2));
kids.push(P(
  'ReDeEP [11] localises hallucination to attention heads and feed-forward components. '
  + 'Sinha [12] reports that embedding-based detection separates synthetic hallucinations well '
  + 'and real ones poorly, the hard cases being those that remain semantically close to a '
  + 'faithful response. Chen et al. [13] vary the embedding model in a RAG pipeline and combine '
  + 'several of them for mathematics question answering, on the observation that different '
  + 'embedders succeed on different queries. '
  + 'Our analysis is correlational and behavioural rather than mechanistic: we do not probe '
  + 'model internals.',
  { after: 160 }));

// ── III. Definitions ───────────────────────────────────────────────
kids.push(H('III. Definitions', H1));
kids.push(H('A. Faithfulness', H2));
kids.push(P(
  'For a generated answer a and retrieved context R = {d1, ..., dk}, the standard aggregate is:',
  { after: 60 }));
kids.push(EQ('F_max(a, R) = max over d in R of NLI(d entails a)          (1)'));
kids.push(P(
  'We also evaluate a claim-level aggregate. Writing C(a) for the assertions in a:',
  { after: 60 }));
kids.push(EQ('F_claim(a, R) = min over c in C(a) of max over d in R of NLI(d entails c)     (2)'));
kids.push(P(
  'The maximum over chunks is appropriate — any retrieved document may support a claim. The '
  + 'maximum over claims, implicit in (1), is not: an answer is only as supported as its '
  + 'least supported assertion, so one unsupported claim should not be masked by several '
  + 'supported ones. Faithfulness is measured against the generated answer and the retrieved '
  + 'context only; the gold answer is never used.',
  { after: 120 }));
kids.push(H('B. Correctness and retrieval hit', H2));
kids.push(P(
  'Correctness is measured against the gold answer and never against the context, so that a '
  + 'grounded wrong answer and an ungrounded right answer remain distinguishable. A retrieval '
  + 'hit means the gold answer string appears in the retrieved context. This is a strong '
  + 'definition of retrieval success, and Section X notes what it does not capture.',
  { after: 120 }));
kids.push(H('C. On differencing retrieval quality and faithfulness', H2));
kids.push(P(
  'An earlier version of this work defined a gap metric as the difference between NDCG@5 and '
  + 'faithfulness, with a normalised variant dividing by retrieval quality. We report the '
  + 'properties of that family here because they are general rather than specific to our '
  + 'implementation. First, the difference has no common unit: a rank-based retrieval metric '
  + 'and an entailment probability are not commensurable. Second, on our data the difference is '
  + 'negative for every model on Natural Questions, since faithfulness (0.85) exceeds retrieval '
  + 'quality (0.79) — the metric presumes generation loses fidelity relative to retrieval, and '
  + 'it does not. Third, any metric of the form 1 - F/RQ reproduces the retrieval ranking '
  + 'whenever F is near-constant across systems, so apparent agreement of such a metric across '
  + 'generators is uninformative: retrieval quality is generator-invariant by construction. We '
  + 'therefore report absolute faithfulness throughout. Retiring the metric also disposes of a '
  + 'known objection to it at the root rather than patching it: a difference cannot separate a '
  + 'system whose retrieval and faithfulness are both high from one where both are low, and the '
  + 'normalised variant rescales that ambiguity without removing it. Reporting the two '
  + 'quantities separately keeps the both-low case visible. The retired implementation is held '
  + 'out of the analysis path that produced the tables reported here.',
  { after: 160 }));

// ── IV. Experimental Setup ─────────────────────────────────────────
kids.push(H('IV. Experimental Setup', H1));
kids.push(P(
  'Two question answering datasets are used, Natural Questions [21] and HotpotQA [22], at 1,000 '
  + 'queries each. Four embedding models index each corpus; retrieval takes the top five chunks of 256 tokens '
  + 'with 32-token overlap; two generators produce an answer from a byte-identical prompt at '
  + 'temperature 0. This yields 4 x 2 x 2 x 1,000 = 16,000 generated answers, each re-scored by '
  + 'every evaluator. Prompt parity across generators is enforced by construction — a single '
  + 'prompt builder, one user turn, no system message — because a prompt difference would '
  + 'confound every cross-generator comparison in Section V.',
  { after: 120 }));
kids.push(table(
  ['Embedding model', 'Paradigm', 'NDCG@5 (NQ)', 'NDCG@5 (HotpotQA)'],
  [
    ['all-mpnet-base-v2', 'contrastive', '0.786', '0.705'],
    ['BGE-M3', 'multilingual', '0.794', '0.809'],
    ['E5-large-instruct', 'instruction-tuned', '0.806', '0.824'],
    ['text-embedding-3-small', 'proprietary', '0.830', '0.767'],
    ['spread', '', '4.4 pts', '11.9 pts'],
  ],
  [2900, 2000, 2050, 2050]));
kids.push(CAP(
  'Table I. Retrieval quality of the four systems in the testbed. The ranking differs between '
  + 'datasets, and the spread is far larger on HotpotQA.'));
kids.push(P(
  'Three evaluators are compared. NLI-max is (1) with a DeBERTa-v3-large [14] inference model, the '
  + 'common choice. Claim-min is (2) with the same inference model and sentence-level claim '
  + 'segmentation, chosen so that no second generator enters the measurement loop. AlignScore '
  + '[3] is an independently trained alignment model. All three score identical inputs.',
  { after: 120 }));
kids.push(P(
  'The answers are byte-identical across evaluators; the premises are not, and the difference '
  + 'is part of what the comparison measures. Premise construction follows each evaluator\'s '
  + 'intended interface. The entailment scorer receives each of the five retrieved chunks '
  + 'separately plus their concatenation truncated to the model\'s 512-token window, and the '
  + 'maximum is taken, since any one retrieved document may entail the answer and the '
  + 'concatenation exceeds the window. The claim-level scorer applies that same construction to '
  + 'each extracted claim. AlignScore receives the concatenated context as a single premise and '
  + 'performs its own splitting internally. Part of the difference reported in Section V is '
  + 'therefore premise granularity rather than evaluator sensitivity; Section V-A bounds that '
  + 'share by re-scoring the same cases with the same entailment model on the concatenated '
  + 'premise alone. The hypothesis is normalised identically for all three: markdown emphasis, '
  + 'headings and list markers are stripped before scoring. This matters because one generator '
  + 'emits markdown in 74.9% of its answers and the other in 0.3%, and the entailment model '
  + 'responds to it — see Section VII-B.',
  { after: 120 }));
kids.push(P(
  'One refusal rule is used throughout. An answer counts as a refusal when, after SQuAD-style '
  + 'normalisation, it begins with one of a fixed list of declining phrases; a single shared '
  + 'implementation applies it, so Sections V through VIII describe the same population. It '
  + 'excludes 4,123 of the 16,000 answers (25.8%): 587 on NQ/Claude (14.7%), 1,162 on '
  + 'NQ/GPT-4o-mini (29.0%), 1,166 on HotpotQA/Claude (29.1%) and 1,208 on HotpotQA/GPT-4o-mini '
  + '(30.2%). Both generators prepend a "Based on the provided context," preamble to everything, '
  + 'refusals included, so the rule strips that preamble before the anchored test; without it '
  + '107 refusals, all of them Claude\'s, would be graded as attempts.',
  { after: 160 }));

// ── V. Evaluator sensitivity ───────────────────────────────────────
kids.push(H('V. How Sensitive Are These Evaluators to a Known Error?', H1));
kids.push(H('A. A controlled falsification', H2));
kids.push(P(
  'A faithfulness score is only useful if it responds when an answer stops being faithful. We '
  + 'construct that condition directly. For each answer containing a numeric value that appears '
  + 'in the retrieved context, we replace that value with one that does not appear in the '
  + 'context, preserving surface format, and re-score against the same chunks. Two eligibility '
  + 'rules are load-bearing: the original value must be present in the context, or the answer '
  + 'was never grounded in it; and the replacement must be absent, or the altered answer is '
  + 'accidentally supported and a low response is correct behaviour.',
  { after: 120 }));
kids.push(P(
  'Cases are taken in dataset order rather than drawn at random: for each system, dataset and '
  + 'generator we use the first 200 eligible answers. The scored set is therefore a prefix of '
  + 'an already filtered population, and 3,184 cases are scored against 5,272 eligible across '
  + 'the grid — 800 of 1,946 on NQ/Claude, 800 of 882 on NQ/GPT-4o-mini, 800 of 1,551 on '
  + 'HotpotQA/Claude and 784 of 893 on HotpotQA/GPT-4o-mini. The two GPT-4o-mini cells are '
  + 'close to a census; the two Claude cells are sampled at two fifths and one half.',
  { after: 120 }));
kids.push(P(
  'Two controls accompany it. An entity substitution replaces a grounded named entity instead '
  + 'of a value, giving a comparable edit. Scoring the untouched answer against another '
  + 'query\'s retrieved context gives the floor of each evaluator\'s scale on this data. 200 '
  + 'paired cases per cell, rebuilt deterministically so all three evaluators see identical '
  + 'inputs.',
  { after: 120 }));
kids.push(table(
  ['Dataset / generator', 'Evaluator', 'Orig', 'Falsified', 'Floor', 'Detected @0.5', 'Detected @floor gate'],
  [
    ['NQ / Claude Haiku 4.5', 'NLI-max', '0.906', '0.852', '0.474', '5.1% [3.8-7.0]', '50.2% [45.7-54.7]'],
    ['', 'Claim-min', '0.679', '0.514', '0.245', '25.1% [21.7-28.9]', '58.3% [53.3-63.1]'],
    ['', 'AlignScore', '0.866', '0.619', '0.331', '24.1% [21.3-27.3]', '77.9% [74.4-81.1]'],
    ['NQ / GPT-4o-mini', 'NLI-max', '0.887', '0.436', '0.506', '56.6% [52.9-60.1]', '85.2% [81.9-88.1]'],
    ['', 'Claim-min', '0.863', '0.390', '0.490', '61.0% [57.4-64.6]', '91.0% [88.1-93.2]'],
    ['', 'AlignScore', '0.912', '0.272', '0.160', '75.4% [72.2-78.4]', '82.1% [79.2-84.7]'],
    ['HotpotQA / Claude', 'NLI-max', '0.702', '0.618', '0.427', '13.9% [11.3-17.0]', '49.7% [42.3-57.2]'],
    ['', 'Claim-min', '0.370', '0.255', '0.161', '34.3% [29.0-40.0]', '48.9% [41.8-56.1]'],
    ['', 'AlignScore', '0.786', '0.454', '0.475', '48.9% [45.2-52.5]', '93.7% [90.7-95.8]'],
    ['HotpotQA / GPT-4o-mini', 'NLI-max', '0.577', '0.186', '0.201', '70.5% [66.2-74.6]', '83.8% [79.7-87.2]'],
    ['', 'Claim-min', '0.571', '0.182', '0.204', '70.6% [66.2-74.6]', '84.2% [80.1-87.6]'],
    ['', 'AlignScore', '0.780', '0.119', '0.234', '89.4% [86.8-91.6]', '97.5% [95.8-98.5]'],
  ],
  [1850, 1250, 780, 950, 780, 1700, 1790]));
kids.push(CAP(
  'Table II. Response of three evaluators to a falsified value, on identical inputs, pooled '
  + 'over the four embedding systems (n = 800 per cell, 784 on HotpotQA/GPT-4o-mini). '
  + '"Detected" is the share of answers that passed a threshold before falsification and fail '
  + 'it after, with Wilson 95% intervals. "Floor" is the mean score of an untouched answer '
  + 'against an unrelated query\'s context; the floor gate is the 95th percentile of that same '
  + 'distribution, per cell and evaluator.'));
kids.push(P(
  'On Claude\'s Natural Questions answers, NLI-max leaves the falsified answer at 0.852 against '
  + 'a floor of 0.474: the altered answer remains comfortably above any usable threshold. On '
  + 'GPT-4o-mini\'s answers the same edit lands at 0.436, essentially at that generator\'s floor '
  + 'of 0.506. The evaluator, the context and the perturbation code are identical; only the '
  + 'generator that wrote the answer differs.',
  { after: 120 }));
kids.push(P(
  'Detection rates are only comparable across evaluators after the scale is normalised, because '
  + 'the evaluators do not share a floor, and we therefore report two gates. On this cell an '
  + 'untouched answer scored against an unrelated context averages 0.474 under NLI-max against '
  + '0.245 under claim-min, so a fixed 0.5 gate sits just above the floor for one evaluator and '
  + 'well above it for the other. The random-context distribution is also heavy-tailed — its '
  + '95th percentile is 0.985 under NLI-max — so an unrelated document clears 0.5 nearly half '
  + 'the time. The second gate is anchored at that 95th percentile, at which "detected" means the '
  + 'falsified answer is no longer better supported than an unrelated document, which is the '
  + 'same statement on every scale. Read that way NLI-max detects 50.2% of falsifications on '
  + 'Claude\'s NQ answers rather than 5.1%, and its gap to GPT-4o-mini narrows from roughly '
  + 'elevenfold to under twofold. The ordering of the three evaluators is unchanged at either '
  + 'gate. The floor-anchored gate is a comparison device and not an operating point: it sits '
  + 'above the mean untouched answer, so a deployed system could not use it.',
  { after: 120 }));

kids.push(H('B. The effect is not verbatim copying', H2));
kids.push(P(
  'A natural explanation is that one generator quotes the retrieved text back, so the evaluator '
  + 'scores a copy of its own premise and a single substituted value cannot move it. The data '
  + 'does not support this. Over the 3,173 falsification cases, the fraction of answer word '
  + '5-grams occurring in the retrieved context as a contiguous token sequence is 0.226 for '
  + 'Claude and 0.235 for GPT-4o-mini: the generator whose scores barely move copies slightly '
  + 'less, not more. The correlation between overlap and the response to falsification is '
  + '+0.046 for Claude (95% CI -0.004 to +0.096) and +0.166 for GPT-4o-mini (+0.118 to +0.212) '
  + '— indistinguishable from zero in the first case and positive in the second, where the '
  + 'explanation requires both to be negative.',
  { after: 120 }));
kids.push(P(
  'The comparison also survives at matched overlap, which the correlation alone does not show. '
  + 'Grouping the cases into four overlap bands leaves Claude between +0.052 and +0.085 and '
  + 'GPT-4o-mini between +0.326 and +0.507; the separation is at least fivefold in every band, '
  + 'including the band above 0.3 where both generators are reusing a substantial part of the '
  + 'retrieved wording. Overlap is measured on tokens with punctuation mapped to spaces, so it '
  + 'is unaffected by the markdown normalisation described in Section VII-B.',
  { after: 160 }));

kids.push(H('C. Answers with more assertions are scored less sensitively', H2));
kids.push(P(
  'Segmenting each answer into sentence-level assertions and grouping by count gives a '
  + 'declining relationship, present for both generators. It is not strictly monotone: the tail '
  + 'buckets are thin and both generators reverse direction once within them.',
  { after: 100 }));
kids.push(table(
  ['Assertions', 'n (Claude)', 'Claude NLI-max', 'Claude Claim-min', 'n (GPT)', 'GPT NLI-max', 'GPT Claim-min'],
  [
    ['1', '91', '+0.102', '+0.102', '680', '+0.485', '+0.485'],
    ['2', '300', '+0.076', '+0.151', '77', '+0.360', '+0.467'],
    ['3', '194', '+0.032', '+0.229', '26', '+0.041', '+0.401'],
    ['4', '103', '+0.024', '+0.162', '10', '+0.204', '+0.192'],
    ['5 or more', '112', '+0.024', '+0.146', '7', '+0.032', '+0.002'],
  ],
  [1150, 1250, 1550, 1620, 1100, 1350, 1400]));
kids.push(CAP(
  'Table III. Drop in score when one grounded value is falsified, by number of assertions in '
  + 'the answer, on Natural Questions. Under NLI-max the response decays as answers accumulate '
  + 'assertions; under claim-level aggregation it does not. The GPT-4o-mini rows beyond two '
  + 'assertions rest on 26, 10 and 7 cases and should be read as such. In the single-assertion '
  + 'row the two aggregates coincide exactly, which is the degenerate case they must reduce to.'));
kids.push(P(
  'The two generators average 3.08 and 1.25 assertions per answer, and median answer lengths of '
  + '55 and 15 words. This largely accounts for the cross-generator difference in Table II: a '
  + 'falsified value is a progressively smaller edit to the hypothesis as correct surrounding '
  + 'text accumulates, and the entailment judgement is dominated by the remainder. It does not '
  + 'account for it entirely — within the single-assertion bucket the two generators still '
  + 'differ, +0.102 against +0.485. As a check on the claim-level implementation, every '
  + 'single-assertion answer scores identically under both aggregates (n = 1,640, largest '
  + 'absolute difference 1.7e-05, which is batch-padding noise), the expected degenerate case. '
  + 'That check is what surfaced the normalisation defect reported in Section VII-B: before '
  + 'markdown was stripped for both aggregates it failed on every one of the 131 formatted '
  + 'answers, by as much as 0.729.',
  { after: 120 }));
kids.push(P(
  'Claim-level aggregation is a partial remedy rather than a repair. Expressed as a fraction of '
  + 'the distance to the random-context floor, the median response on Claude\'s answers moves '
  + 'from 0.01 to 0.03, and the minimum moves in the correct direction in 55% of cases: when '
  + 'the falsified assertion is not already the weakest, the minimum does not move at all. A '
  + 'second dilution remains within assertions — at matched assertion count the two generators '
  + 'still differ (+0.102 against +0.485), and their assertions average 20.5 against 13.8 words.',
  { after: 160 }));

// ── VI. Evaluation population ──────────────────────────────────────
kids.push(H('VI. The Evaluation Population Can Manufacture a Relationship', H1));
kids.push(P(
  'Both generators decline to answer on a substantial share of queries, and the rate depends on '
  + 'retrieval quality. A refusal is correctly not entailed by the context and scores near the '
  + 'floor. Including refusals therefore transfers a retrieval effect into the faithfulness '
  + 'average, at no cost in plausibility.',
  { after: 120 }));
kids.push(table(
  ['HotpotQA, Claude', 'all-mpnet', 'text-emb-3-small', 'BGE-M3', 'E5-instruct'],
  [
    ['NDCG@5', '0.705', '0.767', '0.809', '0.824'],
    ['Refusal rate', '0.376', '0.279', '0.224', '0.207'],
    ['Faithfulness, all rows', '0.548', '0.599', '0.628', '0.638'],
    ['Faithfulness, answered', '0.737', '0.727', '0.731', '0.746'],
  ],
  [2700, 1575, 1725, 1500, 1500]));
kids.push(CAP(
  'Table IV. Pooling refusals into the faithfulness average reproduces the retrieval ranking '
  + 'exactly (spread 0.090). Restricted to answered rows the spread is 0.019 and the ordering '
  + 'does not follow retrieval quality.'));
kids.push(P(
  'Any evaluation population whose membership depends on the system under comparison can '
  + 'transfer an effect from the selection into the measurement. Here the weakest retriever '
  + 'refuses most often, and every refusal lowers its average. All faithfulness results in this '
  + 'paper exclude refusals and are paired on the queries every system answered.',
  { after: 160 }));

kids.push(H('VI-A. Membership Is Set by a Grader, and Graders Disagree', H2));
kids.push(P(
  'Excluding refusals requires deciding which answers are refusals, and that decision is made '
  + 'by a rule rather than given by the data. We compare two defensible rules: a string-matching '
  + 'heuristic keyed to the refusal templates the prompts offer, and a judge model shown the '
  + 'question, the reference answers and the answer. They agree on the large majority of rows '
  + 'and disagree consistently at the margin.',
  { after: 120 }));
kids.push(table(
  ['HotpotQA, Claude', 'all-mpnet', 'text-emb-3-small', 'BGE-M3', 'E5-instruct'],
  [
    ['Refusal rate, heuristic', '0.376', '0.279', '0.224', '0.207'],
    ['Refusal rate, judge', '0.422', '0.332', '0.277', '0.252'],
  ],
  [2700, 1575, 1725, 1500, 1500]));
kids.push(CAP(
  'Table V. Two refusal detectors on the same answers. The judge finds roughly four to five '
  + 'more refusals per hundred answers in every column, so the disagreement is a level shift '
  + 'rather than noise, and it does not reorder the systems.'));
kids.push(P(
  'A uniform shift that preserves the ordering would ordinarily be unremarkable. It is not '
  + 'unremarkable here, because the excluded rows are not a random sample of the population: '
  + 'they are the rows nearest the boundary between refusing and attempting. Substituting one '
  + 'detector for the other moves nineteen rows on Natural Questions with Claude and fifty-five '
  + 'on HotpotQA with Claude, and changes the significance verdict in two of the eight '
  + 'dataset-generator-evaluator cells. Under the entailment aggregate, Natural Questions with '
  + 'Claude moves from no detectable difference (p = 0.223) to a detectable one (p = 0.048); '
  + 'under AlignScore, HotpotQA with Claude moves from p = 0.151 to p = 0.049. Both land within '
  + 'a thousandth of the conventional threshold, which is the point: the verdict in these cells '
  + 'is not a property of the systems.',
  { after: 140 }));
kids.push(P(
  'The count is more stable than the cells. One of four dataset-generator cells changes verdict '
  + 'between evaluators under either detector, but not the same two. We therefore report the '
  + 'count as the finding and treat any individual cell as undetermined.',
  { after: 160 }));

kids.push(H('VI-B. The Correctness Grader Admits Refusals to the Wrong Population', H2));
kids.push(P(
  'The same problem appears once faithfulness is conditioned on correctness. String containment '
  + 'grades an answer correct when a reference answer string occurs in it, and a short reference '
  + 'string can occur inside a refusal. On a 200-answer calibration, ten refusals were graded '
  + 'correct by containment and the judge overturned nine of them. A refusal admitted to the '
  + 'correct population carries a near-floor faithfulness score into the correct-answer average, '
  + 'which lowers the baseline that any correct-versus-incorrect comparison is measured against. '
  + 'Under graded judgement no refusal is graded correct in any of the sixteen cells.',
  { after: 140 }));
kids.push(P(
  'The effect is large enough to change a sign. Measured with containment and a baseline that '
  + 'excluded refusals from the incorrect side only, answers on Natural Questions with Claude '
  + 'appeared more grounded when wrong than when right, by 0.018 to 0.050 across the four '
  + 'systems. Excluding refusals from both sides reduces this to between -0.006 and +0.018. '
  + 'Grading correctness with the judge instead of containment moves it to between -0.051 and '
  + '-0.113, in the opposite direction. Each of the three steps is a correction to the grader '
  + 'rather than a change to the answers, and all three move the quantity the same way.',
  { after: 140 }));
kids.push(P(
  'Tested directly, no positive value of this quantity survives. Across all sixteen cells a '
  + 'two-sample permutation test with Holm correction finds three significant, all negative, and '
  + 'none of the eight positive-signed cells has a confidence interval excluding zero. The '
  + 'clearest single result is E5-instruct on Natural Questions, where the same effect appears '
  + 'under both generators (-0.113 and -0.115). We report no evidence that answers are more '
  + 'grounded when they are wrong. The intervals are wide, between 0.08 and 0.13, because only '
  + '44 to 84 wrong answered rows remain per cell once refusals are excluded; these tests bear '
  + 'on the general claim and not on individual systems.',
  { after: 160 }));

kids.push(H('VI-C. A Contingency Cell That Mostly Counts Refusals', H2));
kids.push(P(
  'The same accounting governs the contingency between retrieval success and correctness. The '
  + 'cell in which retrieval succeeded and the answer was still wrong is the one that bears on '
  + 'whether retrieval quality is sufficient, and most of it consists of answers that were never '
  + 'attempted.',
  { after: 120 }));
kids.push(table(
  ['Retrieval hit and answer incorrect', 'Claude, NQ', 'Claude, HQA', 'GPT, NQ', 'GPT, HQA'],
  [
    ['Share of all queries', '17.4%', '34.0%', '31.7%', '36.3%'],
    ['Of that cell, refusals', '69.3%', '86.8%', '79.0%', '78.1%'],
    ['Attempted and wrong', '5.4%', '4.5%', '6.7%', '7.9%'],
  ],
  [3300, 1425, 1425, 1425, 1425]));
kids.push(CAP(
  'Table VI. Correctness from graded judgement. The raw cell is between 17 and 36 per cent of '
  + 'queries; the share that is a grounded, committed, wrong answer is between 4.5 and 7.9 per '
  + 'cent.'));
kids.push(P(
  'Read without the decomposition, the raw cell would support the claim that good retrieval '
  + 'frequently fails to produce a correct answer. Most of what it counts is a system declining '
  + 'to answer, and refusal rates differ by more than twenty points across the systems compared. '
  + 'The residual is a real phenomenon and it is roughly a fifth the size the raw cell implies.',
  { after: 160 }));

// ── VII. Consequences for the comparison ───────────────────────────
kids.push(H('VII. The Verdict Depends on the Evaluator', H1));
kids.push(P(
  'We now compare the four retrieval systems on faithfulness, refusals excluded, paired on a '
  + 'common query subset, with a paired bootstrap of 10,000 resamples.',
  { after: 100 }));
kids.push(table(
  ['Dataset / generator', 'n', 'NLI-max spread', 'p (Holm)', 'AlignScore spread', 'p (Holm)'],
  [
    ['NQ / Claude', '784', '0.0107', '0.155 (0.391)', '0.0114', '0.0081 (0.0567)'],
    ['NQ / GPT-4o-mini', '628', '0.0130', '0.098 (0.391)', '0.0080', '0.117 (0.391)'],
    ['HotpotQA / Claude', '517', '0.0232', '0.159 (0.391)', '0.0164', '0.063 (0.380)'],
    ['HotpotQA / GPT-4o-mini', '510', '0.0267', '0.065 (0.380)', '0.0461', '0.0005 (0.0040)'],
  ],
  [2400, 800, 1600, 1400, 1700, 1400]));
kids.push(CAP(
  'Table VII. The same comparison under two evaluators, with the uncorrected p first and the '
  + 'Holm-adjusted p in parentheses, corrected over the eight comparisons in the table. One of '
  + 'four cells changes verdict at alpha = 0.05 and survives correction; NQ/Claude is '
  + 'significant uncorrected and not after correction. Queries, answers, '
  + 'refusal handling and test are identical throughout. With 10,000 resamples the smallest '
  + 'attainable non-zero p is 1e-4, so the two significant cells at p = 0.0005 are 5 extreme '
  + 'resamples of 10,000 and not a clamped floor.'));
kids.push(P(
  'The statistic is the spread: the difference in mean faithfulness between the best and the '
  + 'worst of the four systems, with the p-value from a paired bootstrap over that pair, 10,000 '
  + 'resamples, re-centred at zero [23]. The range is the quantity the decision this table '
  + 'informs is exposed to — a practitioner adopts one embedding model, and what they stand to '
  + 'lose is the distance to the best alternative, not the dispersion of the set. It is however '
  + 'a summary of two systems selected for being the extremes, so the p-value beside it is not '
  + 'adjusted for that within-cell selection, and the two intermediate systems do not appear in '
  + 'it at all. We therefore treat the range as descriptive, and report the direction across all '
  + 'four systems separately in Table VIII rather than reading it off the extremes. An omnibus '
  + 'test over the four systems would be the stricter statistic; we do not run one here.',
  { after: 120 }));
kids.push(P(
  'Under the common choice of evaluator, no cell shows a detectable difference. Under an '
  + 'evaluator measured in Section V to be more responsive to known errors, one does, and a '
  + 'second is significant before correction and not after. A result reported only under the '
  + 'first evaluator would be an artifact of instrument resolution rather than a property of '
  + 'the systems. The effect is narrower than the four-cell design can establish, and we report '
  + 'it as one cell rather than as a general property of the grid.',
  { after: 120 }));
kids.push(P(
  'That count of one is itself not invariant to the correction. NQ/Claude under AlignScore is '
  + 'at p = 0.0081 uncorrected, and Holm places it either side of alpha = 0.05 depending on '
  + 'which comparisons are treated as one family: 0.057 over the eight in the table, 0.024 over '
  + 'the four sharing an evaluator, 0.016 over the two in its own cell. Correcting within '
  + 'evaluator is defensible, since the four NLI tests establish which cells are null and so '
  + 'precede the question rather than competing with it, and it would make the count two. We '
  + 'report the eight-comparison family because it is the most conservative of the three and '
  + 'because it was fixed before the correction was applied; adopting a smaller family after '
  + 'observing that it restores a cell would be a selection we could not account for. We state '
  + 'the alternative rather than only the choice, because a count that moves with the '
  + 'multiplicity family is the same kind of dependence on an analysis decision that this paper '
  + 'reports for the choice of evaluator.',
  { after: 120 }));

kids.push(H('A. Direction is dataset-dependent', H2));
kids.push(P(
  'Where a difference exists, it does not point the same way. Spearman correlation between '
  + 'NDCG@5 and faithfulness across the four systems, computed in every generator-evaluator '
  + 'combination:',
  { after: 100 }));
kids.push(table(
  ['Dataset', 'Claude / NLI', 'Claude / Align', 'GPT / NLI', 'GPT / Align'],
  [
    ['Natural Questions', '-1.00', '-0.40', '-0.80', '0.00'],
    ['HotpotQA', '+0.40', '+0.80', '+0.80', '+1.00'],
  ],
  [2400, 1650, 1650, 1650, 1650]));
kids.push(CAP(
  'Table VIII. With four systems these coefficients cannot carry significance individually. The '
  + 'pattern is that all four HotpotQA cells are positive and all four Natural Questions cells '
  + 'are zero or negative, across both generators and both evaluators.'));
kids.push(P(
  'On the multi-hop dataset, better retrieval accompanies more faithful answers; on the '
  + 'single-hop dataset it does not. A plausible reading is that composing an answer from '
  + 'several retrieved pieces makes grounding sensitive to how much of the needed evidence is '
  + 'present, while a single-fact answer is not. We do not test this, and note that the '
  + 'experiment varies embedding model rather than manipulating retrieval quality directly, so '
  + 'the correlation should not be read as an effect of NDCG@5 itself.',
  { after: 160 }));

// ── VIII. Equivalence ──────────────────────────────────────────────
kids.push(H('B. A normalisation defect that produced a result', H2));
kids.push(P(
  'An earlier version of this analysis reported two cells rather than one. The difference was '
  + 'not a change of test, data or population but of text normalisation, and it is worth '
  + 'reporting because it is an instance of the paper\'s own subject.',
  { after: 100 }));
kids.push(P(
  'The entailment scorer originally read the answer exactly as the generator wrote it, while '
  + 'the claim-level scorer stripped markdown from each claim before scoring. The two aggregates '
  + 'were therefore not reading the same hypothesis. The discrepancy is invisible in aggregate '
  + 'and was found by the degenerate-case check in Section V-C: with a single claim and no '
  + 'markdown the two must be numerically identical, and they were, to 1.7e-05; with markdown '
  + 'they differed on every one of 131 cases, by up to 0.729 on a pair of asterisks alone. '
  + 'Markdown is not evenly distributed across the grid — it appears in 74.9% of Claude\'s '
  + 'answers and 0.3% of GPT-4o-mini\'s — so the formatting habits of one generator were '
  + 'entering a comparison between embedding systems.',
  { after: 120 }));
kids.push(P(
  'Stripping markdown for all three evaluators removes the asymmetry and restores the identity '
  + 'on every single-assertion answer. It also removes the significance of the NQ/Claude cell '
  + 'under AlignScore: its spread falls from 0.0179 to 0.0114 and its Holm-adjusted p from '
  + '0.0040 to 0.0567. That cell\'s apparent evaluator-dependence was substantially an artifact '
  + 'of formatting, and the count reported above is the corrected one. We record it rather than '
  + 'silently reporting the smaller number because the failure mode generalises: a preprocessing '
  + 'choice applied to one aggregate and not another is invisible in every summary statistic, '
  + 'survives every significance test, and is detectable only by an identity that the two '
  + 'aggregates must satisfy. Metrics that admit such an identity should be checked against it.',
  { after: 160 }));
kids.push(H('VIII. What an Equivalence Claim Requires', H1));
kids.push(P(
  'Two of the four cells in Table VII show no detectable difference under either evaluator. '
  + 'Absence of a detected difference is not evidence of equivalence, so we test equivalence '
  + 'directly with two one-sided tests [15]. A TOST result depends entirely on the margin, and a '
  + 'margin is a substantive judgement rather than a statistical fact — so we report a curve '
  + 'rather than assert a point.',
  { after: 100 }));
kids.push(table(
  ['Dataset / generator', 'Evaluator', '±0.01', '±0.02', '±0.03', '±0.05', '±0.10'],
  [
    ['NQ / Claude', 'NLI-max', '0/6', '4/6', '6/6', '6/6', '6/6'],
    ['', 'AlignScore', '2/6', '3/6', '6/6', '6/6', '6/6'],
    ['NQ / GPT-4o-mini', 'NLI-max', '0/6', '3/6', '6/6', '6/6', '6/6'],
    ['', 'AlignScore', '1/6', '6/6', '6/6', '6/6', '6/6'],
    ['HotpotQA / Claude', 'NLI-max', '0/6', '0/6', '0/6', '6/6', '6/6'],
    ['', 'AlignScore', '0/6', '3/6', '6/6', '6/6', '6/6'],
    ['HotpotQA / GPT', 'NLI-max', '0/6', '0/6', '1/6', '4/6', '6/6'],
    ['', 'AlignScore', '0/6', '0/6', '0/6', '4/6', '6/6'],
  ],
  [2200, 1400, 1080, 1080, 1080, 1080, 1080]));
kids.push(CAP(
  'Table IX. System pairs judged equivalent, of six tested, across margins. A claim of '
  + 'equivalence is a claim about a particular column.'));
kids.push(P(
  'Section V supplies an external anchor for choosing that column. Falsifying one grounded '
  + 'value moves NLI-max by 0.049 to 0.095 on Claude\'s answers. A margin of ±0.05 on that '
  + 'evaluator is therefore approximately the size of one fabricated fact, which is not a '
  + 'negligible difference. Stating equivalence at ±0.05 without that context asserts more than '
  + 'the instrument supports.',
  { after: 160 }));

// ── IX. Discussion ─────────────────────────────────────────────────
kids.push(H('IX. Discussion', H1));
kids.push(P('For work that reports RAG faithfulness, we suggest the following.', { after: 80 }));
kids.push(BULLET(
  'Name the evaluator in the claim. "Faithfulness did not differ" is incomplete; two of our '
  + 'four cells reverse between evaluators on identical data.'));
kids.push(BULLET(
  'Report the evaluation population explicitly, and whether membership depends on the system '
  + 'being compared. Pooling refusals reproduced the retrieval ranking exactly here (Table IV).'));
kids.push(BULLET(
  'Prefer claim-level aggregation to a maximum over the whole answer, and expect it to help '
  + 'most where answers carry several assertions.'));
kids.push(BULLET(
  'Calibrate the margin against a measured effect before claiming equivalence. A falsification '
  + 'experiment of the kind in Section V costs nothing beyond re-scoring.'));
kids.push(P(
  'For a document question answering system, the practical reading of Table II is that a '
  + 'grounding threshold on a verbose generator detects a minority of falsified values at a '
  + 'fixed gate: 5.1% under the common evaluator, 25.1% under the most responsive one tested. A '
  + 'deterministic check that every value in an answer appears in the retrieved context '
  + 'addresses this class of error directly and is complementary to an entailment score. Run '
  + 'over the same cases as Table II, it recalls every falsified value in all four cells at a '
  + 'false-positive rate of 0.7% to 5.2% on untouched answers, against 5.1% detection for '
  + 'NLI-max on the hardest of those cells. Two qualifications matter. Numbers that are not '
  + 'assertions about the world — the chunk indices the prompt supplies and the markers of an '
  + 'ordered list — are neither asserted of the world nor expected in the retrieved text, and '
  + 'counting them raises the false-positive rate on Claude\'s HotpotQA answers from 5.2% to '
  + 'about 39%. And the residual false positives are quantities the answer derives rather than '
  + 'copies: counts, sums, elapsed years, formats converted. A literal check flags all of those, '
  + 'so it is a complement to an entailment score and not a replacement for one.',
  { after: 120 }));
kids.push(P(
  'Checking symbolic content against the source is not a new proposal, and we do not claim it '
  + 'as one. Goodrich et al. [16] score generated text by extracting subject-relation-object '
  + 'triples and comparing them against the source document. Nan et al. [17] define entity-level '
  + 'consistency metrics for summarisation and evaluate them beside entailment-based '
  + 'alternatives. Zhao et al. [18] address quantity hallucination directly, verifying dates, '
  + 'numbers and monetary amounts in generated summaries against the source. That numbers are a '
  + 'weak point is also documented at the level of the representations these systems are built '
  + 'on: Wallace et al. [19] find number magnitude encoded unevenly across standard embeddings '
  + 'and least reliably by subword models. Entailment is separately a loose proxy for '
  + 'attribution — the AIS framework [20] distinguishes whether a statement is supported by an '
  + 'identified source from whether it is plausible or entailed in a general sense, which is the '
  + 'distinction a structurally similar but differently attributed claim exploits.',
  { after: 120 }));
kids.push(P(
  'What this paper adds is narrower than the check itself: the controlled minimal pair. A single '
  + 'numeric value is replaced in an otherwise byte-identical generation, and the same case is '
  + 'scored by all three evaluators, so the quantity measured is each evaluator\'s response to a '
  + 'known error on identical input rather than its agreement with a human label on some other '
  + 'corpus. The scope of that measurement is numeric values, two proprietary generators and two '
  + 'datasets.',
  { after: 160 }));

// ── X. Limitations ─────────────────────────────────────────────────
kids.push(H('X. Limitations', H1));
kids.push(BULLET(
  'Four embedding models, two English datasets and two proprietary generators. The generator '
  + 'comparison in particular rests on two systems whose answer-length distributions differ '
  + 'substantially; an open-weight generator with a third profile would test the mechanism in '
  + 'Section V-C rather than merely repeat it.'));
kids.push(BULLET(
  'Correctness is graded by a judge model rather than by string overlap. Exact match is 0.000 '
  + 'across all 16,000 answers because the generators do not emit bare answer spans. String '
  + 'containment, the obvious fallback, is not a bound in either direction: against the judge it '
  + 'understates accuracy by 5 to 15 points in twelve cells and overstates it by 3 to 7 points '
  + 'in the four HotpotQA cells under Claude, where long answers mention a reference string '
  + 'while asserting something else. The judge is itself a language model applied to its own '
  + 'outputs, and its agreement with a human annotator on this data has not been established.'));
kids.push(BULLET(
  'Retrieval hit is defined as the gold answer string appearing in the retrieved context. This '
  + 'is a strong definition, and it does not separate retrieval failure from failures of '
  + 'composition, entity resolution or inference once the evidence is present.'));
kids.push(BULLET(
  'The Natural Questions sample retains only queries whose annotated answer falls inside the '
  + 'retained context window. This makes the retrieval task more constrained than open-domain '
  + 'RAG, where evidence may be distributed, partial or implicit.'));
kids.push(BULLET(
  'Pairing on the queries every system answered removes exactly those queries where retrieval '
  + 'quality determined whether an answer was produced. The reported comparison is therefore '
  + 'conditional on that subset, which is narrower than an unconditional comparison.'));
kids.push(BULLET(
  'Effect sizes are small where detected — 0.018 and 0.046 — against retrieval spreads of 4.4 '
  + 'and 11.9 NDCG@5 points. These are detectable differences, not large ones.'));
kids.push(BULLET(
  'The three evaluators receive byte-identical answers but not identical premises. The '
  + 'entailment scorer is given each of the five retrieved chunks separately, plus their '
  + 'concatenation truncated to the model\'s window, and the maximum is taken; the claim-level '
  + 'scorer applies that construction per extracted claim; AlignScore is given the concatenated '
  + 'context as one premise and performs its own splitting. Each is the evaluator used as its '
  + 'authors intend, but it means part of the difference reported in Section V is premise '
  + 'granularity rather than evaluator sensitivity. Re-scoring the same cases with the same '
  + 'entailment model on the concatenated premise bounds that share between none and half of the '
  + 'gap depending on the cell, and at about a quarter of it on Claude\'s answers. The '
  + 'hypothesis, by contrast, is now normalised identically for all three (Section VII-B); it '
  + 'was not in an earlier version, and that difference alone decided one cell.'));
kids.push(BULLET(
  'Which answers count as refusals is fixed by a rule rather than given by the data, and that '
  + 'rule sets the evaluation population for every result in Sections V through VIII. It is '
  + 'therefore a researcher degree of freedom as well as a finding: substituting one defensible '
  + 'detector for another moves the rows nearest the boundary between refusing and attempting, '
  + 'and carries the uncorrected p-value across the conventional threshold in two of eight cells '
  + '(Section VI-A). Those two do not survive correction for multiple comparisons, but the '
  + 'population every reported number is computed on still depends on the choice. The rule used '
  + 'here strips the generators\' standard preamble before matching; without that step 107 '
  + 'refusals, all from one generator, would be graded as attempts.'));
kids.push(BULLET(
  'The falsification cases are the first 200 eligible answers in dataset order for each system, '
  + 'dataset and generator, rather than a random sample of the eligible answers: 3,184 cases '
  + 'scored against 5,272 eligible across the grid. Eligibility requires a grounded numeric '
  + 'value in the answer together with a replacement value absent from the context, so the '
  + 'scored set is a prefix of an already filtered population. Cells whose eligible count is '
  + 'near 200 are close to a census; the largest cells are sampled at about two fifths.'));
kids.push(BULLET(
  'The repository configures seven embedding models and this paper reports four: '
  + 'all-mpnet-base-v2 (contrastive), BGE-M3 (multilingual), E5-large-instruct '
  + '(instruction-tuned) and text-embedding-3-small (proprietary), chosen to span pretraining '
  + 'paradigms and to include one closed-source arm. GTE-large, Instructor-XL and '
  + 'jina-embeddings-v3 [24] are configured and runnable but were not run at this scale: each '
  + 'additional model is a further generation pass over both datasets and both generators, and '
  + 'the grid is a testbed for the evaluator comparison rather than a survey of embedding '
  + 'models.'));
kids.push(BULLET(
  'The correct-versus-incorrect faithfulness comparison rests on 44 to 84 attempted wrong '
  + 'answers per cell. Excluding refusals and grading correctness with a judge both reduce that '
  + 'population, and the resulting confidence intervals are 0.08 to 0.13 wide. The tests reported '
  + 'in Section VI-B bear on the general claim and do not resolve individual systems.'));

// ── XI. Conclusion ─────────────────────────────────────────────────
kids.push(H('XI. Conclusion', H1));
kids.push(P(
  'Holding a 16,000-answer generation grid fixed and varying only the faithfulness evaluator '
  + 'changes the conclusion about whether embedding choice affects faithfulness in one of four '
  + 'dataset-generator cells. The differences between evaluators are measurable directly: a '
  + 'falsified grounded value moves the common entailment aggregate by 0.054 on one generator '
  + 'and 0.451 on the other, and that gap is largely accounted for by how many assertions an '
  + 'answer contains rather than by the retrieval system. Reported without naming the evaluator, the evaluation '
  + 'population, the aggregation and the equivalence margin, a faithfulness result is not '
  + 'reproducible in the sense that matters — another group applying a different standard '
  + 'choice to the same generations would reach a different conclusion.',
  { after: 200 }));

// ── References ─────────────────────────────────────────────────────
kids.push(H('References', H1));
const refs = [
  '[1] A. Salemi and H. Zamani, "Evaluating Retrieval Quality in Retrieval-Augmented Generation," in Proc. SIGIR, 2024, pp. 2395-2400. arXiv:2404.13781.',
  '[2] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAS: Automated Evaluation of Retrieval Augmented Generation," in Proc. EACL (System Demonstrations), 2024, pp. 150-158.',
  '[3] Y. Zha, Y. Yang, R. Li, and Z. Yu, "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function," in Proc. ACL, 2023, pp. 11328-11348.',
  '[4] M. S. Tamber, S. Kazi, V. Sourabh, and J. Lin, "Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards," in Proc. EMNLP (Industry Track), 2025, pp. 799-811. arXiv:2505.04847.',
  '[5] O. Honovich, R. Aharoni, J. Herzig, H. Taitelbaum, D. Kukliansy, V. Cohen, T. Scialom, I. Szpektor, A. Hassidim, and Y. Matias, "TRUE: Re-evaluating Factual Consistency Evaluation," in Proc. NAACL-HLT, 2022, pp. 3905-3920. arXiv:2204.04991.',
  '[6] L. Tang, T. Goyal, A. R. Fabbri, P. Laban, J. Xu, S. Yavuz, W. Kryscinski, J. F. Rousseau, and G. Durrett, "Understanding Factual Errors in Summarization: Errors, Summarizers, Datasets, Error Detectors," in Proc. ACL, 2023, pp. 11626-11644.',
  '[7] C. Niu, Y. Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song, and T. Zhang, "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models," in Proc. ACL, 2024, pp. 10862-10878.',
  '[8] F. S. Bao, M. Li, R. Qu, G. Luo, E. Wan, Y. Tang, W. Fan, M. S. Tamber, S. Kazi, V. Sourabh, M. Qi, R. Tu, C. Xu, M. Gonzales, O. Mendelevitch, and A. Ahmad, "FaithBench: A Diverse Hallucination Benchmark for Summarization by Modern LLMs," in Proc. NAACL-HLT (Short Papers), 2025, pp. 448-461.',
  '[9] L. Tang, P. Laban, and G. Durrett, "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents," in Proc. EMNLP, 2024, pp. 8818-8847. arXiv:2404.10774.',
  '[10] Z. Xiao, S. Zhang, V. Lai, and Q. V. Liao, "Evaluating Evaluation Metrics: A Framework for Analyzing NLG Evaluation Metrics using Measurement Theory," in Proc. EMNLP, 2023, pp. 10967-10982.',
  '[11] Z. Sun, X. Zang, K. Zheng, J. Xu, X. Zhang, W. Yu, Y. Song, and H. Li, "ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability," in Proc. ICLR, 2025. arXiv:2410.11414.',
  '[12] D. Sinha, "The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems," arXiv:2512.15068.',
  '[13] S. Chen, Z. Zhao, and J. Chen, "Confident RAG: Enhancing the Performance of LLMs for Mathematics Question Answering through Multi-Embedding and Confidence Scoring," arXiv:2507.17442.',
  '[14] P. He, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing," in Proc. ICLR, 2023. arXiv:2111.09543.',
  '[15] D. J. Schuirmann, "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability," J. Pharmacokinet. Biopharm., vol. 15, no. 6, pp. 657-680, 1987.',
  '[16] B. Goodrich, V. Rao, P. J. Liu, and M. Saleh, "Assessing The Factual Accuracy of Generated Text," in Proc. ACM SIGKDD, 2019, pp. 166-175. arXiv:1905.13322.',
  '[17] F. Nan, R. Nallapati, Z. Wang, C. N. dos Santos, H. Zhu, D. Zhang, K. McKeown, and B. Xiang, "Entity-level Factual Consistency of Abstractive Text Summarization," in Proc. EACL, 2021, pp. 2727-2733. arXiv:2102.09130.',
  '[18] Z. Zhao, S. B. Cohen, and B. Webber, "Reducing Quantity Hallucinations in Abstractive Summarization," in Findings of EMNLP, 2020, pp. 2237-2249. arXiv:2009.13312.',
  '[19] E. Wallace, Y. Wang, S. Li, S. Singh, and M. Gardner, "Do NLP Models Know Numbers? Probing Numeracy in Embeddings," in Proc. EMNLP-IJCNLP, 2019, pp. 5307-5315. arXiv:1909.07940.',
  '[20] H. Rashkin, V. Nikolaev, M. Lamm, L. Aroyo, M. Collins, D. Das, S. Petrov, G. S. Tomar, I. Turc, and D. Reitter, "Measuring Attribution in Natural Language Generation Models," Computational Linguistics, vol. 49, no. 4, pp. 777-840, 2023. arXiv:2112.12870.',
  '[21] T. Kwiatkowski et al., "Natural Questions: A Benchmark for Question Answering Research," TACL, vol. 7, pp. 453-466, 2019.',
  '[22] Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. W. Cohen, R. Salakhutdinov, and C. D. Manning, "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering," in Proc. EMNLP, 2018, pp. 2369-2380.',
  '[23] R. Dror, G. Baumer, S. Shlomov, and R. Reichart, "The Hitchhiker\'s Guide to Testing Statistical Significance in Natural Language Processing," in Proc. ACL, 2018, pp. 1383-1392.',
  '[24] S. Sturua, I. Mohr, M. K. Akram, M. Gunther, B. Wang, M. Krimmel, F. Wang, G. Mastrapas, A. Koukounas, N. Wang, and H. Xiao, "jina-embeddings-v3: Multilingual Embeddings With Task LoRA," arXiv:2409.10173.',
];
refs.forEach((r) => kids.push(P(r, { after: 60, size: 18 })));

const doc = new Document({ sections: [{ properties: {}, children: kids }] });
Packer.toBuffer(doc).then((buf) => {
  const out = 'paper/RAG_Faithfulness_v6_evaluators.docx';
  fs.writeFileSync(out, buf);
  console.log(`wrote ${out} (${(buf.length / 1024).toFixed(1)} KB)`);
});
