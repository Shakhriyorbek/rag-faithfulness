// build_v6.js — regenerates RAG_Faithfulness_v6_evaluators.docx
//
// v6 rewrites v5 around the 2026-08-25/26 results. The v5 title
// ("Retrieval Quality Predicts Correctness, Not Faithfulness") is contradicted
// by its own data: the verdict flips with the evaluator in 2 of 4 cells, and
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
  + 'faithfulness changes in two of them depending on which evaluator is used. We trace this '
  + 'to measured properties of the evaluators rather than to the retrieval systems. Replacing '
  + 'a grounded value in an answer with one absent from the context — the failure a document '
  + 'question answering system most needs to detect — moves the entailment score by 0.038 on '
  + 'one generator and 0.456 on the other, and is caught by a fixed threshold 4% of the time '
  + 'against 58%. The gap is explained by answer structure: the score falls monotonically as '
  + 'the number of assertions in an answer grows, and the two generators average 2.99 and 1.18 '
  + 'assertions per answer. Claim-level aggregation and an alternative evaluator each recover '
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
  'Salemi and Zamani [1] report that query-document relevance labels correlate only weakly '
  + 'with downstream retrieval-augmented generation performance, and propose evaluating each '
  + 'retrieved document by the downstream result it produces. Our starting point is downstream '
  + 'of theirs: we take as given that retrieval metrics do not directly predict generation '
  + 'quality, and ask which downstream property a given measurement instrument can resolve at '
  + 'all.',
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
kids.push(H('C. Hallucination and grounding analysis', H2));
kids.push(P(
  'ReDeEP [5] localises hallucination to attention heads and feed-forward components. Sinha [6] '
  + 'analyses cases where a generated answer appears supported by the retrieved context without '
  + 'being entailed by it. Chen [7] examines how different retrievers suit different generators. '
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
  + 'therefore report absolute faithfulness throughout.',
  { after: 160 }));

// ── IV. Experimental Setup ─────────────────────────────────────────
kids.push(H('IV. Experimental Setup', H1));
kids.push(P(
  'Four embedding models index each corpus; retrieval takes the top five chunks of 256 tokens '
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
  'Three evaluators are compared. NLI-max is (1) with a DeBERTa-v3-large inference model, the '
  + 'common choice. Claim-min is (2) with the same inference model and sentence-level claim '
  + 'segmentation, chosen so that no second generator enters the measurement loop. AlignScore '
  + '[3] is an independently trained alignment model. All three score identical inputs.',
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
  'Two controls accompany it. An entity substitution replaces a grounded named entity instead '
  + 'of a value, giving a comparable edit. Scoring the untouched answer against another '
  + 'query\'s retrieved context gives the floor of each evaluator\'s scale on this data. 200 '
  + 'paired cases per cell, rebuilt deterministically so all three evaluators see identical '
  + 'inputs.',
  { after: 120 }));
kids.push(table(
  ['Dataset / generator', 'Evaluator', 'Original', 'Falsified', 'Random ctx', 'Detected'],
  [
    ['NQ / Claude Haiku 4.5', 'NLI-max', '0.918', '0.880', '0.522', '4%'],
    ['', 'Claim-min', '0.686', '0.535', '0.249', '23%'],
    ['', 'AlignScore', '0.818', '0.586', '0.342', '29%'],
    ['NQ / GPT-4o-mini', 'NLI-max', '0.893', '0.436', '0.507', '58%'],
    ['', 'Claim-min', '0.866', '0.391', '0.493', '63%'],
    ['', 'AlignScore', '0.918', '0.266', '0.173', '76%'],
    ['HotpotQA / Claude', 'NLI-max', '0.739', '0.660', '0.468', '13%'],
    ['', 'Claim-min', '0.367', '0.256', '0.149', '33%'],
    ['', 'AlignScore', '0.770', '0.466', '0.498', '45%'],
    ['HotpotQA / GPT-4o-mini', 'NLI-max', '0.578', '0.185', '0.222', '70%'],
    ['', 'Claim-min', '0.571', '0.182', '0.225', '70%'],
    ['', 'AlignScore', '0.773', '0.115', '0.223', '90%'],
  ],
  [2450, 1550, 1250, 1250, 1300, 1200]));
kids.push(CAP(
  'Table II. Response of three evaluators to a falsified value, on identical inputs. "Detected" '
  + 'is the share of answers that passed a 0.5 threshold before falsification and fail it '
  + 'after. Random-context scores anchor the floor of each scale.'));
kids.push(P(
  'On Claude\'s Natural Questions answers, NLI-max leaves the falsified answer at 0.880 against '
  + 'a floor of 0.522: the altered answer remains comfortably above any usable threshold. On '
  + 'GPT-4o-mini\'s answers the same edit lands at 0.436, essentially at that generator\'s floor '
  + 'of 0.507. The evaluator, the context and the perturbation code are identical; only the '
  + 'generator that wrote the answer differs.',
  { after: 120 }));

kids.push(H('B. The effect is not verbatim copying', H2));
kids.push(P(
  'A natural explanation is that one generator quotes the retrieved text back, so the evaluator '
  + 'scores a copy of its own premise. This does not hold. Measuring the fraction of answer '
  + 'word 5-grams appearing verbatim in the context gives 0.234 for both generators, and the '
  + 'correlation between overlap and the response to falsification is +0.041 and +0.161 — '
  + 'positive, where the explanation predicts negative.',
  { after: 160 }));

kids.push(H('C. Answers with more assertions are scored less sensitively', H2));
kids.push(P(
  'Segmenting each answer into sentence-level assertions and grouping by count gives a '
  + 'monotone relationship, present for both generators.',
  { after: 100 }));
kids.push(table(
  ['Assertions per answer', 'Claude NLI-max', 'Claude Claim-min', 'GPT NLI-max', 'GPT Claim-min'],
  [
    ['1', '+0.086', '+0.098', '+0.492', '+0.492'],
    ['2', '+0.056', '+0.131', '+0.336', '+0.452'],
    ['3', '+0.018', '+0.188', '+0.137', '+0.277'],
    ['5 or more', '+0.007', '+0.149', '—', '—'],
  ],
  [2400, 1650, 1650, 1650, 1650]));
kids.push(CAP(
  'Table III. Drop in score when one grounded value is falsified, by number of assertions in '
  + 'the answer. Under NLI-max the response decays toward zero as answers accumulate '
  + 'assertions; under claim-level aggregation it does not.'));
kids.push(P(
  'The two generators average 2.99 and 1.18 assertions per answer, and median answer lengths of '
  + '49 and 18 words. This accounts for the cross-generator difference in Table II: a falsified '
  + 'value is a progressively smaller edit to the hypothesis as correct surrounding text '
  + 'accumulates, and the entailment judgement is dominated by the remainder. As a check on the '
  + 'claim-level implementation, single-assertion answers score identically under both '
  + 'aggregates (0.4922 in each case), which is the expected degenerate case.',
  { after: 120 }));
kids.push(P(
  'Claim-level aggregation is a partial remedy rather than a repair. Expressed as a fraction of '
  + 'the distance to the random-context floor, the median response on Claude\'s answers moves '
  + 'from 0.01 to 0.03, and the minimum moves in the correct direction in 55% of cases: when '
  + 'the falsified assertion is not already the weakest, the minimum does not move at all. A '
  + 'second dilution remains within assertions — at matched assertion count the two generators '
  + 'still differ (+0.098 against +0.492), and their assertions average 20.5 against 13.8 words.',
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
  'The general statement is that any evaluation population whose membership depends on the '
  + 'system under comparison can transfer an effect from the selection into the measurement. '
  + 'Here the weakest retriever refuses most often, and every refusal lowers its average. All '
  + 'faithfulness results in this paper exclude refusals and are paired on the queries every '
  + 'system answered. Section X notes what that pairing costs.',
  { after: 160 }));

// ── VII. Consequences for the comparison ───────────────────────────
kids.push(H('VII. The Verdict Depends on the Evaluator', H1));
kids.push(P(
  'We now compare the four retrieval systems on faithfulness, refusals excluded, paired on a '
  + 'common query subset, with a paired bootstrap of 10,000 resamples.',
  { after: 100 }));
kids.push(table(
  ['Dataset / generator', 'n', 'NLI-max spread', 'p', 'AlignScore spread', 'p'],
  [
    ['NQ / Claude', '792', '0.0090', '0.223', '0.0179', '0.0005'],
    ['NQ / GPT-4o-mini', '628', '0.0129', '0.099', '0.0080', '0.118'],
    ['HotpotQA / Claude', '538', '0.0190', '0.270', '0.0126', '0.151'],
    ['HotpotQA / GPT-4o-mini', '510', '0.0267', '0.065', '0.0461', '0.0005'],
  ],
  [2700, 900, 1800, 1000, 1800, 800]));
kids.push(CAP(
  'Table V. The same comparison under two evaluators. Two of four cells change verdict at '
  + 'alpha = 0.05. Queries, answers, refusal handling and test are identical throughout.'));
kids.push(P(
  'Under the common choice of evaluator, no cell shows a detectable difference. Under an '
  + 'evaluator measured in Section V to be more responsive to known errors, two do. A result '
  + 'reported only under the first would be an artifact of instrument resolution rather than a '
  + 'property of the systems.',
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
  'Table VI. With four systems these coefficients cannot carry significance individually. The '
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
kids.push(H('VIII. What an Equivalence Claim Requires', H1));
kids.push(P(
  'Two of the four cells in Table V show no detectable difference under either evaluator. '
  + 'Absence of a detected difference is not evidence of equivalence, so we test equivalence '
  + 'directly with two one-sided tests [8]. A TOST result depends entirely on the margin, and a '
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
  'Table VII. System pairs judged equivalent, of six tested, across margins. A claim of '
  + 'equivalence is a claim about a particular column.'));
kids.push(P(
  'Section V supplies an external anchor for choosing that column. Falsifying one grounded '
  + 'value moves NLI-max by 0.033 to 0.094 on Claude\'s answers. A margin of ±0.05 on that '
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
  + 'grounding threshold on a verbose generator detects a minority of falsified values: 4% under '
  + 'the common evaluator, 29% under the most responsive one tested. A deterministic check that '
  + 'every value in an answer appears in the retrieved context addresses this class of error '
  + 'directly and is complementary to an entailment score. We have not evaluated such a check '
  + 'here.',
  { after: 160 }));

// ── X. Limitations ─────────────────────────────────────────────────
kids.push(H('X. Limitations', H1));
kids.push(BULLET(
  'Four embedding models, two English datasets and two proprietary generators. The generator '
  + 'comparison in particular rests on two systems whose answer-length distributions differ '
  + 'substantially; an open-weight generator with a third profile would test the mechanism in '
  + 'Section V-C rather than merely repeat it.'));
kids.push(BULLET(
  'Correctness figures are provisional. Exact match is 0.000 across all 4,000 Claude Natural '
  + 'Questions answers because the generators do not emit bare answer spans, and token F1 is '
  + '0.121 for the same reason, so correctness currently rests on string containment, an upper '
  + 'bound. A graded judgement is implemented but not yet run. Table VIII is reported for '
  + 'completeness and should not be cited as a measurement.'));
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
kids.push(P('', { after: 60 }));
kids.push(table(
  ['Contingency (Claude, provisional)', 'NQ', 'HotpotQA'],
  [
    ['hit and correct', '71.8%', '67.2%'],
    ['hit and incorrect', '23.3%', '30.2%'],
    ['miss and correct', '0.5%', '0.2%'],
    ['miss and incorrect', '4.4%', '2.4%'],
  ],
  [4200, 2400, 2400]));
kids.push(CAP(
  'Table VIII. Retrieval hit against correctness. Computed with string containment, an upper '
  + 'bound on correctness, and therefore provisional — see Limitations.'));

// ── XI. Conclusion ─────────────────────────────────────────────────
kids.push(H('XI. Conclusion', H1));
kids.push(P(
  'Holding a 16,000-answer generation grid fixed and varying only the faithfulness evaluator '
  + 'changes the conclusion about whether embedding choice affects faithfulness in two of four '
  + 'dataset-generator cells. The differences between evaluators are measurable directly: a '
  + 'falsified grounded value moves the common entailment aggregate by 0.038 on one generator '
  + 'and 0.456 on the other, and the gap is explained by how many assertions an answer contains '
  + 'rather than by the retrieval system. Reported without naming the evaluator, the evaluation '
  + 'population, the aggregation and the equivalence margin, a faithfulness result is not '
  + 'reproducible in the sense that matters — another group applying a different standard '
  + 'choice to the same generations would reach a different conclusion.',
  { after: 200 }));

// ── References ─────────────────────────────────────────────────────
kids.push(H('References', H1));
const refs = [
  '[1] A. Salemi and H. Zamani, "Evaluating Retrieval Quality in Retrieval-Augmented Generation," in Proc. SIGIR, 2024. arXiv:2404.13781.',
  '[2] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAS: Automated Evaluation of Retrieval Augmented Generation," in Proc. EACL (System Demonstrations), 2024.',
  '[3] Y. Zha, Y. Yang, R. Li, and Z. Yu, "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function," in Proc. ACL, 2023.',
  '[4] M. S. Tamber et al., "FaithJudge," in Proc. EMNLP Industry Track, 2025.',
  '[5] Z. Sun et al., "ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation," in Proc. ICLR, 2025.',
  '[6] D. Sinha, "The Semantic Illusion in Retrieval-Augmented Generation," arXiv:2512.15068.',
  '[7] S. Chen, "Each to Their Own: Matching Retrievers to Generators," arXiv:2507.17442.',
  '[8] D. J. Schuirmann, "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability," J. Pharmacokinet. Biopharm., vol. 15, no. 6, pp. 657-680, 1987.',
  '[9] T. Kwiatkowski et al., "Natural Questions: A Benchmark for Question Answering Research," TACL, vol. 7, pp. 453-466, 2019.',
  '[10] Z. Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering," in Proc. EMNLP, 2018.',
  '[11] P. He, X. Liu, J. Gao, and W. Chen, "DeBERTa: Decoding-enhanced BERT with Disentangled Attention," in Proc. ICLR, 2021.',
  '[12] R. Dror, G. Baumer, S. Shlomov, and R. Reichart, "The Hitchhiker\'s Guide to Testing Statistical Significance in Natural Language Processing," in Proc. ACL, 2018.',
  '[13] S. Sturua et al., "jina-embeddings-v3: Multilingual Embeddings With Task LoRA," arXiv:2409.10173.',
];
refs.forEach((r) => kids.push(P(r, { after: 60, size: 18 })));

const doc = new Document({ sections: [{ properties: {}, children: kids }] });
Packer.toBuffer(doc).then((buf) => {
  const out = 'paper/RAG_Faithfulness_v6_evaluators.docx';
  fs.writeFileSync(out, buf);
  console.log(`wrote ${out} (${(buf.length / 1024).toFixed(1)} KB)`);
});
