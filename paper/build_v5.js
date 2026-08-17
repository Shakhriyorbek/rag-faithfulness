const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
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

// Paragraph with mixed bold/plain runs: pass ["plain ", ["bold", true], ...]
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

const kids = [];

// ── Title block ────────────────────────────────────────────────────
kids.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 120 },
  children: [new TextRun({
    text: 'Retrieval Quality Predicts Correctness, Not Faithfulness, in Retrieval-Augmented Generation',
    bold: true, size: 32, font: 'Times New Roman' })],
}));
kids.push(P('Shakhriyorbek Boltabaev', { align: AlignmentType.CENTER, after: 0, size: 22 }));
kids.push(P('University of Szeged, Department of Computer Science', { align: AlignmentType.CENTER, after: 0, size: 20 }));
kids.push(P('Email: shakhriyorbekboltabaev@gmail.com', { align: AlignmentType.CENTER, after: 60, size: 20 }));
kids.push(P('Draft — 2026-08-17. All numbers are measured; see §5 for what has and has not been run.',
  { align: AlignmentType.CENTER, italics: true, size: 18, after: 240 }));

// ── Abstract ───────────────────────────────────────────────────────
kids.push(H('Abstract', HeadingLevel.HEADING_1));
kids.push(P(
  'Retrieval-Augmented Generation (RAG) is a standard approach for grounding large language model outputs in external knowledge [1]. Systems are widely assumed to produce more faithful outputs when retrieval quality improves. We test that assumption directly by holding the generator and prompt fixed and varying only the retrieval embedding model. Across four embedding models spanning 11.9 NDCG@5 points, two QA benchmarks (Natural Questions and HotpotQA), two generators (Claude Haiku 4.5 and GPT-4o-mini), and 16,000 generated answers, we find that retrieval quality strongly predicts whether the model answers correctly and whether it abstains, but does not predict how faithful its assertions are to the retrieved context. Using paired two one-sided tests (TOST), faithfulness is statistically equivalent across 22 of 24 embedding-model pairs at a margin of 0.05 entailment probability, while retrieval quality is equivalent for 0 of 6 pairs at a margin of 0.02 NDCG@5. Moving from the weakest to the strongest retriever on HotpotQA is worth 13.5 points of answer accuracy and 17 points of abstention rate, and 0.019 of faithfulness, which is not statistically distinguishable from zero. We further report that 23.3% (NQ) and 30.2% (HotpotQA) of queries whose answer was present in the retrieved context are nevertheless answered incorrectly, while the converse case — a correct answer without a retrieval hit — occurs in under 0.5%. Retrieval quality is therefore close to necessary and clearly not sufficient. We document one measurement pitfall in detail: pooling abstentions into mean faithfulness produces an apparent effect of embedding architecture that reverses under a per-response analysis.',
  { after: 160 }));
kids.push(PR([['Keywords: ', true],
  'Retrieval-Augmented Generation, Faithfulness, Hallucination, Embedding Models, Semantic Entailment, Equivalence Testing'],
  { after: 240 }));

// ── 1. Introduction ────────────────────────────────────────────────
kids.push(H('1. Introduction', HeadingLevel.HEADING_1));
kids.push(P('Retrieval-Augmented Generation (RAG) grounds large language model (LLM) outputs in external knowledge sources [1]. By retrieving semantically relevant documents at inference time, RAG systems are expected to reduce hallucination and improve factual accuracy [10]. The prevailing assumption in both research and industry is that better retrieval leads to more faithful generation.'));
kids.push(P('This assumption bundles together two properties that can be measured separately. Whether an answer is correct is a claim about the world; whether it is faithful is a claim about the relationship between the answer and the context the model was given [9]. A RAG system can be right for the wrong reasons, or wrong while remaining entirely grounded in what it was shown. If retrieval quality governs one of these and not the other, then improving retrieval metrics buys something narrower than the literature usually implies.'));
kids.push(P('We test this by holding the generator, the prompt, the chunking, and the retrieval depth fixed, and varying only the embedding model used for retrieval. This isolates the upstream stage. Our headline finding is negative with respect to our original hypothesis and, we argue, more useful: across a retrieval-quality range of 11.9 NDCG@5 points, faithfulness does not move. Correctness and abstention move substantially.'));
kids.push(P('Because a null result is only as informative as its statistical power, we do not rest the claim on a failure to reject. We use paired two one-sided tests (TOST) to establish equivalence in the positive sense: we show that the faithfulness difference between embedding models is bounded within a small margin, on the same queries, with the same generator.'));
kids.push(P('We make three contributions. First, we separate which downstream property of a RAG system retrieval quality governs (answer correctness and abstention rate) from which it does not (faithfulness of the answer to the context), with equivalence testing in both directions. Second, we quantify the necessity and sufficiency of retrieval quality per query, finding a pronounced asymmetry: retrieval is nearly necessary and clearly not sufficient. Third, we document a measurement pitfall that inverts the headline conclusion — pooling abstentions into mean faithfulness manufactures an apparent architecture effect that disappears under per-response analysis.'));

// ── 2. Background ──────────────────────────────────────────────────
kids.push(H('2. Background and Related Work', HeadingLevel.HEADING_1));
kids.push(H('2.1 Retrieval-Augmented Generation', HeadingLevel.HEADING_2));
kids.push(P('RAG was introduced by Lewis et al. [1] as a method for combining parametric knowledge in LLMs with non-parametric document retrieval. A standard pipeline consists of a query encoder, a retrieval module that finds the k most similar document chunks by cosine similarity using dense vector search [14], and a generator LLM that produces an answer conditioned on the retrieved context. A survey of RAG architectures is provided by Gao et al. [10].'));

kids.push(H('2.2 Hallucination and Faithfulness', HeadingLevel.HEADING_2));
kids.push(P('Hallucination refers broadly to generated content that is factually incorrect or unsupported by the input context [15]. In RAG systems, faithfulness refers to whether the generated answer is semantically entailed by the retrieved context, irrespective of whether that context is itself factually correct. This distinction between faithfulness and factuality is central to our study and is formalized by Tamber et al. [9].'));
kids.push(P('Sinha [5] demonstrates that embedding similarity fails as a post-hoc faithfulness detector, showing that faithful and hallucinated responses become difficult to separate in embedding space. Sun et al. [6] apply mechanistic interpretability to study how LLMs balance retrieved context against parametric memory. Tan et al. [13] propose a contrastive likelihood reward for reinforcement learning that trains generators to favour retrieved evidence. All three address faithfulness at the generation stage; we ask whether it can be shaped upstream, at the retrieval embedding stage.'));

kids.push(H('2.3 Embedding Models for Retrieval', HeadingLevel.HEADING_2));
kids.push(P('Dense retrieval using pre-trained embedding models is the dominant retrieval paradigm in RAG [14]. Model families differ in training objective. Contrastive models such as Sentence-BERT [4] and GTE [11] maximize similarity between semantically related pairs. Instruction-tuned models such as Instructor-XL [12] and E5-instruct [2] condition the embedding on a task instruction. Multilingual models such as BGE-M3 [3] are trained across languages, and jina-embeddings-v3 [8] uses task-specific LoRA adapters. The closest prior work is Chen et al. [7], which combines model strengths through Mixture-Embedding RAG and Confident RAG but measures general response quality rather than faithfulness.'));

kids.push(H('2.4 Evaluating Retrieval for Downstream Quality', HeadingLevel.HEADING_2));
kids.push(P('Salemi and Zamani [22] address the question closest to ours. They show that standard query–document relevance labels correlate only weakly with downstream RAG performance, and propose eRAG, which scores each retrieved document by running the LLM on that document alone and using the downstream result as the document’s relevance label. Their finding — that relevance judgements are a poor proxy for downstream quality — is the premise our work starts from rather than a result we claim.'));
kids.push(P('Our contribution is to separate the downstream properties that this observation lumps together. Salemi and Zamani measure downstream performance as task quality; we ask which component of task quality is affected. We find that retrieval quality does predict correctness and abstention — so relevance labels are not uninformative — while leaving faithfulness statistically equivalent. We additionally test equivalence rather than reporting the absence of a significant difference.'));

// ── 3. Problem formulation ─────────────────────────────────────────
kids.push(H('3. Problem Formulation', HeadingLevel.HEADING_1));
kids.push(H('3.1 Definitions', HeadingLevel.HEADING_2));
kids.push(P('Let q denote a query, D a document corpus, and E an embedding model. A RAG system retrieves the top-k chunks R = {d₁, ..., dₖ} by cosine similarity in the space defined by E, then generates a = LLM(q, R). We define faithfulness F(a, R) as the probability that the generated answer a is entailed by the retrieved context R, where a is the model’s own output and not the gold-standard answer:'));
kids.push(EQ('F(a, R) = Pɴʟɪ(entailment | R, a) ∈ [0, 1]          (1)'));
kids.push(P('This choice is deliberate. Measuring entailment between retrieved context and the gold-standard answer would conflate faithfulness with retrieval quality, since it would test whether the correct answer appears in the retrieved documents rather than whether the generator grounded its output in what it was given. An answer can be faithful yet incorrect, or correct yet unfaithful. Keeping the two separable is what allows the analysis in §6.5.'));
kids.push(P('We define correctness C(a, a*) as whether the generated answer conveys the gold answer a*. §4.4 discusses why this is harder to measure than it appears for verbose generators.'));

kids.push(H('3.2 The Retrieval-Faithfulness Gap and its limits', HeadingLevel.HEADING_2));
kids.push(P('Our earlier formulation of this study introduced the Retrieval-Faithfulness Gap, defined for an embedding model E as'));
kids.push(EQ('RFG(E) = NDCG@5(E) − F(E)          (2)'));
kids.push(P('with a normalized variant intended to distinguish a system where both terms are high from one where both are low:'));
kids.push(EQ('nRFG(E) = (NDCG@5(E) − F(E)) / NDCG@5(E)          (3)'));
kids.push(P('We report both, but as diagnostics rather than as the primary quantity, for three reasons that emerged from the measurements and which we state here so the results in §6.6 can be read against the design intent.'));
kids.push(BULLET('The difference is between quantities on different scales: a ranking metric and an entailment probability. Their subtraction has no natural unit.'));
kids.push(BULLET('Empirically the gap is negative for every model we measured on Natural Questions (§6.6): faithfulness exceeds NDCG@5. The metric was designed on the assumption that generation loses fidelity relative to retrieval quality, which the data does not support.'));
kids.push(BULLET('Because nRFG = 1 − F/NDCG@5 and NDCG@5 does not depend on the generator, any comparison of nRFG rankings across generators is dominated by the shared retrieval term whenever F varies little. In our data the nRFG ranking reproduced the NDCG@5 ranking exactly for both generators (§6.7).'));

kids.push(H('3.3 Necessity and sufficiency', HeadingLevel.HEADING_2));
kids.push(P('We adopt as the organizing question whether good retrieval quality is necessary and sufficient for a good response. This decomposes per query into a two-by-two contingency between whether retrieval succeeded and whether the answer was correct, shown in Table 1. We define a retrieval hit as the retrieved context containing the gold answer span, not merely the gold document; §4.2 describes how relevance is made answer-bearing.'));
kids.push(table(
  ['', 'answer correct', 'answer incorrect'],
  [['retrieval hit', 'as the premise expects', 'retrieval NOT SUFFICIENT'],
   ['retrieval miss', 'retrieval NOT NECESSARY', 'as the premise expects']],
  [2400, 3300, 3300]));
kids.push(CAP('Table 1: The necessity/sufficiency contingency. Each cell also carries a mean faithfulness, computed over answered responses only (§4.3).'));

kids.push(H('3.4 Research questions', HeadingLevel.HEADING_2));
kids.push(BULLET('RQ1: Does the choice of embedding model affect downstream faithfulness, and can any such effect be bounded rather than merely tested for significance?'));
kids.push(BULLET('RQ2: Which downstream properties does retrieval quality govern — correctness, abstention, faithfulness?'));
kids.push(BULLET('RQ3: Is good retrieval necessary, sufficient, both, or neither, measured per query?'));
kids.push(BULLET('RQ4: Do the answers hold across generators of different capability?'));

// ── 4. Methodology ─────────────────────────────────────────────────
kids.push(H('4. Methodology', HeadingLevel.HEADING_1));
kids.push(H('4.1 Embedding models', HeadingLevel.HEADING_2));
kids.push(P('We evaluate four embedding models spanning three training paradigms, listed in Table 2. They were selected to maximize architectural variation while remaining within a compute and API budget; three further models (GTE-large, Instructor-XL, jina-embeddings-v3) are integrated in the pipeline but not yet run, and no claim in this paper depends on them.'));
kids.push(table(
  ['Model', 'Training objective', 'Dim.'],
  [['all-mpnet-base-v2 [4]', 'Contrastive (symmetric)', '768'],
   ['BGE-M3 [3]', 'Multilingual contrastive', '1024'],
   ['multilingual-e5-large-instruct [2]', 'Instruction-tuned contrastive', '1024'],
   ['text-embedding-3-small', 'Contrastive (proprietary)', '1536']],
  [4200, 3400, 1400]));
kids.push(CAP('Table 2: Embedding models evaluated.'));

kids.push(H('4.2 Datasets and relevance', HeadingLevel.HEADING_2));
kids.push(P('We evaluate on Natural Questions [17], single-hop factoid QA over Wikipedia, and HotpotQA [18], multi-hop QA requiring synthesis across documents. We sample 1,000 queries from each. QASPER [19] is integrated but not yet run.'));
kids.push(PR([['Relevance is answer-bearing. ', true],
  'A chunk of a gold document counts as relevant only if it actually carries the gold span, so that a retrieval hit means the generator was shown the answer rather than merely the right document. This distinction matters directly for Table 1: under document-level relevance, a query whose answer lies outside the retained context window is scored as a retrieval hit while the generator correctly reports that it cannot answer, manufacturing spurious mass in the hit-by-incorrect cell. On HotpotQA, 2 of 1,000 queries required a fallback to document-level relevance because a gold sentence straddled every chunk boundary.']));
kids.push(PR([['Sampling. ', true],
  'For Natural Questions we draw until we obtain 1,000 queries whose annotated short answer appears in the retained 500-token context window, which is centred on the annotated answer span rather than taken from the head of the document. Of 1,857 candidates examined, 845 had no short answer (the usual rate for this benchmark) and 12 had their answer outside the window and were discarded. For HotpotQA, 95.6% of gold answers appear verbatim in the gold context; the remainder are yes/no and comparison answers, which are retained.']));

kids.push(H('4.3 Faithfulness measurement', HeadingLevel.HEADING_2));
kids.push(P('Faithfulness is measured with a DeBERTa-v3-large cross-encoder NLI model [20], scoring each retrieved chunk separately as premise against the generated answer as hypothesis and aggregating by maximum. Per-chunk scoring avoids truncating the premise: concatenating five 256-token chunks exceeds the 512-token encoder limit, so a concatenated premise silently discards the later chunks. The entailment class index is resolved from the model configuration at load time rather than assumed. AlignScore [16] is integrated as a second signal but not yet run; every faithfulness number in this paper is NLI-based.'));
kids.push(PR([['Abstentions are excluded from faithfulness means. ', true],
  'When a generator responds that it cannot answer from the provided context, that response is correctly not entailed by the context and receives a low entailment score (0.25–0.32 in our data). Abstention rate is itself driven by retrieval quality, so pooling abstentions into a mean gives the weaker retriever a lower faithfulness score as an arithmetic consequence of abstaining more often. §6.3 quantifies the size of this artifact, which is large enough to reverse the paper’s conclusion. All faithfulness comparisons are additionally paired on the subset of queries that every model answered, because which queries a model abstains on depends on its own retrieval and the resulting per-model subsets are not exchangeable.']));

kids.push(H('4.4 Correctness measurement', HeadingLevel.HEADING_2));
kids.push(P('Correctness is harder to measure than it appears here, and we state the difficulty rather than resolving it. Exact match assumes the prediction is roughly the length of the gold span. Our generators produce full sentences: exact match is 0.000 across all 4,000 Claude responses on Natural Questions, because the model never emits a bare answer span. Token-level F1 is similarly depressed by verbosity, at 0.121.'));
kids.push(P('We therefore report containment — whether a normalized gold answer appears within the generated answer — which is an upper bound, since a long answer may mention the gold string while asserting something else. With exact match at zero, the interval between the lower and upper bound is uninformative, and the correctness figures in §6.4 and §6.5 should be read as provisional. An LLM-judge evaluation over the existing generations is the natural remedy and is not yet run. Faithfulness results are unaffected, as they do not use the correctness label.'));
kids.push(P('Normalization replaces punctuation with spaces rather than deleting it. Deletion glues adjacent tokens together and both invents and destroys matches: the benchmark text stores documents as token lists rejoined with spaces, so a gold answer of the form "Röntgen’s" must match a context reading "Röntgen ’s".'));

kids.push(H('4.5 Generators and prompt parity', HeadingLevel.HEADING_2));
kids.push(P('We evaluate two closed-source generators of different capability tiers: Claude Haiku 4.5 and GPT-4o-mini. Llama-3-8B-Instruct is integrated but not yet run. Both generators receive the byte-identical prompt: a single user turn carrying the full template, with no system message and no few-shot examples, at temperature 0 and a 256-token limit. Prompt parity is enforced by test rather than by convention, because RQ4 compares rankings across generators and any prompt difference would confound that comparison. The instruction is deliberately not hoisted into the system parameter of either API, idiomatic though that would be.'));

kids.push(H('4.6 Equivalence testing', HeadingLevel.HEADING_2));
kids.push(P('A non-significant difference is not evidence of equivalence. Since our central claim is that faithfulness does not differ across embedding models, we test it in the form that carries the burden of proof correctly, using two one-sided tests (TOST). The null hypothesis is that the two models differ by at least a margin δ; rejecting it at α licenses the claim of equivalence within δ. Two one-sided tests are conducted against +δ and −δ and the larger p-value decides.'));
kids.push(P('All tests are paired: every embedding model is evaluated on identical queries, and pairing removes per-query difficulty, which is the dominant source of variance. We apply the same procedure to retrieval quality, where it serves the opposite purpose — testing whether the models can be described as matched. We report margins explicitly, since the choice of margin is a judgement and not a property of the data.'));

// ── 5. Experimental setup ──────────────────────────────────────────
kids.push(H('5. Experimental Setup', HeadingLevel.HEADING_1));
kids.push(P('Documents are chunked at 256 tokens with 32-token overlap, preserving original text via offset-based chunking. Retrieval is top-5 by cosine similarity over an exact index; a pool of 20 is retained per query for the re-ranking ablation. Temperature is 0. The random seed is fixed at 42. Statistical significance of differences is assessed by paired bootstrap resampling with 10,000 iterations [21]; equivalence is assessed by paired TOST (§4.6).'));
kids.push(P('The experiments reported here comprise 4 embedding models × 2 datasets × 2 generators × 1,000 queries = 16,000 generated answers, produced at a total API cost of $13.46 with no API errors. Experiments integrated but not yet run, and on which no claim here depends: the zero-retrieval and oracle conditions, the context-arrangement ablation, per-document utility (eRAG, leave-one-out, Shapley), the entailment-similarity geometric analysis, faithfulness-aware re-ranking, AlignScore, Llama-3-8B, and QASPER.'));

// ── 6. Results ─────────────────────────────────────────────────────
kids.push(H('6. Results', HeadingLevel.HEADING_1));

kids.push(H('6.1 The models are not matched on retrieval quality', HeadingLevel.HEADING_2));
kids.push(P('Our earlier framing assumed these embedding models achieve near-identical retrieval quality, so that downstream differences could be attributed to architecture at fixed retrieval performance. Table 3 shows this does not hold.'));
kids.push(table(
  ['Model', 'NQ NDCG@5', 'NQ Recall@5', 'HotpotQA NDCG@5', 'HotpotQA Recall@5'],
  [['all-mpnet-base-v2', '0.7862', '0.9222', '0.7051', '0.7268'],
   ['BGE-M3', '0.7944', '0.8912', '0.8093', '0.8333'],
   ['multilingual-e5-large-instruct', '0.8064', '0.9278', '0.8237', '0.8587'],
   ['text-embedding-3-small', '0.8300', '0.9486', '0.7666', '0.7953']],
  [3000, 1500, 1500, 1500, 1500]));
kids.push(CAP('Table 3: Retrieval quality, n = 1,000 queries per dataset. The spread is 4.4 NDCG@5 points on NQ and 11.9 on HotpotQA.'));
kids.push(P('Paired TOST at a margin of 0.02 NDCG@5 finds 0 of 6 model pairs equivalent on either dataset. On HotpotQA five of six pairs differ at p < 0.0001. The description "matched retrieval quality" is therefore not available to us at any defensible margin, and we do not use it.'));
kids.push(P('We note in passing that the ranking is dataset-dependent: text-embedding-3-small is the strongest retriever on Natural Questions and the third strongest on HotpotQA. Retrieval quality rankings measured on one benchmark do not transfer.'));

kids.push(H('6.2 Faithfulness is equivalent across embedding models', HeadingLevel.HEADING_2));
kids.push(P('Table 4 reports mean faithfulness per embedding model, computed over answered responses only and paired on the queries every model answered.'));
kids.push(table(
  ['Dataset / generator', 'mpnet', 'BGE-M3', 'e5-instruct', 'text-emb-3', 'n', 'spread', 'p'],
  [['NQ / Claude', '0.9248', '0.9235', '0.9179', '0.9159', '792', '0.009', '0.223'],
   ['NQ / GPT-4o-mini', '0.8949', '0.8858', '0.8820', '0.8824', '628', '0.013', '0.099'],
   ['HotpotQA / Claude', '0.7368', '0.7310', '0.7461', '0.7271', '538', '0.019', '0.270'],
   ['HotpotQA / GPT-4o-mini', '0.6647', '0.6709', '0.6911', '0.6644', '510', '0.027', '0.065']],
  [2100, 1000, 1000, 1100, 1150, 700, 950, 1000]));
kids.push(CAP('Table 4: Faithfulness (NLI entailment, per-chunk maximum), answered responses only, paired on a common subset. p is from a paired bootstrap between the highest- and lowest-scoring model in each row.'));
kids.push(P('No cell shows a significant difference. Since this is a null result, we state it in the stronger form. Paired TOST at a margin of 0.05 entailment probability finds 22 of 24 model pairs equivalent across the four dataset-by-generator cells. On Natural Questions all six pairs are equivalent for both generators at p < 0.0001. The two pairs that do not reach equivalence both occur in the HotpotQA / GPT-4o-mini cell, which also has the smallest common subset.'));
kids.push(PR([['This is the central result. ',true],'Embedding models separated by up to 11.9 NDCG@5 points produce faithfulness that is statistically equivalent within 0.05, on the same queries, under two generators of different capability.']));

kids.push(H('6.3 A measurement artifact that reverses this conclusion', HeadingLevel.HEADING_2));
kids.push(P('We report this in detail because the naive analysis produces the opposite result and it is the analysis a reader is most likely to perform.'));
kids.push(P('Computing mean faithfulness over all responses, rather than over answered responses, yields the pattern in Table 5: on HotpotQA, faithfulness tracks retrieval quality rank for rank, with a spread of 0.090 — an apparently clean confirmation that embedding architecture drives faithfulness, on precisely the multi-hop dataset where such an effect was predicted.'));
kids.push(table(
  ['HotpotQA / Claude', 'mpnet', 'text-emb-3', 'BGE-M3', 'e5-instruct', 'spread'],
  [['NDCG@5', '0.705', '0.767', '0.809', '0.824', '0.119'],
   ['Faithfulness, pooled', '0.548', '0.599', '0.628', '0.638', '0.090'],
   ['Faithfulness, answered only', '0.737', '0.727', '0.731', '0.746', '0.019'],
   ['Abstention rate', '0.376', '0.279', '0.224', '0.207', '0.169']],
  [2600, 1300, 1300, 1300, 1300, 1200]));
kids.push(CAP('Table 5: The pooled mean orders models exactly by retrieval quality; the answered-only mean does not. Models are ordered by NDCG@5.'));
kids.push(P('The pooled ordering is an arithmetic consequence of the abstention row. Abstentions are correctly not entailed and score 0.25–0.32; the weakest retriever abstains on 37.6% of queries against 20.7% for the strongest, so pooling awards it a lower mean without any difference in the groundedness of what it actually asserts. Restricted to answered responses the spread falls from 0.090 to 0.019, the ordering no longer matches retrieval quality, and significance disappears.'));
kids.push(P('A second, smaller error compounds this: comparing each model on its own answered responses compares different query sets, since which queries a model abstains on is determined by its own retrieval. Pairing on the common answered subset removes it.'));

kids.push(H('6.4 What retrieval quality does govern', HeadingLevel.HEADING_2));
kids.push(P('The embedding model is far from irrelevant. It governs whether the system answers at all, and whether the answer is right.'));
kids.push(table(
  ['HotpotQA / Claude', 'NDCG@5', 'Accuracy (containment)', 'Abstention'],
  [['all-mpnet-base-v2', '0.7051', '0.599', '0.376'],
   ['text-embedding-3-small', '0.7666', '0.655', '0.279'],
   ['BGE-M3', '0.8093', '0.707', '0.224'],
   ['multilingual-e5-large-instruct', '0.8237', '0.734', '0.207']],
  [3300, 1700, 2400, 1600]));
kids.push(CAP('Table 6: On HotpotQA, accuracy and abstention track retrieval quality monotonically. Accuracy is an upper bound (§4.4).'));
kids.push(P('Moving from the weakest to the strongest retriever is worth 13.5 points of accuracy and 16.9 points of abstention rate, against 1.9 points of faithfulness that are not distinguishable from zero. The practical reading is that better retrieval makes the system answer more often and get it right more often; it does not make what the system asserts more grounded in the context.'));

kids.push(H('6.5 Necessity and sufficiency', HeadingLevel.HEADING_2));
kids.push(P('Table 7 populates the contingency of §3.3, pooling the four embedding models for the Claude generator (4,000 query-rows per dataset).'));
kids.push(table(
  ['Cell', 'NQ', 'HotpotQA'],
  [['hit × correct — as the premise expects', '71.8%', '67.2%'],
   ['hit × incorrect — retrieval NOT sufficient', '23.3%', '30.2%'],
   ['miss × correct — retrieval NOT necessary', '0.5%', '0.2%'],
   ['miss × incorrect — as the premise expects', '4.4%', '2.4%']],
  [5000, 2000, 2000]));
kids.push(CAP('Table 7: Necessity/sufficiency contingency, Claude generator, four embedding models pooled. Correctness is provisional (§4.4).'));
kids.push(P('The asymmetry is pronounced and consistent across both datasets. Retrieval sufficiency fails for 23.3% of queries on NQ and 30.2% on HotpotQA: the answer was present in the retrieved context and the system nevertheless answered incorrectly, with the failure rate rising on multi-hop questions. Retrieval necessity, by contrast, almost never fails — 20 and 9 queries respectively out of 4,000. On these benchmarks a correct answer essentially requires a retrieval hit, while a retrieval hit is far from delivering one.'));

kids.push(H('6.6 RFG behaves contrary to its design', HeadingLevel.HEADING_2));
kids.push(P('Table 8 reports the gap metrics of §3.2 on Natural Questions with the Claude generator.'));
kids.push(table(
  ['Model', 'NDCG@5', 'Faithfulness', 'RFG', 'nRFG'],
  [['all-mpnet-base-v2', '0.7862', '0.8532', '−0.0670', '−0.0853'],
   ['BGE-M3', '0.7944', '0.8484', '−0.0540', '−0.0680'],
   ['multilingual-e5-large-instruct', '0.8064', '0.8507', '−0.0443', '−0.0550'],
   ['text-embedding-3-small', '0.8300', '0.8585', '−0.0285', '−0.0344']],
  [3200, 1450, 1650, 1350, 1350]));
kids.push(CAP('Table 8: The gap is negative for every model: faithfulness exceeds retrieval quality.'));
kids.push(P('The metric was constructed to quantify how much faithfulness falls short of what retrieval quality would predict. On this data it does not fall short at all. We report the metric for completeness and to document that the assumption behind it was not borne out, and we do not use it to rank models.'));

kids.push(H('6.7 Hypotheses', HeadingLevel.HEADING_2));
kids.push(P('We stated five hypotheses before running these experiments and report their outcomes as measured.'));
kids.push(BULLET('H1 (instruction-tuned models show a smaller gap than contrastive models): not supported. multilingual-e5-large-instruct does not outperform the contrastive models on the gap metrics, and the underlying faithfulness differences are equivalent in any case.'));
kids.push(BULLET('H2 (the gap is largest on HotpotQA): not evaluable as stated, because the gap is negative throughout. The related sufficiency failure rate does rise on HotpotQA, from 23.3% to 30.2%.'));
kids.push(BULLET('H3 (the model ranking is consistent across generators): not supported. The Spearman correlation between the faithfulness rankings induced by Claude and by GPT-4o-mini is 0.000. We note that the rankings being compared are themselves differences within an equivalence band, so the correlation should be read as the absence of a stable ranking rather than as a conflict between generators.'));
kids.push(BULLET('H4 (entailment-similarity alignment is higher for instruction-tuned models): not yet run.'));
kids.push(BULLET('H5 (faithfulness-aware re-ranking reduces the gap by at least 15% relative): not yet run, and no longer well posed, since it was defined as a relative reduction in a quantity that is negative.'));
kids.push(PR([['A methodological note on H3. ', true],
  'Evaluated on nRFG rather than on faithfulness, H3 appears supported with a Spearman correlation of 1.00. This is an artifact: nRFG = 1 − F/NDCG@5, both generators are evaluated against the identical retrieval and therefore the identical NDCG@5 term, and faithfulness varies far less than retrieval quality does. The nRFG ranking consequently reproduced the NDCG@5 ranking exactly for both generators. The apparent agreement is agreement about retrieval, reported as agreement about generators.']));

// ── 7. Discussion ──────────────────────────────────────────────────
kids.push(H('7. Discussion', HeadingLevel.HEADING_1));
kids.push(P('Taken together, the results suggest that the retrieved context determines what the generator has available to be right about, while the generator’s propensity to stay within that context is a property of the generator rather than of the retriever. Faithfulness levels differ substantially between our two generators — roughly 0.92 for Claude Haiku 4.5 against 0.88 for GPT-4o-mini on Natural Questions — while differing negligibly between embedding models under either. If faithfulness is the target, these results point to the generation stage and to prompt design rather than to embedding selection.'));
kids.push(P('This does not make retrieval quality unimportant; it makes its contribution specific. A 13.5-point accuracy difference is a large practical effect, and it is obtained purely by changing the embedding model. The finding is that the mechanism runs through what gets retrieved and whether the model can answer at all, not through how faithfully it treats what it receives.'));
kids.push(P('The sufficiency result deserves emphasis independently. Between a quarter and a third of queries whose answer was present in the retrieved context were answered incorrectly. Since faithfulness in those cases remains high, these are not cases of the generator ignoring its context. They are cases of the generator using the context and still failing — through multi-hop composition, distractor interference, or misreading. Retrieval metrics cannot see this failure mode, and improving them will not address it.'));

// ── 8. Limitations ─────────────────────────────────────────────────
kids.push(H('8. Limitations', HeadingLevel.HEADING_1));
kids.push(BULLET('Correctness is measured by containment, an upper bound, with exact match at zero for our verbose generators (§4.4). The accuracy figures and the contingency in Table 7 are provisional pending an LLM-judge evaluation. The faithfulness results do not depend on the correctness label.'));
kids.push(BULLET('Faithfulness is measured by a single NLI model. AlignScore is integrated but not run, so we cannot report the metric-robustness analysis across faithfulness measures that our design calls for.'));
kids.push(BULLET('Four embedding models and two English-language datasets. Three further models and QASPER are integrated but not run. The equivalence claim is bounded by the margin and the sample size stated, not general.'));
kids.push(BULLET('Both generators are closed-source models accessed by API; an open-weight generator (Llama-3-8B) is integrated but not run, so we cannot separate effects of the generator family from effects of capability.'));
kids.push(BULLET('The controlled conditions that would anchor the contingency table — zero-retrieval as a parametric-knowledge floor and oracle retrieval as a ceiling — are implemented but not run, so Table 7 has no floor or ceiling reference.'));
kids.push(BULLET('Fixed chunk size of 256 tokens. The interaction between chunking and faithfulness is a distinct question we do not address.'));

// ── 9. Conclusion ──────────────────────────────────────────────────
kids.push(H('9. Conclusion', HeadingLevel.HEADING_1));
kids.push(P('We set out to test whether embedding architecture affects the faithfulness of RAG generation, and found that it does not, within a bound we state rather than infer from a failed significance test. Across four embedding models spanning 11.9 NDCG@5 points, two datasets, and two generators, faithfulness is statistically equivalent in 22 of 24 model pairs, while the same models differ substantially in retrieval quality, answer accuracy, and abstention rate.'));
kids.push(P('The practical implication is that embedding selection is an intervention on correctness and coverage, not on grounding. The methodological implication is that faithfulness means computed over pooled responses conflate abstention rate with groundedness, and will report an architecture effect that a per-response analysis does not support.'));

// ── References ─────────────────────────────────────────────────────
kids.push(H('References', HeadingLevel.HEADING_1));
const refs = [
  '[1] Lewis, P., Perez, E., Piktus, A., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.',
  '[2] Wang, L., Yang, N., Huang, X., et al. (2024). "Improving Text Embeddings with Large Language Models." ACL 2024. [E5-instruct]',
  '[3] Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2024). "C-Pack: Packaged Resources to Advance General Chinese Embedding." SIGIR 2024. [BGE-M3]',
  '[4] Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019.',
  '[5] Sinha, D. (2025). "The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems." arXiv:2512.15068.',
  '[6] Sun, Z., et al. (2025). "ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability." ICLR 2025.',
  '[7] Chen, S., et al. (2025). "Each to Their Own: Exploring the Optimal Embedding in RAG." arXiv:2507.17442.',
  '[8] Sturua, S., Mohr, I., Akram, M. K., et al. (2024). "jina-embeddings-v3: Multilingual Embeddings With Task LoRA." arXiv:2409.10173.',
  '[9] Tamber, M. S., Bao, F. S., Xu, C., et al. (2025). "Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards (FaithJudge)." EMNLP 2025 Industry Track.',
  '[10] Gao, Y., et al. (2025). "A Survey of Retrieval-Augmented Generation." arXiv:2506.00054.',
  '[11] Li, Z., et al. (2023). "Towards General Text Embeddings with Multi-stage Contrastive Learning." arXiv:2308.03281. [GTE]',
  '[12] Su, H., Shi, W., Kasner, Z., et al. (2023). "One Embedder, Any Task: Instruction-Finetuned Text Embeddings." ACL Findings 2023. [Instructor-XL]',
  '[13] Tan, Z., Jiao, Y., Yang, D., et al. (2026). "CTRL-RAG: Contrastive Likelihood Reward Based Reinforcement Learning for Context-Faithful RAG Models." arXiv:2603.04406.',
  '[14] Johnson, J., Douze, M., & Jégou, H. (2021). "Billion-Scale Similarity Search with GPUs." IEEE Trans. Big Data. [FAISS]',
  '[15] Ji, Z., Lee, N., Frieske, R., et al. (2023). "Survey of Hallucination in Natural Language Generation." ACM Computing Surveys.',
  '[16] Zha, Y., Yang, Y., Li, R., & Hu, X. (2023). "AlignScore: Evaluating Factual Consistency with A Unified Alignment Function." ACL 2023.',
  '[17] Kwiatkowski, T., et al. (2019). "Natural Questions: A Benchmark for Question Answering Research." TACL 2019.',
  '[18] Yang, Z., et al. (2018). "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering." EMNLP 2018.',
  '[19] Dasigi, P., et al. (2021). "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers." NAACL 2021. [QASPER]',
  '[20] He, P., Gao, J., & Chen, W. (2023). "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training." ICLR 2023.',
  '[21] Dror, R., Baumer, G., Shlain, M., & Reichart, R. (2018). "Deep Dominance — How to Properly Compare Deep Neural Models." ACL 2018.',
  '[22] Salemi, A. & Zamani, H. (2024). "Evaluating Retrieval Quality in Retrieval-Augmented Generation." SIGIR 2024. arXiv:2404.13781.',
  '[23] Schuirmann, D. J. (1987). "A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability." Journal of Pharmacokinetics and Biopharmaceutics. [TOST]',
];
refs.forEach((r) => kids.push(new Paragraph({
  spacing: { after: 60, line: 240 },
  indent: { left: 340, hanging: 340 },
  children: [new TextRun({ text: r, size: 18, font: 'Times New Roman' })],
})));

const doc = new Document({ sections: [{ children: kids }] });
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(process.argv[2], buf);
  console.log('written', process.argv[2]);
});
