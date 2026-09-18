# Paper claims → code

Every method claim in `paper/acl/main.tex`, with the file and function that
implements it. An automated code check reported 22 of 23 claims as "not
verifiable" against this repository while finding no discrepancy; the cause was
that the checker could not locate the implementations, so this file states them
explicitly.

Line numbers are for commit `440ac21` and may drift; the symbol names will not.

| # | Paper claim | Implementation |
|---|---|---|
| 1 | NLI-max is the maximum entailment score over retrieved chunks | `src/nli.py::NLIScorer.score_chunks` (l.139) |
| 2 | Claim-min is the minimum over claims of the maximum over chunks | `src/claim_faithfulness.py::score_claims` (l.122) |
| 3 | Faithfulness is scored against the generated answer, never the gold answer | `src/faithfulness.py` l.55, l.103, l.167 — the hypothesis is always `g['generated_answer']` |
| 4 | NQ and HotpotQA, 1,000 queries each | `src/config.py::DATASETS`; `src/datasets_loader.py::load_nq`, `::load_hotpotqa` |
| 5 | Top-5 chunks of 256 tokens, 32-token overlap | `src/config.py` l.23–27: `TOP_K=5`, `CHUNK_SIZE=256`, `CHUNK_OVERLAP=32`; applied in `src/embed_index.py` l.52 |
| 6 | Four embedding models carry the matched comparison | `src/config.py::EMBEDDING_MODELS` (l.162) |
| 7 | Three generators, byte-identical prompt, temperature 0 | `src/generate.py::build_prompt` (l.45); parity pinned by `tests/test_openai_generator.py::TestPromptParity` |
| 8 | Each evaluator receives premises in its own intended interface | `src/nli.py::score_chunks` (per-chunk + truncated concatenation, max-pooled); AlignScore arm in `src/faithfulness.py` passes the concatenated context as one premise |
| 9 | Refusals: SQuAD-normalised prefix match against a fixed phrase list, after stripping a shared preamble | `src/abstention.py::is_abstention` (l.97) — single shared implementation, imported by every caller |
| 10 | Falsification replaces a grounded numeric value with one absent from the context | `src/perturbation_check.py::build_number_case` (l.184) |
| 11 | Eligibility: original value present in context, replacement absent | `src/perturbation_check.py::num_in_text` (l.126) — boundary-matched, not substring |
| 12 | First 200 eligible answers in dataset order, deterministic | `src/perturbation_check.py::build_cases` (l.256) |
| 13 | Two gates: fixed 0.5 and floor-anchored 95th percentile | `src/perturbation_check.py`, reported by `src/perturb_report.py` |
| 14 | Sentence-level assertion segmentation, grouped by count | `src/claim_faithfulness.py::split_claims` (l.71); grouping in `src/perturb_report.py` |
| 15 | Paired bootstrap; statistic is best-vs-worst spread, re-centred at zero | `src/results.py::bootstrap_significance` (l.34); cell assembly in `src/results.py::faithfulness_by_model` (l.281) |
| 16 | Holm correction over the comparison family | `src/results.py::holm_bonferroni` (l.120) |
| 17 | Spearman of NDCG@5 against mean faithfulness across systems | `src/results.py`, reported by `src/compare_evaluators.py` |
| 18 | TOST equivalence, margin swept and reported as a curve | `src/results.py::tost_equivalence` (l.160); `src/compare_evaluators.py::MARGINS` (l.42) = `[0.01, 0.02, 0.03, 0.05, 0.10]` |
| 19 | Floor/ceiling on the Claude arm; oracle supplies answer-bearing chunks only | `src/conditions.py::build_oracle_contexts` (l.196), `--oracle-source qrels` |
| 20 | Context arrangement: order, position, subset, distractor-only | `src/context_ablation.py::build_context` (l.125), `::run_ablation` (l.170) |
| 21 | Geometric analysis computes both NLI(d, gold) and NLI(d, query) | `src/esa_analysis.py::run_esa` (l.54), `::_correlations` (l.43) |
| 22 | Re-rank top-20 by query entailment at λ=0.6, then regenerate | `src/rerank.py::rerank_candidates` (l.22) |
| 23 | QASPER as a boundary condition, five embedders × three generators | `src/datasets_loader.py::load_qasper` (l.303); `src/config.py::DATASETS` |
| 24 | Entity substitution reproduces the numeric result | `src/perturbation_check.py::build_entity_case` (l.208) |
| 25 | Deterministic value check: 100% recall at 0.8–5.2% FPR | `src/perturbation_check.py`, `--scorer numeric`; content-vs-citation split in `value_role()` |
| 26 | Evaluator selection table (detection, MDE, refusal behaviour) | `scripts/entity_substitution_table.py`, `scripts/power_analysis.py`, `scripts/refusal_scores.py` |
| 27 | Verbosity control: one model, two prompts, weights and retrieval held fixed | `src/generate.py::build_prompt(style=)` + `PROMPT_STYLES`; run via `run_pipeline --open-label --open-prompt-style`; pinned by `tests/test_prompt_styles.py` |
| 28 | Verbosity control results (Table 5) | `scripts/verbosity_control_table.py` |

## Reading order

Start at `src/run_pipeline.py`, which sequences every phase, then `src/config.py`
for the constants above. The three evaluators are `src/nli.py`,
`src/claim_faithfulness.py` and the AlignScore arm of `src/faithfulness.py`. All
statistics are in `src/results.py`.

## Two things that are easy to get wrong

- **Checkpoints are scoped** to `checkpoints/n{N}_{CORPUS_VERSION}`. A run
  against the wrong scope silently reuses another run's results. Override with
  `RAG_SCOPE_N` or `RAG_CHECKPOINT_DIR`.
- **The NLI entailment index is resolved from `model.config.label2id`**, never
  hardcoded. `cross-encoder/nli-deberta-v3-large` orders labels
  `{0: contradiction, 1: entailment, 2: neutral}`, so the intuitive `probs[2]`
  is *neutral*. See `src/nli.py`.

`tests/` has 225 passing tests covering the aggregations, the abstention rule,
the boundary-matched containment test, the Shapley axioms and prompt parity.
