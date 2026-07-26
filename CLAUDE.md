# RAG Faithfulness Research — Project Context

> This file is read automatically by Claude Code at the start of every session.
> It carries the full context of a long prior conversation so work can resume cold.

---

## 1. What this project is

**Paper title:** *Beyond Retrieval Quality: How Embedding Architecture Affects Faithfulness in RAG Systems*

**Author:** Shakhriyorbek Boltabaev (Aiden), MSc Computer Science, University of Szeged (SZTE)
**Supervisor:** Dr. Gábor Berend (SZTE NLP group / RGAI) — actively engaged, has reviewed two drafts

**Core research question:** Do embedding models with near-identical retrieval quality scores (NDCG@5, Recall@5) produce different *faithfulness* rates in downstream RAG generation? If so, why?

**Key contributions:**
1. **RFG** (Retrieval-Faithfulness Gap) metric — Eq. (2)
2. **nRFG** normalized variant — Eq. (3), added to fix a design flaw the supervisor found
3. **ESA** (Entailment-Similarity Alignment) geometric analysis — Eq. (4)
4. Faithfulness-aware **re-ranking** strategy — Eq. (5)

**Experimental scale:** 7 embedding models × 3 QA datasets × 2 generators × 1000 queries = **42,000 triples**

---

## 2. Current status

| Area | Status |
|------|--------|
| Paper draft (IEEE format) | ✅ Complete, all supervisor feedback addressed |
| References verified | ⚠️ 5 wrong attributions fixed (see §6), but ref **[8] cites a non-existent "jina-embeddings-v5-text"** — must become jina-embeddings-v3 (arXiv:2409.10173) in the next paper pass |
| Pipeline code | ✅ **All of §8 built in `src/`** (2026-07-23); notebook is now a reference artifact only |
| Notebook | ✅ Patched: title/"Mechanistic"/eq numbers fixed, `SIMULATED_RESULTS` removed, NLI index resolved from config |
| Server access | ✅ Granted by Berend, SSH key installed |
| Experiments | ❌ **Not started** — all results in the paper are still simulated |
| Venue | ⚠️ EMNLP 2026 May 25 ARR deadline **has passed**. Target July ARR cycle or COLING 2026. |

**The single biggest gap:** every number, table, and figure in the paper is placeholder/simulated data. Real experiments have never been run.

**Critical bug found & fixed during the 2026-07-23 audit (do not regress):**
the notebook hardcoded `probs[2]` as the NLI "entailment" probability, but
`cross-encoder/nli-deberta-v3-large` uses label order
`{0: contradiction, 1: entailment, 2: neutral}` (verified against the HF
config) — index 2 is **neutral**. All NLI code must resolve the index from
`model.config.label2id` (`src/nli.py` does this). Also fixed: qrels matched
by lossy chunk-string equality (now stable chunk IDs), NQ corpus had no
distractors (now pooled), NLI premise truncation ignored chunks 3–5 (now
per-chunk scoring), `intfloat/e5-large-instruct` does not exist (now
`intfloat/multilingual-e5-large-instruct`).

---

## 3. Server access (SZTE)

```
laptop ──ssh──> hop (193.225.250.29) ──ssh gpu1──> gpu1 (NVIDIA V100)
                user: sboltabaev                    alias pre-configured by Berend
```

- SSH config at `server/ssh_config` → copy to `~/.ssh/config`, then `ssh szte-gpu` connects in one hop
- Server maintenance outage was expected during the first week of access (early July 2026).

**Always use `tmux` for long jobs** — SSH dies when the laptop sleeps and would kill a multi-hour run.

### Verified server inventory (2026-07-26) — both former UNKNOWNs resolved

| Fact | Value |
|------|-------|
| hop hostname | `nlp` (193.225.250.29), key auth with `~/.ssh/id_ed25519` works |
| gpu1 hostname | `nlp-large-1` |
| **GPU** | **`GRID V100DX-32C`, 32768 MiB (~30.5 GB free), driver 580.167.08** |
| **→ Llama-3-8B** | **fp16 fits — 8-bit NOT needed.** `bitsandbytes` is now an optional dep |
| **Internet from gpu1** | **YES** — `huggingface.co` and `pypi.org` both return HTTP/2 200. Models download directly; no staging via the hop |
| CPU / RAM / disk | 16 cores / 62 GB / 284 GB free on `/` |
| Python | 3.10.12 at `/usr/bin/python3`, **no conda**, **torch not installed** |
| tmux / git | both present |

**Gotchas learned the hard way:**
- **ICMP is blocked** — `ping` always fails, this says nothing about reachability. Test with TCP 22 instead.
- **The gateway rate-limits SSH.** ~6 connections in a few minutes got the IP temporarily blocked (TCP 22 went from open to refused). Use **one** long-lived session plus `ControlMaster` multiplexing (already in `server/ssh_config`); never script rapid reconnects.
- `-J`/ProxyJump from this Windows box failed at least once where nested `ssh hop → ssh gpu1` succeeded. If `ssh szte-gpu` misbehaves, fall back to two hops.
- Local Git Bash has `scp` but **no `rsync`**.
- UNVERIFIED (rate-limited before testing): whether `$HOME` is shared between hop and gpu1. If it is, `scp` to the hop is enough. Check with `ssh hop 'touch ~/x' && ssh gpu1 'ls ~/x'`.

---

## 4. Supervisor feedback — all 7 points and their resolution

Berend's review (received ~June 2026). All are addressed **in the paper**; items 3 and 6 also require **code that does not exist yet**.

| # | His comment | Resolution | Code needed? |
|---|-------------|------------|--------------|
| 1 | Improve notation precision; don't overclaim | Removed all causal language ("causally"→removed from title), dropped "first to"/"most comprehensive to date"/"core novel"/"pioneered"; every equation now defines variables + domains | No |
| 2 | Add Jina embeddings | Added `jina-embeddings-v3` as 7th model, paradigm "distilled"; counts updated to 7 models / 42,000 triples; ref [8] | **Yes** — in `config.py` ✅ done |
| 3 | RFG may be volatile across measurement choices — test it | New §5.3: 3 retrieval metrics × 3 faithfulness metrics = 9 RFG variants, Spearman correlation of induced rankings, 9×9 matrix | **Yes** — in `metrics.py` ✅ done |
| 4 | RFG can't distinguish both-high from both-low | Acknowledged in §3.2; introduced **nRFG = (RQ − F)/RQ** as primary metric. Verified: his example (0.9,0.85) vs (0.3,0.25) gives identical RFG=0.05 but nRFG 0.056 vs 0.167 | **Yes** — in `metrics.py` ✅ done |
| 5 | Is faithfulness vs generated answer or gold answer? | §3.1 now explicit: **generated answer**, with rationale — measuring vs gold would collapse faithfulness into retrieval quality | No |
| 6 | Re-ranking not justified by the analysis; hesitant to call it "mechanistic" | §4.5 renamed **"Geometric Analysis"**; §4.6 re-ranking now justified *independently*; §4.5.1 commits to also computing correlation with **NLI(d, q)** — the signal actually available at re-rank time | **Yes** — not yet implemented |
| 7 | ESA doesn't extend Zhu et al. | Claim removed; Zhu reference deleted, slot [8] reused for Jina | No |

**Critical nuance on #6:** the ESA analysis measures `corr(cos(q,d), NLI(d, gold_answer))`, but re-ranking uses `NLI(d, q)` because no answer exists yet at retrieval time. The paper now promises to compute *both* correlations. **This experiment must actually be run** or the paper's logic remains open.

---

## 5. Key design decisions (don't accidentally reverse these)

- **Faithfulness is measured between retrieved context and the GENERATED answer**, never the gold answer. Gold answers are used only for retrieval-quality evaluation (qrels). Reversing this breaks the paper's central argument.
- **ESA deliberately uses the gold answer** (unlike F) because it measures a static property of the embedding space, independent of any generator. This asymmetry is intentional and is explained in §4.5.1.
- **nRFG is the primary metric**; raw RFG is a secondary diagnostic and must always be reported alongside absolute faithfulness so the both-low case stays visible.
- **Never call §4.5 "mechanistic analysis"** — it is correlational/geometric. True mechanistic analysis (à la ReDeEP) probes attention heads and FFNs; this does not.
- **Avoid overclaiming language** in any new text: no "causally", "first to", "novel", "comprehensive", "pioneered", "state-of-the-art". Supervisor explicitly flagged this and recommended Nicholas Carlini's writing guide.

---

## 6. Reference corrections already made — DO NOT REINTRODUCE

Five references in the original draft had fabricated or wrong author attributions. All were verified against real sources and fixed:

| Ref | Was (WRONG) | Is (CORRECT) |
|-----|-------------|--------------|
| [5] Semantic Illusion | "Zhang, T. et al." | **Sinha, D.** — arXiv:2512.15068 |
| [6] ReDeEP | "Wu, Z. et al." | **Sun, Z. et al.** — ICLR 2025 |
| [7] Each to Their Own | "Chen, J.", arXiv:2507.xxxxx | **Chen, S.** — arXiv:**2507.17442** |
| [9] FaithJudge | "Es, S. et al." | **Tamber, M. S. et al.** — EMNLP 2025 Industry Track |
| [13] CTRL-RAG | "Luo, X.", arXiv:2602.xxxxx, "contrastive likelihood training" | **Tan, Z. et al.** — arXiv:**2603.04406**, "Contrastive Likelihood **Reward Based Reinforcement Learning**" |

In-text mentions were updated too (Zhang→Sinha, Wu→Sun, Es→Tamber, Luo→Tan).

---

## 7. Known bugs already fixed in the dataset loaders

The HuggingFace dataset APIs do not match the naive assumptions the original code made. All three loaders were rewritten:

- **Natural Questions** — `annotations.short_answers[i]` has `text`, `start_token`, `end_token` as **lists**, not scalars. Original code did `tokens[start_token:end_token]` and crashed with `TypeError: slice indices must be integers`. Fix: use `sa["text"][0]` directly. Also filter out HTML tokens via `tokens["is_html"]`.
- **HotpotQA** — `supporting_facts` is a dict of parallel lists (`title`, `sent_id`); `sent_id` is an index into `context.sentences[i]`, not a sentence. Must look up `(title, sent_id)` pairs against the context.
- **QASPER** — `paper["qas"]` is a **dict of lists** (columnar), not a list of dicts. Index with `qas["question"][i]`, `qas["answers"][i]`. Handle both `free_form_answer` and `extractive_spans`. Needs `trust_remote_code=True` on newer `datasets` versions.

---

## 8. Build status (all code items ✅ as of 2026-07-23)

1. ✅ **`src/datasets_loader.py`** — 3 loaders with §7 fixes; corpus docs carry gold-provenance; NQ pools contexts so distractors exist
2. ✅ **`src/embed_index.py`** — offset-based chunking (original text preserved), stable chunk IDs, FAISS-or-numpy exact index, per-family encoder handling (Instructor pairs, jina trust_remote_code, E5-instruct template, OpenAI cost-tracked), Phase A retrieves top-20 pool
3. ✅ **`src/generate.py`** — GPT-4o-mini (resumable, budget-capped) + Llama-3-8B (VRAM-probed 8-bit, chat template, greedy, token-sliced)
4. ✅ **`src/faithfulness.py`** — per-chunk NLI (max/mean/concat) + optional AlignScore, both generators
5. ✅ **`src/esa_analysis.py`** — both `NLI(d, gold_a)` and `NLI(d, q)` (closes supervisor point 6)
6. ✅ **`src/rerank.py`** — Eq. (5) ablation: re-rank top-20 → regenerate → re-score; λ sweep helper
7. ✅ **`src/run_pipeline.py`** — `--smoke-test` (N=50, 3 models, NQ) / `--full`; per-phase selection; asserts NDCG@5 > 0 (B3 guard)
8. ✅ **`src/figures.py`** + **`src/results.py`** — Fig. 1–4 (incl. 9-variant robustness heatmap), nRFG-primary assembly, bootstrap H1–H5 summary
   - supporting: `src/utils.py` (checkpoints, seeding, unified cost tracker), `src/nli.py` (config-resolved entailment index)

**Still open:**
9. ❌ **Run the experiments** (smoke test → reduced 3×NQ → full 7×3×2)
10. ❌ **Paper update** — replace every simulated number with real results; rewrite §6 from "Expected Results" (H1–H5) into actual Results + Discussion, reporting honestly which hypotheses failed; fix ref [8] (jina v3, see §2); resolve §4.5.2 "cross-attention" wording (Llama-3 is decoder-only — self-attention over context tokens, and no code implements this analysis yet)

---

## 9. Hypotheses under test

- **H1** Instruction-tuned models show lower RFG than contrastive-only
- **H2** RFG is largest on HotpotQA (multi-hop)
- **H3** Model ranking by RFG is consistent across GPT-4o-mini and Llama-3
- **H4** ESA is higher for instruction-tuned models
- **H5** Faithfulness-aware re-ranking cuts the worst model's RFG by ≥15% relative

Report failures honestly — the supervisor values this over inflated claims.

---

## 10. Conventions

- Checkpoint everything to `checkpoints/` as pickle; every phase must be resumable
- Cost-track all OpenAI API calls; print a running total
- Chunking: 256 tokens, 32 overlap. Retrieval: top-k=5. Temperature: 0.
- Bootstrap significance: paired, n=10,000
- Random seed: 42
- Never hardcode API keys — read `OPENAI_API_KEY` and `HF_TOKEN` from env
