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
- **Access granted by Berend on 2026-07-02.** His maintenance-outage window ("during the next week", ~Jul 2–9) **has passed** — it is no longer a blocker. The hop reported 12 days uptime on 2026-07-26, i.e. stable since ~Jul 14.
- Berend's instruction: the hop is an **entry point only**, "not primarily meant for conducting experiments". Never compute there — jump straight to gpu1.
- The authorized key is `ssh-ed25519 AAAAC3...t/T0 shakhriyorbekboltabaev@gmail.com`, which matches the local `~/.ssh/id_ed25519` (verified 2026-07-26).

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
| tmux / git / rsync | all present on gpu1 (laptop has `scp` but no `rsync`) |
| `python3 -m venv` | ❌ **broken** — `ensurepip` missing, `python3.10-venv` needs sudo. Use `pip install --user` |

**⚠️ torch on this GPU — `torch.cuda.is_available()` is NOT a sufficient check.**
The V100 is **Volta, compute capability 7.0**; the default PyPI wheel ships
kernels for **sm_75 and up only**. It imports fine, reports
`cuda: True`, and then dies on the first real kernel launch with
`CUDA error: no kernel image is available for execution on the device`.
Always verify with a real matmul and confirm `torch.cuda.get_arch_list()`
contains `sm_70`. Reinstalling needs `--force-reinstall` — pip treats
`2.13.0+cu130` and `2.13.0+cu126` as the same version and silently skips.

**Gotchas learned the hard way:**
- **`gpu1` is an alias that exists ONLY in the hop's `~/.ssh/config`** (`host gpu1 → hostname 192.168.0.206`). It is not in DNS or `/etc/hosts`. `ssh gpu1` works from the hop because SSH reads that config, but **ProxyJump uses `ssh -W gpu1:22`, which does a literal DNS lookup and ignores Host aliases** → `Temporary failure in name resolution`. A working `~/.ssh/config` must set `HostName 192.168.0.206` for `szte-gpu`. This was the cause of the "ProxyJump flakiness" noted earlier — it was never flaky, it was always wrong.
- **`ControlMaster` does not work on Windows OpenSSH** (needs Unix domain sockets). Including it makes every connection fail. Linux/macOS only.
- **ICMP is blocked** — `ping` always fails, this says nothing about reachability. Test with TCP 22 instead.
- **The gateway rate-limits SSH.** ~6 connections in a few minutes got the IP temporarily blocked (TCP 22 went from open to refused). Use **one** long-lived session plus `ControlMaster` multiplexing (already in `server/ssh_config`); never script rapid reconnects.
- `-J`/ProxyJump from this Windows box failed at least once where nested `ssh hop → ssh gpu1` succeeded. If `ssh szte-gpu` misbehaves, fall back to two hops.
- Local Git Bash has `scp` but **no `rsync`**.

### Code sync — solved 2026-08-13 (was the long-standing blocker)

- **`$HOME` is NOT shared between hop and gpu1** (`HOME_SHARED=no`, tested).
  This was UNVERIFIED for weeks. Copying to the hop does nothing for gpu1.
- **The laptop key is now in `gpu1:~/.ssh/authorized_keys`.** Before this,
  `ssh szte-gpu` failed `Permission denied (publickey)` because Berend
  authorized the key on the *hop* only — that is why scp to gpu1 never worked.
  `ssh szte-gpu` now connects directly, and scp/git work over it.
- **Code and data are separate directories on gpu1** (note hyphen vs underscore):

  | Path | Holds | Git |
  |---|---|---|
  | `~/rag-faithfulness/` | code, a clone of the bare repo | yes |
  | `~/rag_faithfulness/` | `checkpoints/`, `hf_cache/`, `outputs/` | no |

  `config.BASE_DIR` still resolves to `~/rag_faithfulness`, so the code finds
  every existing checkpoint. Old loose copies are parked at
  `~/rag_faithfulness/src.superseded-2026-08-13` (verified to contain nothing
  the repo lacks; nothing deleted).
- **Sync loop:** `git push gpu1 main` on the laptop, `git pull` on gpu1.
  Remote `gpu1` → `szte-gpu:rag-faithfulness.git` (bare repo on gpu1). Pushing
  over SSH avoids putting a GitHub credential on the server.
- `pytest` installed via `pip install --user` (venv is broken — `ensurepip`
  missing, needs sudo). Suite passes 60/60 on gpu1.

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

## 4b. Berend's second letter — 2026-08-11 (the reframing)

He restates the paper's target claim as: **good retrieval quality is necessary
and sufficient for a high-quality response**, and the paper's job is to show it
is not. Two separable branches:

- **A — not sufficient.** Good retrieval does not imply a good response.
- **B — not necessary.** A good response is possible despite imperfect retrieval.

| # | His ask | Where it lives now |
|---|---------|--------------------|
| 1 | Zero-retrieval generation (parametric knowledge) | `src/conditions.py` C1 |
| 2 | Oracle RAG as a glass ceiling | `src/conditions.py` C2 |
| 3 | Watch for retrieval BEATING the oracle | `conditional.anchor_table` — `pct_of_oracle > 1.0` is surfaced, never clipped |
| 4 | Subset/ordering of relevant snippets | `src/context_ablation.py` |
| 5 | Faithfulness only matters when the answer is correct | `src/correctness.py` + `src/conditional.py` |
| 6 | Shapley-style per-document utility | `src/doc_utility.py` |
| 7 | Justify n=1000, or scale up | `conditions.py --emit-filter`, applied via `conditional.py --filtered` |

**⚠️ The paper he linked is a positioning problem, not just a citation.**
DOI `10.1145/3626772.3657957` is **Salemi & Zamani, "Evaluating Retrieval
Quality in Retrieval-Augmented Generation", SIGIR 2024 (arXiv:2404.13781)** —
*not* "The Power of Noise", which is the natural wrong guess. Its abstract
states that query–document relevance labels correlate only weakly with
downstream RAG performance. **That is this paper's premise, already published.**
Their eRAG (score each retrieved doc alone, use the downstream result as its
relevance label) is the singleton term of a Shapley value, which is why points
5 and 6 of his letter are one idea.

What survives as this paper's own contribution: comparison **across embedding
architectures at matched retrieval quality**; **faithfulness** specifically,
conditioned on correctness; and the controlled floor/ceiling/arrangement
conditions. What does not: "retrieval quality does not predict downstream
quality" as a headline. Position as confirming and extending them, with a real
related-work paragraph. Read the full PDF first — only the abstract has been read.

**Open question put to Berend (2026-08-13 reply draft):** whether nRFG stays
the primary metric. It subtracts a faithfulness score from a ranking metric,
and only means anything under exactly the reading he questioned. Recommended
position: demote nRFG to a diagnostic, make the necessary/sufficient 2×2 the
paper's spine.

---

## 5. Key design decisions (don't accidentally reverse these)

- **The generator is Claude (`claude-haiku-4-5`), not GPT-4o-mini** (changed 2026-07-26). Haiku 4.5 was chosen as the closest analog in capability tier and cost, preserving the design intent of a small, widely-deployed closed-source model — a frontier model would likely be more faithful across the board and could compress the very RFG differences the paper measures. **Paper §4.4 and §5.2 still say GPT-4o-mini and must be updated; tell Berend.** Estimated cost for the full 21,000-query grid: ~$38 (vs ~$5–6 for GPT-4o-mini); `--smoke-test` prints a measured projection before you commit.
- **An OpenAI key is still required and is NOT interchangeable with the Anthropic one.** `text-embedding-3-small` is one of the 7 *embedding models under study*, and Anthropic has no embeddings API. Dropping it means 6 models and new paper counts.
- **`claude-haiku-4-5` API specifics:** `temperature=0` IS accepted (sampling params are only removed on Opus 4.7+/Opus 5/Sonnet 5/Fable 5), so the paper's temperature-0 protocol is unchanged. `output_config.effort` **errors** on this model — never pass it. `thinking` is omitted deliberately (no thinking): the experiment measures grounding, not reasoning depth, and it keeps prompt parity with Llama-3.
- **Both generators get a byte-identical prompt** (one user turn with the full template). The instruction is deliberately NOT hoisted into Claude's `system` parameter, idiomatic though that would be — H3 compares rankings across generators, so a prompt difference would confound it.
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

### First real run on gpu1 — 2026-07-26 (smoke test, N=50, NQ, 3 models)

**Phases A+B PASSED in 1m51s.** B1 and B3 both verified on real hardware:

```
B1  labels: {0: contradiction, 1: entailment, 2: neutral} -> index 1
    entail 0.9970 / contra 0.0000

B3  [NQ] all-mpnet-base-v2 : NDCG@5 0.9483  Recall@5 0.9400  MRR@5 0.980
    [NQ] BGE-M3            : NDCG@5 0.9076  Recall@5 0.8950  MRR@5 0.984
    [NQ] E5-large-instruct : NDCG@5 0.9564  Recall@5 0.9667  MRR@5 0.970
```

**Read these numbers with two caveats:**
- At N=50 the NQ corpus is only 50 docs / 152 chunks, so retrieval is easy
  and NDCG is inflated. **Expect it to drop at N=1000** (1000 docs). That is
  correct behaviour, not a regression.
- The B4 fix is confirmed working: each query now competes against 49
  distractor documents. Before the fix the index held only the query's own
  gold context, which would have made retrieval trivially perfect.
- The model spread (0.908–0.956) is small — which *is* the paper's premise
  (near-identical retrieval quality), but it is not yet meaningful at N=50.

### First end-to-end RFG — 2026-07-26 (smoke, N=50, NQ, Claude Haiku 4.5)

Phases C+D ran clean: **150 requests, 0 errors, $0.2782**.
**Cost probe: $0.001855/query → $38.95 projected for the full 21,000.**
(Measured 1,465 input tok/query vs the 1,400 assumed — the ~$38 budget holds.)

| Model | Paradigm | NDCG@5 | Faith (nli_max) | RFG | nRFG |
|---|---|---|---|---|---|
| BGE-M3 | multilingual | 0.9076 | 0.7810 | 0.1266 | **0.1395** |
| all-mpnet-base-v2 | contrastive | 0.9483 | 0.7324 | 0.2159 | **0.2277** |
| E5-large-instruct | instruction-tuned | 0.9564 | 0.6906 | 0.2658 | **0.2779** |

**Retrieval and faithfulness rankings are INVERTED** — the best retriever is
the least faithful. That is the paper's central premise showing up in real
data for the first time.

⚠️ **NOT EVIDENCE YET.** n=50, one dataset, one generator, no error bars, no
significance test, and a trivially easy 152-chunk corpus. Do not show these
to Berend as a finding. The next real gate is Rung 2 (3 models × NQ, N=1000,
~$5.60, ~2-4 h).

⚠️ **H1 is currently contradicted.** H1 predicts instruction-tuned < contrastive
in RFG; here E5-large-instruct (instruction-tuned, nRFG 0.2779) is *worse*
than all-mpnet-base-v2 (contrastive, 0.2277). If this survives to N=1000 with
significance, **report H1 as failed** — §9 says Berend values that over an
inflated claim.

### Modules added for Berend's 2026-08-11 letter (2026-08-13) — all untested against real data

11. ✅ **`src/correctness.py`** — EM + token-F1 re-scoring of existing generation
    checkpoints; free. **Ungradable ≠ incorrect:** `[ERROR: ...]` rows and
    `generated_answer=None` (C2 oracle gaps) yield `correct=None`, never False —
    grading an API 429 as a wrong answer deflates whichever condition hit rate
    limits, invisibly. Abstentions ("I do not know") are `correct=False` but
    flagged separately via `abstained`.
12. ✅ **`src/conditions.py`** — C1 no-RAG floor, C2 oracle ceiling,
    `--prompt-probe` (isolates the C1 closed-book-prompt confound),
    `--emit-filter`. **C2 defaults to `--oracle-source qrels`**, not
    `gold_context`: the latter is strictly easier than perfect retrieval (on
    HotpotQA it hands over supporting sentences stripped of their paragraphs)
    and would be a ceiling on something the paper never measures.
13. ✅ **`src/conditional.py`** — the deliverable. Per-query join of retrieval
    hit × correctness × faithfulness; the necessary/sufficient 2×2;
    `faith_gap = faithfulness(incorrect) − faithfulness(correct)`; floor →
    embedders → ceiling anchors with `pct_of_oracle`.
14. ✅ **`src/context_ablation.py`** — 8 conditions over the oracle set: order,
    subset, position-at-constant-length, `noise_only`. Distractors are the
    reference embedder's own top-20 misses (hard negatives), not random text.
15. ✅ **`src/doc_utility.py`** — eRAG / leave-one-out / exact Shapley over the
    top-5. `--mode shapley` enumerates all 2⁵=32 subsets, which *contains* the
    eRAG and LOO subsets — one run yields all three. Shapley verified against
    the efficiency, dummy and symmetry axioms in `tests/`.
16. ✅ **`src/preflight.py`** — free read-only launch gate. ANTHROPIC key fatal;
    OPENAI/HF non-fatal (a 3-model NQ run needs neither).

Also fixed in the same pass, beyond patches 0001/0002:
- `metrics.nrfg` returned **−7.8e8** for `nrfg(0.0, 0.78)` — the old
  `rq = max(1e-9, rq)` guard divided by 1e-9. Now NaN. RQ=0 never happens in
  aggregate NDCG but happens constantly **per query**, and the conditional
  analysis is per query.
- QASPER loader silently substituted the paper's first 3 paragraphs when
  annotated evidence was a figure/table. Now flagged as
  `QASample.gold_is_fallback` and excluded from C2.

### ⚠️ Two measurement bugs found by running the FREE analysis on the pilot (2026-08-13)

Both would have manufactured a false positive for the paper's headline claim.
Found for $0, before the paid run. **Do not regress either.**

**B6 — NQ's answer was outside its own context 34% of the time.**
`load_nq` kept the first 500 non-HTML tokens; the annotated short answer fell
outside that window for **17/50 pilot queries**. The answer was therefore not in
the corpus at all — no retriever could surface it — yet qrels marked chunks of
that document relevant, so **NDCG@5 reported ~0.95** while the generator
correctly said "I cannot answer based on the provided context". That fabricates
*hit × incorrect* rows, the exact cell N1 rests on.
Fixed: the window is **centred on the answer span** (`start_token` remapped
through the HTML filter), and **NQ relevance is now answer-bearing** via
`gold_sentences`, matching how HotpotQA already worked. `hit` now means "the
model was shown the answer", not "the right document was retrieved".
`CORPUS_VERSION` (`datasets_loader.py`) is in every dataset/chunks/qrels
checkpoint key so old caches cannot be silently reused — **bump it whenever
corpus or relevance semantics change**.

**B7 — EM/token-F1 measured verbosity, not correctness.**
Pilot accuracy read **2%**; the model was actually right on **58%**. The RAG
prompt yields "Based on the provided context, **Wilhelm Conrad Röntgen** of
Germany received…" against gold "Wilhelm Conrad Röntgen, of Germany" — correct,
but EM=0 and F1=0.28 because precision dies on every extra token.
Fixed: `correctness.py` gains `contains_answer()` and `CORRECT_MODE='contains'`.
EM is the **lower** bound, containment the **upper** — report both, never
containment alone as "accuracy".

**⚠️ Abstentions invert the paper's headline statistic.** ~30% of answers are
"I cannot answer based on the provided context", which is correctly NOT entailed
by the context (NLI ≈ 0.28–0.41). Pooled into "incorrect", they drag its mean
down and `faith_gap` comes out **negative** (−0.21…−0.27) — the paper would
conclude faithfulness tracks correctness. **Excluding abstentions the sign
flips to +0.02…+0.16**: committed wrong answers are as grounded as right ones
or more (0.83–0.99 vs 0.82–0.90). `conditional.py` reports `faith_gap` over
answered rows only; `faith_gap_pooled` is kept solely to keep the artifact
auditable. **Never report the pooled column alone.**

**Still open:**
9. ❌ **Run the experiments** — smoke test passes end to end; Rung 2 (3×NQ at
   N=1000), then Rung 5 (the Berend conditions, ~$15), ESA, rerank, Llama-3,
   AlignScore, and the full 7×3 grid
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
