# RAG Faithfulness Research — Project Context

> This file is read automatically by Claude Code at the start of every session.
> It carries the full context of a long prior conversation so work can resume cold.

---

## 1. What this project is

**Paper title (v6, 2026-08-26):** *When the Evaluator Decides the Result*
Superseded: v5 *"Retrieval Quality Predicts Correctness, Not Faithfulness"* —
contradicted by its own data (false on HotpotQA, and evaluator-dependent).
Older still: *"Beyond Retrieval Quality: How Embedding Architecture Affects
Faithfulness in RAG Systems"*.

**Author:** Shakhriyorbek Boltabaev (Aiden), MSc Computer Science, University of Szeged (SZTE)
**Supervisor:** Dr. Gábor Berend (SZTE NLP group / RGAI) — actively engaged, has reviewed two drafts

**Core research question (v6):** does the *choice of faithfulness evaluator*
decide whether a RAG faithfulness result exists at all? The embedding grid is
now the **testbed**, not the claim: it supplies matched-retrieval-quality
systems whose faithfulness verdict can then be shown to flip with the metric.

**Key contributions (v6):**
1. Same 16,000 answers scored by **three evaluators** (NLI-max, claim-min,
   AlignScore) — 2 of 4 nulls break under AlignScore alone
2. A **falsification probe**: replace one grounded number with a value absent
   from the context and re-score. Detection ranges 4%–90% across evaluators and
   generators
3. The **mechanism**: the score decays with the number of assertions an answer
   contains (Claude 2.99/answer, GPT-4o-mini 1.18). Verbatim copying was tested
   as the alternative explanation and rejected
4. Consequence for equivalence testing: one fabricated fact moves NLI-max by
   0.033–0.094, so a TOST margin of ±0.05 is about the size of one fabrication.
   The margin is reported as a **curve** (±0.01 to ±0.10), not a point

Carried over as secondary: **RFG/nRFG** (diagnostics now, not the spine — see
§5), **ESA** and **re-ranking** (never run; keep-or-drop pending with Berend).

**Experimental scale actually run:** 4 embedding models × 2 QA datasets (NQ,
HotpotQA) × 2 generators × 1000 queries = **16,000 generations**, $13.46, zero
API errors. Not 42,000 triples — do not quote that number; 7 models and QASPER
are configured but were never run.

---

## 2. Current status

| Area | Status |
|------|--------|
| Paper draft (IEEE format) | ✅ Complete, all supervisor feedback addressed |
| References verified | ⚠️ 5 wrong attributions fixed (see §6), but ref **[8] cites a non-existent "jina-embeddings-v5-text"** — must become jina-embeddings-v3 (arXiv:2409.10173) in the next paper pass |
| Pipeline code | ✅ All of §8 built in `src/`; notebook is a reference artifact only |
| Server access | ✅ Granted by Berend. **Fedora is now the only machine that can deploy** — the Windows laptop's key was revoked on both hosts 2026-08-28 |
| Experiments | ✅ **Rung 2 done** — 16,000 real generations, NLI + AlignScore + claim-level, perturbation probe, TOST margin sweep |
| Correctness | ✅ **LLM judge over the full grid** (2026-08-30, $29.27, 0 errors). Containment is retired as the correctness signal — always run analyses with `--correct-source judge` |
| Paper v6 | ✅ `paper/RAG_Faithfulness_v6_evaluators.docx`, built by `paper/build_v6.js` |
| Format | ⚠️ Draft is **IEEE**; **ACL is required for any ARR submission** |
| Venue | ⚠️ EMNLP 2026 May 25 ARR deadline passed. Target July ARR cycle or COLING 2026. |

**Backup:** 266 MB / 168 files at `~/rag-backup/checkpoints` on Fedora — the only
copy of $13.46 of paid work outside gpu1. Refresh it after every paid run:
`rsync -avz szte-gpu:rag_faithfulness/checkpoints/ ~/rag-backup/checkpoints/`

**Correctness is now measured, and it moved the paper.** The judge grades 5-15
points higher than containment in every one of the 16 cells (false negatives
8.7-15.1% against false positives 0-7.9%), and it never grades a refusal
correct, so B12 is fully resolved. Two draft claims did not survive: **"more
faithful when wrong" is false on NQ for both generators** (faith_gap -0.051 to
-0.115), and the "retrieval NOT SUFFICIENT" cell is **~80% refusals**, putting
the grounded-and-wrong rate at 4.5-7.9% rather than 23-45%. See
`reports/2026-08-30_judge_full_grid.md`. Faithfulness scoring is untouched, so
the evaluator-dependence spine stands.

**The evaluator-dependence spine was re-checked after the judge run and holds**
(`reports/2026-08-30_evaluator_dependence_recheck.md`): the 2026-08-26 numbers
reproduce exactly, and 2 of 4 nulls still break under AlignScore. But the
*identity* of those two cells depends on whether abstentions are decided by
`is_abstention` or by the judge — NQ/Claude stops disagreeing, HotpotQA/Claude
starts, both at p≈0.048. **`faithfulness_by_model` therefore defaults to
`correct_source='contains'` on purpose**, unlike `conditional.py` which defaults
to `judge`: this analysis needs only `abstained` (string matching, deterministic),
not `correct`. Do not "align" the two defaults. Report the count as the finding,
never an individual cell.

**Significance is now done, and it killed the claim.** Two-sample permutation
test (`results.permutation_test` — NOT `bootstrap_significance`, which is paired
and asserts equal lengths; the wrong/right groups are disjoint and unequal),
Holm-corrected over 16 cells: **3 significant, all negative**. No positive cell
survives, none has a CI excluding zero. NQ/E5-large-instruct replicates across
generators at -0.113/-0.115. `n_wrong` is 44-84 per cell so the CIs are ±0.08 to
±0.13 — well powered against the general claim, weak on individual cells.
See `reports/2026-08-30_faith_gap_significance.md`. Full ordering in
**`PAPER_TODO.md`** — that file, not this one, is the to-do.

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

> **The laptop is Fedora as of 2026-08-28**, and its key is the only one still
> authorized — the Windows machine's key was revoked on both hosts. The
> Windows-specific notes below (`ControlMaster` unsupported, no `rsync`) are
> kept as history; on Fedora both work and `ControlMaster` should be **on**,
> because it is what keeps the gateway from rate-limiting you. Setup:
> `README.md`; moving machines again: `DEVICE_MIGRATION.md`.

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
- **GPT-4o-mini was restored as a SECOND generator on 2026-08-14** (Berend asked to see the OpenAI result too). It does not replace Claude — both run. This is strictly better than the July swap: H3 asks whether the faithfulness *ranking* of embedders survives a change of generator, and two closed-source arms plus Llama-3 test that far better than one. It also means §4.4/§5.2/H3, which still name GPT-4o-mini, are now partly true rather than simply wrong. Phase `cgpt`, label `gpt4omini`, ~$1.10 for 4,000 queries. Both API generators share one driver (`generate._run_api_generation`) so the budget cap, fail-fast guards and resume logic cannot drift apart.
- **Prompt parity is load-bearing and tested.** All three generators send the byte-identical string from `build_prompt()` — one user turn, no system message, temperature 0. `tests/test_openai_generator.py::TestPromptParity` pins this. Do not "improve" one arm with a system prompt or few-shot examples; it would confound H3.
- **An OpenAI key is required and is NOT interchangeable with the Anthropic one.** It now gates two things: `text-embedding-3-small`, one of the 7 *embedding models under study* (Anthropic has no embeddings API), and phase `cgpt`. Dropping the embedder means 6 models and new paper counts.
- **`claude-haiku-4-5` API specifics:** `temperature=0` IS accepted (sampling params are only removed on Opus 4.7+/Opus 5/Sonnet 5/Fable 5), so the paper's temperature-0 protocol is unchanged. `output_config.effort` **errors** on this model — never pass it. `thinking` is omitted deliberately (no thinking): the experiment measures grounding, not reasoning depth, and it keeps prompt parity with Llama-3.
- **Both generators get a byte-identical prompt** (one user turn with the full template). The instruction is deliberately NOT hoisted into Claude's `system` parameter, idiomatic though that would be — H3 compares rankings across generators, so a prompt difference would confound it.
- **Faithfulness is measured between retrieved context and the GENERATED answer**, never the gold answer. Gold answers are used only for retrieval-quality evaluation (qrels). Reversing this breaks the paper's central argument.
- **ESA deliberately uses the gold answer** (unlike F) because it measures a static property of the embedding space, independent of any generator. This asymmetry is intentional and is explained in §4.5.1.
- **nRFG is a diagnostic, not the primary metric** (demoted 2026-08-13, confirmed by v6). It subtracts a faithfulness score from a ranking metric and only means anything under the reading Berend questioned. Report it alongside absolute faithfulness so the both-low case stays visible; the paper's spine is evaluator sensitivity, not nRFG.
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

## 8. Build status

1. ✅ **`src/datasets_loader.py`** — 3 loaders with §7 fixes; corpus docs carry gold-provenance; NQ pools contexts so distractors exist
2. ✅ **`src/embed_index.py`** — offset-based chunking (original text preserved), stable chunk IDs, FAISS-or-numpy exact index, per-family encoder handling (Instructor pairs, jina trust_remote_code, E5-instruct template, OpenAI cost-tracked), Phase A retrieves top-20 pool
3. ✅ **`src/generate.py`** — Claude + GPT-4o-mini (one shared driver, resumable, budget-capped) + `LocalHFGenerator` for the open-weight arm (VRAM-probed 8-bit, chat template, greedy, token-sliced)
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
EM is the **lower** bound; containment was assumed to be the **upper** one.
**It is not — measured 2026-08-30** on the 200-row judge calibration
(Claude/all-mpnet/NQ): judge **0.800** > containment **0.776** > EM **0.000**.
Containment errs both ways — 5.85% false positives, **8.29% false negatives** —
so it understates accuracy on net and the truth is *outside* the EM–containment
interval. Treat containment as a cheap proxy, not a bound, and never present it
alone as "accuracy".

**⚠️ Abstentions invert the paper's headline statistic.** ~30% of answers are
"I cannot answer based on the provided context", which is correctly NOT entailed
by the context (NLI ≈ 0.28–0.41). Pooled into "incorrect", they drag its mean
down and `faith_gap` comes out **negative** (−0.21…−0.27) — the paper would
conclude faithfulness tracks correctness. **Excluding abstentions the sign
flips to +0.02…+0.16**: committed wrong answers are as grounded as right ones
or more (0.83–0.99 vs 0.82–0.90). `conditional.py` reports `faith_gap` over
answered rows only; `faith_gap_pooled` is kept solely to keep the artifact
auditable. **Never report the pooled column alone.**

> **Superseded at N=1000 — see B12 in §8.** That "+0.02…+0.16" came from a
> baseline that excluded abstentions on the *wrong* side only. With both sides
> symmetric, Claude/NQ is −0.006…+0.018, i.e. nothing. The abstention warning
> above still holds; the positive gap it claimed to recover does not.

### ⚠️ Three defects found on the first real N=1000 attempt (2026-08-14)

**B8a — every checkpoint key was scoped to nothing, so the full run reused
the pilot.** Keys are named for their content (`retrieval_BGE-M3_NQ`), not
for the run, so N=50/v1 and N=1000/v2 collided on all of them. Each phase
skips work whose checkpoint exists, so the first real run printed
`[phase A] all-mpnet-base-v2: all datasets done, skipping`, loaded the
pilot's `retrieval_quality_all.pkl`, and **exited 0 in seconds having done
nothing**. Had it reached the paid phases it would have skipped generation
too and handed 50 stale rows to the analysis as the full run.
Fixed by scoping the **directory**, not the keys:
`checkpoints/n{N}_{CORPUS_VERSION}/`. `utils.set_scope(n)` is called by
`run_pipeline` before any phase. Overrides: `RAG_SCOPE_N`, or
`RAG_CHECKPOINT_DIR` for an absolute path — the pre-scope pilot checkpoints
still sit flat in `checkpoints/`, so reading them needs the latter.

**B8b — 187 of 1000 NQ answers were falsely reported absent from their own
context.** NQ documents are token lists re-joined with spaces, so the context
holds `Röntgen 's` and `1,020 - 1,080 kg` while the answer holds
`Röntgen's` and `1,020–1,080 kg`. A whitespace-only containment test called
those different. Measured: 248/1000 "missing", of which **187 were this
artifact and only 61 genuinely absent**. Worse, `build_qrels` used the same
strict test to choose relevant chunks, found none, and **fell back to marking
the whole document relevant — re-creating B6 for a quarter of the data**.
Fixed with `src/textnorm.py`, one canonical containment rule (lowercase,
punctuation→space, collapse whitespace, unicode-safe) now used by the loader,
qrels, and `correctness.contains_answer` (which had the same defect: its
punctuation-*deleting* normalizer glued `Röntgen's`→`röntgens` and failed
against `röntgen s`). The doc-level fallback still exists for genuinely
straddling gold sentences but is **counted and printed**, never silent.

**B8c — the 61 genuinely-absent NQ queries were kept.** The loader warned
and moved on. Those queries cannot test retrieval at all, yet qrels marked
their document relevant, so they manufacture `hit × incorrect` rows — the
exact cell the paper's headline rests on. NQ now **draws until n usable
queries** are collected, discards the rest, records counts in
`LoadedDataset.stats`, and asserts the invariant that every retained sample
contains its own answer. `CORPUS_VERSION` → **v3**.

**Added the same day:** `results.tost_equivalence` / `results.equivalence_table`
— paired TOST on per-query NDCG (margin ±0.02, `EQUIV_MARGIN_NDCG`), backed by
new `per_query_rq_{model}_{dataset}` checkpoints from Phase B. This closes D8,
the one reviewer attack nothing covered: "matched retrieval quality" was only
ever supported by a non-significant difference, which is not evidence of
equivalence. `matched` requires **both** no detectable difference **and** TOST
equivalence.

### Modules added for the v6 measurement spine (2026-08-25/26)

17. ✅ **`src/perturbation_check.py`** — the falsification probe, `--scorer {nli,claim,align}`
18. ✅ **`src/claim_faithfulness.py`** — min over claims of max over chunks
19. ✅ **`src/compare_evaluators.py`** — per-evaluator comparison + TOST margin sweep
20. ✅ **`src/llm_judge.py`** — correctness grading, two judges, per-model pricing

### ⚠️ Three code gaps found on 2026-08-30 (fixed — do not regress)

**B9 — the LLM judge was write-only.** `llm_judge.py` wrote
`{name}_judged_{judge}` and *nothing read it*: `conditional.py` and
`results.py` went straight to `{name}_scored`. Running the ~$32 full-grid judge
would have changed no number in the paper. Fixed: `correctness.load_correctness`
is now the single resolver (`CORRECT_SOURCES = judge|contains|em|f1`), used by
`conditional.build_query_frame(correct_source=...)`,
`results.compare_faithfulness` and `run_pipeline --correct-source`. It overlays
the judge **per row** and **prints coverage** (`judged N, heuristic N,
ungradable N`) — a run that quietly fell back to containment is otherwise
indistinguishable from a judged one. Three judged-row cases must stay distinct:
no judged row → heuristic; `source_ungradable` → `None`, never `False`; a
judge-side `None` (429, unparseable) → heuristic, because it is retryable.
The legacy field name `ungradable` is also honoured — the backup's only judged
file is 4 rows of `AuthenticationError 401` frozen in the pre-fix schema.

**B10 — `conditional.GENERATORS` was `('claude', 'llama3')`.** `llama3` has
**zero** checkpoints; `gpt4omini` has **8,000** paid ones. The 2×2, the
conditional table and the anchors therefore described Claude only. Adding
`gpt4omini` doubles the frame (4,000 → 8,000 rows on NQ) and moves the headline
cell: *hit × incorrect* is 23.3% for Claude but **46.5%** for GPT-4o-mini on NQ.
Any generator added to `generate.py` must also be added to `GENERATORS` in
`conditional.py`, `faithfulness.py`, `results.py`, `perturbation_check.py` and
`claim_faithfulness.py`, or the arm is generated and then never scored.
`tests/test_openai_generator.py::TestOpenWeightGenerator` pins this.

**B11 — `correctness._generation_checkpoint_names` matched `*_judged_*`.**
Judged files match `generated_*.pkl` and hold verdicts, not generations, so
`run_phase_correctness` would have scored them into junk
`*_judged_*_scored` checkpoints. Now excluded, as `llm_judge.py` already did.

**B12 — `faith_gap` had a one-sided baseline (found 2026-08-30 by running the
free analysis on gpu1 with both generator arms).** It compared
`faith(wrong, ANSWERED)` against `faith(correct, ALL)`. Containment grades some
abstentions **correct** — a short gold answer string occurs inside "I cannot
answer based on the provided context" — 193/2893 of Claude's correct rows on NQ
(6.7%) and 315/2695 on HotpotQA (11.7%), each scoring ~0.37–0.51 NLI. They
depressed the correct-side mean and manufactured a positive gap.

Symmetric, **Claude/NQ goes from +0.018…+0.050 on all four embedders to
−0.006…+0.018** — from "more faithful when WRONG", the sharpest claim in the
draft, to nothing. HotpotQA keeps its sign and most of its magnitude.
GPT-4o-mini has **zero** abstained-correct rows on NQ and never moved, which is
exactly why the two generators looked like they disagreed. `faith_gap` now
excludes abstentions on both sides, `faith_gap_pooled` pools both sides, and
`n_correct_abstained` is reported so the containment false positives stay
visible. Those same rows also inflate `accuracy` — another reason to run the
judge rather than ship containment.

**B13 — the tables pooled datasets.** `conditional_faithfulness` grouped by
model only and `necessity_sufficiency` not at all, so NQ and HotpotQA were
averaged. That made `faith_gap` look like a *generator* effect; split by
dataset, NQ is ~0 for both generators and HotpotQA carries all of it, in
**opposite directions** (Claude −0.087…−0.182, GPT-4o-mini +0.030…+0.093).
Both tables now emit one block per dataset, with an `ALL` block kept for
continuity. Read the per-dataset blocks.

**The hit × incorrect cell is mostly refusals.** Abstention share of that cell:
Claude/NQ 24.9%, Claude/HotpotQA 56.2%, GPT-4o-mini/NQ 53.6%,
GPT-4o-mini/HotpotQA 60.1%. Its raw size is therefore not comparable across
generators or datasets and is not on its own evidence that retrieval was
insufficient — `share_answered` subtracts the refusals.

**Still open:**
21. ❌ **The experiments in `PAPER_TODO.md` §3** — LLM judge (~$32, top item),
   open-weight generator (free), claim-level over the full grid (free),
   literal value grounding (not built), C1/C2 (~$3–4), QASPER (~$5),
   ESA/re-ranking (decision pending with Berend). Shapley/`doc_utility.py`
   is **dropped** at Berend's explicit direction.
22. ❌ **Paper update** — v6 exists but references are **unverified** (five were
   wrong in earlier drafts, see §6); fix ref [8] (jina v3, see §1); convert
   IEEE → ACL; read the Salemi & Zamani PDF (only the abstract has been read);
   resolve §4.5.2 "cross-attention" wording (decoder-only models use
   self-attention over context tokens, and no code implements that analysis)

---

## 9. Hypotheses — and what the data did to them

H1–H5 were written before any experiment ran. Rung 2 settled several of them,
mostly against the hypothesis. Report that honestly; the supervisor values it
over an inflated claim, and it is why the paper was reframed rather than
rewritten to fit.

| | claim | outcome at N=1000 |
|---|---|---|
| **H1** | instruction-tuned show lower RFG than contrastive-only | **not supported** — the faithfulness differences between embedders are mostly null, and where they are not, the sign depends on the evaluator |
| **H2** | RFG largest on HotpotQA (multi-hop) | **dataset does matter, but not as RFG** — all four HotpotQA cells correlate *positively* with NDCG@5 (rho +0.40…+1.00); all four NQ cells are zero or negative |
| **H3** | model ranking by RFG survives a change of generator | **partly** — but the more interesting finding is that it does not survive a change of *evaluator*, which is now the paper |
| **H4** | ESA higher for instruction-tuned models | **not run** |
| **H5** | faithfulness-aware re-ranking cuts the worst model's RFG ≥15% | **not run** |

The v6 spine is **measurement validity**: same queries, same answers, same
test, different evaluator, different verdict (2 of 4 nulls break under
AlignScore at p=0.0005). Details in
`reports/2026-08-26_evaluator_dependence.md` and
`reports/2026-08-25_perturbation_check.md`.

---

## 10. Conventions

- Checkpoint everything to `checkpoints/` as pickle; every phase must be resumable
- Cost-track all OpenAI API calls; print a running total
- Chunking: 256 tokens, 32 overlap. Retrieval: top-k=5. Temperature: 0.
- Bootstrap significance: paired, n=10,000
- Random seed: 42
- Never hardcode API keys — read `OPENAI_API_KEY` and `HF_TOKEN` from env
