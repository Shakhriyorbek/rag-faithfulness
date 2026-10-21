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
1. Same 32,000 answers scored by **three evaluators** (NLI-max, claim-min,
   AlignScore) — **4 comparisons in 3 of 8 cells** break, surviving Holm over 24.
   Two are AlignScore; the other two are Mistral-7B under **NLI-max and
   claim-min**, so the dissent is no longer AlignScore-specific (see B28)
2. A **falsification probe**: replace one grounded number with a value absent
   from the context and re-score. Detection ranges **5.1%–89.4%** across
   evaluators and generators. Also run with **entity** substitution, which
   reproduces the generator ordering in 5 of 6 cells (the two error types are
   NOT interchangeable — see B26)
3. The **mechanism**: the score decays with the number of assertions an answer
   contains (NQ, answered rows: Claude 3.08, qwen 1.77, GPT-4o-mini 1.25).
   Verbatim copying was tested as the alternative explanation and rejected.
   **Now established causally within one model** — see B25. ⚠️ Every assertion
   figure in the paper is **answered-only**; an all-rows figure (1.117/2.728)
   was mixed into Limitations until 2026-10-07 and is fixed
4. Consequence for equivalence testing: one fabricated fact moves NLI-max by
   0.033–0.094, so a TOST margin of ±0.05 is about the size of one fabrication.
   The margin is reported as a **curve** (±0.01 to ±0.10), not a point

Carried over as secondary: **RFG/nRFG** (retired — see §5). **ESA** and
**re-ranking** are now RUN and in the paper; both are negative results and both
are Claude-arm only (see B24, item 21).

**Experimental scale actually run (as of 2026-10-07):**
- **Main grid:** 4 embedders × 2 datasets (NQ, HotpotQA) × 3 generators × 1000
  = **24,000 generations**, plus the Mistral-7B arm (2026-10-15) for
  **32,000**. This is the grid every headline rests on.
- **jina-embeddings-v3:** a 5th embedder, both datasets, all 3 generators —
  reported as a *positive control*, not a matched system (B24b).
- **QASPER:** 5 embedders × 3 generators × 819 queries — a *boundary condition*,
  not a third dataset. $5.22.
- **qwen_verbose:** 5 embedders × 2 datasets, 10,000 answers, $0 — the W2
  verbosity control (B25).
- Plus C1/C2 anchors, context ablation, ESA, re-ranking (all Claude-arm only).
Total paid ≈ **$48**. Never quote 42,000 triples. GTE-large and Instructor-XL
are still the only configured models never run.

---

## 2. Current status

| Area | Status |
|------|--------|
| Paper draft | ✅ Complete and current. **Source is `paper/acl/main.tex`**; the docx builder is superseded. Two PDFs: `main-preprint.pdf` (named, **this is the one to send Berend**) and `main.pdf` (anonymous + line-numbered). Two automated review passes done — **3.5 Borderline Conference, soundness 4** |
| References verified | ✅ 24 verified 2026-09-03 (§6, §6b); **31 entries now**, all cited, none orphaned. 6 added 2026-09-16..10-07 and each checked against source: Wallat (ICTIR 2025), Ju (arXiv:2604.25313), Xu (Findings EMNLP 2025), Steen (ACL 2023), ARES (**NAACL 2024**, not 2023), RAGVUE (EACL 2026 demos). ✅ **`dasigi-etal-2021` (QASPER) verified 2026-10-16** against ACL Anthology 2021.naacl-main.365 — title, six authors in order, booktitle, pages 4599-4610, publisher and DOI all match; `month = jun` added. **All 31 entries have now been through the B3 check** |
| Pipeline code | ✅ All of §8 built in `src/`; notebook is a reference artifact only |
| Server access | ✅ Granted by Berend. **Fedora is now the only machine that can deploy** — the Windows laptop's key was revoked on both hosts 2026-08-28 |
| Experiments | ✅ **All run.** 24,000-answer main grid + jina control + QASPER + qwen_verbose + C1/C2 + ablation + ESA + re-ranking. Nothing paid remains. See `reports/2026-09-16_all_results.md` and `reports/2026-10-07_reviews_and_verbosity_control.md` |
| Correctness | ✅ **LLM judge over all 32 matched cells.** 24 cells 2026-08-30 ($29.27); the mistral arm 2026-10-16 (8,000 rows, **$13.74**, 0 errors, 0 ungradable — see B29). Containment is retired as the correctness signal — always run analyses with `--correct-source judge`. The judge covers the matched grid **only**: jina, QASPER and the C1/C2/ablation conditions are still containment-graded, so their accuracy figures are not comparable with the grid's |
| Paper v6 | ✅ `paper/acl/main.tex` → `main.pdf` (anonymous, for ARR) and `main-preprint.pdf` (named, for Berend). `paper/build_v6.js` + its docx are **superseded — do not edit them**, an edit there will not reach the manuscript |
| Format | ✅ **ACL** (`paper/acl/`, `./build.sh` via tectonic). **30 pages (the two compact variants are 31), 0 undefined refs, 0 overfull boxes, 0 rendered ??.** The ARR 8-page body limit is NOT currently enforced — the paper deliberately carries every result for Berend. Body is ~14 pages, so **a submission to a venue with a hard page limit needs the supplementary sections moved to the appendix first** (Floor/Ceiling 587 w, ESA/re-rank 283 w, QASPER 400 w, §12.2 ~350 w; ACL appendices do not count) |
| Venue | ⚠️ **The ARR 12 October 2026 deadline has passed and was not used** — the goal became a complete draft for Berend instead. Next ARR cycle or a direct venue submission is an open decision. The old plan (one ARR cycle feeding NAACL 2027 and COLING 2027, primary chosen at commitment) is in `reports/2026-09-05_berend_report.md` §2.3 and still describes the mechanism |

**Backup:** 450 MB / 636 files at `~/rag-backup/checkpoints` on Fedora — the only
copy of $42.73 of paid work outside gpu1. **Verified byte-identical to gpu1 on
2026-09-08.** Refresh after any run that changes a number in the paper, not only
after a paid one:
`rsync -avz szte-gpu:rag_faithfulness/checkpoints/ ~/rag-backup/checkpoints/`
A dry run (`-avzn`) first shows exactly what would move; the 09-08 refresh
transferred one 358-byte file, so a "stale backup" claim is worth checking
before it is acted on.

**Correctness is now measured, and it moved the paper.** Containment is biased
against the judge and **the sign of the bias is not constant**: it understates
accuracy by **5-16 points in 20 of the 24 cells where both signals exist** but
*overstates* by **2-6 points** in the four HotpotQA cells under Claude, where
long multi-hop answers mention a reference string while asserting something
else (false positives 8.9-13.0% there against 5.8-7.3% false negatives). ⚠️ The
earlier figures (5-15 in 12 cells, overstating 3-7) were a 2-generator
measurement and the overstatement predates D4, which moved 80 Claude/HotpotQA
rows to refusal; re-measured over all 24 judged cells 2026-10-16. **mistral has
no `_scored` checkpoints**, so it is not in this comparison — the free
correctness phase was never run on it, and running it would rewrite the 24
published `_scored` files. Containment is a bound in **neither** direction — do
not describe it as one. The judge never grades a refusal correct, so B12 is
fully resolved. Two draft claims did not survive: **"more
faithful when wrong" is false on NQ for both generators** (faith_gap -0.051 to
-0.115), and the "retrieval NOT SUFFICIENT" cell is **~80% refusals**, putting
the grounded-and-wrong rate at 4.5-7.9% rather than 23-45%. See
`reports/2026-08-30_judge_full_grid.md`. Faithfulness scoring is untouched, so
the evaluator-dependence spine stands.

**The evaluator-dependence spine was re-checked after the judge run and holds**
(`reports/2026-08-30_evaluator_dependence_recheck.md`): the 2026-08-26 numbers
reproduce exactly, and 2 of 4 nulls still break under AlignScore (**superseded:
now 4 of 24, in 3 of 8 cells — see B28**). But the
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
laptop ──ssh──> hop (SZTE_HOP_IP) ──ssh gpu1──> gpu1 (NVIDIA V100)
                user: <your-username>                    alias pre-configured by Berend
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
| hop hostname | `nlp` (SZTE_HOP_IP), key auth with `~/.ssh/id_ed25519` works |
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
- **`gpu1` is an alias that exists ONLY in the hop's `~/.ssh/config`** (`host gpu1 → hostname GPU1_PRIVATE_IP`). It is not in DNS or `/etc/hosts`. `ssh gpu1` works from the hop because SSH reads that config, but **ProxyJump uses `ssh -W gpu1:22`, which does a literal DNS lookup and ignores Host aliases** → `Temporary failure in name resolution`. A working `~/.ssh/config` must set `HostName GPU1_PRIVATE_IP` for `szte-gpu`. This was the cause of the "ProxyJump flakiness" noted earlier — it was never flaky, it was always wrong.
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
related-work paragraph. **Full PDF read 2026-09-05** (`reports/2026-09-05_salemi_zamani_full_read.md`); the DOI is
verified against Crossref (pp. 2395-2400). **The sentence above is too strong.** Their
weak-correlation result covers *human provenance* and *LLM relevance* labels; their
strongest baseline is **answer-containment** labelling — the scheme our qrels use after
B6/B8b — which holds tau 0.31-0.40 on the QA sets. It is measured at **k=50**, where their
own Fig. 1 puts the correlation at its lowest (it declines with k); we use k=5. And `E_M`
is always a match against gold `y`, so **every number they publish is a correctness
number** from a fine-tuned 60M FiD reader that cannot abstain. Less of our premise is
pre-empted than this section assumed; the positioning conclusion is unchanged.

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
| [7] "Each to Their Own" | "Chen, J.", arXiv:2507.xxxxx | **the title was also wrong — see §6b** |
| [9] FaithJudge | "Es, S. et al." | **Tamber, M. S. et al.** — EMNLP 2025 Industry Track (**title also wrong — see §6b**) |
| [13] CTRL-RAG | "Luo, X.", arXiv:2602.xxxxx, "contrastive likelihood training" | **Tan, Z. et al.** — arXiv:**2603.04406**, "Contrastive Likelihood **Reward Based Reinforcement Learning**" |

In-text mentions were updated too (Zhang→Sinha, Wu→Sun, Es→Tamber, Luo→Tan).

### 6b. Four MORE wrong entries found 2026-09-03 (revisions B3) — bibliography rebuilt

The §6 pass fixed *authors*. It did not check *titles*, and four were wrong. Every
one of the 24 entries has now been verified against arXiv/ACL Anthology directly
(ID resolves, venue and status confirmed, abstract read to confirm it says what
it is cited for). The paper's bibliography is the verified list; this table is
what changed.

| Ref | Was (WRONG) | Is (CORRECT, verified) |
|-----|-------------|------------------------|
| [13] (was [7]) | "Each to Their Own: Matching Retrievers to Generators", Chen, S. | **arXiv:2507.17442 is a different paper entirely**: S. Chen, Z. Zhao, J. Chen, "Confident RAG: … Mathematics Question Answering through Multi-Embedding and Confidence Scoring". The ID was right, the title was invented. In-text description corrected and now names the maths-QA domain |
| [4] (was [9]) | "FaithJudge" as the paper title | FaithJudge is the **framework**, not the title. Paper: Tamber, Kazi, Sourabh, Lin, **"Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards"**, EMNLP 2025 Industry Track, pp. 799-811, arXiv:2505.04847 |
| [12] (was [6]) | "The Semantic Illusion in Retrieval-Augmented Generation" | Full title: **"The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems"**, D. Sinha, arXiv:2512.15068 |
| [14] (was [11]) | DeBERTa — He, Liu, Gao, Chen, ICLR 2021 | The scorer is `nli-deberta-**v3**-large`, a different paper: **DeBERTaV3**, He, Gao, Chen, ICLR 2023, arXiv:2111.09543 |

Also confirmed correct and unchanged: [11] ReDeEP (Sun et al., ICLR 2025,
arXiv:2410.11414 — full title adds "via Mechanistic Interpretability") and
[24] jina-embeddings-v3 (Sturua et al., arXiv:2409.10173), which closes the
"jina-embeddings-v5-text" item flagged in §2.

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

### ⚠️ Six defects found on 2026-09-01 by the external audit brief (fixed — do not regress)

Full write-up with every before/after number:
`reports/2026-09-01_fixes_implementation.md`.

**B14 — `perturbation_check.num_in_text` matched a number as a SUBSTRING.**
`num_in_text('1000', 'the total was 10000 USD')` was True. Both falsification
eligibility rules run through it, so ungrounded values entered the sample (a
value the context never supported cannot lose entailment when falsified, so
those cases arrive with a ~0 delta and drag the mean toward the reported
conclusion) and eligible cases were silently discarded. Now matched at
numeric/word boundaries — `\b` is WRONG here, it treats `,` and `.` as
boundaries and reinstates the bug on `1,000` and `3.55`. Eligible cases over
the grid: 5,123 → 5,371.

**B15 — `textnorm.contains` had the same defect one level up.**
`contains('the budget was 10000 dollars', '1000')` and
`contains('Alice went to Paris', 'Ali')` were True. It decides the NQ retained
sample, `build_qrels` relevance and `correctness.contains_answer`. Now a
token-sequence test (both sides space-padded). Impact measured: **0** NQ
samples change status, 7/1000 NQ queries change their relevant chunk set (NDCG
moves in the 4th decimal, no TOST verdict moves), 166/16,000 correctness rows
flip and agreement with the judge improves 0.8340 → 0.8360. An existing test
asserted the defect as deliberate ("gold spans are frequently sub-token"); the
justification is wrong, because `squash()` splits on punctuation, so the real
sub-token cases are separate tokens and still match.
**`CORPUS_VERSION` was deliberately NOT bumped** — the corpus and the retained
samples are byte-identical and bumping strands 16,000 paid generations. The
old qrels/per-query/aggregate checkpoints are archived in
`checkpoints/n1000_v3/superseded_2026-09-01/`.

**B16 — `split_claims` never split a sentence that OPENS with a digit.** The
boundary test required `nxt.isupper()`. Selective on numeric-heavy answers,
i.e. exactly the ones the falsification probe uses. Fixed with
`nxt.isdigit()`. Impact is small — 32/16,000 answers, means 2.992 → 2.996
(Claude/NQ) and 1.176 → 1.176 (GPT/NQ) — so **Table III's gradient is not an
artifact of it**.

**B17 — there were TWO `is_abstention()` functions and they disagreed on 365
of 16,000 rows.** `perturbation_check` matched 9 phrases as a substring
ANYWHERE; `correctness` matched 4 phrases anchored with `startswith` on
SQuAD-normalised text. Section V and Section VI therefore called different
populations "answered". The canonical rule now lives in **`src/abstention.py`**
and both import it; `correctness`'s anchored rule is canonical because 253 of
the 365 disagreements (69%) are substantive answers carrying a mid-text hedge.
Adopting it changes no published number (`correctness`-only was 0 rows).
`normalize_answer` moved to `textnorm.squad_normalize` (aliased in
`correctness`) so `abstention.py` can share it without a circular import.

**B18 — the NLI hypothesis carries markdown for one aggregate and not the
other.** `nli.score_chunks` scores the answer exactly as written, `**bold**`
included; `claim_faithfulness.score_claims` runs `strip_markdown` over claims
first. On single-claim answers the two therefore read different strings: 244 of
1,615 such cases differ, by up to **0.73** on one `**Graduados**`. Markdown is
**74.9%** of Claude's answered rows and **0.3%** of GPT-4o-mini's, so the
paper's cross-generator sensitivity gap is partly confounded with formatting.
`src/markdown_check.py` sizes the confound; **the fix — strip markdown in both
places, or neither — is NOT applied**, because it would move every `nli_max`
number for Claude. Decide before the next paper pass.

**B19 — a chunk citation is not an asserted quantity.** The prompt numbers the
retrieved chunks and Claude cites them ("According to Chunk 4, …"); it also
writes ordered lists. `numeric_grounding_check` counted those as ungrounded
values, which alone accounted for a false-positive rate of 39.5% on
HotpotQA/Claude (6.0% after excluding them via `content_numbers()`). The same
contamination is in the falsification sample: `build_number_case` picks the
first grounded number, and that is a citation or list marker in **9.5%** of
HotpotQA/Claude and **5.0%** of NQ/Claude cases. Case construction was
deliberately NOT changed — every scorer must see the same cases — so
`perturb_report.by_value_role` reports the delta split by what the falsified
value actually is.

### The four open decisions — RESOLVED 2026-09-03

Full write-up, including the Part A/B revisions pass and every before/after
number: `reports/2026-09-03_revisions_and_decisions.md`.

`reports/2026-09-01_fixes_implementation.md` closed with four decisions. All
four are now taken. Two required re-running the falsification probe and the
evaluator comparison; they were made together and re-run **once**, because both
change which cases exist.

**D1 — markdown is now stripped for EVERY evaluator, and it cost a cell.**
`nli.score_chunks` scored the answer as written, `**bold**` included, while
`claim_faithfulness` stripped it first, so the two aggregates read different
hypotheses. Sized first with `markdown_check` (n=300/cell): stripping lowers
Claude's `nli_max` by 0.005-0.012 on NQ and 0.021-0.043 on HotpotQA, moving
>0.05 on 4.7-9.7% and 17.7-19.3% of answers respectively.

`strip_markdown` moved to **`textnorm`** (`claim_faithfulness` imports `nli`,
so `nli` cannot import it back — same reason `squad_normalize` moved in B17)
and is applied in `nli.score_chunks`, in the AlignScore arm of
`faithfulness.py`, and once at `_Scorer.score` in `perturbation_check` —
that last one matters, because `nli_concat` and `align` call their models
directly and would otherwise still read the raw string. It is idempotent.

**⚠️ THE HEADLINE IS NOW 1 OF 4 CELLS, NOT 2.** NQ/Claude under AlignScore goes
from spread 0.0179 / p_holm 0.0040 to **0.0114 / p_holm 0.0567** — significant
uncorrected (p=0.0081), not after Holm. Only HotpotQA/GPT-4o-mini survives
(p_holm 0.0040). That cell's apparent evaluator-dependence was substantially a
formatting artifact from one generator. Written up as **Section VII-B**, a
correction reported as a finding, at the author's direction.

The single-assertion identity confirms the fix: the markdown group went from
**131 of 131 differing, max 0.729** to **7 of 140, max 1.5e-05** — the same
batch-padding noise as the unformatted group. Table III's single-assertion row
is now numerically equal under both aggregates (+0.102 Claude, +0.485 GPT).

**Do not un-strip one arm without re-running phases D and E and the whole
probe.** Superseded checkpoints: `checkpoints/n1000_v3/superseded_2026-09-03_markdown/`.

**D2 — case construction is now content-only.** `build_number_case` took the
first GROUNDED number, which is a chunk citation or ordered-list marker in 9.5%
of HotpotQA/Claude cases; falsifying "Chunk 4" into "Chunk 7" is not a
falsification and those cases carried a **negative** mean delta (-0.0625). It
now takes the first CONTENT number (`value_role() == 'content'`). Filtering the
old rows showed the direction in advance: HotpotQA/Claude +0.0734 -> +0.0850,
NQ/Claude +0.0426 -> +0.0442, GPT cells unmoved. The re-run restores n to 200
per cell instead of dropping 6.6-8.8% of Claude's sample.

**D3 — `CORPUS_VERSION` stays at v3. This is a deliberate, documented deviation
from the §8 convention**, which says to bump whenever corpus or relevance
semantics change. The F2 fix changed the chunk-relevance rule for **7 of 1000**
NQ queries; NDCG moves in the 4th decimal and **no TOST verdict moves**. The
corpus, the chunks and the retained samples are byte-identical. Bumping re-keys
`checkpoints/n{N}_{CORPUS_VERSION}` and would strand 16,000 paid generations
plus the judge run — **$42.73 re-spent to change a 4th decimal**. The old
qrels/per-query/aggregate checkpoints are archived at
`checkpoints/n1000_v3/superseded_2026-09-01/`. If the relevance rule changes
again for a reason that moves a reported number, bump then.

**D4 — the canonical refusal rule now strips the preamble.** Both generators
prepend "Based on the provided context," to everything, refusals included, so
the bare anchored rule graded genuine refusals as attempts. `is_abstention` now
strips a leading "Based on ...," / "According to ...," before the anchored test,
over the wider phrase list. **Audited over all 16,000 generations before
adoption: 107 rows move from attempt to refusal (0.67%), ALL Claude's — 80
HotpotQA, 27 NQ, none GPT-4o-mini.** 105 open with "I cannot answer this
question". Mid-answer hedges are still answers. The pre-change rule survives as
`is_abstention_bare` so the difference stays measurable. This moves the
answered-only population behind Sections V-VIII, which is why the probe and
`compare_evaluators` were re-run rather than carried over.

(The report's count for D4 was 112; the audit measures **107**. The difference
is rows excluded as `[ERROR]`/None before the rule is applied. 107 is the
number produced by running the adopted code over the grid.)

---

### The two 2026-09-04 closures (report: `reports/2026-09-04_copying_and_multiplicity.md`)

The 2026-09-03 session closed by flagging two unfinished items. Both are done.

**B20 — the verbatim-copying control had no home in `src/`, so it went stale.**
Section V-B's refutation (overlap 0.234 both generators, corr +0.041/+0.161)
came from an uncommitted 2026-08-25 script, which is why it was the only
Section V number not regenerated after D1/D2 — both of which move the deltas it
correlates against. Rebuilt as **`src/copying_check.py`**. **The conclusion
holds and the figures reproduce to within 0.005**: overlap 0.226 Claude / 0.235
GPT-4o-mini, corr +0.046 [-0.004, +0.096] and +0.166 [+0.118, +0.212] over
3,173 cases. Three corrections to the draft: the overlaps are **not identical**
(and GPT-4o-mini copies slightly MORE, the wrong way for the hypothesis);
**Claude's correlation is indistinguishable from zero**, so "positive where the
explanation predicts negative" was too strong for that arm; and the real
refutation is the **matched-overlap bands** — Claude +0.052..+0.085 against
GPT-4o-mini +0.326..+0.507, a fivefold separation in *every* band including
above 0.3. Two choices are now explicit because they decide the number:
`squash` tokenisation (which neutralises markdown, so `raw/squash` and
`stripped/squash` are byte-identical and the measure is **provably insensitive
to D1**), and answers under 5 tokens **excluded, never scored 0** — GPT-4o-mini
writes far more of them, so scoring them 0 would manufacture the very
cross-generator difference the control tests for.

**B21 — the headline count depends on the multiplicity family, and that is now
printed.** NQ/Claude/align is p=0.0081 uncorrected and Holm puts it either side
of 0.05: **p_holm 0.0567 over the 8 tests in the table (k=1), 0.0243 within
evaluator (k=2), 0.0162 within cell (k=2)**. Within-evaluator is defensible on
the statistics — the 4 NLI tests establish which cells are null, so they
precede the question rather than compete with it, and the family at risk of
false positives is the 4 AlignScore rejections. **Table-wide is kept anyway**:
it was fixed in advance (A5) and is the most conservative, and switching to a
smaller family after seeing that it restores a cell is a forking path whatever
its a-priori merit. `compare_evaluators._report_holm_scope_sensitivity` now
prints all three families and any swing cell on every run; Section VII states
the alternative. **The headline is still 1 of 4 cells.**

**✅ The backup is current (checked 2026-09-08).** This section previously
warned it was a pre-D1 snapshot from 2026-09-01. That was wrong by the time it
mattered: an `rsync -avzn` against gpu1 on 2026-09-08 found a single 358-byte
file out of date (`n1000_v3/hypothesis_summary.pkl`), and everything behind the
current paper — the D1/D2 re-run and the whole qwen arm — was already mirrored.
The rule is still widened to "after any run that changes a number in the paper".

---

**B22 — claim-level now covers the full grid, and the evaluator family is 12.**
`claim_scores_*` exist for all 16 cells (16,000 rows, 0 NaN, 280,548 NLI pairs).
`compare_evaluators --metrics` already defaulted to `nli,align,claim`, so claim
was being **silently skipped for want of checkpoints** — running it changes
Table VII from 8 rows to 12 and every `p_holm` with it. **The headline count is
unchanged at 1 of 4**; the surviving cell (HotpotQA/GPT-4o-mini, AlignScore)
goes p_holm 0.0040 -> **0.0060** on the larger family. Table-wide Holm is kept:
applying it to a 12-row table is the same rule, not a new one, because the paper
always specified three evaluators — 12 is the design family and the earlier 8
was incomplete data.

Three things the full grid adds:
- **The same cell breaks under TWO evaluators.** claim-min gives p=0.0070 there
  (second-smallest in the table), so the result is not AlignScore-specific.
- **claim-min has a larger spread and smaller p than nli-max in ALL FOUR cells.**
  Same DeBERTa checkpoint — the aggregation choice is doing the work.
- **Equivalence moves with the evaluator too**: at ±0.03 on NQ/Claude, nli-max
  calls 6/6 pairs equivalent and claim-min 2/6.

The Table III mechanism reproduces at n=16,000: the claim_min/whole_max gap is
~0.24-0.27 for Claude (2.4-3.0 assertions/answer) and ~0.01-0.02 for
GPT-4o-mini (1.0-1.2). **The paper is NOT yet updated for any of this** — see §5
of the report.

**⚠️ `nli.NLIScorer` batch size is now adaptive** (`config.NLI_BATCH_SIZE`,
`RAG_NLI_BATCH` to pin). gpu1 is shared: a co-tenant holding 27 of 32 GB leaves
room for batch 2, not 16, and the hardcoded 16 dies on the first forward pass.
It shrinks on OOM and grows back. **The OOM retry re-runs the failing slice
rather than skipping it** — `score_claims` slices the returned probabilities by
block, so a short return would attribute one claim's score to another and every
`claim_min` would be silently wrong. Pinned by `tests/test_nli_batching.py`.

---

**B23 — the open-weight arm is run, and it moved the headline UP.**
Qwen2.5-7B-Instruct, 8 cells / 8,000 answers, fp16 on the V100, $0. Scored with
all three evaluators + correctness + the falsification probe. Report:
`reports/2026-09-05_open_weight_arm.md`.

**Two bugs had to be fixed first, both because this arm had never run:**
- `LocalHFGenerator.generate` **failed on the first call**.
  `apply_chat_template(return_tensors='pt')` returns a `BatchEncoding` in
  transformers 5.x (gpu1 has 5.14.1) and a bare tensor in 4.x, so a dict was
  passed positionally into `model.generate`. Now `return_dict=True` + `**enc`.
- `run_phase_llama(model_names=None)` meant a **hardcoded 3 models**, not all,
  unlike every other phase. `run_pipeline` passes None when `--models` is
  omitted, so the first pass produced **6 cells, not 8**, dropping
  text-embedding-3-small. That is *incomparable*, not merely smaller: this
  arm's embedder spread would be over 3 systems against the others' 4, and
  nothing downstream would say so. Both pinned by tests.
- Needed `pip install --user accelerate` (for `device_map='auto'`).
- **`config.ACTIVE_GENERATORS`** is now the one list every analysis CLI builds
  its `--generators` default from. All four were hardcoded to
  `claude,gpt4omini` — B10 one layer up.

**Results.** Assertions/answer: Claude 2.728, **qwen 1.584**, GPT-4o-mini 1.117.
NLI-max falsification detection lands between the other two in both datasets
(NQ 30.0% vs 4.8%/51.1%; HotpotQA 29.8% vs 9.9%/40.3%) — **predicted before the
arm was run**. Evaluator ordering AlignScore > Claim-min ≳ NLI-max is identical
in all six cells.

**⚠️ TABLE VII IS NOW 18 ROWS AND THE HEADLINE IS 2 OF 6, NOT 1 OF 4.**
HotpotQA/GPT-4o-mini/align p_holm **0.0090** and HotpotQA/qwen/align p_holm
**0.0187** both survive Holm over 18. Adding a generator made the family
stricter and the count still rose.

**The strongest result in the project: under AlignScore on HotpotQA all three
generators give the IDENTICAL embedder ordering** (all-mpnet < text-emb <
BGE-M3 < E5-large-instruct); under NLI-max no two agree. State it carefully —
the generators are independent but the *retrieval is shared*, so this is three
generators agreeing about a property of the retrieval, NOT three independent
replications.

**Nuance for the paper:** conditioning on assertion count, qwen tracks
GPT-4o-mini (+0.410 vs +0.447 at one assertion) and **Claude is the outlier**
(+0.129). The aggregate intermediacy comes from qwen's assertion
*distribution*. The residual is Claude-specific, not a smooth verbose/terse
axis — only a third generator makes that distinguishable.

Copying dies a third time: qwen has the **lowest** overlap (0.194 vs 0.226 /
0.235) with intermediate sensitivity. Numeric check: 100% recall in all 8 qwen
cells at 1.5-5.5% FPR. **Judge correctness NOT run on this arm (~$15).**

**B24 — the paper IS now updated for B22 and B23 (2026-09-08), and the update
found two stale tables.** Every reported number was regenerated from
`~/rag-backup/checkpoints/n1000_v3` in one pass rather than spliced from the two
reports. The 12 pre-existing Table VII rows, both Table III columns, the copying
control, Table II's 12 rows and the refusal counts all reproduced **exactly** —
so the two that did not moved for a substantive reason:

- **Table IV was pre-D1 and pre-D4**, and the mechanism behind it is
  **generator-dependent**. Under NLI-max, Claude's HotpotQA refusals average
  0.272-0.353 against 0.693-0.712 for its attempts, but qwen's average
  0.846-0.903 and GPT-4o-mini's 0.905-0.940 — *above* their attempts. So pooling
  refusals makes the pooled column track NDCG@5 **positively** for Claude
  (rho +0.99) and **negatively** for qwen and GPT (-0.98, -0.92); answered-only
  spread collapses to 0.003-0.029 in all three. Do not restore the old claim
  that pooling "reproduces the retrieval ranking" — it reproduces it *or
  inverts it*, and the sign is set by how the evaluator scores that generator's
  refusal sentence.
- **Table VIII had two stale cells** (Claude/align -0.40 → **-0.20**,
  HotpotQA/Claude/nli +0.40 → **+0.20**). The clean "all HotpotQA positive, all
  NQ zero-or-negative" pattern holds in all 12 NLI-max and AlignScore
  combinations and **breaks under claim-min** (NQ/qwen +0.63, HotpotQA/qwen
  -0.40). The +0.63 is not a typo: 4 systems can only give multiples of 0.2
  *without ties*, and two systems tie there.

Also settled in that pass: eligible/scored falsification cases are **4,784 of
7,609** (was 3,184 of 5,272); refusals **5,776 of 24,000 (24.1%)**; markdown on
answered rows **76.0% / 1.3% / 0.3%** (the old 74.9% predates D4); the
single-assertion identity check is **n=2,472, max 1.66e-05**; the numeric
grounding check is 100% recall in all six pooled cells at **0.8-5.2% FPR**.
**Content is now ~11.5 pages against ARR's 8** — cutting is the next job.

---

**B25 — the verbosity mechanism is now CAUSAL, shown within one model.**
Report: `reports/2026-10-07_reviews_and_verbosity_control.md`. The review's W2
was that assertion count is confounded with model family. Fix: run Qwen2.5-7B
twice over identical queries, weights and retrieved contexts with **one line**
added to the prompt. 10 cells, 10,000 answers, $0, label `qwen_verbose`.

Assertions 1.77 -> **4.90** (NQ) and 1.54 -> **4.99** (HotpotQA); words 34 -> 108
and 26 -> 110. That overshoots Claude's 3.08, so the arm EXTENDS the axis.

**Detection falls in all six evaluator x dataset cells, no CI overlapping:**
NLI-max 35.1->10.4 and 54.8->17.4; claim-min 48.1->28.9 and 66.8->43.6;
AlignScore 51.8->16.0 and 74.5->18.5.

Three things it turned up that the run was not bought for:
- **AlignScore is not an NLI model and falls furthest** (74.5->18.5, mean
  response 0.494->0.127). So the structural bias attaches to **scalar
  aggregation over a multi-assertion answer**, not to entailment — a
  fine-tuned judge inherits it. This is the backbone of the new §2.4.
- **claim-min keeps sensitivity and loses thresholdability.** Degrades least,
  but absolute means collapse 0.693->0.234 (NQ) and 0.424->0.076 (HotpotQA).
  A minimum over five assertions sits near zero whatever the answer says.
- **The scale floor rises**: unrelated-document baseline 0.245 -> **0.509** on
  HotpotQA, i.e. ABOVE the fixed 0.5 gate. Verbosity degrades a fixed threshold
  twice over. Floor-anchored gating, reached by manipulation.

⚠️ **The pre-registered prediction FAILED and that is the finding.** An arm at
4.9 assertions was predicted to fall below the 5.1% of a generator at 3.08. It
did not — **10.4%**. Answer structure explains most of the gap, not all, and
the residual is **generator-specific** (same conclusion as B23 reached from the
opposite direction). Do not claim monotonicity in assertion count across models.

⚠️ **`qwen_verbose` is NOT a fourth generator.** Same weights. Holding the model
fixed is what makes it a clean test of the length/identity confound and exactly
what stops it saying anything about models trained differently. Counting it as a
generator puts the same weights behind two rows. The verbosity **axis** is now
measured 1.17–4.99; the model **population** is still three.

**B26 — entity substitution and power were already measurable and unreported.**
- **Entity cases had always been built** (`build_entity_case`) and the
  checkpoints had always held them. The review's heaviest weakness (only one
  error type) cost $0 to answer. Generator ordering reproduces in **5 of 6**
  cells; NQ/claim-min inverts the two more verbose generators. The two error
  types are **not interchangeable** — they agree within 0.05 for the tersest
  generator and diverge in opposite directions for the other two.
- **Power analysis:** MDE at 80% power is **0.012–0.060**, and in **16 of 18
  cells it exceeds the spread actually observed**. Both significant cells clear
  their own MDE. New: **AlignScore has the lowest MDE in all six cells** because
  its paired per-query differences are far less noisy (sd 0.12–0.30 vs
  0.20–0.48) on identical answers. So part of the evaluator-dependence is
  **resolving power at fixed n**, not only response to a known error. The design
  does not separate them and §7 now says so.

**B27 — two population/convention slips, same class as B13.**
- Limitations gave the generator span as 1.12–2.73 (**all rows, incl. refusals**)
  while the abstract gives 3.08/1.77/1.25 (**answered-only**). Fixed 2026-10-07
  to 1.17–2.77. **Every assertion figure in the paper is now answered-only.**
- A draft sentence blended a pooled across-dataset word count with a
  HotpotQA-only one. Now per-dataset.
  **Rule: never quote a pooled figure beside a per-dataset one for the same
  quantity.** This is the third instance; B13 was the first.

**B28 — the fourth generator is run, and the dissent is no longer
AlignScore's.** Mistral-7B-Instruct-v0.3, 8 matched cells + 2 jina cells,
10,000 answers, $0, label `mistral`, run through the same `LocalHFGenerator`
via `--open-label`/`--open-model-id` so prompt and decoding are byte-identical
to the qwen arm. Scored with all three evaluators and the falsification probe
(number + entity, 3 scorers x 10 cells). Chain log: `outputs/logs/mistral_chain.log`.

**⚠️ THE HEADLINE IS NOW 4 COMPARISONS IN 3 OF 8 CELLS, OVER 24.** The family
grew 18 -> 24 and the count rose anyway, for the second time. Survivors:
HotpotQA/mistral/**nli** p_holm 0.0072 (spread **0.0796**, the largest in the
table), HotpotQA/gpt4omini/align 0.0115, HotpotQA/qwen/align 0.0242,
HotpotQA/mistral/**claim** 0.0378. The two pre-existing cells reproduced
(0.0005 and 0.0011 uncorrected).

**The important structural change.** Every break before this was AlignScore's,
which invited the reading that AlignScore is simply the sharper instrument.
Mistral breaks under the two NLI-based aggregates and **not** under AlignScore
(p_holm 0.3330). So *which* evaluator dissents is generator-dependent.

**But it is still a resolving-power story, not a disagreement story**, and the
mixed model is what settles that: the system x evaluator interaction is
**undetectable in all 8 cells** (p 0.31-0.96). Both mistral cells sit at 0.96
(NQ 0.9649, HotpotQA 0.9616), so the weakest evidence of interaction anywhere
is on the arm with the largest spread. ⚠️ **tab:interaction carried only the 6
three-generator rows until 2026-10-16** — B28 recorded the 8-cell result and
the table was never extended; the two mistral rows are now in it (n=8,304 and
n=5,076). After
standardising each evaluator's scale the three see the same pattern across
systems; they differ in whether it clears significance. Do not write
"the evaluators disagree"; write "no detectable interaction".

Everything else that moved, all re-measured at 8 cells:
- **Standardised effect** (C1): AlignScore > NLI-max in **5 of 8**, was 5 of 6.
  Mistral reverses it in BOTH datasets, and the largest value in the table is
  NLI-max's 0.177 on HotpotQA/mistral.
- **Omnibus** (C3b): permutation re-selecting extremes **2 of 24** (was 1 of
  18), Friedman **1 of 24** (was 0) — HotpotQA/mistral/nli is the first cell in
  the project to survive all three tests. Friedman reproduced to 4 dp;
  permutation p for the gpt cell moved 0.0008 -> 0.0015 because a fourth
  generator consumes the shared RNG stream, not because data changed.
- **Power**: MDE 0.012-0.064, median 0.036, above the observed spread in **20
  of 24**. AlignScore still has the lowest MDE in **all 8** cells. Exactly the
  4 Holm survivors clear their own MDE and no others do.
- **Refusals**: **8,055 of 32,000 (25.2%)**, 11.8% (NQ/qwen) to **39.0%**
  (HotpotQA/mistral — the highest cell in the grid). The 3-generator subset
  still sums to **5,776**, so the old figure reproduced exactly.
- **Assertions, answered-only**: mistral NQ **1.724**, HotpotQA **1.410**. The
  published 3.08 / 1.77 / 1.25 reproduced to 3 dp. The generator span stays
  **1.17-2.77** pooled per generator, so Limitations needed no change.
- **Refusal-rule sensitivity**: all three rules select the **same 4
  comparisons**; nothing breaks under one rule and not another.
- **Entity substitution**: the number-vs-entity generator ordering reproduces
  in **3 of 6 rows**, down from 5 of 6. Every failure is a transposition of
  qwen and mistral, which sit within 0.03 of each other. The ordering of the
  **extremes** (Claude lowest, GPT-4o-mini highest) holds in every row.
- **Pairing / red flag 3**: spread moves by at most **0.0221** (the published
  0.022 reproduced), ordering changes in 11 of 24. ⚠️ The old claim that the
  ordering is unchanged in *every* surviving cell is **false** now:
  HotpotQA/mistral/claim-min reorders, so no ordering is read off it.

**The jina positive control, re-run at four generators and five systems**, goes
from 4 comparisons in 3 of 8 cells to **8 in 5 of 8** — not to every cell, as it
did at three generators (6 of 6). Three cells stay null (NQ/gpt4omini,
NQ/mistral, HotpotQA/claude). What changes is *which* evaluator participates:
with jina in, NLI-max breaks on NQ/qwen and claim-min in three further cells.
Admitting one genuinely worse retriever makes the common evaluator register
differences it otherwise calls null.

⚠️ **`qwen_verbose` is still not a generator** (B25) and `mistral` is not a
prompt control. `config.ACTIVE_GENERATORS` now has four labels and
`OPEN_MODEL2_ID/LABEL` name the arm; seven `scripts/` and four `src/` lists had
hardcoded generator tuples and all were threaded. Mistral's jina cells had claim-min and
AlignScore but no NLI, because generation outran the chain's 8-cell gate; **NLI
was backfilled 2026-10-16** (`--phase d` — `--phase e` is AlignScore and
silently skips, exit 0, when the align env is absent), so all four generators
now cover all five embedders under all three evaluators. **The LLM judge was run on mistral
2026-10-16 and the correctness results are now four-generator** — see B29.

**B29 — the judge now covers mistral, and the correctness results are
four-generator.** 8 matched cells, 8,000 rows, `claude-opus-5` at
`effort: low` — the same model and the same byte-identical prompt as the 24
existing cells, because correctness figures across arms are otherwise not
comparable. **$13.74** total ($0.7159 calibration + $13.02), **0 errors, 0
ungradable**, 2026-10-16. Run as **four parallel workers over disjoint
checkpoints** (one per embedder, `--budget 6` each): sequential was 37 rows/min
and ~3.4 h. Two cells must never be shared between workers — each worker saves
its whole `done` dict, so the later save would drop the other's rows and pay
for them twice. Judged files were written laptop-side and rsynced **up** to
gpu1 (the reverse of the usual direction); both trees verified at 8,000 rows.

**Judge coverage is the matched grid and nothing else.** 32 judged checkpoints
= 4 generators x 4 systems x 2 datasets. jina (13 files), QASPER (15), C1
`norag_*` (6), C2 `oracle_*` (6) and the 48 `ctx_*` ablation files are **still
containment-graded**, so the accuracy figures in the control, QASPER and
floor/ceiling sections are not comparable with the grid's. The Limitations item
that used to say "three of the four generators" now says this instead.

What moved:
- **`faith_gap` over one family of 32: 8 significant, all negative** (was 5 of
  24). 5 positive-signed cells, none with a CI excluding zero. Correcting
  **within each arm's 8 tests gives 9**, so the count depends on the family;
  the paper reports the stricter one and names both.
- **E5-instruct on NQ is negative under all four generators** (-0.105, -0.115,
  -0.123, -0.125) and survives the family correction under **three** — Claude's
  p_holm is 0.0720, just outside.
- **mistral is the least accurate arm**: NQ 0.675-0.709, HotpotQA 0.433-0.535.
  Its HotpotQA hit-and-wrong cell is **48.4%**, the largest in the grid.
- The 2x2 ranges all widened: cell size **17-48%** (was 17-36), refusal share
  of that cell **50-87%** (was 69-87), committed-and-wrong **4.5-11.4%** (was
  4.5-7.9). The open-weight arms refuse least inside the cell and so leave the
  largest committed-wrong share, which means cell size and interpretable
  remainder do **not** order the generators the same way.
- `n_correct_abstained` is **0 in all 32 cells** — the judge still never grades
  a refusal correct.

**⚠️ Six errors in the published paper found while doing this, none of them
about mistral.** All fixed 2026-10-16.
1. **QASPER's evaluator-ordering claim was wrong twice over.** It said the main
   grid orders "AlignScore above claim-min above NLI-max" and that QASPER keeps
   it "for GPT-4o-mini alone". By **mean score** the grid orders
   **AlignScore > NLI-max > claim-min** in all 8 cells — claim-min is a minimum
   over claims and is lowest everywhere, never in the middle. The sentence had
   imported the **falsification-response** ordering (B23, where claim-min *is*
   second) and compared it against absolute means. On QASPER the ordering holds
   for **Qwen and GPT-4o-mini** (the latter by 0.003) and inverts for **Claude
   alone**, which reads NLI-max 0.747 / AlignScore 0.625 / claim-min 0.249. Two
   quantities under one ordering claim — the B27 class.
2. **The `faith_gap` count's multiplicity family was mislabelled.** "Over all
   twenty-four judged cells ... finds six" is the **per-arm** correction summed
   across arms; over one family of 24 it is **five**. Same class as B21.
3. **"survives correction in each"** was true only under the per-arm family.
4. **`tab:interaction` had 6 rows, not 8.** B28 recorded the 8-cell result and
   the table was never extended. Added: NQ/mistral n=8,304 LR=1.41 p=0.9649,
   HotpotQA/mistral n=5,076 LR=1.47 p=0.9616.
5. **The omnibus appendix contradicted itself**: its closing line said "2 under
   a paired test, 1 under permutation, 0 under Friedman" while the paragraph
   above it said 4 / 2 / 1. The closing line was the 18-comparison version.
6. **"Intervals are 0.064 to 0.129 wide"** are **half**-widths. The widths are
   0.129 to 0.259. Now written as $\pm$0.064 to $\pm$0.129.
Also stale and fixed: jina described as "all three generators" twice (it is
four), "four of the six cells are null" (five of eight), the entity prose
saying "five of the six" against its own caption's three of six, and the
containment-bias range (above).

**Two traps worth remembering from this session.**
- **`conditional.py` defaults `--models` to every configured model**, which
  silently admits jina and changes the published numbers: Claude/NQ's
  hit-and-wrong cell reads **18.2%** with jina in and **17.4%** without. Always
  pass the matched four when reproducing a table.
- **`pkill -f "llm_judge.py --judge claude"` matched its own shell**, because
  the pattern appears in the `bash -c` command line. It killed the wrapper
  along with the target. Use a pattern that cannot match itself
  (`llm_judge[.]py`).

**`cost_tracker.summary()` under-reported judge runs** and is fixed: judge
tokens accumulate in their own counters, so a judge-only run printed
`Claude in/out toks: 0 / 0` and `Claude cost: $0.0000` beside a correct
non-zero total. It now prints a Judge tokens line and a Judge cost line.

**B30 — the three open prose items are closed, and the prose pass found two
more published errors.** 2026-10-16, after B29. Nothing here needed compute or
money; two of the three were pure writing and the third was a lookup.

**Item 3, the citation.** `dasigi-etal-2021` is verified against the live
Anthology page for 2021.naacl-main.365: title, all six authors in order,
booktitle, year, pages and DOI match, and `month = jun` was missing and is
added. The bib entry carries the date and what was checked. Every one of the 31
entries has now had the B3 treatment.

**Item 2, review items C4, C9 and C10.** The review text is in
`Review of When the Evaluator.txt`; C1-C3, C5-C8 and C11 were answered in
earlier commits and these three were the remainder.
- **C4** — the spread statistic was justified as "what a practitioner is
  exposed to". It is not: someone who adopts one system loses the distance from
  it to the best, which equals max-minus-min only when they adopted the worst of
  the four. It is now presented as a deliberately extreme sensitivity check,
  the largest shortfall the choice can produce, and the paragraph points at the
  two omnibus tests that pay for the selection. **It also said "An omnibus test
  would be stricter; we do not run one", which Appendix~`app:omnibus`
  contradicts** — that appendix runs two. Fixed.
- **C9** — the claim-min recommendation now carries the cost it was omitting:
  absolute scores collapse to 0.234 and 0.076 at the verbose arm's verbosity, so
  a fixed grounding gate ported onto it rejects almost everything, and
  "resisting the structural bias and remaining thresholdable are separate
  properties" is now in the bullet rather than 200 lines away. The Definitions
  section also presented "an answer is only as supported as its weakest
  assertion" as definitional; it is now stated as the design choice it is, with
  the note that a minimum is the most severe aggregation available and is what
  drives those near-zero scores.
- **C10** — the constructed-versus-natural tension is reconciled where the
  margin is set. If natural hallucinations are harder to separate (Sinha), one
  real fabrication moves the score by less than the 0.049-0.095 we measure, so a
  margin anchored on constructed errors is **permissive, never strict**. The
  sign is fixed even though we cannot size it. Stated in `sec:equivalence`, in
  the Discussion bullet and cross-referenced from the Limitations item.

**Item 1, Berend's sentence-level pass.** Measured rather than eyeballed, with
a sentence-length scan over `paper.tex` + `sections/`. Sentences of 45+ words
went **28 to 0** and 40+ went **49 to 11**, the remainder being dense table
captions. 61 edits in all, none of which changes a number: the rule was to split
clauses and never drop a hedge, a population label or a scope qualifier, since
in this paper those are the content. Two comma splices and one comma standing
where a full stop belongs (Limitations item 1, "assertions per answer, The
four generators") were fixed on the way, and the two Limitations items that both
opened "Correctness is graded by..." now open differently.

**⚠️ Two more published errors, found by rewriting the sentences that carried
them.** Neither is about Mistral; both are the B27 class again.
7. **`sections/direction.tex` contradicted itself.** Paragraph 1 was updated for
   the fourth generator ("claim-min fails in four of eight, all four on the two
   open-weight generators") and **paragraph 2 was left at the three-generator
   version** ("two of its six combinations ... both on the open-weight
   generator"). Rewritten from `tab:8`: Qwen inverts the split on both datasets
   (+0.63 NQ, -0.40 HotpotQA) and Mistral-7B is positive on NQ (+0.80) and flat
   on HotpotQA. Drop claim-min and 15 of the remaining 16 follow the split; drop
   the two open-weight generators and all 12 do. Also "twenty-four
   generator-evaluator combinations" now uses the glossary's own unit name,
   dataset-generator-evaluator.
8. **"The evaluator ordering is stable across these six cells" is false under
   either quantity, and its own table shows it.** Counted off the published
   `tab:2`: the full ordering AlignScore > claim-min > NLI-max holds in **4 of 6
   by mean drop** (NLI-max edges claim-min on HotpotQA under Qwen, 0.270 against
   0.264, and under GPT-4o-mini, 0.391 against 0.389) and **5 of 6 by detection**
   (claim-min edges AlignScore on NQ/Claude, 25.1% against 24.1%). Two
   quantities under one ordering claim, which is exactly error 1 of B29. What
   survives, counted over all eight cells: **AlignScore has the largest mean
   drop in every one**, and **claim-min detects more than NLI-max in every one**;
   the full three-way ordering holds in **7 of 8 by detection and 5 of 8 by mean
   drop**. The entity probe reproduces the pattern, 5 of 6 by detection with the
   same NQ/Claude exception. The paper now names the quantity every time it
   states the ordering, in the body and in `tab:choose`'s caption.
   **The QASPER cross-reference attached to that claim was also wrong twice
   over**: it said the ordering "inverts for two of the three generators" on
   QASPER, but **the falsification probe was never run on QASPER** (0
   checkpoints, verified) so the response ordering was never measured there, and
   what `sec:qasper` actually reports is the **mean-score** ordering inverting
   for **one** generator. The paper now says the probe did not leave these two
   datasets.

**`tab:2full` was not "in full".** The falsification probe covers four
generators and both falsification tables carried three, while the Conclusion
quotes four values (0.054 / 0.251 / 0.261 / 0.451) — the fourth had no table
behind it. Mistral-7B's six rows are now in `tab:2full`, and **all eighteen
published rows were reproduced from the checkpoints to the last digit before
the new six were added** (orig, falsified, floor, detection at 0.5, Wilson
interval and floor-gate detection all matched). Mistral: NQ 0.818/0.567 det
34.3%, 0.695/0.411 det 48.2%, 0.856/0.415 det 51.8%; HotpotQA 0.604/0.317 det
52.5%, 0.470/0.209 det 59.6%, 0.768/0.287 det 71.5%. **n = 595 on HotpotQA**,
not 800: it has the highest refusal rate in the grid, so fewer answers carry a
grounded value to falsify. **`tab:2` now carries all four generators too** (2026-10-18,
at the author's direction). The first attempt, one row per generator-evaluator,
was reverted: 24 rows in a single column overflowed it by 20.6pt and cost
**three pages**. What works is the original six-row shape widened to four
column blocks --- `table*`, `\footnotesize`, `\tabcolsep` 4pt, 14 columns ---
which costs **one page** and no overfull box. Generators in both falsification
tables are now ordered by assertions per answer, most first (Claude, Qwen,
Mistral-7B, GPT-4o-mini), so the two extremes the bolding marks are the outer
blocks. All 72 values in the rendered table were checked back against the
checkpoint-derived figures: 0 mismatches. Two prose consequences: "the
open-weight generator falls between at 0.574" no longer identifies one arm and
now names both (Qwen 0.574 against a floor of 0.470, Mistral 0.567 against
0.492), and the intermediate-position prediction is attributed to Qwen, which is
the arm it was made for.

**The last two three-generator tables are closed (2026-10-18), and the ninth
published error came out of the second one.** Found by auditing all 17 table
environments for generator coverage; `tab:refusalrule` lists only the cells
that move, and `tab:ablation` and `tab:choose` are Claude-arm or per-evaluator
by design, so these two were the whole remainder.

Reproduce-then-extend, as for `tab:2full`. **All 30 published values of `tab:3`
and all 90 of `tab:9` reproduced exactly** from `~/rag-backup/checkpoints/n1000_v3`,
and so did the entire 24-comparison spread table behind `tab:7` — the four Holm
survivors at 0.0072 (HotpotQA/mistral/nli, spread 0.0796), 0.0115, 0.0242 and
0.0378, which is B28 to 4 dp. After the edit, 180 rendered values were checked
back against the reference: **0 mismatches**. Pages 30/30/31/31, 0 undefined
refs, 0 overfull boxes, 0 rendered `??`; 230 tests pass.

- **`tab:3`** (`app:assertions`) now has 13 columns. Mistral-7B responds by
  +0.385, +0.135, +0.077, +0.031, +0.082 under NLI-max and +0.385, +0.246,
  +0.080, +0.075, +0.139 under claim-min, on n = 422, 201, 97, 50, 30. **The
  old caption was Claude-only**: "under claim-level aggregation it does not
  [decay]" is true of Claude alone — Qwen, Mistral and GPT-4o-mini all still
  decline under claim-min, less steeply than under NLI-max. On Mistral at three
  and four assertions the two aggregates nearly coincide (0.077/0.080,
  0.031/0.075). Caption rewritten to say which generator does what.
- **`tab:9`** (`app:tost`) now has 24 rows. NQ/Mistral is 0/6, 0/6, 4/6, 6/6,
  6/6 under NLI-max, and **HotpotQA/Mistral is the only cell in the table where
  $\pm$0.10 still leaves a pair non-equivalent** — under NLI-max (5/6) and
  claim-min (5/6) both. That is the arm with the largest spread in `tab:7`,
  so the two readings agree.

**⚠️ Published error 9, and it is not about Mistral.** The body sentence "on
HotpotQA nothing is equivalent below $\pm$0.05 in any cell" was **already false
in the three-generator table it cited**: HotpotQA/Claude under AlignScore is
2/6 at $\pm$0.02 and 5/6 at $\pm$0.03, and HotpotQA/GPT-4o-mini is 1/6 at
$\pm$0.03 under both NLI aggregates. Counted over all 18 HotpotQA rows now,
**four contain an equivalent pair below $\pm$0.05** and none is equivalent at
$\pm$0.01. Same class as errors 1, 7 and 8: a claim stated over a table whose
own cells contradict it.

**Five prose figures were stale behind these two tables** and are fixed. The
abstract quoted the single-assertion response as "the three generators ...
0.102, 0.409 and 0.485" (now "the four generators ... between 0.102 and 0.485");
median answer lengths were "55, 25 and 15 words" for four generators (now 55,
25, 20, 15); single-assertion word means were "23.9, 19.5 and 16.1" (now
23.9, 19.5, 17.2, 16.1 — the three published values reproduced exactly on the
NQ probe sample, and are **not** the all-answered figures, which are 23.6/17.4/
14.7/15.6); the "third generator makes that residual legible" paragraph now
reads Mistral as a **replication** of Qwen (1.72 against 1.77 assertions,
+0.385 against +0.409 at one assertion, overlapping intervals — two separately
trained models agreeing on both quantities, with Claude the outlier at +0.102);
and the single-assertion identity check is **3,249 answers, max 2.33e-05** over
the four-generator grid. The published 2,472 at 1.66e-05 was confirmed to be
the three-generator figure before it was replaced, so the "7 of 140 formatted"
sentence is now marked as the grid as it then stood (it is 12 of 152 now).

**⚠️ A run has now silently done nothing, or silently dropped an arm, EIGHT
times**, each because a hardcoded default excluded what was being asked for:
B10 (`GENERATORS`), the QASPER `--datasets` default, `faithfulness.py` /
`claim_faithfulness.py` having no `--generators` flag, and five more found
while adding Mistral (B28) — three `scripts/` generator lists, a **second**
hardcoded list inside `entity_substitution_table.py` that kept the new arm in
the pickle but dropped it from the printed table, and `refusal_scores.py`
resolving `checkpoints/` relative to the cwd, so on gpu1 every load failed into
a bare `except: continue` and produced an empty frame. All are fixed, the
selected labels are **printed at startup**, and the two scripts that silently
returned nothing now raise with the resolved path named. Any new arm must have
its label threaded through *and* echoed in the first lines of the log.
**Two traps specific to gpu1:** the code tree (`~/rag-faithfulness`) and the
data tree (`~/rag_faithfulness`) differ only by hyphen/underscore, so any
script using a relative checkpoint path needs `RAG_CHECKPOINT_DIR`; and
`python3 -I` hides the `pip install --user` site-packages, so numpy/statsmodels
vanish under it.

---

**B31 — Part 1A of the 2026-10-19 consistency audit: every remaining
three-generator figure is gone (2026-10-19).** A full read of the PDF found 17
places still carrying three- (or two-) generator numbers. All were recomputed
from `~/rag-backup/checkpoints/n1000_v3`, and every already-published value
reproduced exactly before anything was extended. Pages still 30/30/31/31, 0
overfull, 0 undefined, 0 `??`.
- **`tab:entity` gains the Mistral column** (NQ 0.203/26.9, 0.233/37.1,
  0.399/45.4; HotpotQA 0.312/54.6, 0.266/60.0, 0.429/64.4), columns now in the
  `tab:2` order. Number-vs-entity generator ordering is still 3 of 6, but one
  of the three (NQ/AlignScore) holds only because Qwen and Mistral **tie** on
  the numeric probe at 51.8%. On NQ/NLI-max the entity swap is NOT a near-tie
  (Mistral 26.9% vs Qwen 10.8%). Evaluator ordering by entity detection is **7
  of 8** cells (was "5 of 6"); the exception is still NQ/Claude.
- **`tab:refusalrule` was stale twice over**: 2 rows per rule at the old
  18-family p_holms (0.0187/0.0090). It now has all 4 survivors per rule, and
  the adopted rows equal `tab:7`. The judge-rule paired n on HotpotQA/Mistral is
  **404**, not the published 423. "All three rules are anchored phrase matching"
  was false (the third is the judge).
- **⚠️ D4's "107 refusals, all Claude's" is a two-generator figure.** Over the
  four-generator grid the canonical rule differs from the bare one on **759**
  rows: Claude 107, Qwen 50, **Mistral 602**, GPT 0. And it is almost all the
  **wider phrase list**, not the preamble strip (626 need only the list, 2 only
  the strip): Mistral opens 378 refusals with "The provided context does not".
  "All four generators prepend the preamble" was false (Claude 97.5% of attempts,
  Qwen 12.5%, Mistral 17.6%, GPT 0.2%).
- **§6.1's detector comparison was the pre-D1 2026-08-30 recheck** (two arms,
  two evaluators; 19/55 rows, p 0.223→0.048). Recomputed over all 24: 68/173
  rows move on NQ/HotpotQA Claude; the same two comparisons cross uncorrected
  0.05 (NQ/Claude/NLI 0.155→0.032, HotpotQA/Claude/align 0.063→0.022); the judge
  adds up to 3.4 refusals per hundred, and reorders no cell.
- Mistral figures added: markdown on attempts **0.2%**; copying overlap
  **0.165** (lowest of four), r +0.046 [-0.008, +0.102]; refusals score below
  attempts under all three evaluators, so Table 7 says "2 of 4 gens". Probe
  cases are **6,179 scored of 9,286 eligible**; HotpotQA/Mistral is a full census
  (595/595). MDE ranges: NLI-max .021–.061, claim-min .024–.064.
- Also fixed: the Limitations survivor spreads (0.041–0.080), the "three NQ
  cells", §7.3's "just outside" (p_holm 0.154), the Limitations item claiming the
  contingency table was two-arm (it is 32 cells; item deleted), `tab:power`'s
  "the two" (three), H.1's single "exception", "largest" → "exceeds NLI-max",
  QASPER "three of the four generators, Mistral-7B excepted", and the abstract's
  "up to 200 answers".
**Part 1B, same day — the logic contradictions.** Main PDFs now **31 pages**
(compact 31), 0 overfull/undefined/`??`.
- "The ordering holds at either gate" was false: at the floor-anchored gate
  the evaluator ordering breaks (AlignScore lowest on NQ/GPT, claim-min < NLI-max
  on HotpotQA/Claude); the generator extremes hold in 5 of 6 rows (NQ/AlignScore
  is the exception).
- `tab:3` non-monotonicity described per generator (Claude never reverses, GPT
  reverses twice), not "each once".
- "Most responsive" on Claude/NQ is **claim-min 25.1%**, then AlignScore 24.1%.
- "Shortening buys more than switching evaluators" is now scoped to Qwen (24.7 /
  37.4 lost vs 16.7 / 19.7 gained) and flagged as **inferred**: length was only
  ever varied upward. "It is free" dropped (it moves refusals; correctness
  unmeasured).
- "Answer structure is therefore causal" → within-one-model wording that names
  the refusal/floor side effects. No "causal" remains in the paper.
- The claim-level recommendation is scoped to checking individual answers.
- §8 anchor: claim-min 0.093–0.179, AlignScore 0.243–0.364, per system over both
  datasets (same population as NLI-max's 0.049–0.095; the old 0.165/0.247 were
  NQ-pooled).
- QASPER's generator reversal holds under NLI-max only (main grid: Claude 0.904 /
  0.706); claim-min already ranks GPT first.
- **`\movedref` / `\movedrefs`** print "Section(s)" in the full build and
  "Appendix/Appendices" in the compact one. Use them for any reference into a
  `sections/*.tex` label; 21 references converted ("Appendix 2.1" was wrong in
  the main build, the rest were wrong in the compact one).
- Small: six → eight tests (App. K), the geometric analysis uses no generator
  (re-ranking is the Claude/one-system arm), "quadruples" → "almost fivefold".
**Part 2, same day — the style pass.** Pages unchanged (31/31/31/31), clean.
- Vague attribution removed: "established frameworks often separate refusals",
  "re-ranking is often proposed", "the sample sizes people use" (now 400–800
  paired queries per cell).
- Repetition cut: §7.1's doubled "precondition failing", §7's doubled "all four
  surviving comparisons are on HotpotQA", the third "statement about
  resolution, not about the systems".
- §9's distractor floor now explained (9.0/15.0% sits BELOW no-retrieval
  32.2/20.9% because the model refuses 66.7/91.3% of the time), not "consistent".
- §2.1's "will appear in any claim-decomposed metric" scoped to min-aggregation,
  "we tested one".
- Two em-dash asides, "OWN", `one\'s` (rendered "oneś"), mixed numerals
  ("20 of the twenty-four" ×3, "4 / 2 / 1"), prose en-dash ranges, and two
  Title-Case subsection headings fixed. Convention: Title Case for top-level
  sections, sentence case for subsections and appendices.
- Of 77 "X, not Y"/"rather than" sentences, only the repeated or ornamental ones
  were trimmed (8); the rest state method choices and stay.
**Part 3, same day — and it found a second mechanism.** Main PDFs 31 pages,
compact 32, clean.
- **⚠️ The "Claude-specific residual" travels with an attribution preamble.**
  Claude opens 97.5% of attempts with "Based on / According to the provided
  context," (Qwen 12.5%, Mistral 17.6%, GPT 0.2%). Single-assertion NQ, NLI-max:
  Qwen +0.457 [.418,.495] without it vs **+0.129** [.076,.187] with it; Mistral
  +0.426 vs **+0.162** [.091,.238]; Claude (all with it) **+0.102** [.064,.146],
  inside both. 2+ assertions with it: 0.048 / 0.045 / 0.041. Same n as `tab:3`
  (91, 413, 422, 680).
- **Direct test RUN 2026-10-19** (`scripts/preamble_strip_rescore.py`, gpu1,
  copied as a snapshot into `~/rag_faithfulness/outputs/preamble_rescore/` so
  gpu1's git tree was not touched; result `checkpoints/n1000_v3/
  preamble_strip_{nli,claim,align}.pkl`). Rebuilds the probe's own 2,722 cases
  whose answer opens with the clause; as-written scores reproduce the
  checkpoints with **max diff 0.0**. NLI-max, clause stripped from both answers:
  single-assertion Claude NQ +0.102 → **+0.272** (diff [+0.098, +0.237]),
  HotpotQA +0.149 → **+0.470**; detection 10.1→42.5% and 20.4→80.6%; Mistral NQ
  +0.162 → +0.291. Closes the gap to unprefixed answers on HotpotQA, ~45% of it
  on NQ. The clause also inflates the untouched answer (0.950 vs 0.788). **At 2+
  assertions stripping changes ~nothing** (Claude NQ +0.038, else ~0), so the
  earlier "2+ answers with the clause agree again" reading was wrong and is
  gone: dilution carries the 2+ effect.
- **claim-min and AlignScore finished 2026-10-19 19:26 UTC**, both reproducing
  the checkpoints with max diff 0.0; all three result files are in the laptop
  backup. Claim-min = NLI-max on 1 assertion; at 2+ stripping lifts Claude's
  detection 28.0→39.1% (NQ) and 42.5→62.4% (HotpotQA), mean +0.032/+0.041.
  **AlignScore is shielded too**, less: Claude single-assertion +0.497→+0.656
  (NQ, diff [+0.120,+0.196]) and +0.678→+0.782 (HotpotQA); ~+0.025 at 2+. So
  the clause is not an NLI-model quirk. In §5.4.
- ⚠️ The 2026-10-21 backup dry run also listed **pre-existing analysis files
  that differ between gpu1 and the backup** (`conditional_faithfulness_*`,
  `faith_gap_significance_*`, `necessity_sufficiency_*`, `query_frame.pkl`, …).
  They were deliberately NOT synced: which side holds the version behind the
  paper is unverified. Check before any full `rsync`.
- **The verbosity result survives the preamble.** qwen_verbose raised the
  preamble share 24.2→47.2% (NQ), 8.2→27.6% (HotpotQA), but among no-preamble
  answers NLI-max detection still falls 41.0→10.4% and 57.8→17.1%.
- **Table 4's sign is the refusal wording.** HotpotQA: GPT refuses with the
  bare template 100% and Qwen 92.5%, NLI-max 0.93 on it; Claude and Mistral
  explain what the context lacks in 100% of refusals, NLI-max 0.31 / 0.36.
- Worked example added to §5.1 (nq_98, all-mpnet: "March 8" → "March 13";
  Claude NLI 0.985→0.983 vs GPT 0.905→0.068; AlignScore 0.971→0.054 /
  0.986→0.018). §12.2 "barely matters" replaced with 56.6% vs 75.4%. "Most
  widely used" / "most such scores" / "the configuration people deploy" are
  softened, since no citation supports them.

### Code moves and reporting changes of 2026-09-01

- **`src/metrics.py` is now `src/legacy/rfg.py`.** The gap metric is retired
  (paper §III-C). `assemble_results` writes the `RFG`/`nRFG` columns only under
  `run_pipeline --legacy-rfg`; `robustness_analysis` and the nRFG rows of
  `hypothesis_summary` skip themselves when the columns are absent.
- **H1 is deleted from `hypothesis_summary`.** It fed two UNPAIRED groups
  (different models aligned by DataFrame row order) into
  `bootstrap_significance`, which pairs by index; the assert passed only
  because the lengths matched.
- **`compare_evaluators` now Holm-corrects the table it prints** and reports
  each p as extreme/total resamples. **This changes a documented conclusion**:
  the two p-values that made the two disagreeing cells depend on the
  abstention source (0.0481 and 0.0487) are **0.2886 after Holm**, so the
  verdict flip disappears and the spine result is stable to that choice. The
  headline is p = 0.0005 (5/10,000 resamples), **p_holm = 0.0040**, and 2 of 4
  cells still break under AlignScore. The paragraph in §2 above and the
  corresponding hedge in the paper are superseded.
- New: `src/abstention.py`, `src/perturb_report.py` (Table III with n per row,
  the single-assertion identity check, the value-role breakdown),
  `src/markdown_check.py`. `perturbation_check` gains `--scorer nli_concat`
  (premise-granularity diagnostic) and `--scorer numeric` (deterministic value
  grounding), a floor-anchored detection gate, Wilson CIs on every detection
  rate, and rejection of random-context donors that support the answer.

**Still open:**
21. ✅ **Every experiment in `PAPER_TODO.md` §3 has now run** (2026-09-16).
   LLM judge (2026-08-30, all 24 cells incl. qwen), open-weight generator
   (2026-09-05), claim-level full grid (2026-09-04), literal value grounding
   (2026-09-01), **C1/C2**, **context ablation**, **ESA**, **re-ranking**,
   **QASPER** and **jina-embeddings-v3** (all 2026-09-12..16).
   Shapley/`doc_utility.py` is dropped at Berend's direction. Nothing paid
   remains. Full derivation: `reports/2026-09-16_all_results.md`.

22. ✅ **Paper update — the draft now contains every result we have** (2026-09-16).
   Previously open items: — references ✅ **all 24 verified 2026-09-03** (§6b),
   which also closed ref [8] (now [24], jina v3); Section V-B and the
   multiplicity disclosure ✅ refreshed 2026-09-04 (B20/B21); the Salemi & Zamani
   PDF ✅ **read in full, and §II-A rewritten, 2026-09-05**
   (`reports/2026-09-05_salemi_zamani_full_read.md`) — DOI verified against Crossref;
   the docx is rebuilt and exactly one paragraph moved; IEEE → ACL ✅ **converted
   2026-09-05** (`paper/acl/`, compiles clean, 0 undefined citations, all 234 prose
   fragments verified to survive verbatim). all closed. **The ARR page limit no longer
   applies** — the goal is a complete draft for Berend, and the paper is 22
   pages. The six orphaned arms were added in commit 03951bd: jina as a
   positive control (2 of 6 -> 6 of 6 when included, which is the premise
   breaking, not a better result), QASPER as a boundary condition where the
   evaluator ordering inverts, C1/C2 anchors (retrieval beats the oracle on NQ
   in 3 of 4 systems), context ablation, ESA and re-ranking. **All 13 red flags
   and all 14 of Berend's points are closed or explicitly stated.**
   Still open: put the six new sections in the author's own voice, and verify
   the one new citation (`dasigi-etal-2021`, QASPER) — it has NOT been checked
   against the ACL Anthology the way the other 24 were in the B3 pass.

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
test, different evaluator, different verdict. ⚠️ **The count is now 4
comparisons in 3 of 8 cells over a 24-comparison family** (B28), not the 2 of 4
at p=0.0005 recorded below — that figure predates the markdown fix (D1) and both
open-weight arms. The progression 2/4 -> 1/4 -> 2/6 -> 3/8 is documented in D1,
B22, B23 and B28; **3 of 8 cells / 4 of 24 comparisons is current**. Details in
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
