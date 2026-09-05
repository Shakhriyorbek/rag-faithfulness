# The open-weight generator arm — Qwen2.5-7B-Instruct

**Run dates:** 2026-09-04 14:44 UTC → 2026-09-05 00:18 UTC · **Cost:** $0 (local GPU)
**Scope:** `checkpoints/n1000_v3` on gpu1
**Commits:** `8af53fe` (generate fix), `6351d95` (full scale), `442359b` (CLI
defaults), `fcb7ed8` (model default)
**Output:** 8 generation cells (8,000 answers), 8 each of NLI / AlignScore / claim
scores, correctness, 32 falsification-probe checkpoints. Backed up 2026-09-05.

Closes item 2 of `PAPER_TODO.md` §3 — the last never-run experiment Berend
asked for. It was requested for reproducibility; under the v6 framing it is
also a prediction test, because the paper claims the evaluator's blindness is
mediated by how many assertions an answer contains, and a third generator with
a third verbosity profile either lands where that predicts or it does not.

---

## Headline

1. **The prediction holds.** Qwen writes 1.58 assertions per answer against
   Claude's 2.73 and GPT-4o-mini's 1.12, and its falsification detection lands
   between them in every cell — 30.0% and 29.8% against Claude's 4.8%/9.9% and
   GPT-4o-mini's 51.1%/40.3%.
2. **The headline count rises from 1 of 4 to 2 of 6.** The new arm adds a second
   AlignScore break (HotpotQA/qwen, p_holm 0.0187) on a Holm family half again
   as large.
3. **The embedder ranking replicates exactly across all three generators** under
   AlignScore on HotpotQA, and under NLI-max it does not replicate at all.
4. **Conditioning on assertion count, Qwen matches GPT-4o-mini rather than
   sitting between.** The intermediate aggregate comes from Qwen's assertion
   *distribution*. Claude is the outlier, not one end of a smooth axis.
5. The copying hypothesis dies a third time: Qwen has the **lowest** overlap of
   the three (0.194) and intermediate sensitivity.

---

## 1. Two bugs had to be fixed first — the arm had never run

**`LocalHFGenerator.generate` did not work at all.** It failed on the first
call with `AttributeError` inside `model.generate`:
`apply_chat_template(return_tensors='pt')` returned a bare tensor in
transformers 4.x and returns a `BatchEncoding` in 5.x (gpu1 runs 5.14.1), so a
dict was being passed as the first positional argument. Now passes
`return_dict=True` explicitly and unpacks with `**enc`, which works on both
majors and passes the attention mask through instead of letting `generate()`
infer one.

**`run_phase_llama(model_names=None)` meant a hardcoded three models**, not
"all", unlike every other phase. `run_pipeline` passes `None` when `--models`
is omitted, so the first pass silently produced **6 cells instead of 8**,
dropping `text-embedding-3-small`. That is not a smaller run but an
incomparable one: this arm's embedder spread would have been computed over 3
systems while the other two generators' is over 4, and nothing downstream would
have said so — `faithfulness_by_model` simply skips a model with no checkpoint.
Caught by inspecting completed cells mid-run. Both fixes are pinned by tests.

Also required: `pip install --user accelerate` (needed by `device_map='auto'`,
absent on the box).

---

## 2. Verbosity profile — the input to the prediction

Assertions per answer, all 8,000 answers per generator:

| generator | NQ | HotpotQA | ALL | median words (NQ) | markdown | refusals (NQ) |
|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | 2.996 | 2.460 | **2.728** | 55 | 74.9% | 14.7% |
| **Qwen2.5-7B** | **1.760** | **1.409** | **1.584** | **24** | **1.8%** | **12.3%** |
| GPT-4o-mini | 1.176 | 1.058 | **1.117** | 15 | 0.3% | 29.0% |

Qwen sits between the two API arms on assertion count and length. Its markdown
rate is GPT-like, so the D1 formatting confound that cost a cell in the
AlignScore result is effectively absent for this arm.

---

## 3. The prediction test — Table II with three generators

200 paired cases per cell, pooled over the four embedders. "Floor" is the mean
score of an untouched answer against an unrelated query's context.

| dataset / generator | evaluator | orig | falsified | floor | det@0.5 | det@floor |
|---|---|---|---|---|---|---|
| NQ / Claude | NLI-max | 0.906 | 0.852 | 0.474 | **4.8%** | 29.8% |
| NQ / **Qwen** | NLI-max | 0.836 | 0.574 | 0.470 | **30.0%** | 32.4% |
| NQ / GPT-4o-mini | NLI-max | 0.887 | 0.436 | 0.506 | **51.1%** | 53.4% |
| HotpotQA / Claude | NLI-max | 0.702 | 0.618 | 0.427 | **9.9%** | 10.5% |
| HotpotQA / **Qwen** | NLI-max | 0.547 | 0.277 | 0.245 | **29.8%** | 24.6% |
| HotpotQA / GPT-4o-mini | NLI-max | 0.577 | 0.186 | 0.201 | **40.3%** | 38.9% |

Ordered exactly as assertion count predicts, on both datasets, and predicted
before the arm was run rather than after.

The falsified score against the floor tells the same story: Claude's altered
answer sits at 0.852 against a floor of 0.474 — far above any usable
threshold — GPT-4o-mini's lands at 0.436 against a floor of 0.506, i.e. below
its own floor, and Qwen's 0.574 against 0.470 is between.

**Evaluator ordering is identical for all three generators** —
AlignScore > Claim-min ≳ NLI-max — in every cell:

| cell | NLI-max | Claim-min | AlignScore |
|---|---|---|---|
| NQ / Claude | 4.8% | 17.2% | **23.5%** |
| NQ / Qwen | 30.0% | 34.0% | **48.1%** |
| NQ / GPT-4o-mini | 51.1% | 53.9% | **71.0%** |
| HotpotQA / Claude | 9.9% | 12.0% | **43.8%** |
| HotpotQA / Qwen | 29.8% | 28.4% | **61.1%** |
| HotpotQA / GPT-4o-mini | 40.3% | 39.8% | **71.2%** |

### The nuance worth putting in the paper

Conditioning on assertion count, Qwen does **not** sit between — it tracks
GPT-4o-mini, and Claude is the outlier:

| assertions | Claude | Qwen | GPT-4o-mini |
|---|---|---|---|
| 1 | +0.1293 (n=235) | **+0.4103** (n=846) | +0.4473 (n=1411) |
| 2 | +0.0853 (n=583) | +0.1424 (n=447) | +0.2556 (n=127) |
| 3 | +0.0439 (n=473) | +0.0371 (n=194) | +0.0372 (n=29) |
| 4 | +0.0335 (n=166) | +0.0654 (n=66) | +0.2042 (n=10) |
| 5+ | +0.0314 (n=143) | +0.0684 (n=47) | +0.0323 (n=7) |

All three converge from three assertions on. At a single assertion Qwen is
+0.410 against GPT-4o-mini's +0.447 and Claude's +0.129. So the aggregate
intermediacy comes from Qwen's assertion *distribution*, not from an intrinsic
midpoint, and the residual the paper already acknowledges is **specifically
Claude's** rather than a smooth verbose/terse axis. The third generator is what
makes that distinguishable.

---

## 4. Table VII is now 18 rows, and 2 of 6 cells break

Holm over the 18 comparisons printed.

| dataset | generator | metric | n | spread | p | p_holm | verdict |
|---|---|---|---|---|---|---|---|
| **HotpotQA** | **GPT-4o-mini** | **align** | 510 | 0.0461 | 0.0005 | **0.0090** | **DIFFERS** |
| **HotpotQA** | **qwen** | **align** | 499 | 0.0406 | 0.0011 | **0.0187** | **DIFFERS** |
| HotpotQA | Claude | align | 517 | 0.0164 | 0.0633 | 0.8229 | null |
| HotpotQA | GPT-4o-mini | claim | 510 | 0.0378 | 0.0070 | 0.1120 | null |
| NQ | Claude | align | 784 | 0.0114 | 0.0081 | 0.1215 | null |
| NQ / HotpotQA | qwen | nli, claim | 790 / 499 | ≤0.0181 | ≥0.29 | 1.0000 | null |

The remaining twelve rows are null. Adding a generator made the family stricter
(12 → 18) and the count still went up.

### The strongest result: the ranking replicates, and only under AlignScore

Under **AlignScore on HotpotQA**, all three generators produce the identical
ordering of all four embedding systems:

```
all-mpnet-base-v2 < text-embedding-3-small < BGE-M3 < E5-large-instruct
```

Under **NLI-max on the same answers**, no two generators agree:

```
Claude       BGE-M3   < text-emb  < all-mpnet < E5
GPT-4o-mini  text-emb < all-mpnet < BGE-M3    < E5
Qwen         text-emb < all-mpnet < E5        < BGE-M3
```

Two of the three AlignScore cells clear Holm; the third (Claude) is at
p = 0.0633.

**State it carefully.** The generators are independent, but the *retrieval* is
shared — same embedders, same chunks, same queries. This is therefore three
generators agreeing about a property of the retrieval, not three independent
replications of an effect. That is still exactly the claim the paper makes, and
it should be worded that way rather than as independent replication.

On NQ nothing breaks for any generator and the AlignScore orderings do vary, so
the effect is HotpotQA-specific — consistent with Table VIII, where only
HotpotQA correlates positively with NDCG@5.

---

## 5. Controls

**Copying, three arms.** Qwen has the *lowest* 5-gram overlap of the three and
intermediate sensitivity, which is the wrong way round for the hypothesis a
third time:

| generator | mean overlap | corr(overlap, Δ) 95% CI |
|---|---|---|
| Claude | 0.226 | +0.046 [−0.004, +0.096] |
| GPT-4o-mini | 0.235 | +0.166 [+0.118, +0.212] |
| **Qwen** | **0.194** | +0.178 [+0.127, +0.229] |

At matched overlap Qwen sits between the other two in all four bands (e.g.
[0.3, 1.0]: Claude +0.085, Qwen +0.364, GPT-4o-mini +0.507).

**Deterministic value grounding.** 100% recall in all 8 qwen cells at a
1.5–5.5% false-positive rate, in line with Claude (1.5–6.0%) and GPT-4o-mini
(0.5–3.0%).

---

## 6. Operational

- 8,000 generations at ~2.0 s/answer in fp16 on the V100; ~4.3 h across two
  passes. Phase D 36 min, claim-level 89 min, AlignScore 26 min.
- The hop was unreachable for several hours on 2026-09-05 (TCP 22 timing out
  with no banner, no ControlMaster socket involved). The tmux jobs were
  unaffected. Backup taken as soon as it returned.
- **The backup now holds the qwen work** (80 files, 450 MB total). Before that
  it existed only on gpu1 — free to regenerate, but ~5 h of GPU time.

## 7. Not done

- **Judge correctness for this arm** (~$15) — not run. Correctness uses the
  free heuristics, which is what `compare_evaluators` needs for `abstained`.
- The paper is **not** updated for any of this. Tables II, III and VII all
  change, and Section VII-B's multiplicity paragraph needs its numbers redone
  against a family of 18.
