# RUNBOOK — how to actually run this project

Practical manual for getting from "code exists" to "the paper has real numbers".
Read `CLAUDE.md` for *what* the project is; this file is *how you work on it*.

---

## 0. Where things stand

| | |
|---|---|
| Pipeline code | ✅ built and unit-tested in `src/` |
| Server | ✅ reachable, 32 GB V100, has internet |
| Experiments | ❌ **never run** — every number in the paper is simulated |
| Your next action | **§3 Setup**, then the **§5 ladder** |

The whole point of the ladder in §5: find out whether the core finding exists
after a few hours, not after a few days.

---

## 1. Server facts (verified 2026-07-26)

```
laptop ──ssh──> nlp (193.225.250.29) ──ssh gpu1──> nlp-large-1
                user: sboltabaev                   GRID V100DX-32C, 32 GB
```

- **GPU is 32 GB** → Llama-3-8B runs in **fp16, no 8-bit needed**. `bitsandbytes` optional.
- **gpu1 has direct internet** → HuggingFace/pip download straight to the node.
- 16 cores, 62 GB RAM, 284 GB free disk. Python 3.10.12, no conda, torch not installed.
- `tmux`, `git`, `pip3`, `rsync` present; `python3 -m venv` works. `~/rag_faithfulness` does not exist yet.

### Three things that will waste your time if you forget them

1. **`ping` never works** — ICMP is blocked. Test reachability with TCP 22:
   ```powershell
   Test-NetConnection -ComputerName 193.225.250.29 -Port 22
   ```
2. **The gateway rate-limits SSH.** ~6 connections in a few minutes = temporary
   IP block. Open **one** session and stay in it; `ControlMaster` in
   `server/ssh_config` multiplexes the rest. If everything suddenly times out,
   you are probably blocked — wait ~15 min, don't retry in a loop.
3. **Long jobs need `tmux`.** Closing the laptop kills a bare SSH session and
   takes the run with it.

---

## 2. The mental model

Three machines, three jobs:

| Machine | Does |
|---|---|
| **Laptop** | Edit code, run `tests/test_local_smoke.py`, read results, write the paper |
| **hop (`nlp`)** | Nothing but forwarding. Don't compute here |
| **gpu1 (`nlp-large-1`)** | Every phase. Embeddings, generation, NLI, AlignScore |

Everything is checkpointed to `checkpoints/*.pkl`. **Any phase can be killed and
resumed** — it skips what's already done. This is what makes the ladder safe.

---

## 3. One-time setup

### 3.1 Laptop — SSH config

```bash
cp server/ssh_config ~/.ssh/config
```

Then verify (this also warms the ControlMaster connection):

```bash
ssh szte-gpu 'hostname && nvidia-smi --query-gpu=name,memory.total --format=csv'
```

Expect `nlp-large-1` and `GRID V100DX-32C, 32768 MiB`. If it hangs at "banner
exchange", use the two-step fallback: `ssh szte-hop` then `ssh gpu1`.

### 3.2 gpu1 — environment

Inside **one** SSH session:

```bash
mkdir -p ~/rag_faithfulness && cd ~/rag_faithfulness
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
```

> **`python3 -m venv` does not work on gpu1** — `ensurepip` is missing and
> installing `python3.10-venv` needs sudo. Use `pip install --user` (below),
> or ask Berend for the package.

```bash
python3 -m pip install --user -r requirements.txt
```

### ⚠️ torch on the V100: `is_available()` is NOT a sufficient check

The V100 is **Volta, compute capability 7.0**. The default PyPI torch wheel
ships kernels for **sm_75 and up only**. On this GPU it imports fine and
`torch.cuda.is_available()` returns **`True`** — then the first real kernel
launch dies with:

```
CUDA error: no kernel image is available for execution on the device
```

Verify with an actual matmul, and check that `sm_70` is in the arch list:

```bash
python3 -c "import torch; print(torch.cuda.get_arch_list()); x=torch.randn(512,512,device='cuda'); torch.cuda.synchronize(); print('OK', float((x@x).sum()))"
```

`get_arch_list()` **must contain `sm_70`**. If it doesn't, reinstall against an
older CUDA index (`--force-reinstall` is required — pip treats `2.13.0+cu130`
and `2.13.0+cu126` as the same version and will otherwise skip the install):

```bash
python3 -m pip install --user --force-reinstall torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
```

If cu126 still lacks `sm_70`, step down (cu121, then an older torch) until the
matmul passes. Recent PyTorch releases have been dropping Volta.

### 3.3 Secrets — never commit these

```bash
export ANTHROPIC_API_KEY='sk-ant-...'   # generator (phase c)
export OPENAI_API_KEY='sk-...'          # text-embedding-3-small ONLY
export HF_TOKEN='hf_...'                # Llama-3 is a gated model
```

Put them in `~/.bashrc` on gpu1 so tmux panes inherit them. `HF_TOKEN` also
requires accepting the Llama-3 license once on the HuggingFace website.

**Both API keys are needed, and they are not interchangeable.** Claude is the
generator; `text-embedding-3-small` is one of the seven *embedding models under
study* and Anthropic has no embeddings API. Dropping the OpenAI key means
dropping to six models and changing the paper's counts.

---

## 4. Getting code onto gpu1 — solved 2026-08-13

**This is now one command.** The setup below was done once and does not need
repeating.

### The sync loop

```bash
git push gpu1 main
```
then on gpu1:
```bash
cd ~/rag-faithfulness && git pull
```

### Why it works now, and what had to change

Three things were fixed on 2026-08-13:

1. **Your laptop key is now authorized on gpu1**, not just on the hop. Until
   then `ssh szte-gpu` (ProxyJump) failed with `Permission denied (publickey)`,
   which is why `scp` straight to gpu1 never worked and every transfer had to
   be piped through the hop. The key was appended to `gpu1:~/.ssh/authorized_keys`;
   `ssh szte-gpu` now connects directly in one command.

2. **`$HOME` is NOT shared between hop and gpu1.** This was flagged UNVERIFIED
   for weeks. It is now tested and answered: `HOME_SHARED=no`. Copying to the
   hop does *nothing* for gpu1 — do not rely on it.

3. **Code and data are now separate directories.**

   | Path | Holds | Git |
   |---|---|---|
   | `~/rag-faithfulness/` | the code — a real clone of the bare repo | yes |
   | `~/rag_faithfulness/` | `checkpoints/`, `hf_cache/`, `outputs/` | no |

   Note the hyphen/underscore distinction; they are different directories.
   `config.BASE_DIR` still resolves to `~/rag_faithfulness`, so running
   `python3 ~/rag-faithfulness/src/...` finds every existing checkpoint. A
   `git pull` in the code directory cannot touch data, and data cannot pollute
   `git status`.

   The old loose copies were moved to `~/rag_faithfulness/src.superseded-2026-08-13`
   and `tests.superseded-2026-08-13`. They were verified to contain nothing the
   repo lacks before being moved, and nothing was deleted — remove them once
   you are satisfied.

### Remotes

| Remote | URL | Use |
|---|---|---|
| `origin` | `github.com/Shakhriyorbek/rag-faithfulness` (private) | backup, history |
| `gpu1` | `szte-gpu:rag-faithfulness.git` (bare, on gpu1) | deploying to the server |

gpu1 has internet, so `git pull` from GitHub would also work — but it is a
private repo and would need a credential on the server. Pushing to the bare
repo over SSH avoids putting any token there.

**Never copy `checkpoints/`** — large, regenerable, gitignored. Pull *results*
back instead:
```bash
scp -r szte-gpu:~/rag_faithfulness/outputs ./outputs
```

### Verified on gpu1 after the sync

All six new modules import, the offline suite passes 60/60, `BASE_DIR` resolves
to the data directory with all 13 pilot checkpoints visible, and the dependency
set is present: torch 2.7.1+cu126 (the sm_70 build), transformers 5.14.1,
pandas 2.3.3, scipy 1.15.3, numpy 1.26.4, anthropic 0.120.0. `pytest` was
installed with `pip install --user` (the venv route is broken — `ensurepip` is
missing and needs sudo).

---

## 5. The execution ladder

Climb in order. **Each rung is a decision gate** — check the stated condition
before spending time on the next one. Always inside `tmux`:

```bash
tmux new -s rag        # detach: Ctrl-B then D    reattach: tmux attach -t rag
```

(There is no venv on gpu1 — `ensurepip` is missing and `python3.10-venv`
needs sudo. Packages are installed with `pip install --user`, so just call
`python3`.)

### ⚠️ Checkpoints are scoped — check the scope line before you trust a run

Every run prints its checkpoint directory as its **first line**:

```
  [scope] checkpoints -> /home/sboltabaev/rag_faithfulness/checkpoints/n1000_v3
```

`n{N}_{CORPUS_VERSION}`. Checkpoint keys are named for their content
(`retrieval_BGE-M3_NQ`), not for the run, so before this existed the N=50
pilot and the N=1000 run collided on every key — and since each phase skips
work whose checkpoint exists, the first real N=1000 attempt printed
`all datasets done, skipping` for every model and **exited 0 in seconds
having done nothing** (2026-08-14). If a phase finishes suspiciously fast,
read that line first.

- `RAG_SCOPE_N=50` — work in another N's scope
- `RAG_CHECKPOINT_DIR=~/rag_faithfulness/checkpoints` — absolute override.
  Needed to read the **pre-scope pilot** checkpoints, which still sit flat in
  `checkpoints/` and are not visible to any scoped run.

Bumping `CORPUS_VERSION` (config.py) therefore forces a clean rebuild of
phases A and B rather than silently reusing a corpus with different
semantics. Bump it whenever corpus or relevance semantics change.

### Rung 1 — Smoke test (~15 min, ~$0.05)

```bash
python src/run_pipeline.py --smoke-test
```
50 queries, 3 models, NQ only, phases a→b→c→d→report.

**Gate:** it must complete without an assertion. The pipeline asserts
`NDCG@5 > 0` — that is the guard for the qrels bug that silently zeroed
retrieval quality in the old notebook. Also confirm the NLI load prints
`entailment index: 1`. If it prints `2`, something re-broke the label
resolution and every faithfulness number would be garbage.

### Rung 2 — Reduced run: 3 models × NQ, N=1000 (~2–4 h, ~$1)

```bash
python src/run_pipeline.py --full --datasets NQ \
  --models all-mpnet-base-v2,E5-large-instruct,BGE-M3
```

Phases a,b alone are **free** (GPU only) — run them first and read the
retrieval numbers before authorizing `--phases c,d --yes`, which is where the
~$5.60 goes.

**Gate — this is the one that matters.** Look at the nRFG spread across the
three models.
- **Spread is visible and models rank differently than NDCG@5 does** → the
  paper's premise survives contact with data. Continue.
- **All three nRFG values are within noise** → *stop and think.* The core claim
  may not hold on NQ. That is a real finding, not a failure — bring it to
  Berend before burning days on the full grid. Single-hop NQ is also the least
  likely dataset to show the effect, so HotpotQA is the natural next probe.

**Second gate, free, run it as soon as phase b finishes:**

```bash
python3 -c "import sys; sys.path.insert(0,'src'); import results; print(results.equivalence_table('NQ').to_string())"
```

The paper claims these embedders reach *matched* retrieval quality. A
non-significant difference does not show that — it is absence of evidence.
`equivalence_table` runs a **paired TOST** (margin ±0.02 NDCG@5,
`results.EQUIV_MARGIN_NDCG`) and marks a pair `matched` only when it is both
statistically indistinguishable **and** equivalent within the margin. If no
pair comes out `matched`, the premise sentence has to be rewritten before the
full grid, not after.

Also read the loader's sampling line — the paper has to state it (Berend
point 8):

```
scanned 1120 candidates: 54 had no short answer, 66 had the answer outside
the retained window (dropped), 408 re-centred on the answer span (B6)
```

### Rung 3 — Full grid, Claude (~12–20 h, **~$38 est.**)

**Read the cost probe from Rung 1 before starting this.** `--smoke-test`
extrapolates measured cost to the full grid and prints:

```
COST PROBE: measured $0.00XXXX/query over 150 queries -> $XX.XX projected for 21,000
```

Trust that number over the estimate — it is measured on your real chunk
lengths. Then:

```bash
python src/run_pipeline.py --full
```

7 models × 3 datasets. The budget cap raises a `RuntimeError` at $60 rather
than draining the account; partial checkpoints survive, so raise
`budget_limit` and rerun to resume.

**Generator note:** Claude (`claude-haiku-4-5`) replaced GPT-4o-mini on
2026-07-26. Haiku 4.5 was chosen as the closest analog in capability tier and
cost, preserving the paper's design intent — a small, widely-deployed
closed-source model. A frontier model would likely be *more* faithful across
the board and could compress the very RFG differences the paper measures.
**Paper §4.4 and §5.2 must be updated**, and Berend should hear about the
change before submission.

### Rung 4 — The analyses that answer the supervisor (~4–6 h)

```bash
python src/run_pipeline.py --full --phases esa      # closes point 6
python src/run_pipeline.py --full --phases llama    # H3, GPU
python src/run_pipeline.py --full --phases e        # AlignScore, GPU
python src/run_pipeline.py --full --phases rerank   # H5
python src/run_pipeline.py --full --phases report
```

`esa` is the highest-value one: the paper *promises* both `NLI(d, gold_a)` and
`NLI(d, q)` correlations, and until it runs, §4.5→§4.6 has an open gap.

AlignScore needs a source install and a checkpoint download:
```bash
pip install git+https://github.com/yuh-zha/AlignScore.git
```
Phase `e` skips cleanly if it's absent — NLI alone is enough to make progress.

### Rung 5 — Berend's 2026-08-11 letter (~$18 on top of Rung 3)

His letter reframes the target claim as *good retrieval quality is necessary
and sufficient for a high-quality response*, and asks for controlled conditions
that test each half separately. Everything below exists to produce that
evidence. Run in this order — the free steps first, so you see the shape of the
data before paying for the next thing.

**5a — correctness (free, no API).** Run this against whatever generations
already exist. Nothing else in Rung 5 means anything without it.
```bash
python src/correctness.py
```
Read `abstention_rate` beside `correct_rate`: a floor made of "I do not know"
is a different phenomenon from a floor made of confident wrong answers.

**5b — the anchors (~$7 at N=1000 × 3 datasets).** Both are
*embedder-independent* — computed once, not once per model.
```bash
python src/conditions.py --condition both --dry-run     # inspect cost first
python src/conditions.py --condition both --yes
python src/conditions.py --prompt-probe 100 --yes       # ~$0.03, see below
python src/correctness.py
python src/conditions.py --emit-filter
```
- **C1** is the no-retrieval parametric floor. It *cannot* reuse the RAG
  prompt — that template says "answer using ONLY the provided context" and
  offers a refusal string, so with an empty context it would measure
  willingness to refuse. `--prompt-probe` runs the RAG template with an empty
  context on 100 queries so the size of that confound is measured rather than
  argued about. State it in the paper either way.
- **C2** is the oracle ceiling and defaults to `--oracle-source qrels`: the
  chunks a perfect retriever would return under the *same* qrels NDCG@5 is
  scored against. The alternative, `gold_context`, is strictly easier —
  on HotpotQA it hands over the supporting sentences stripped of their
  paragraphs, which no retriever could ever return. Running both is
  informative; reporting only the second would be a ceiling on something the
  paper never measures.
- `--emit-filter` writes the sampling justification: keep only queries the
  model gets **wrong** without retrieval. That is a principled answer to
  "why 1000 instances", and the retained fraction is itself a finding about
  benchmark contamination.

**5c — score and read the grid (free).**
```bash
python src/run_pipeline.py --full --phases dcond   # NLI for the conditions
python src/conditional.py                          # the three tables
python src/conditional.py --filtered               # same, filter applied
```
`conditional.py` prints the table the letter is really asking for:

| | answer correct | answer incorrect |
|---|---|---|
| **retrieval hit** | as the premise expects | **retrieval NOT SUFFICIENT** |
| **retrieval miss** | **retrieval NOT NECESSARY** | as the premise expects |

Mean faithfulness inside the *hit × incorrect* cell is the paper's sharpest
number: grounded in correctly retrieved text, and wrong anyway. `faith_gap` in
the second table is `faithfulness(incorrect) − faithfulness(correct)`; if it is
positive, the model is **more** faithful when wrong, and the pooled
faithfulness column should not be reported on its own again.

**5d — context ablation (~$4.40 for 8 conditions × 300 queries × 1 dataset).**
His third suggestion: hold relevance perfect, vary only subset and order.
```bash
python src/context_ablation.py --datasets NQ --limit 300 --dry-run
python src/context_ablation.py --datasets NQ --limit 300 --yes
python src/correctness.py
python src/context_ablation.py --datasets NQ --report
```
Distractors are the reference embedder's own top-20 misses — hard negatives a
real retriever returns, not random text. Two comparisons carry the argument:
`gold_reversed − gold_all` (identical information, different order) and the
spread across `gold1_first / middle / last` (relevance *and* context length
held constant, only position moves). `noise_only` doubles as a floor measured
through the real RAG prompt, which is the confound C1 cannot avoid.

**5e — document utility (~$3.40 for 100 queries).** His Shapley suggestion and
the paper he linked are the same idea at two levels of generality.
```bash
python src/doc_utility.py --mode shapley --datasets NQ --limit 100 --dry-run
python src/doc_utility.py --mode shapley --datasets NQ --limit 100 --yes
python src/correctness.py
python src/doc_utility.py --report --datasets NQ
```
`--mode shapley` enumerates all 2⁵ = 32 subsets per query, so **eRAG and
leave-one-out come out of the same 32 generations at no extra cost** — running
them separately would only waste money. The report correlates each document's
utility against its qrels relevance label; a weak correlation is
"good retrieval is not sufficient" stated per document instead of per system.
Watch the negative-utility count: documents the qrels call relevant that make
the answer *worse* are the single most quotable observation available here.

**Cost summary for Rung 5**

| Step | Requests | Cost |
|---|---|---|
| 5a correctness | 0 | free |
| 5b C1 + C2 (N=1000 × 3 datasets) | 6,000 | ~$7.00 |
| 5b prompt probe | 300 | ~$0.03 |
| 5c conditional tables | 0 | free |
| 5d context ablation (8 × 300, NQ) | 2,400 | ~$4.40 |
| 5e Shapley (100 queries, NQ) | 3,200 | ~$3.40 |
| **total** | **~11,900** | **~$15** |

---

## 6. Reading the output

`outputs/` gets `full_results.csv` plus Fig. 1–4. The console report prints
per-model NDCG@5 / faithfulness / RFG / nRFG, the robustness verdict, and the
H1–H5 table.

Two conventions to hold onto when you write these up:

- **nRFG is primary**; raw RFG is a secondary diagnostic and must always appear
  next to absolute faithfulness, so a both-low system stays visible.
- **Report failed hypotheses as failed.** Berend values that over inflated
  claims, and the honest version is the publishable one.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Everything times out at once | SSH rate limit. Wait ~15 min. Use ControlMaster, don't loop |
| `ping` fails | Normal, ICMP blocked. Meaningless signal |
| `ssh szte-gpu` hangs at banner exchange | ProxyJump flaking. Two-step: `ssh szte-hop`, then `ssh gpu1` |
| `torch.cuda.is_available()` is `False` | CPU wheel installed. Reinstall from a CUDA index |
| `no kernel image is available for execution on the device` | torch wheel lacks V100 `sm_70` kernels — see §3.2. `is_available()` returning `True` does **not** rule this out |
| `python3 -m venv` fails on `ensurepip` | `python3.10-venv` not installed (needs sudo). Use `pip install --user` |
| Anthropic 400 mentioning `effort` | `output_config.effort` errors on Haiku 4.5 — don't pass it |
| Generation cost climbing faster than projected | Check the probe's assumptions: longer chunks mean more input tokens. Lower `budget_limit` and re-probe |
| CUDA OOM on Llama-3 | Shouldn't happen at 32 GB. Check for another process with `nvidia-smi`; else set `LLAMA_FORCE_8BIT = True` in `config.py` |
| Llama-3 download 401/403 | `HF_TOKEN` unset, or the gated license was never accepted |
| Assertion `NDCG@5 == 0` | The B3 qrels/ID mismatch is back. Do not "fix" by loosening the assert — it exists to catch exactly this |
| NLI prints `entailment index: 2` | Label resolution broke. Index 2 is *neutral* for this checkpoint. All faithfulness output is invalid until fixed |
| Run died overnight | Just rerun the same command — phases resume from checkpoints |
| Want to redo one phase | Delete its `checkpoints/*.pkl` and rerun with `--phases` |

Fast local regression check before touching pipeline logic (no GPU, no
downloads, no network):
```bash
python tests/test_local_smoke.py
```

---

## 8. Moving to a new machine

### ⚠️ The one thing you cannot regenerate: your SSH private key

`~/.ssh/id_ed25519` is the key **Berend authorized by hand** on the hop
(`ssh-ed25519 AAAAC3...t/T0`). It is deliberately not in git. Lose it and you
lose all server access until he authorizes a replacement — which means waiting
on a busy person.

Copy **`~/.ssh/id_ed25519`** and **`~/.ssh/id_ed25519.pub`** to the new machine
by hand (encrypted USB, password manager, or a secure transfer). Then:

```bash
chmod 600 ~/.ssh/id_ed25519
```

Windows: the file must not be world-readable, or SSH refuses it. If it
complains, fix inheritance via *Properties → Security → Advanced → Disable
inheritance*, leaving only your own account.

### Getting the repo

Two copies exist. Either works:

```bash
# From the bundle you carried across
git clone rag-faithfulness-2026-07-26.bundle rag-faithfulness

# Or straight from gpu1 (needs the SSH key first)
ssh szte-hop
git clone ~/rag-faithfulness.git       # on the hop, or scp it down
```

A bare repo also lives at `gpu1:~/rag-faithfulness.git` and can serve as a
real `git remote` once your laptop key is authorized on gpu1 (see §4).

### Then

```bash
cp server/ssh_config ~/.ssh/config     # edit per §3.1 — Windows must NOT use ControlMaster
ssh szte-gpu-shell                     # verify access
```

### What you do NOT need to move

- **`checkpoints/`** — all live on gpu1 under `~/rag_faithfulness/`, along with
  the installed Python environment. Nothing is recomputed.
- **`key.txt` / your API key** — rotate it instead. Any key that has been in a
  chat window, a shell history, or a plaintext file on shared infrastructure
  should be considered public.

---

## 9. Still open after the experiments run

1. **Paper §6** — rewrite "Expected Results / H1–H5" into real Results +
   Discussion.
2. **Ref [8]** cites a non-existent "jina-embeddings-v5-text" → must become
   jina-embeddings-v3, arXiv:2409.10173.
3. **§4.5.2** promises Llama-3 *cross*-attention analysis. Llama-3 is
   decoder-only, so it has self-attention over context tokens — and no code
   implements it. Either cut the claim or scope it honestly.
4. **Reformat IEEE → ACL** before any ARR submission.
5. **Confirm the venue with Berend** — the EMNLP 2026 May deadline has passed;
   July ARR or COLING 2026 are the live options.
