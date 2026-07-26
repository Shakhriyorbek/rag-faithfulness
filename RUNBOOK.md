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

Install torch first (CUDA build), then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Sanity check before anything else:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print `True` and the V100 name. If `False`, you installed the CPU wheel —
reinstall torch from the cu121 index.

### 3.3 Secrets — never commit these

```bash
export OPENAI_API_KEY='sk-...'
export HF_TOKEN='hf_...'          # needed: Llama-3 is a gated model
```

Put them in `~/.bashrc` on gpu1 so tmux panes inherit them. `HF_TOKEN` also
requires accepting the Llama-3 license once on the HuggingFace website.

---

## 4. Getting code onto gpu1

gpu1 has `rsync`, but **your laptop does not** — so remote rsync is out. No
GitHub remote exists yet either. Pick one:

**Option A — GitHub (recommended, gpu1 has internet).** Create a private repo,
then the loop is `git push` on the laptop, `git pull` on gpu1. Best for
iterating, and it backs up your work.

**Option B — `scp`, no third party.**
```bash
scp -r src tests requirements.txt szte-gpu:~/rag_faithfulness/
```

**Worth checking once** (may make this trivial): if `$HOME` is shared between
hop and gpu1, copying to either is enough.
```bash
ssh szte-hop 'touch ~/sharetest' && ssh szte-gpu 'ls ~/sharetest && rm ~/sharetest'
```

**Never copy `checkpoints/`** — they are large, regenerable, and gitignored.
Pull *results* back the other way instead:
```bash
scp -r szte-gpu:~/rag_faithfulness/outputs ./outputs
```

---

## 5. The execution ladder

Climb in order. **Each rung is a decision gate** — check the stated condition
before spending time on the next one. Always inside `tmux`:

```bash
tmux new -s rag        # detach: Ctrl-B then D    reattach: tmux attach -t rag
source .venv/bin/activate
```

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

**Gate — this is the one that matters.** Look at the nRFG spread across the
three models.
- **Spread is visible and models rank differently than NDCG@5 does** → the
  paper's premise survives contact with data. Continue.
- **All three nRFG values are within noise** → *stop and think.* The core claim
  may not hold on NQ. That is a real finding, not a failure — bring it to
  Berend before burning days on the full grid. Single-hop NQ is also the least
  likely dataset to show the effect, so HotpotQA is the natural next probe.

### Rung 3 — Full grid, GPT-4o-mini (~12–20 h, ~$5–6)

```bash
python src/run_pipeline.py --full
```
7 models × 3 datasets. Watch the running cost line; the budget cap raises a
`RuntimeError` at $15 rather than draining the account. Partial checkpoints
survive it — raise `budget_limit` and rerun to resume.

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
| `torch.cuda.is_available()` is `False` | CPU wheel installed. Reinstall from the cu121 index |
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

## 8. Still open after the experiments run

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
