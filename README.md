# RAG Faithfulness — Beyond Retrieval Quality

Experiment pipeline for the paper on whether embedding models with different
retrieval quality produce different **faithfulness** in downstream RAG
generation.

**Short answer from the data so far: they do not.** Retrieval quality governs
whether the model answers and whether it is right; it does not govern how
grounded its assertions are. See
[`reports/2026-08-14_rung2_results.md`](reports/2026-08-14_rung2_results.md).

- **Moving to a new machine (SSH keys first)** → [`DEVICE_MIGRATION.md`](DEVICE_MIGRATION.md)
- **What still has to change to finish the paper** → [`PAPER_TODO.md`](PAPER_TODO.md)
- **Project context, decisions, and every bug not to reintroduce** → [`CLAUDE.md`](CLAUDE.md)
- **Full operational detail, the execution ladder, troubleshooting** → [`RUNBOOK.md`](RUNBOOK.md)
- **This file** → getting a new machine to the point where it can run things

---

## 1. How the machines split up

```
your laptop  ──ssh──►  hop (193.225.250.29)  ──►  gpu1 (nlp-large-1, V100 32GB)
  git, ssh, editing         entry point only        ALL compute lives here
```

The laptop runs **no experiments**. It edits code, pushes to git, and drives
gpu1 over SSH. So "set up a new device" mostly means "get SSH and git working";
gpu1 already holds the code, the data and every checkpoint.

| location | holds | git |
|---|---|---|
| `gpu1:~/rag-faithfulness/` | code, a clone of the bare repo | yes |
| `gpu1:~/rag_faithfulness/` | `checkpoints/`, `hf_cache/`, `outputs/`, `logs/` | no |

Note the hyphen vs underscore — different directories. `config.BASE_DIR`
resolves to the underscore one.

---

## 2. New device setup

### 2.1 The one thing you cannot regenerate: the SSH key

`~/.ssh/id_ed25519` is authorized on **both** the hop and gpu1. It is not in
this repo and never will be.

- **Moving machines?** Copy it by hand from the old machine to
  `~/.ssh/id_ed25519` on the new one, then `chmod 600 ~/.ssh/id_ed25519`
  (on Windows, ensure only your user has access or SSH refuses it).
- **Lost it?** Generate a new keypair and ask Dr. Berend to authorize the
  public key on the hop, then append it to `gpu1:~/.ssh/authorized_keys`
  yourself. Berend originally authorized it on the hop **only**, which is why
  direct `ssh szte-gpu` failed for weeks.

### 2.2 SSH config

```bash
cp server/ssh_config ~/.ssh/config
```

Read the comments in that file before editing it. Two things that have already
cost days:

- `HostName` for `szte-gpu` **must be the private IP `192.168.0.206`**, not the
  alias `gpu1`. ProxyJump runs `ssh -W gpu1:22`, which does a literal DNS
  lookup and ignores Host aliases defined on the hop.
- **`ControlMaster` is Linux/macOS only.** Windows OpenSSH has no Unix domain
  sockets; enabling it makes every connection fail. It is commented out.

Verify:

```bash
ssh szte-gpu 'hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

Expect `nlp-large-1` and `GRID V100DX-32C, 32768 MiB`.

> **The gateway rate-limits SSH.** Roughly six connections in a few minutes got
> the source IP temporarily blocked. Use one long-lived session; never script
> rapid reconnects. `ping` always fails — ICMP is blocked — so test with TCP 22.

### 2.3 Clone

```bash
git clone https://github.com/Shakhriyorbek/rag-faithfulness.git
```

Add the gpu1 remote so you can deploy without putting a GitHub credential on
the university server:

```bash
git remote add gpu1 szte-gpu:rag-faithfulness.git
```

### 2.4 Local Python (optional)

Only needed to run the offline test suite while editing. Heavy phases never run
locally.

```bash
pip install pytest numpy pandas scipy
python -m pytest tests/ -q
```

Expect **87 passed, 11 skipped** — the skips are tests needing `openai` or
`torch`, which are absent locally by design. On gpu1 all 98 run.

---

## 3. Deploying and running

### Sync loop

```bash
git push gpu1 main
```

then on gpu1:

```bash
cd ~/rag-faithfulness && git pull
```

### API keys — never commit these

`key.txt` is gitignored and is **not** in the repo, so a new device will not
have it. Keys are needed only on gpu1, and only for generation phases.

| key | gates |
|---|---|
| `ANTHROPIC_API_KEY` | Claude generation (phase `c`), C1/C2, re-ranking |
| `OPENAI_API_KEY` | `text-embedding-3-small` embedder **and** GPT-4o-mini (phase `cgpt`) |
| `HF_TOKEN` | Llama-3-8B only (gated repo) |

Set them inside your tmux session on gpu1 without echoing or storing them:

```bash
read -s -p "key: " ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY && echo "set (${#ANTHROPIC_API_KEY} chars)"
```

An Anthropic key is ~108 chars. If you keep them in a file instead, **strip
CRLF** — a file written on Windows leaves a trailing `\r` on each key and every
API call returns 401:

```bash
export ANTHROPIC_API_KEY=$(sed -n '1p' ~/.rag_keys | tr -d '\r\n')
```

Delete any such file when finished (`shred -u ~/.rag_keys`) and rotate keys
that have lived on a shared machine.

### Always use tmux

```bash
ssh szte-gpu
tmux new -s rag          # detach: Ctrl-B then D    reattach: tmux attach -t rag
```

The run survives your laptop sleeping or shutting down; it does **not** survive
a reboot of gpu1. Checkpoints make any interrupted run resumable — re-issue the
same command and finished work is skipped.

### Free launch gate

```bash
cd ~/rag-faithfulness && python3 src/preflight.py
```

Read-only, costs nothing, and fails loudly on a missing or malformed key.
`ANTHROPIC_API_KEY` is fatal; OpenAI and HF are warnings.

### Run

```bash
python3 -u src/run_pipeline.py --full --datasets NQ --models all-mpnet-base-v2,E5-large-instruct,BGE-M3,text-embedding-3-small --phases a,b,c,cgpt,d,correct,cond --yes
```

Phases: `a` embed+retrieve · `b` retrieval quality · `c` Claude **[$]** ·
`cgpt` GPT-4o-mini **[$]** · `d` NLI faithfulness · `correct` grading ·
`cond` the necessary/sufficient grid · `c1`/`c2` floor/ceiling **[$]** ·
`esa`, `rerank` **[$]**, `llama`, `e`, `report`.

Paid phases refuse to run without `--yes`. Generation aborts after 5
consecutive API errors or a 20% error rate, and hard-stops at a $60 spend cap.

---

## 4. ⚠️ Two things that will silently corrupt a run

### Checkpoint scope

Every run prints its checkpoint directory as its **first line**:

```
  [scope] checkpoints -> /home/sboltabaev/rag_faithfulness/checkpoints/n1000_v3
```

Checkpoint keys are named for their content (`retrieval_BGE-M3_NQ`), not for
the run, so scoping is done by **directory**: `n{N}_{CORPUS_VERSION}`. Before
this existed, a full N=1000 run reused the N=50 pilot's checkpoints, printed
"all datasets done, skipping", and **exited 0 having done nothing**. If a phase
finishes suspiciously fast, read that line first.

- `RAG_SCOPE_N=50` — work in another N's scope
- `RAG_CHECKPOINT_DIR=...` — absolute override; needed to read the pre-scope
  pilot checkpoints, which still sit flat in `checkpoints/`

Bump `CORPUS_VERSION` in `src/config.py` whenever corpus or relevance semantics
change; that forces a clean rebuild instead of silently reusing a cache built
under different rules.

### torch on the V100

The V100 is **Volta, sm_70**, and the default PyPI wheel ships sm_75+ kernels
only. It imports fine, reports `cuda: True`, then dies on the first real kernel
launch. `torch.cuda.is_available()` is **not** a sufficient check — confirm
`torch.cuda.get_arch_list()` contains `sm_70` and run a real matmul.
`preflight.py` does both. Reinstalling needs `--force-reinstall`; pip treats
`+cu130` and `+cu126` as the same version and silently skips.

Also: `python3 -m venv` is broken on gpu1 (`ensurepip` missing, needs sudo).
Use `pip install --user`.

---

## 5. Where the work stands

**Done** — Rung 2, $13.46 spent of ~$55, zero API errors:

| | NQ | HotpotQA |
|---|---|---|
| queries × embedders × generators | 1000 × 4 × 2 | 1000 × 4 × 2 |
| NDCG@5 spread | 4.4 pts | 11.9 pts |
| faithfulness pairs equivalent (±0.05) | 6/6 both generators | 6/6 Claude, 4/6 GPT |
| hit × incorrect ("not sufficient") | 23.3% | 30.2% |

**Not yet run:** LLM-judge correctness (blocks the accuracy numbers — EM is
0.000, see the results report), C1/C2 floor and ceiling, ESA, re-ranking,
Llama-3, QASPER.

**Reading order for someone new:** this file → `CLAUDE.md` §5 (design decisions
not to reverse) → `reports/2026-08-14_rung2_results.md` (what the data says) →
`RUNBOOK.md` (how to run the rest).
