# Moving to a new Linux device

The Windows laptop is being returned. Everything that matters is either in git
or on gpu1 — the **only** thing that cannot be regenerated is SSH access.

Do §1 **before you hand the Windows machine back.** Step 1.2 needs a working
connection from the old device, and once it is gone the fallback is asking
Dr. Berend to re-authorize a key.

---

## 1. SSH access — do this first

### The recommended path: enrol a NEW key, don't move the old one

Copying a private key between machines means it exists in more places, usually
including a USB stick or a cloud folder along the way. Generating a fresh key
on the Linux box and authorizing it is safer and takes two extra minutes.

Your key is authorized on **both** the hop and gpu1 — Berend authorized it on
the hop, and you appended it to gpu1 yourself on 2026-08-13. So you can enrol
the new key on both hosts yourself, from the old machine, while it still works.

#### 1.1 — On the NEW Linux device: generate a keypair

```bash
ssh-keygen -t ed25519 -C "shakhriyorbekboltabaev@gmail.com" -f ~/.ssh/id_ed25519
chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_ed25519
```

Set a passphrase when prompted. Then print the **public** key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy that one line. It starts `ssh-ed25519 AAAAC3...`. The public key is safe
to paste anywhere; the file *without* `.pub` is the private key and must never
leave this machine.

#### 1.2 — On the OLD Windows machine: authorize it on both hosts

Paste the public key from 1.1 in place of `PASTE_PUBLIC_KEY_HERE`. Run each
line separately and confirm it succeeded before the next — the gateway
rate-limits connections.

```bash
ssh szte-hop 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo "PASTE_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && tail -2 ~/.ssh/authorized_keys'
```

```bash
ssh szte-gpu 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo "PASTE_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && tail -2 ~/.ssh/authorized_keys'
```

**Both are required.** `$HOME` is not shared between the hop and gpu1 — this
cost weeks of confusion in July. Authorizing only the hop is exactly the state
that made `ssh szte-gpu` fail with `Permission denied (publickey)`.

#### 1.3 — Fallback: move the existing key by hand

Only if you have already lost access. Copy `~/.ssh/id_ed25519` **and**
`id_ed25519.pub` from the Windows machine (`C:\Users\<you>\.ssh\`) to
`~/.ssh/` on Linux by a means you control — a USB stick you then wipe, not
email or cloud sync. Then:

```bash
chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_ed25519 && chmod 644 ~/.ssh/id_ed25519.pub
```

If you have lost access entirely, the hop key must be re-authorized by
Dr. Berend; only gpu1 can be fixed by you afterwards.

#### 1.4 — SSH config

```bash
cp server/ssh_config ~/.ssh/config && chmod 600 ~/.ssh/config
```

Then **uncomment the last block** — connection multiplexing is Linux/macOS
only and was disabled for Windows. This is the one thing that gets *better* on
Linux, and it matters because the gateway blocks your IP after roughly six
connections in a few minutes:

```
Host szte-*
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

One-liner:

```bash
sed -i 's/^# \(Host szte-\*\)/\1/; s/^#     \(ControlMaster\|ControlPath\|ControlPersist\)/    \1/' ~/.ssh/config
```

Do **not** change `HostName 192.168.0.206` for `szte-gpu` to `gpu1`. The alias
exists only inside the hop's own config, and ProxyJump does a literal DNS
lookup that ignores it.

#### 1.5 — Verify

```bash
ssh szte-gpu 'hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

Expect `nlp-large-1` and `GRID V100DX-32C, 32768 MiB`. `ping` will always fail
— ICMP is blocked — so never use it as a test.

#### 1.6 — After it works, revoke the old key

Only once 1.5 succeeds from the Linux box. Find the old key's line and delete
it from `~/.ssh/authorized_keys` on **both** hosts:

```bash
ssh szte-hop 'cp ~/.ssh/authorized_keys ~/.ssh/authorized_keys.bak && nano ~/.ssh/authorized_keys'
ssh szte-gpu 'cp ~/.ssh/authorized_keys ~/.ssh/authorized_keys.bak && nano ~/.ssh/authorized_keys'
```

Keep the backup until you have reconnected successfully afterwards. Locking
yourself out of the hop means emailing Berend.

---

## 2. The repository

```bash
git clone https://github.com/Shakhriyorbek/rag-faithfulness.git
cd rag-faithfulness
git remote add gpu1 szte-gpu:rag-faithfulness.git
```

Everything is pushed to both remotes. The gpu1 remote lets you deploy without
putting a GitHub credential on the university server.

Deploy loop:

```bash
git push gpu1 main
ssh szte-gpu 'cd ~/rag-faithfulness && git pull'
```

---

## 3. Local Python

Only for running the offline test suite while editing. **No heavy phase ever
runs locally** — no torch, no models, no API calls.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pytest numpy pandas scipy
python -m pytest tests/ -q
```

Expect **0 failures**. The passed/skipped split depends on which optional
packages happen to be present: with only the four above you get 135 passed /
15 skipped; with `transformers` also installed, 139 / 11; on gpu1, where
`torch` and `openai` are present too, all 150 run. Tests needing a heavy
dependency skip rather than fail, so a count is not the thing to check — a
failure is.

(`python3 -m venv` works here. It is broken on *gpu1* — `ensurepip` missing,
needs sudo — which is why gpu1 uses `pip install --user`.)

---

## 4. What is NOT in the repo

| thing | where it lives | if lost |
|---|---|---|
| **SSH private key** | your machine only | §1 |
| **API keys** | `~/.rag_keys` on gpu1, mode 600 | regenerate from the provider consoles |
| **Checkpoints** (16k generations, all scores) | `gpu1:~/rag_faithfulness/checkpoints/n1000_v3/` | **~$13.46 of API spend to recreate** — back these up |
| **AlignScore install** | `gpu1:~/align_env/` | recipe in `RUNBOOK.md` |
| **AlignScore checkpoint** (455 MB) | `gpu1:~/rag_faithfulness/alignscore/` | re-download, see RUNBOOK |
| **HF model cache** | `gpu1:~/rag_faithfulness/hf_cache/` | re-downloads automatically |

**Back up the checkpoints.** They are gitignored and they are the only copy of
paid work. Linux has `rsync`, which the Windows box did not:

```bash
mkdir -p ~/rag-backup/checkpoints && rsync -avz --progress szte-gpu:rag_faithfulness/checkpoints/ ~/rag-backup/checkpoints/
```

`mkdir -p` first: rsync creates the final path component but not intermediate
parents, and fails with `mkdir ... No such file or directory` without it.
(`rsync --mkpath` does the same job on rsync 3.2.3+.)

---

## 5. API keys on gpu1

Needed only for generation phases. Never commit them — `.gitignore` covers
`key*.txt`, `*.key`, `.env`, `*secret*`, `*token*`.

| key | gates |
|---|---|
| `ANTHROPIC_API_KEY` | Claude generation, C1/C2, re-ranking |
| `OPENAI_API_KEY` | `text-embedding-3-small` **and** GPT-4o-mini |
| `HF_TOKEN` | Llama-3 / Qwen / Gemma (gated repos) |

Set them inside your tmux session without storing them:

```bash
read -s -p "key: " ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY && echo "set (${#ANTHROPIC_API_KEY} chars)"
```

If you keep them in a file, **strip CRLF** — a file written on Windows leaves a
trailing `\r` and every call returns 401:

```bash
export ANTHROPIC_API_KEY=$(sed -n '1p' ~/.rag_keys | tr -d '\r\n')
```

Since keys have been sitting on a shared university machine, rotating them
once the experiments finish is worth doing.

---

## 6. First thing to run on the new setup

```bash
ssh szte-gpu
tmux new -s rag
cd ~/rag-faithfulness && git pull
python3 src/preflight.py
```

`preflight.py` is free and read-only. It checks keys and — importantly —
verifies torch with a real matmul and confirms `sm_70` is in
`torch.cuda.get_arch_list()`. `torch.cuda.is_available()` is **not** a
sufficient check on this V100: the default PyPI wheel ships sm_75+ kernels,
imports fine, reports `cuda: True`, then dies on the first real kernel launch.

---

## 6b. Node — needed to rebuild the paper, and it did NOT survive the migration

`paper/build_v6.js` renders the .docx. Its dependency (`docx`) is **vendored in
`paper/node_modules`**, so nothing needs installing from npm — but the Fedora
box had **no node runtime at all** after the move from Windows, and this was
only noticed on 2026-09-05 when the paper first needed rebuilding. Installed
user-locally, no sudo:

```bash
curl -sSLO https://nodejs.org/dist/v24.20.0/node-v24.20.0-linux-x64.tar.xz
curl -sSL  https://nodejs.org/dist/v24.20.0/SHASUMS256.txt -o SHASUMS256.txt
grep "node-v24.20.0-linux-x64.tar.xz" SHASUMS256.txt | sha256sum -c -   # verify
tar -xJf node-v24.20.0-linux-x64.tar.xz -C ~/.local/opt/
export PATH="$HOME/.local/opt/node-v24.20.0-linux-x64/bin:$PATH"        # add to ~/.bashrc
```

**Run the build from the repo root, not from `paper/`** — the script writes to
the relative path `paper/RAG_Faithfulness_v6_evaluators.docx` and dies with
`ENOENT` if the cwd is already `paper/`:

```bash
cd ~/Documents/rag-faithfulness && node paper/build_v6.js
```

The .docx is **tracked in git**, so a source edit that is not rebuilt leaves the
committed paper stale. Rebuild and commit both together.

---

## 7. Where to pick the work up

Read in this order:

1. `PAPER_TODO.md` — what has to change to finish the paper
2. `reports/2026-08-26_evaluator_dependence.md` — the current headline
3. `reports/2026-08-25_perturbation_check.md` — the measurement results
4. `CLAUDE.md` §5 — design decisions not to reverse
5. `RUNBOOK.md` — how to run everything

The next experiment is the **LLM judge** (`PAPER_TODO.md` §3.1). It is cheap
and it blocks every accuracy number in the paper.
