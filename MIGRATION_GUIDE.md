# Migration Guide — Moving this project to Claude Code

## Recommendation: Claude Code, not Cowork

The remaining work is roughly 85% software engineering: writing eight Python modules, running them over SSH on a remote GPU box, debugging CUDA/OOM issues, and iterating on experiment results. Cowork is built for multi-step knowledge work by non-developers; this is a coding project with a paper attached.

Use **Claude Code** for the pipeline, and come back to a chat interface (or Claude in Word) later for the prose-writing pass on the Results and Discussion sections.

### Where to run Claude Code

**Recommended: locally on your laptop, in a git repo.**
Develop and smoke-test with `N_QUERIES=50` locally on the Ryzen, then `scp` or `git pull` to gpu1 for the full run. This keeps your work version-controlled and survives server maintenance windows.

**Alternative: on gpu1 itself**, if it has Node.js and internet. Convenient for tight debug loops against the real GPU, but you lose local git history unless you push. Check `node --version` on gpu1 first.

---

## Setup steps

### 1. Create the repo locally

```powershell
cd $env:USERPROFILE\Documents
mkdir rag-faithfulness
cd rag-faithfulness
git init
```

Copy every file from this handoff package in, preserving the folder layout below.

```
rag-faithfulness/
├── CLAUDE.md                    ← auto-loaded by Claude Code every session
├── MIGRATION_GUIDE.md           ← this file
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py                ← 7 models incl. Jina, metric grid
│   └── metrics.py               ← RFG, nRFG, 9-variant robustness
├── server/
│   ├── ssh_config               ← → ~/.ssh/config
│   └── first_login_checks.sh    ← run on gpu1 first
├── paper/
│   └── RAG_Faithfulness_IEEE_v3.docx
├── notebooks/
│   └── rag_faithfulness_research_v5.ipynb   ← source for porting logic
├── checkpoints/                 ← gitignored, created at runtime
└── outputs/                     ← gitignored, figures + results
```

### 2. Install and start Claude Code

```powershell
npm install -g @anthropic-ai/claude-code
cd $env:USERPROFILE\Documents\rag-faithfulness
claude
```

`CLAUDE.md` is picked up automatically — you don't need to paste any history.

### 3. First prompt to give it

> Read CLAUDE.md. We're resuming a research project. Start with item 1 in §8: port the three dataset loaders out of `notebooks/rag_faithfulness_research_v5.ipynb` into `src/datasets_loader.py`, keeping the bug fixes described in §7 exactly as they are. Then write a smoke test that loads 10 samples from each dataset and prints them.

Work through §8 in order. Items 1 and 2 block everything downstream.

---

## First real milestone

Before writing any more pipeline code, get the server facts:

```bash
ssh szte-gpu
bash first_login_checks.sh
```

Two answers change the code:
- **V100 16GB vs 32GB** → whether Llama-3-8B needs 8-bit quantization
- **Does gpu1 reach the internet?** → whether HuggingFace models download directly or must come via the hop

Feed that output to Claude Code and let it finalize `src/generate.py` accordingly.

---

## Suggested milestone order

1. Server inventory (above)
2. `datasets_loader.py` + smoke test — verify all three datasets load
3. `embed_index.py` — start with **3** models on **NQ only** to prove the path works
4. `generate.py` GPT-4o-mini branch — watch the cost tracker, stop if it projects past $15
5. `faithfulness.py` — NLI first (cheap), AlignScore second
6. Assemble a first RFG table on the reduced setup. **If the RFG differences show up here, the paper has a result.**
7. Only then scale to 7 models × 3 datasets
8. `esa_analysis.py` with both NLI variants — this closes supervisor point 6
9. `rerank.py` ablation
10. Figures, then the paper rewrite

Running the reduced setup first means you find out whether the core finding exists after a few hours instead of a few days.

---

## What to keep in the chat interface instead

- Rewriting §6 from hypotheses into real Results/Discussion prose
- Any further email drafts to Berend or Jelasity
- Reformatting the paper from IEEE to the ACL template — required before any ARR submission, and the .docx work is easier in a chat session

---

## Timeline note

The EMNLP 2026 May 25 ARR deadline has passed. Realistic targets now are the July ARR cycle (for a later EMNLP/Findings commitment) or COLING 2026. Confirm the target with Berend before locking a schedule — he offered server access, which signals he's invested, and he's the right person to pick the venue.
