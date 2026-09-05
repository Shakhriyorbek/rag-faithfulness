# ACL manuscript

The paper's source, converted from the IEEE `.docx` builder on 2026-09-05.

**This directory is now the paper.** `paper/build_v6.js` and
`paper/RAG_Faithfulness_v6_evaluators.docx` are superseded and should not be
edited — an edit there will not reach the manuscript. They are kept only so the
IEEE rendering stays available for comparison; delete them once you are happy
with this version.

| file | what it is |
|---|---|
| `main.tex` | the paper |
| `custom.bib` | the 24 references, generated from the list verified 2026-09-03 |
| `acl.sty`, `acl_natbib.bst` | official ACL style files, unmodified |
| `acl_latex.tex` | the official template, kept for reference only — not built |
| `build.sh` | builds both PDFs |

## Building

```bash
./build.sh
```

Produces:

- **`main.pdf`** — `review` mode: anonymous, line-numbered. This is the file
  ARR wants.
- **`main-preprint.pdf`** — `preprint` mode: named, page-numbered. The reading
  copy to send to a supervisor.

The only toolchain requirement is `tectonic`, a single self-contained binary
(no TeX Live install, no sudo):

```bash
curl -sSL https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-gnu.tar.gz \
  | tar -xz -C ~/.local/bin tectonic
```

The first run downloads the LaTeX package bundle and takes a few minutes; later
runs take seconds. Overleaf also compiles this directory as-is.

## What the conversion changed

Format only — **no number, and no sentence of argument, was altered.** All 234
prose fragments were checked to survive the conversion verbatim.

- IEEE numbered citations `[1]` → ACL author-year (`\citet` / `\citep`), and
  the reference list → `custom.bib`. Author lists and titles are carried over
  exactly as verified on 2026-09-03; they were **not** re-typed from memory.
- Roman/letter section numbers (`IV.`, `VI-B.`) → LaTeX-numbered sections, with
  every cross-reference converted to `\ref` so they cannot drift again.
- Limitations moved after the Conclusion and made unnumbered, as ACL requires.
  One lead-in sentence was written for it — the only new prose in the file.
- Nine tables became `table`/`table*` floats with real captions.

## ⚠️ Over the ARR page limit

Content (Sections 1–10) runs to about **9.4 pages against ARR's 8-page limit**
for a long paper. Limitations and References do not count toward the limit, so
the 12-page total is not the number to look at. **Roughly 1.5 pages have to
come out before submission** — exceeding the limit is a desk reject, not a
reviewer complaint.

## ⚠️ The content is the v6 draft, not the current results

No result was updated during the conversion. The paper still does not reflect
the claim-level full grid (B22) or the open-weight arm (B23): Table VII here is
the 8-row version and the headline reads "1 of 4 cells" where the current data
says **2 of 6** over an 18-row table. The Limitations section still lists the
missing open-weight generator as a limitation, which it no longer is. See
`PAPER_TODO.md`.
