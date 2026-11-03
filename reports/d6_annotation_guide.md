# D6 — checking the LLM judge against human labels

The third review: "The LLM judge is never checked against human annotation."
This is the check. It costs nothing but annotation time (about 200 items,
roughly 1.5–2 hours for one person).

## What to do

1. Open `reports/d6/annotation_sheet.csv` in a spreadsheet (LibreOffice Calc,
   Google Sheets or Excel). Each row is one item: a question, the reference
   answer(s), and a generated answer.
2. Fill the `label` column with exactly one of `CORRECT`, `INCORRECT`,
   `REFUSAL`. Use `note` for anything worth recording (an ambiguous reference,
   an answer that is right but not the reference, a wrong reference).
3. Save as CSV with the same columns. Do not reorder or delete rows.
4. Run:

       RAG_CHECKPOINT_DIR=~/rag-backup/checkpoints/n1000_v3 \
           python3 scripts/d6_human_agreement.py

   A partly labelled sheet works; unlabelled rows are skipped.

A second annotator is worth having if one is available (Berend's group, a
colleague). They label a copy of the same sheet independently, without seeing
the first annotator's labels, and the script takes it with
`--second path/to/their_sheet.csv`. Their agreement with each other is the
ceiling any judge could reach against one person.

## The rule — the judge's own definitions, verbatim

The judge (claude-opus-5, effort low) received exactly this, with the three
fields filled in. Label by the same definitions, so that a disagreement means
the judge applied them wrongly, not that the two of you used different rules:

> The answer is CORRECT if it conveys the reference answer, even if it is
> phrased differently, adds extra correct detail, or is embedded in a longer
> sentence. The answer is INCORRECT if it contradicts the reference, states a
> different value, or omits the reference answer entirely. If the answer
> declines to answer or says the information is not available, that is
> REFUSAL.

Practical readings of that rule:

- **Grade against the reference, not against what you know.** If the answer is
  plausibly true but is not the reference, it is INCORRECT, and say so in
  `note`. The judge was told the same.
- **Extra detail does not make an answer wrong** unless the extra detail
  contradicts the reference.
- **Decline-then-answer.** An answer that says the context is insufficient and
  then gives an answer anyway: label by what it asserts. If it ends by
  conveying the reference, CORRECT; if it commits to a different answer,
  INCORRECT; if it only declines, REFUSAL. Write `decline-then-answer` in
  `note` for these — the paper reports 220 such rows, and this is where the
  judge's refusal flag and the string rule part company.
- **Several references** are separated by `|`. Conveying any one of them is
  enough.
- **Do not look anything up.** You see what the judge saw: question, reference,
  answer. The retrieved context, the generator and the system are hidden on
  purpose.

## What is hidden, and why

The sheet carries an opaque id (`D6-001` …) and the items are shuffled across
datasets and generators. The judge's verdict, the generator, the retrieval
system and the query id are in the checkpoint `d6_key.pkl` in the checkpoint
directory. Do not open it before labelling is finished; the agreement figure is
only meaningful if the labels are blind to the verdict.

## Design (fixed before any label was collected)

- Population: the 32 judged cells of the matched grid, judged rows only.
- Sample: 25 rows from each of the 8 dataset × generator cells, pooled over the
  four systems, `random.Random(42)`; 200 items.
- Judge labels in the sample, for reference: 121 CORRECT, 20 INCORRECT,
  59 REFUSAL. INCORRECT is thin, so its agreement will carry a wide interval;
  that is a property of the population, not of the draw.
- Measures (`scripts/d6_human_agreement.py`): raw agreement with a Wilson
  interval and Cohen's kappa, over the three labels, over correct vs not
  (what every accuracy figure uses) and over refusal vs not (what `faith_gap`
  and the 2×2 take from the judge); the confusion matrix; agreement per
  generator and per dataset; every disagreement listed by item.
