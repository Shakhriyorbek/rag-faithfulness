# Draft reply to Dr. Berend — 2026-08-28

> **Notes before sending — not part of the message.**
>
> Deliberately short (~450 words). His first complaint about the last reply was
> that it was a wall of text, and the second was that it referred to things
> never discussed. Both are addressed: the TOST misattribution is corrected in
> the opening line, and nothing here assumes context he does not have.
>
> The prose is intentionally plainer than the previous draft. He said the
> writing reads as genAI output with surface polish and no substance. Rewrite
> any sentence below that sounds unlike you before sending — that is the point,
> and it matters more than the wording being optimal.
>
> Attach `paper/RAG_Faithfulness_v6_evaluators.docx`.
>
> Two questions are asked. Do not add a third; he answers what is asked and a
> longer list gets a shorter reply.

---

**Subject:** Update — new results, and a correction

Dear Dr. Berend,

First a correction. In my last message I wrote that the paired TOST was
something you had pointed me towards. That was wrong. You never mentioned it —
I added it myself and then credited it to you. I am sorry, and I have gone back
through the draft for anything else attributed to you that you did not say.

Since then I ran a second faithfulness evaluator, AlignScore, which a reviewer
would expect. It changed the result. Under the NLI metric I had been using,
none of the four dataset-generator cells showed a difference between embedding
models. Under AlignScore two of them do, at p = 0.0005. The queries, the
answers and the test are identical; only the evaluator changed.

I also tested the metric itself. I took answers that were grounded, replaced
one number with a value not present in the retrieved context, and re-scored.
On Claude's answers the NLI score dropped by 0.04, and a 0.5 threshold caught
4% of them. On GPT-4o-mini's shorter answers the same edit dropped 0.46 and
was caught 58% of the time. The difference comes from how many separate
statements an answer contains — Claude averages 3.0, GPT-4o-mini 1.2 — and the
score decays as that number grows. I checked whether Claude was simply quoting
the context back, and it is not: the verbatim overlap is the same for both.

This has made me rewrite the paper. The old title said retrieval quality
predicts correctness and not faithfulness. My own data contradicts it — the
answer depends on the evaluator, and on HotpotQA the correlation is positive.
The new version is about the measurement, with the embedding grid kept as the
testbed. I want to be clear that this is not me chasing the data a third time:
it is the part that you and the ChatGPT review independently picked out as the
strongest, so I have made it the paper.

Two questions.

Is that reframing acceptable? The alternative is to keep the embedding
comparison as the main claim and report the evaluator sensitivity as a
secondary result.

Should ESA and the re-ranking ablation stay? Neither has been run. Under the
new framing they are optional, and I would rather drop them than include them
thinly.

One practical note: my university laptop is being returned, so I have moved to
my own Linux machine. Everything is on the server and in git; nothing is lost.
Next I plan to run an LLM judge for correctness — exact match is 0.000 because
the models never produce bare answer spans, so accuracy currently rests on
string containment — and then an open-weight generator as you suggested, Qwen
or Gemma. That one now does real work rather than only reproducibility, since
the effect above depends on answer length and a third generator would test it.

Best regards,
Shakhriyorbek
