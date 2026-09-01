"""
legacy/ — code kept for reproducibility of superseded results.

Nothing in this package is part of the v6 measurement spine. It is here so a
reader can re-derive the numbers in the earlier drafts, not because the
pipeline needs it.

  rfg.py   RFG = RetrievalQuality - Faithfulness, and nRFG = RFG / RQ.
           RETIRED — see Section III-C of the paper. The gap metric subtracts
           a faithfulness score from a ranking metric, and its cross-generator
           agreement turned out to be an algebraic artifact: nRFG = 1 - F/RQ
           collapses onto the RQ ranking whenever the spread in F is small,
           which is exactly the regime this grid is in. Reached the pipeline
           only through `--legacy-rfg`.
"""
