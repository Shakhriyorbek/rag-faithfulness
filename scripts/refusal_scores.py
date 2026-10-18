#!/usr/bin/env python3
"""Mean evaluator score on refusals vs attempted answers (paper Table 11 row 5).

Answers the selection criterion: if refusals cannot be excluded from the
evaluation population, which evaluator degrades predictably? Read-only.
"""
import sys, pickle, os
sys.path.insert(0, 'src')
import pandas as pd, numpy as np
from abstention import is_abstention

B = os.environ.get('RAG_CHECKPOINT_DIR', 'checkpoints/n1000_v3') + '/'
M4 = ['all-mpnet-base-v2', 'BGE-M3', 'E5-large-instruct', 'text-embedding-3-small']
EV = [('nli_scores', 'nli_max', 'NLI-max'),
      ('claim_scores', 'claim_min', 'Claim-min'),
      ('align_scores', 'align_score', 'AlignScore')]

rows = []
for pre, col, lab in EV:
    for g in ['claude', 'qwen', 'gpt4omini', 'mistral']:
        for ds in ['NQ', 'HotpotQA']:
            ref, att = [], []
            for m in M4:
                try:
                    d = pd.DataFrame(pickle.load(open(f'{B}{pre}_{g}_{m}_{ds}.pkl', 'rb')))
                    gen = pd.DataFrame(pickle.load(open(f'{B}generated_{g}_{m}_{ds}.pkl', 'rb')))
                except Exception:
                    continue
                if 'generated_answer' not in d.columns:
                    d = d.merge(gen[['query_id', 'generated_answer']], on='query_id')
                c = [x for x in d.columns
                     if x in (col, 'nli_max', 'claim_min', 'align_score', 'score')][0]
                ab = d['generated_answer'].map(is_abstention)
                ref += list(d[ab][c].dropna())
                att += list(d[~ab][c].dropna())
            if ref:
                rows.append(dict(evaluator=lab, generator=g, dataset=ds,
                                 refusal=np.mean(ref), attempt=np.mean(att),
                                 gap=np.mean(ref) - np.mean(att)))

t = pd.DataFrame(rows)
if t.empty:
    raise SystemExit(
        'no checkpoints found under %r -- every pickle open failed and was\n'
        'swallowed by the per-file except. Set RAG_CHECKPOINT_DIR to the\n'
        'absolute data path; the code and data trees are siblings with\n'
        'different names (rag-faithfulness vs rag_faithfulness).' % B)
print('mean score on refusals:')
print(t.pivot_table(index='evaluator', columns='generator', values='refusal').round(3).to_string())
print('\nrefusal minus attempt (positive = refusals score HIGHER than real answers):')
print(t.pivot_table(index='evaluator', columns='generator', values='gap').round(3).to_string())
