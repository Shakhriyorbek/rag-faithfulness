#!/usr/bin/env python3
"""Minimum detectable effect at 80% power (paper Appendix I, Table 12).
Read-only over scored checkpoints. Run from the repo root."""
import sys; sys.path.insert(0,'src')
from utils import set_scope; set_scope(1000)
import numpy as np, pandas as pd
from scipy import stats
from results import faithfulness_by_model
M4=['all-mpnet-base-v2','BGE-M3','E5-large-instruct','text-embedding-3-small']
rows=[]
za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
for ds in ['NQ','HotpotQA']:
    for g in ['claude','qwen','gpt4omini']:
        for metric,lab in [('nli','NLI-max'),('align','AlignScore'),('claim','Claim-min')]:
            r=faithfulness_by_model(ds,g,M4,answered_only=True,metric=metric,
                                    correct_source='contains')
            if not r: continue
            pq=r['per_query']; b,w=r['best'],r['worst']
            n=min(len(pq[b]),len(pq[w]))
            d=np.asarray(pq[b][:n],float)-np.asarray(pq[w][:n],float)
            sd=d.std(ddof=1); se=sd/np.sqrt(n)
            rows.append(dict(dataset=ds,generator=g,evaluator=lab,n=n,
                spread=r['spread'], sd_diff=sd, se=se, mde80=(za+zb)*se))
t=pd.DataFrame(rows)
t.to_pickle('/tmp/claude-1000/-home-shakhriyorbekboltabaev-Documents-rag-faithfulness/84936299-5b8f-46c6-a4b0-665b55951e53/scratchpad/power_table.pkl')
pd.set_option('display.width',200)
print(t.round(4).to_string(index=False))
print('\nMDE at 80% power, two-sided alpha=0.05, paired:')
print('  range %.4f - %.4f   median %.4f' % (t.mde80.min(), t.mde80.max(), t.mde80.median()))
print('  cells where observed spread < MDE (underpowered for what we saw): %d/%d'
      % ((t.spread<t.mde80).sum(), len(t)))
print('\nby evaluator:'); print(t.groupby('evaluator')[['mde80','spread']].agg(['min','max']).round(4).to_string())
