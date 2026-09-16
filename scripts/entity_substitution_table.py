#!/usr/bin/env python3
"""Table: response to a falsified ENTITY (paper Section 5.2, Table 4).
Read-only over perturb_* checkpoints. Run from the repo root."""
import pickle, glob, os, numpy as np, pandas as pd
B='/home/shakhriyorbekboltabaev/rag-backup/checkpoints/n1000_v3/'
MODELS=['all-mpnet-base-v2','BGE-M3','E5-large-instruct','text-embedding-3-small']
GENS=[('claude','Claude'),('qwen','Qwen'),('gpt4omini','GPT-4o-mini')]
EVAL=[('nli','NLI-max',''),('claim','Claim-min','claim_'),('align','AlignScore','align_')]
def wilson(k,n,z=1.96):
    if n==0: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (c-h, c+h)
rows=[]
for ds in ['NQ','HotpotQA']:
    for g,gl in GENS:
        for ek,el,pre in EVAL:
            rs=[]
            for m in MODELS:
                p=f'{B}perturb_{pre}{g}_{m}_{ds}.pkl'
                if os.path.exists(p): rs+=pickle.load(open(p,'rb'))
            if not rs: continue
            d=pd.DataFrame(rs)
            for cond in ['number','entity']:
                sub=d[d[cond].notna() & d['orig'].notna()]
                if not len(sub): continue
                det=((sub['orig']>=0.5)&(sub[cond]<0.5)).sum()
                elig=(sub['orig']>=0.5).sum()
                lo,hi=wilson(det,elig)
                rows.append(dict(dataset=ds,generator=gl,evaluator=el,cond=cond,
                    n=len(sub),orig=sub['orig'].mean(),pert=sub[cond].mean(),
                    delta=sub['orig'].mean()-sub[cond].mean(),
                    det=det/elig if elig else np.nan,lo=lo,hi=hi,n_elig=elig))
t=pd.DataFrame(rows)
t.to_pickle('/tmp/claude-1000/-home-shakhriyorbekboltabaev-Documents-rag-faithfulness/84936299-5b8f-46c6-a4b0-665b55951e53/scratchpad/entity_table.pkl')
pd.set_option('display.width',220)
for ds in ['NQ','HotpotQA']:
    print('='*100); print(ds)
    piv=t[t.dataset==ds].pivot_table(index=['evaluator','cond'],columns='generator',
        values=['delta','det'],aggfunc='first')
    print(piv.reindex(columns=['Claude','Qwen','GPT-4o-mini'],level=1).round(3).to_string())
print()
print('=== ENTITY detail (n, eligible, Wilson CI) ===')
print(t[t.cond=='entity'][['dataset','generator','evaluator','n','n_elig','orig','pert','delta','det','lo','hi']].round(4).to_string(index=False))
