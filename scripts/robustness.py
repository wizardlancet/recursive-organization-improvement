"""Exploratory structural sensitivity of the synthetic audit probe; no fitting.
Reproduces Appendix I. Vectorized across seeds, sequential across periods.
"""
from pathlib import Path
import sys,csv,json,re
ROOT=Path(__file__).resolve().parents[1]
if (ROOT.parent/'.paper_runtime').exists():sys.path.insert(0,str(ROOT.parent/'.paper_runtime'))
import numpy as np
from scipy.special import betainc
N,T,B=200,160,32
POLICIES=['No audit','Fixed 0.05','Fixed 0.10','Fixed 0.40','Periodic dense','Adaptive audit']
def simulate(ua,ub,uq,probs,window,policy):
    count=ua.shape[0];ea=ua<probs[None,:,None];eb=ub<.08
    nh=np.zeros((count,T),dtype=int);eh=nh.copy()
    score=np.zeros(count);audits=np.zeros(count);switches=np.zeros(count);safe_count=np.zeros(count)
    prev=None
    for t in range(T):
        n=nh[:,max(0,t-window):t].sum(1);e=eh[:,max(0,t-window):t].sum(1)
        a=1+e;b=9+n-e;safe=a/(a+b)>.14
        if policy=='Adaptive audit':
            risk=1-betainc(a,b,.14);q=np.where((risk>.1)&(risk<.9),.4,.025)
        elif policy=='Periodic dense':q=np.full(count,.4 if t%12<3 else .025)
        elif policy=='No audit':q=np.zeros(count)
        else:q=np.full(count,float(policy.split()[-1]))
        audit=uq[:,t,:]<q[:,None];nh[:,t]=audit.sum(1);eh[:,t]=(audit&ea[:,t,:]).sum(1)
        change=np.zeros(count) if prev is None else (prev!=q).astype(float)
        errors=np.where(safe[:,None],eb[:,t,:],ea[:,t,:]).sum(1)
        score+=np.where(safe,.82,1)*B-3*errors-.2*nh[:,t]-.5*change
        audits+=nh[:,t];switches+=change;safe_count+=safe;prev=q
    return score/(T*B),audits,switches,safe_count/T
def arrays(seeds):
    rows=[]
    for seed in seeds:
        rng=np.random.default_rng(seed);rows.append([rng.random((T,B)) for _ in range(3)])
    return tuple(np.array([row[i] for row in rows]) for i in range(3))
def writecsv(name,rows):
    with (ROOT/'results'/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
seeds=np.arange(20360928,20360928+N);ua,ub,uq=arrays(seeds)
raw=[];summary=[];table=[]
for scenario,block in [('Stationary',None),('Slow shifts',40),('Fast shifts',10)]:
    probs=np.full(T,.04) if block is None else np.where((np.arange(T)//block)%2==0,.04,.25)
    for window in [6,12,24]:
        values={}
        for policy in POLICIES:
            v,a,s,b=simulate(ua,ub,uq,probs,window,policy);values[policy]=v
            for i in range(N):raw.append(dict(scenario=scenario,window=window,policy=policy,seed=int(seeds[i]),value=float(v[i]),audited=int(a[i]),switches=int(s[i]),safe_route_share=float(b[i])))
        for policy,v in values.items():
            d=v-values['Fixed 0.10'];summary.append(dict(scenario=scenario,window=window,policy=policy,mean=float(v.mean()),ci95_halfwidth=float(1.96*v.std(ddof=1)/N**.5),difference_vs_fixed010=float(d.mean()),paired_ci95_halfwidth=float(1.96*d.std(ddof=1)/N**.5)))
        a=next(r for r in summary if r['scenario']==scenario and r['window']==window and r['policy']=='Adaptive audit')
        best=max([p for p in POLICIES if p.startswith('Fixed') or p=='No audit'],key=lambda p:values[p].mean())
        table.append(f"{scenario} & {window} & {values['Fixed 0.10'].mean():.4f} & {a['mean']:.4f} & ${a['difference_vs_fixed010']:+.4f} \\pm {a['paired_ci95_halfwidth']:.4f}$ & {best.replace('No audit','0').replace('Fixed ','')} \\\\")
writecsv('robustness_runs.csv',raw);writecsv('robustness_summary.csv',summary)
(ROOT/'sections/robustness_rows.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
paper=ROOT/'sections/appendices_extended.tex'
if paper.exists():
    s=paper.read_text(encoding='utf-8')
    replacement='% BEGIN GENERATED ROBUSTNESS ROWS\n'+'\n'.join(table)+'\n% END GENERATED ROBUSTNESS ROWS'
    s=re.sub(r'% BEGIN GENERATED ROBUSTNESS ROWS[\s\S]*?% END GENERATED ROBUSTNESS ROWS',lambda _:replacement,s)
    paper.write_text(s,encoding='utf-8')
# Independent regression check against the original scalar implementation's outputs.
test_seeds=[20260928,20260929];ta,tb,tq=arrays(test_seeds)
base=list(csv.DictReader((ROOT/'results/longitudinal_runs.csv').open(encoding='utf-8')))
probs=np.where((np.arange(T)//40)%2==0,.04,.25)
for policy in POLICIES:
    original='Fixed dense' if policy=='Fixed 0.40' else policy
    scores=simulate(ta,tb,tq,probs,12,policy)[0]
    expected=[float(next(r for r in base if r['scenario']=='Regime shifts' and r['policy']==original and int(r['seed'])==seed)['value']) for seed in test_seeds]
    assert np.allclose(scores,expected,atol=1e-12,rtol=0),(policy,scores,expected)
assert len(raw)==10800 and all(np.isfinite(r['value']) for r in raw)
assert all(r['switches']==0 for r in raw if r['policy'].startswith('Fixed') or r['policy']=='No audit')
checks={'trajectories':len(raw),'seeds':N,'seed_start':int(seeds[0]),'windows':[6,12,24],'drift_blocks':[None,40,10],'policies':POLICIES,'scalar_vectorized_regression_cases':12,'regression_tolerance':1e-12,'finite_results':True,'fixed_policy_no_switches':True}
(ROOT/'qa/robustness_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('\n'.join(table));print(json.dumps(checks))
