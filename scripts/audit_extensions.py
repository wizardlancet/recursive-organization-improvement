"""Prior/threshold/generation-cost checks for the original audit model.
Adds a finite-horizon expected-value-of-sample-information (EVSI) controller.
EVSI is an explicitly approximate sliding-window Bayesian policy, not an optimal
nonstationary bandit solver. These analyses are supplementary to coverage study.
"""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT.parent/'.paper_runtime').exists():sys.path.insert(0,str(ROOT.parent/'.paper_runtime'))
import numpy as np
from scipy.special import betainc,betaln,gammaln
T,B=160,32

def arrays(seeds):
    r=[]
    for s in seeds:
        rng=np.random.default_rng(s);r.append([rng.random((T,B)) for _ in range(3)])
    return [np.array([x[j] for x in r]) for j in range(3)]

def simulate(data,scenario,prior,threshold,gen,policy,horizon=8):
    ua,ub,uq=data;N=len(ua);probs=np.full(T,.04) if scenario=='Stationary' else np.where((np.arange(T)//40)%2==0,.04,.25)
    errors_a=ua<probs[None,:,None];errors_b=ub<.08
    nh=np.zeros((N,T),int);eh=nh.copy();v=np.zeros(N);bs=np.zeros(N);ac=np.zeros(N);sw=np.zeros(N);prev=None
    for t in range(T):
        n=nh[:,max(0,t-12):t].sum(1);e=eh[:,max(0,t-12):t].sum(1)
        a=prior[0]+e;b=prior[1]+n-e;p=a/(a+b);safe=p>threshold
        if policy=='EVSI':
            utility=[];candidate=[0,1,4,16]
            current=np.where(safe,.58,1-3*p)
            for m in candidate:
                k=np.arange(m+1);logpmf=gammaln(m+1)-gammaln(k+1)-gammaln(m-k+1)+betaln(a[:,None]+k,b[:,None]+m-k)-betaln(a,b)[:,None]
                after=(a[:,None]+k)/(a+b+m)[:,None]
                posterior_value=np.where(after>threshold,.58,1-3*after)
                gain=(np.exp(logpmf)*posterior_value).sum(1)-current
                switching=0 if prev is None else .5*(prev!=m/B)/B
                utility.append(horizon*gain-(.2+gen*safe)*m/B-switching)
            count=np.array(candidate)[np.stack(utility,1).argmax(1)];q=count/B
            # Uniform sample without replacement, selected using common random ranks.
            rank=np.argsort(np.argsort(uq[:,t],axis=1),axis=1);audit=rank<count[:,None]
        else:
            if policy=='Fixed 0.10':q=np.full(N,.1)
            else:
                risk=1-betainc(a,b,threshold);q=np.where((risk>.1)&(risk<.9),.4,.025)
            audit=uq[:,t]<q[:,None]
        nh[:,t]=audit.sum(1);eh[:,t]=(audit&errors_a[:,t]).sum(1)
        change=np.zeros(N) if prev is None else (prev!=q).astype(float)
        realized=np.where(safe[:,None],errors_b[:,t],errors_a[:,t]).sum(1)
        v+=(np.where(safe,.82,1)*B-3*realized-(.2+gen*safe)*nh[:,t]-.5*change)/B
        bs+=safe;ac+=nh[:,t];sw+=change;prev=q
    return v/T,bs/T,ac,sw

def write(name,rows):
    with (ROOT/'results'/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    dev=arrays(range(710000,710064));tuning=[]
    for h in [1,4,8,16]:
        for sc in ['Stationary','Regime shifts']:
            v,*_=simulate(dev,sc,(1,9),.14,0,'EVSI',h)
            tuning.append(dict(horizon=h,scenario=sc,mean=float(v.mean())))
    horizon=max([1,4,8,16],key=lambda h:np.mean([r['mean'] for r in tuning if r['horizon']==h]));write('audit_evsi_development.csv',tuning)
    seeds=list(range(810000,810096));data=arrays(seeds);rows=[];summary=[]
    for prior in [(1,1),(1,9),(1,19)]:
        for threshold in [.14,.16]:
            for gen in [0,.2]:
                for scenario in ['Stationary','Regime shifts']:
                    values={}
                    for policy in ['Fixed 0.10','Adaptive audit','EVSI']:
                        v,b,a,s=simulate(data,scenario,prior,threshold,gen,policy,horizon);values[policy]=v
                        for j,seed in enumerate(seeds):rows.append(dict(prior=f'{prior[0]}:{prior[1]}',threshold=threshold,candidate_generation_cost=gen,scenario=scenario,policy=policy,seed=seed,value=float(v[j]),safe_route_share=float(b[j]),audited=int(a[j]),switches=int(s[j])))
                    for policy,v in values.items():
                        d=v-values['Fixed 0.10'];summary.append(dict(prior=f'{prior[0]}:{prior[1]}',threshold=threshold,candidate_generation_cost=gen,scenario=scenario,policy=policy,mean=float(v.mean()),ci95=float(1.96*v.std(ddof=1)/len(v)**.5),difference_vs_fixed010=float(d.mean()),paired_ci95=float(1.96*d.std(ddof=1)/len(d)**.5)))
    write('audit_extensions_runs.csv',rows);write('audit_extensions_summary.csv',summary)
    # Regression against the preserved scalar experiment for the original parameters.
    base=list(csv.DictReader((ROOT/'results/longitudinal_runs.csv').open()))
    for policy in ['Fixed 0.10','Adaptive audit']:
        actual=simulate(arrays([20260928,20260929]),'Regime shifts',(1,9),.14,0,policy)[0]
        expected=[float(next(r for r in base if r['policy']==policy and r['scenario']=='Regime shifts' and int(r['seed'])==seed)['value']) for seed in [20260928,20260929]]
        assert np.allclose(actual,expected,rtol=0,atol=1e-12)
    checks={'selected_evsi_horizon':horizon,'development_cases':len(tuning)*64,'test_trajectories':len(rows),'regression_cases':4,'parameters_fixed_before_run':True,'interpretation':'Changing threshold to .16 is a misspecified routing-rule sensitivity, not a change in the payoff optimum .14.'}
    (ROOT/'qa/audit_extensions_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
