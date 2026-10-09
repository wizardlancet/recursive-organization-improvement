"""Study 1 prospective design: discover, retain evidence, and reassess under fixed budgets.
All outcomes are synthetic. No true risk is supplied to a decision function.
"""
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import csv,json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1]
if (ROOT.parent/'.paper_runtime').exists():sys.path.insert(0,str(ROOT.parent/'.paper_runtime'))
import numpy as np
from contracts import State,Patch,apply_patch
P=json.loads((ROOT/'examples/learning_protocol.json').read_text())
T=P['rounds'];K=3;B=P['block_label_budget'];L=P['block_length'];POP=P['production_tasks_per_round'];CAT=P['program_catalog'];ARMS=P['arms']

@dataclass(frozen=True)
class Config:
    share:float=.2
    trials:int=1
    pool:bool=True
    program_memory:bool=True

def risk_matrix(env,t):
    special=[.08,.46]
    shift=int(env.split()[-1]) if env in ['Workflow reversal 20','Workflow reversal 28'] else 24
    if env=='Uniform gain' or (env.startswith('Workflow reversal') and t>=shift):special=[.08,.08]
    return np.array([[.20,.20],special,[.16,.16]])

@lru_cache(maxsize=512)
def bank(env,t,stage,program,trial,seed,size):
    rng=np.random.default_rng(np.random.SeedSequence([seed,t,stage,program,trial]))
    return rng.random((size,K,2,256))<risk_matrix(env,t)[None,:,:,None]

def observed(labels,n):
    assert n.min()>=0 and n.max()<=labels.shape[-1]
    totals=np.take_along_axis(labels.cumsum(-1),np.maximum(n,1)[...,None]-1,axis=-1)[...,0]
    return np.where(n>0,totals,0)

def posterior(e,n,he,hn):return ((1+he+e)/(2+hn+n)).mean(-1)

def screen(labels,budget,program,he,hn):
    """Budget is total label units across all workflows, possibly per replicate."""
    size=labels.shape[0];budget=np.broadcast_to(np.asarray(budget,dtype=int),(size,))
    per=budget//K
    if program=='Successive rejects':
        # A pair observes both strata; pairs are independent bounded losses.
        pairs=budget//2;logbar=.5+1/2+1/3
        first=np.ceil((pairs-K)/(logbar*K)).astype(int)
        second=np.ceil((pairs-K)/(logbar*(K-1))).astype(int)
        first=np.maximum(first,1);second=np.maximum(second,first)
        assert np.all(first+2*second<=pairs)
        n=np.broadcast_to(first[:,None,None],(size,K,2)).copy()
        e=observed(labels,n);v=posterior(e,n,he,hn);rejected=v.argmax(1)
        alive=np.ones((size,K),bool);alive[np.arange(size),rejected]=False
        n=np.where(alive[:,:,None],second[:,None,None],n)
        e=observed(labels,n);v=posterior(e,n,he,hn);v=np.where(alive,v,np.inf)
    elif program=='Biased':
        n=np.maximum(1,np.floor(per[:,None,None]*np.array([.95,.05]))).astype(int)
        n=np.broadcast_to(n,(size,K,2)).copy();e=observed(labels,n);v=posterior(e,n,he,hn)
    elif program=='Balanced':
        n=np.broadcast_to((per//2)[:,None,None],(size,K,2)).copy();e=observed(labels,n);v=posterior(e,n,he,hn)
    elif program=='Neyman':
        pilot=np.minimum(2,np.maximum(1,per//4));pn=np.broadcast_to(pilot[:,None,None],(size,K,2))
        pe=observed(labels,pn);prob=(1+he+pe)/(2+hn+pn);sd=np.sqrt(prob*(1-prob))
        n=pn+np.floor((per-2*pilot)[:,None,None]*sd/sd.sum(-1,keepdims=True)).astype(int)
        e=observed(labels,n);v=posterior(e,n,he,hn)
    else:raise ValueError(program)
    assert (n.sum((1,2))<=budget).all()
    return v.argmin(1),e,n

def memory_at(entries,t,mode,shape):
    if mode=='Reset':return np.zeros(shape),np.zeros(shape)
    valid=[x for x in entries if mode=='Cumulative' or x[0]>=t-8]
    if not valid:return np.zeros(shape),np.zeros(shape)
    return sum(x[1] for x in valid),sum(x[2] for x in valid)

def simulate(env,mode,arm,cfg=Config(),size=128,seed=920000):
    shape=(size,K,2);history=[];program_history=[];active=np.zeros(size,int)
    states=[State(0,frozenset({('review_board','evaluation.coverage_program')}),100) for _ in range(size)]
    if arm=='Trial-matched Balanced':active[:]=1
    version=np.zeros(size,int);patches=[]
    arrays={key:np.zeros((size,T)) for key in ['net','harm','regret','expense','meta_expense','labels','program','selected','changes']}
    operational=np.full(size,B//L);review_records=[]
    for t in range(T):
        he,hn=memory_at(history,t,mode,shape)
        meta_n=np.zeros(shape);meta_e=np.zeros(shape);meta_spent=np.zeros(size);changed=np.zeros(size)
        review=t%L==0 and (arm in ['Repeated discovery','Trial-matched Balanced'] or (arm=='Discover once' and t==0))
        if t%L==0:block_spent=np.zeros(size)
        if review:
            allowance=int(B*cfg.share);trial_budget=allowance//(len(CAT)*cfg.trials)
            validation=max(1,int(.2*trial_budget)//2);screen_budget=trial_budget-2*validation
            scores=np.zeros((size,len(CAT)))
            for j,program in enumerate(CAT):
                for q in range(cfg.trials):
                    labels=bank(env,t,1,j,q,seed,size)
                    chosen,e,n=screen(labels,screen_budget,program,he,hn)
                    val=bank(env,t,2,j,q,seed,size)
                    vn=np.zeros(shape,int);vn[np.arange(size),chosen,:]=validation
                    ve=observed(val,vn)
                    scores[:,j]+=1-3*ve.sum((1,2))/(2*validation)-.22*n.sum((1,2))/POP
                    meta_e+=e+ve;meta_n+=n+vn;meta_spent+=(n+vn).sum((1,2))
            scores/=cfg.trials
            # Each review has the same trial count within a configuration.
            if not cfg.program_memory or mode=='Reset':valid=[]
            else:valid=[x for x in program_history if mode=='Cumulative' or x[0]>=t-8]
            estimate=(scores+sum((x[1] for x in valid),np.zeros_like(scores)))/(1+len(valid))
            winner=estimate.argmax(1)
            if arm=='Trial-matched Balanced':winner[:]=1
            changed=(winner!=active).astype(float)
            for i in np.flatnonzero(changed):
                evidence=frozenset({f'review:{seed}:{i}:{t}'})
                patch=Patch(f'patch:{seed}:{i}:{t}','evaluation.coverage_program',states[i].version,'review_board',evidence,True,.5)
                states[i]=apply_patch(states[i],patch,evidence);version[i]=states[i].version
                if i==0:patches.append(dict(round=t+1,from_program=CAT[active[i]],to_program=CAT[winner[i]],version=int(version[i])))
            active=winner;program_history.append((t,scores))
            review_records.append((t,active.copy()))
        if t%L==0:operational=(B-meta_spent.astype(int))//L
        if cfg.pool:he=he+meta_e;hn=hn+meta_n
        labels=bank(env,t,0,0,0,seed,size)
        if arm in CAT or arm=='Successive rejects':
            chosen,e,n=screen(labels,operational,arm,he,hn)
            active[:]=CAT.index(arm) if arm in CAT else 3
        else:
            chosen=np.zeros(size,int);e=np.zeros(shape);n=np.zeros(shape,int)
            for j,program in enumerate(CAT):
                c,ee,nn=screen(labels,operational,program,he,hn);mask=active==j
                chosen[mask]=c[mask];e[mask]=ee[mask];n[mask]=nn[mask]
        history.append((t,e+(meta_e if cfg.pool else 0),n+(meta_n if cfg.pool else 0)))
        count=n.sum((1,2))+meta_spent;expense=(.22*count+.5*changed)/POP
        truth=risk_matrix(env,t).mean(1);risk=truth[chosen]
        arrays['net'][:,t]=1-3*risk-.02-expense
        arrays['harm'][:,t]=risk>truth[0]+1e-12
        arrays['regret'][:,t]=3*(risk-truth.min())
        arrays['expense'][:,t]=expense;arrays['meta_expense'][:,t]=(.22*meta_spent+.5*changed)/POP
        arrays['labels'][:,t]=count;arrays['program'][:,t]=active;arrays['selected'][:,t]=chosen;arrays['changes'][:,t]=changed
        block_spent+=count
        assert np.all(block_spent<=B+1e-9)
        assert np.allclose(arrays['net'][:,t],1-3*truth.min()-.02-arrays['regret'][:,t]-expense,atol=1e-12)
    rows=[]
    for i in range(size):
        r=dict(environment=env,memory=mode,arm=arm,stream_seed=seed,replicate=i,meta_share=cfg.share,trials=cfg.trials,pool=cfg.pool,program_memory=cfg.program_memory)
        for k in ['net','harm','regret','expense','meta_expense','labels','changes']:r[k]=float(arrays[k][i].mean())
        r['late_net']=float(arrays['net'][i,-8:].mean());rows.append(r)
    periods=[dict(environment=env,memory=mode,arm=arm,round=t+1,net=float(arrays['net'][:,t].mean()),harm=float(arrays['harm'][:,t].mean()),regret=float(arrays['regret'][:,t].mean()),balanced_share=float((arrays['program'][:,t]==1).mean()),neyman_share=float((arrays['program'][:,t]==2).mean()),specialized_share=float((arrays['selected'][:,t]==1).mean())) for t in range(T)]
    conditional=[]
    for t,programs in review_records:
        for j,name in enumerate(CAT):
            mask=programs==j
            conditional.append(dict(environment=env,memory=mode,arm=arm,review_round=t+1,program=name,n=int(mask.sum()),share=float(mask.mean()),net=float(arrays['net'][mask,t:t+L].mean()) if mask.any() else None,harm=float(arrays['harm'][mask,t:t+L].mean()) if mask.any() else None))
    return rows,periods,conditional,patches

def write(name,rows):
    with (ROOT/'results'/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def aggregate(rows):
    group=['environment','memory','arm','meta_share','trials','pool','program_memory'];out=[]
    keys=list(dict.fromkeys(tuple(r[k] for k in group) for r in rows))
    for key in keys:
        rr=[r for r in rows if tuple(r[k] for k in group)==key];o=dict(zip(group,key));o['n']=len(rr)
        for field in ['net','harm','regret','expense','meta_expense','late_net']:
            v=np.array([r[field] for r in rr]);o[field]=float(v.mean());o[field+'_ci95']=float(1.96*v.std(ddof=1)/len(v)**.5)
        out.append(o)
    return out

def main():
    if '--quick' in sys.argv:
        for mode in P['memories']:
            for arm in ARMS:simulate('Workflow reversal',mode,arm,size=4,seed=999000)
        print('18 smoke cases: budget, accounting and patch checks passed');return
    base=[];periods=[];conditional=[];patches={}
    for env in P['environments']:
        for mode in P['memories']:
            for arm in ARMS:
                rr,pp,cc,pt=simulate(env,mode,arm);base+=rr;periods+=pp;conditional+=cc
                patches[env+' / '+mode+' / '+arm]=pt
        bank.cache_clear();print('Completed base:',env,flush=True)
    write('learning_runs.csv',base);write('learning_summary.csv',aggregate(base));write('learning_periods.csv',periods);write('learning_conditional.csv',conditional)
    sensitivity=[]
    for env,mode in [('Stationary harm','Cumulative'),('Workflow reversal','Window 8')]:
        for share in P['allocation_grid']['meta_share']:
            for trials in P['allocation_grid']['trials']:
                for arm in ['Discover once','Repeated discovery']:
                    rr,*_=simulate(env,mode,arm,Config(share,trials),size=96,seed=930000);sensitivity+=rr
        for arm in CAT+['Successive rejects']:
            rr,*_=simulate(env,mode,arm,size=96,seed=930000);sensitivity+=rr
        for cfg in [Config(program_memory=False),Config(pool=False)]:
            rr,*_=simulate(env,mode,'Repeated discovery',cfg,size=96,seed=930000);sensitivity+=rr
        bank.cache_clear();print('Completed sensitivity:',env,flush=True)
    write('learning_sensitivity_runs.csv',sensitivity);write('learning_sensitivity_summary.csv',aggregate(sensitivity))
    report={'protocol_sha256':hashlib.sha256((ROOT/'examples/learning_protocol.json').read_bytes()).hexdigest(),'base_trajectories':len(base),'sensitivity_trajectories':len(sensitivity),'budget_checks_passed':True,'cost_identities_passed':True,'test_tuning':'none','randomization_unit':'replicate within keyed stream seed','program_changes_first_replicate':patches}
    (ROOT/'qa/learning_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print({k:v for k,v in report.items() if k!='program_changes_first_replicate'})

if __name__=='__main__':main()
