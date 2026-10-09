"""Purchased observations identify organizations; hidden truth only scores them."""
from pathlib import Path
from collections import deque
import argparse,json,time
import numpy as np
from model import (catalog,hard_cost,exact_catalog,draw,execute,Topology,reports,
                   private_stats,fit_private,trace_stats,fit_trace)
from oracle_experiment import write_csv
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text());C=P['learning']

class Memory:
    def __init__(self,mode):self.mode=mode;self.rows=deque()
    def add(self,x):
        if self.mode=='reset':self.rows.clear()
        self.rows.append(x)
        # Eight PREVIOUS rounds plus current, matching the preceding supplement.
        if self.mode=='window8' and len(self.rows)>9:self.rows.popleft()
        return sum(self.rows)

def randomized_stats(panel,eh,ea,assignment):
    blind=(assignment%2==0)
    st=[private_stats(panel['h'][:,blind],panel['a'][:,blind])]
    for s,offset in [('H',0),('A',2)]:
        up,private,post=reports(Topology('exposed_vote',s,5),panel,eh,ea)
        for cell,rev in [(offset,private),(offset+1,post)]:
            use=assignment==cell
            ds=(rev[:,use]!=up[:,use,None]).sum((-2,-1))
            st.append(np.stack([np.full(len(ds),5*use.sum()),ds],axis=-1))
    return np.concatenate(st,axis=-1)

def fit_randomized(st):
    ph,pa,rho=fit_private(st[...,:4]);etas=[]
    for start in (4,8):
        db=(st[...,start+1]+.5)/(st[...,start]+1)
        de=(st[...,start+3]+.5)/(st[...,start+2]+1)
        etas.append(np.clip(1-de/db,0,1))
    return ph,pa,rho,*etas

def env(w,round_index):
    e={k:w[k] for k in ('p_h','p_a','rho','eta_h','eta_a')}
    if round_index>=C['shift_round']-1:e.update(w.get('after',{}))
    return e

def run(smoke=False):
    dest=ROOT/('results_smoke' if smoke else 'results');dest.mkdir(exist_ok=True)
    R=4 if smoke else C['replicates'];T=4 if smoke else C['rounds']
    worlds=C['worlds'][:1] if smoke else C['worlds']
    ts=[t for t in catalog(C['max_reviewers']) if hard_cost(t)<=C['production_hard_budget']+1e-10]
    policies=[(p,m) for p in C['policies'] for m in C['memories']]+[(b,'none') for b in C['fixed_baselines']]
    shape=(len(worlds),len(policies),R,T)
    names=['error','production_cost','acquisition_cost','changed','net','regret','hit','choice']
    data={k:np.zeros(shape,dtype=np.int16 if k=='choice' else float) for k in names}
    estimates=np.full(shape+(5,),np.nan)
    oracle_choices=np.zeros((len(worlds),T),dtype=int)
    start=time.time()
    for wi,w in enumerate(worlds):
        histories=[Memory(m) for _,m in policies];last=np.full((len(policies),R),ts.index(Topology('exposed_vote','A',3)))
        for r in range(T):
            e=env(w,r);truth=exact_catalog(ts,*e.values())
            true_loss=3*truth['error']+.02*truth['cost'];best=true_loss.min();oracle_choices[wi,r]=int(true_loss.argmin())
            rng=np.random.default_rng(np.random.SeedSequence([P['seed'],wi,r]))
            # Each row is an independent replication; common potential tasks couple policies.
            panel=draw(rng,(R,240),e['p_h'],e['p_a'],e['rho'],5)
            take=lambda n:{k:v[:,:n] for k,v in panel.items()}
            nt=int(C['acquisition_budget_per_round']//P['cost_model']['full_trace_cost'])
            traced=trace_stats(take(nt),e['eta_h'],e['eta_a'])
            nr=int(C['acquisition_budget_per_round']//4.8)
            assignment=rng.permutation((np.arange(nr)+r)%4)
            randomized=randomized_stats(take(nr),e['eta_h'],e['eta_a'],assignment)
            rcost=float(np.where(assignment%2==0,4.8,4.5).sum())
            # Uniform direct experiments reserve each trial's worst cost; unused
            # capacity is not borrowed. Debit only the costs actually incurred.
            direct=np.zeros((R,len(ts),3));dcost=np.zeros(R);reserved=0.;j=0
            while True:
                idx=(r*83+j)%len(ts);t=ts[idx];reserve=1+hard_cost(t)
                if reserved+reserve>C['acquisition_budget_per_round']+1e-10:break
                single={key:val[:,j] for key,val in panel.items()}
                realized=execute(t,single,e['eta_h'],e['eta_a'])
                direct[:,idx,0]+=1;direct[:,idx,1]+=realized['error'];direct[:,idx,2]+=realized['cost']
                dcost+=1+realized['cost'];reserved+=reserve;j+=1
            for pi,(policy,memory) in enumerate(policies):
                acq=np.zeros(R)
                if policy in C['fixed_baselines']:
                    choice=np.full(R,[t.name for t in ts].index(policy))
                elif policy=='Final_uniform':
                    st=histories[pi].add(direct.copy())
                    pred=3*(1+st[...,1])/(2+st[...,0])+.02*(st[...,2]+np.array([hard_cost(t) for t in ts]))/(1+st[...,0])
                    choice=pred.argmin(-1);acq=dcost
                else:
                    obs=randomized if policy=='Randomized_exposure' else traced
                    st=histories[pi].add(obs.copy())
                    fit=fit_randomized(st) if policy=='Randomized_exposure' else fit_trace(st,policy=='Ignore_echo')
                    estimates[wi,pi,:,r,:]=np.stack(fit,axis=-1)
                    prediction=exact_catalog(ts,*fit)
                    choice=(3*prediction['error']+.02*prediction['cost']).argmin(-1)
                    acq=np.full(R,rcost if policy=='Randomized_exposure' else nt*P['cost_model']['full_trace_cost'])
                loss=true_loss[choice];changed=(choice!=last[pi]).astype(float);last[pi]=choice
                metric=dict(error=truth['error'][choice],production_cost=truth['cost'][choice],acquisition_cost=acq,
                            changed=changed,net=1-loss-(acq+C['change_cost']*changed)/C['production_tasks_per_round'],
                            regret=loss-best,hit=np.abs(loss-best)<1e-10,choice=choice)
                for key,val in metric.items():data[key][wi,pi,:,r]=val
            if (r+1)%12==0:print(f'{w["name"]}: round {r+1}/{T}, elapsed {time.time()-start:.1f}s',flush=True)
    meta={'worlds':[w['name'] for w in worlds],'policies':[list(x) for x in policies],
          'topologies':[t.name for t in ts],'replicates':R,'rounds':T,'smoke':smoke,
          'production_evaluation':'Exact expected risk and cost of selected topology; production_tasks_per_round amortizes acquisition only',
          'window8':'8 previous rounds + current','elapsed_seconds':time.time()-start}
    np.savez_compressed(dest/'learning_raw.npz',**data,estimates=estimates,oracle_choices=oracle_choices)
    (dest/'learning_metadata.json').write_text(json.dumps(meta,indent=2))
    rows=[];periods=[]
    for wi,w in enumerate(worlds):
        for pi,(p,m) in enumerate(policies):
            for rep in range(R):
                row=dict(world=w['name'],policy=p,memory=m,replicate=rep)
                for key in names[:-1]:
                    row[key]=float(data[key][wi,pi,rep].mean())
                    row[key+'_late']=float(data[key][wi,pi,rep,-min(16,T):].mean())
                rows.append(row)
            for r in range(T):
                row=dict(world=w['name'],policy=p,memory=m,round=r+1)
                for key in names[:-1]:
                    v=data[key][wi,pi,:,r];row[key]=float(v.mean());row[key+'_ci95']=float(1.96*v.std(ddof=1)/np.sqrt(R))
                for q,k in enumerate(('p_h','p_a','rho','eta_h','eta_a')):
                    v=estimates[wi,pi,:,r,q];row[k+'_estimate']=float(v.mean()) if np.all(np.isfinite(v)) else ''
                periods.append(row)
    write_csv(dest/'learning_runs.csv',rows);write_csv(dest/'learning_periods.csv',periods)
    print(meta,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();run(args.smoke)
