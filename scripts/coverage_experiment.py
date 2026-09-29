"""Finite-catalog procedure revision for workflow evaluation. Fully synthetic.
Protocol: examples/coverage_protocol.json. No hidden probabilities enter decisions.
Use --quick for invariant checks only; default runs the frozen development/test grid.
"""
from pathlib import Path
import json,csv,sys,hashlib
from dataclasses import dataclass,replace
ROOT=Path(__file__).resolve().parents[1]
if (ROOT.parent/'.paper_runtime').exists():sys.path.insert(0,str(ROOT.parent/'.paper_runtime'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contracts import State,Patch,apply_patch
PROTOCOL=json.loads((ROOT/'examples/coverage_protocol.json').read_text())
PROGRAMS=PROTOCOL['programs'];ARMS=PROTOCOL['arms'];T=24;P=4096;K=3

@dataclass(frozen=True)
class Config:
    screen:int=48
    interval:int=8
    trials:int=3
    validation:int=16
    prior_a:float=1
    prior_b:float=1
    cost_ratio:int=1
    price:float=.2
    generation:float=.02
    flip:float=0
    pilot:int=4

def csvout(name,rows):
    with (ROOT/'results'/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def truth(scenario,t):
    specialized=[.08,.46]
    if scenario=='Moving harm' and (t//10)%2: specialized=specialized[::-1]
    if scenario=='Uniform gain':specialized=[.08,.08]
    return np.array([[.20,.20],specialized,[.16,.16]])

def label_bank(rng,probs,n,size,flip):
    # The sampler is the environment; consumers receive labels only.
    labels=rng.random((size,K,2,n))<probs[None,:,:,None]
    if flip:labels ^= rng.random(labels.shape)<flip
    return labels

def select_workflow(labels,budget,program,cfg):
    size=labels.shape[0];cost=np.array([1,cfg.cost_ratio]);m=min(cfg.pilot,max(1,int(budget//(2*cost.sum()))))
    if program=='Biased':
        n=np.broadcast_to(np.maximum(1,np.floor(budget*np.array([.95,.05])/cost)).astype(int),(size,K,2)).copy()
    elif program=='Balanced':n=np.full((size,K,2),int(budget//cost.sum()),int)
    elif program=='Neyman':
        e=labels[:,:,:,:m].sum(-1);p=(cfg.prior_a+e)/(cfg.prior_a+cfg.prior_b+m)
        v=p*(1-p);remaining=budget-m*cost.sum()
        extra=remaining*np.sqrt(v/cost)/np.sqrt(v*cost).sum(-1,keepdims=True)
        n=m+np.floor(extra).astype(int)
    else:raise ValueError(program)
    assert (n>=1).all() and (n*cost).sum(-1).max()<=budget+1e-9
    assert n.max()<=labels.shape[-1]
    counts=np.take_along_axis(labels.cumsum(-1),n[...,None]-1,axis=-1)[...,0]
    estimate=((cfg.prior_a+counts)/(cfg.prior_a+cfg.prior_b+n)).mean(-1)
    chosen=estimate.argmin(-1)
    return chosen,(n*cost).sum((1,2)),n.sum((1,2)),estimate,n

def simulate(scenario,arm,cfg,seeds,record=False):
    size=len(seeds)
    # One vectorized RNG per run batch: fixed seed identifiers for paired streams.
    # Each trajectory is generated independently below, then stacked.
    rngs=[np.random.default_rng(int(s)) for s in seeds]
    costs=np.array([1,cfg.cost_ratio])
    meta_cap=len(PROGRAMS)*cfg.trials*(K*cfg.screen+cfg.validation*costs.sum())
    block_cap=cfg.interval*K*cfg.screen+meta_cap
    fixed_budget=int(block_cap//(cfg.interval*K))
    nmax=max(fixed_budget,cfg.screen,cfg.validation)
    active=np.zeros(size,int);versions=np.zeros(size,int)
    net=np.zeros((size,T));harm=net.copy();regret=net.copy();expense=net.copy();metacost=net.copy();switches=net.copy()
    choice=np.zeros((size,T),int);program_history=np.zeros((size,T),int);traces=[];patches=[]
    # A freezing ablation preserves the first selected program after round zero.
    revise=arm in ['Procedure revision','Freeze after first revision']
    states=[State(0,frozenset({('review_board','evaluation.coverage_program')}),1e12) for _ in seeds]
    for t in range(T):
        probs=truth(scenario,t)
        screen=np.concatenate([label_bank(r,probs,nmax,1,cfg.flip) for r in rngs],axis=0)
        review=t%cfg.interval==0
        mc=np.zeros(size);mg=np.zeros(size);sw=np.zeros(size)
        # Generate trial arrays for every arm, retaining identical production streams.
        if review:
            scores=[]
            for program in PROGRAMS:
                total=np.zeros(size)
                for trial in range(cfg.trials):
                    bank=np.concatenate([label_bank(r,probs,cfg.screen,1,cfg.flip) for r in rngs],axis=0)
                    validation=np.concatenate([label_bank(r,probs,cfg.validation,1,cfg.flip) for r in rngs],axis=0)
                    picked,cost,generated,_,_=select_workflow(bank,cfg.screen,program,cfg)
                    # Only the chosen workflow's fresh validation labels are queried.
                    observed=validation[np.arange(size),picked].mean((1,2))
                    total+=1-3*observed-(cfg.price*cost+cfg.generation*generated)/P
                    if revise and (arm=='Procedure revision' or t==0):
                        mc+=cost+cfg.validation*costs.sum();mg+=generated+2*cfg.validation
                scores.append(total/cfg.trials)
            if revise and (arm=='Procedure revision' or t==0):
                scores=np.stack(scores,axis=1);winner=scores.argmax(1);sw=(winner!=active).astype(float)
                for i in range(size):
                    if sw[i]:
                        evidence=frozenset({f'trial:{int(seeds[i])}:{t}'})
                        patch=Patch(f'patch:{int(seeds[i])}:{t}','evaluation.coverage_program',states[i].version,'review_board',evidence,True,.5)
                        states[i]=apply_patch(states[i],patch,evidence);versions[i]=states[i].version
                        if record and i==0:patches.append(dict(round=t,old_program=PROGRAMS[active[i]],new_program=PROGRAMS[winner[i]],version=int(versions[i]),scores=scores[i].tolist(),evidence=list(evidence),switch_cost=.5))
                active=winner
        if arm=='Fixed workflow':
            picked=np.zeros(size,int);cost=np.zeros(size);generated=cost.copy();ns=np.zeros((size,K,2),int);est=np.full((size,K),np.nan)
        elif revise:
            picked=np.zeros(size,int);cost=np.zeros(size);generated=np.zeros(size);ns=np.zeros((size,K,2),int);est=np.zeros((size,K))
            for j,program in enumerate(PROGRAMS):
                selected,c,g,e,n=select_workflow(screen,cfg.screen,program,cfg);mask=active==j
                picked[mask]=selected[mask];cost[mask]=c[mask];generated[mask]=g[mask];ns[mask]=n[mask];est[mask]=e[mask]
        else:
            program={'Biased search':'Biased','Balanced search':'Balanced','Neyman search':'Neyman'}[arm]
            active[:]=PROGRAMS.index(program)
            picked,cost,generated,est,ns=select_workflow(screen,fixed_budget,program,cfg)
        # External scoring: probabilities and expected production value are not feedback.
        risk=probs.mean(1);value=1-3*risk[picked]
        total_expense=cfg.price*(cost+mc)+cfg.generation*(generated+mg)+.5*sw
        net[:,t]=value-cfg.generation-total_expense/P
        harm[:,t]=risk[picked]>risk[0]+1e-12
        regret[:,t]=3*(risk[picked]-risk.min())
        expense[:,t]=total_expense/P;metacost[:,t]=(cfg.price*mc+cfg.generation*mg+.5*sw)/P
        switches[:,t]=sw;choice[:,t]=picked;program_history[:,t]=active
        if record:
            traces.append(dict(round=t,program=PROGRAMS[active[0]],version=int(versions[0]),chosen_workflow=int(picked[0]),posterior_errors=est[0].tolist(),sample_counts=ns[0].tolist(),label_units=float(cost[0]+mc[0]),generation_count=float(generated[0]+mg[0]),net_value=float(net[0,t]),meta_expense_per_task=float(metacost[0,t])))
        # Cap check per block: meta allocation at its start + screening each round.
        if t%cfg.interval==0:block_spent=np.zeros(size)
        block_spent+=cost+mc
        assert np.all(block_spent<=block_cap+1e-9)
    rows=[]
    for i,seed in enumerate(seeds):rows.append(dict(scenario=scenario,arm=arm,seed=int(seed),net_value=float(net[i].mean()),harmful_adoption=float(harm[i].mean()),routing_regret=float(regret[i].mean()),evaluation_expense=float(expense[i].mean()),procedure_expense=float(metacost[i].mean()),program_changes=int(switches[i].sum()),specialized_share=float((choice[i]==1).mean())))
    periods=[dict(scenario=scenario,arm=arm,round=t,net_value=float(net[:,t].mean()),net_ci95=float(1.96*net[:,t].std(ddof=1)/size**.5),harmful_adoption=float(harm[:,t].mean()),balanced_share=float((program_history[:,t]==1).mean()),neyman_share=float((program_history[:,t]==2).mean())) for t in range(T)]
    return rows,periods,{'events':traces,'patches':patches,'block_cap':int(block_cap),'fixed_screen_budget':int(fixed_budget)}

def summarize(rows,group='condition'):
    out=[]
    conditions=list(dict.fromkeys((r[group],r['scenario']) for r in rows))
    for condition,scenario in conditions:
        values={arm:[r for r in rows if r[group]==condition and r['scenario']==scenario and r['arm']==arm] for arm in dict.fromkeys(r['arm'] for r in rows)}
        reference={r['seed']:r['net_value'] for r in values['Balanced search']}
        for arm,rr in values.items():
            if not rr:continue
            x=np.array([r['net_value'] for r in rr]);delta=np.array([r['net_value']-reference[r['seed']] for r in rr])
            h=np.array([r['harmful_adoption'] for r in rr])
            out.append({group:condition,'scenario':scenario,'arm':arm,'n':len(rr),'net_value':float(x.mean()),'ci95':float(1.96*x.std(ddof=1)/len(x)**.5),'harmful_adoption':float(h.mean()),'harm_ci95':float(1.96*h.std(ddof=1)/len(x)**.5),'difference_vs_balanced':float(delta.mean()),'paired_ci95':float(1.96*delta.std(ddof=1)/len(x)**.5),'evaluation_expense':float(np.mean([r['evaluation_expense'] for r in rr])),'program_changes':float(np.mean([r['program_changes'] for r in rr]))})
    return out

def main():
    if '--quick' in sys.argv:
        for arm in ARMS:
            rows,_,trace=simulate('Blind harm',arm,Config(),[991,992],True)
            assert all(np.isfinite(r['net_value']) for r in rows)
        print('Budget, visibility interface, finite-score and patch-path smoke checks passed.');return
    dev=[]
    for pilot in PROTOCOL['pilot_grid']:
        for scenario in PROTOCOL['scenarios']:
            rows,_,_=simulate(scenario,'Neyman search',Config(pilot=pilot),range(410000,410064))
            dev.append(dict(pilot=pilot,scenario=scenario,mean=float(np.mean([r['net_value'] for r in rows]))))
    chosen=max(PROTOCOL['pilot_grid'],key=lambda p:np.mean([r['mean'] for r in dev if r['pilot']==p]));cfg=Config(pilot=chosen)
    csvout('coverage_development.csv',dev)
    # The single baseline choice is made on disjoint development streams.
    rows=[];periods=[];traces={}
    for scenario in PROTOCOL['scenarios']:
        for arm in ARMS+['Freeze after first revision']:
            rr,pp,tr=simulate(scenario,arm,cfg,range(510000,510256),True)
            for r in rr:r['condition']='base'
            rows+=rr;periods+=pp;traces[scenario+' / '+arm]=tr
    summary=summarize(rows);csvout('coverage_runs.csv',rows);csvout('coverage_summary.csv',summary);csvout('coverage_periods.csv',periods)
    # Explicit null for the fixed workflow, which has no posterior estimate.
    for tr in traces.values():
        for e in tr['events']:e['posterior_errors']=[None if not np.isfinite(x) else x for x in e['posterior_errors']]
    (ROOT/'examples/coverage_trace.json').write_text(json.dumps(traces,indent=2,allow_nan=False),encoding='utf-8')
    variations={'prior_1_9':replace(cfg,prior_b=9),'prior_1_19':replace(cfg,prior_b=19),'budget_24':replace(cfg,screen=24),'budget_96':replace(cfg,screen=96),'cost_ratio_3':replace(cfg,cost_ratio=3),'label_noise_010':replace(cfg,flip=.1),'generation_000':replace(cfg,generation=0),'generation_010':replace(cfg,generation=.1),'review_4':replace(cfg,interval=4),'review_12':replace(cfg,interval=12)}
    sensitivity=[]
    for condition,config in variations.items():
        for scenario in ['Blind harm','Moving harm']:
            for arm in ARMS:
                rr,_,_=simulate(scenario,arm,config,range(610000,610128))
                for r in rr:r['condition']=condition
                sensitivity+=rr
    csvout('coverage_sensitivity_runs.csv',sensitivity);csvout('coverage_sensitivity_summary.csv',summarize(sensitivity))
    checks={'protocol_sha256':hashlib.sha256((ROOT/'examples/coverage_protocol.json').read_bytes()).hexdigest(),'selected_pilot':chosen,'development_trajectories':len(dev)*64,'test_trajectories':len(rows),'sensitivity_trajectories':len(sensitivity),'common_block_budget':traces['Blind harm / Procedure revision']['block_cap'],'fixed_screen_budget':traces['Blind harm / Procedure revision']['fixed_screen_budget'],'budget_assertions_passed':True,'external_probabilities_used_only_by_sampler_and_scorer':True,'outer_program_selector':'fixed finite-catalog empirical comparison','base_results':summary}
    (ROOT/'qa/coverage_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    plot(summary,periods)
    print(json.dumps(checks,indent=2))

def plot(summary,periods):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    colors=['#777777','#D55E00','#0072B2','#009E73','#CC79A7'];markers=['s','o','^','D','P']
    fig,axs=plt.subplots(1,2,figsize=(7,3.5),layout='constrained')
    for j,arm in enumerate(ARMS):
        selected=[next(r for r in summary if r['scenario']==sc and r['arm']==arm) for sc in PROTOCOL['scenarios']]
        x=np.arange(3)+(j-2)*.13
        for ax,y,err in [(axs[0],'net_value','ci95'),(axs[1],'harmful_adoption','harm_ci95')]:
            ax.errorbar(x,[r[y] for r in selected],yerr=[r[err] for r in selected],fmt=markers[j],color=colors[j],ms=4,capsize=2,label=arm,ls='none')
    for ax in axs:ax.set(xticks=range(3),xticklabels=['Blind\nharm','Moving\nharm','Uniform\ngain'])
    axs[0].set(title='a  Costed organizational value',ylabel='Expected net value / task')
    axs[1].set(title='b  Harmful workflow adoption',ylabel='Fraction of rounds',ylim=(-.005,.20))
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    save(fig,'fig7_coverage')
    fig,axs=plt.subplots(1,2,figsize=(7,3.2),layout='constrained')
    for j,arm in enumerate(ARMS[1:],1):
        rr=[r for r in periods if r['scenario']=='Moving harm' and r['arm']==arm]
        axs[0].plot([r['round']+1 for r in rr],np.cumsum([r['net_value'] for r in rr])/np.arange(1,25),color=colors[j],ls=['-','--',':','-.'][j-1],label=arm)
    rr=[r for r in periods if r['scenario']=='Moving harm' and r['arm']=='Procedure revision']
    x=np.arange(1,25);b=np.array([r['balanced_share'] for r in rr]);n=np.array([r['neyman_share'] for r in rr])
    for y,label,color,ls in [(1-b-n,'Biased','#D55E00','-'),(b,'Balanced','#0072B2','--'),(n,'Neyman','#009E73',':')]:axs[1].plot(x,y,label=label,color=color,ls=ls)
    for ax in axs:
        for t in [1,9,17]:ax.axvline(t,color='#aaaaaa',lw=.6,ls=':')
        ax.set(xlabel='Organizational decision round',xlim=(1,24))
    axs[0].set(title='a  Cumulative value under moving harm',ylabel='Cumulative mean net value / task');axs[0].legend(frameon=False,fontsize=7)
    axs[1].set(title='b  Retained evaluation programs',ylabel='Fraction of runs',ylim=(0,1));axs[1].legend(frameon=False,fontsize=7)
    save(fig,'fig8_revision')

def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
