"""Offline paired contrasts and post-hoc phase diagnostics; policies unchanged.
Run after experiments.py and robustness.py. Phase summaries exclude periods 0-39.
"""
from pathlib import Path
import sys,csv,json
ROOT=Path(__file__).resolve().parents[1]
if (ROOT.parent/'.paper_runtime').exists():sys.path.insert(0,str(ROOT.parent/'.paper_runtime'))
import numpy as np
from scipy.special import betainc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(name):return list(csv.DictReader((ROOT/'results'/name).open(encoding='utf-8')))
def write(name,rows):
    with (ROOT/'results'/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

base=read('longitudinal_runs.csv'); contrasts=[]
for scenario in ['Stationary','Regime shifts']:
    policies=list(dict.fromkeys(r['policy'] for r in base))
    lookup={p:{int(r['seed']):float(r['value']) for r in base if r['scenario']==scenario and r['policy']==p} for p in policies}
    seeds=sorted(lookup['Adaptive audit'])
    for p in policies:
        assert set(lookup[p])==set(seeds)
        d=np.array([lookup['Adaptive audit'][s]-lookup[p][s] for s in seeds]);half=1.96*d.std(ddof=1)/len(d)**.5
        contrasts.append(dict(scenario=scenario,comparator=p,n=len(d),difference=float(d.mean()),ci95_low=float(d.mean()-half),ci95_high=float(d.mean()+half),ci95_halfwidth=float(half)))
write('paired_contrasts.csv',contrasts)

N,T,B=200,160,32
arrays=[]
for seed in range(20360928,20360928+N):
    rng=np.random.default_rng(seed);arrays.append([rng.random((T,B)) for _ in range(3)])
ua,ub,uq=[np.array([a[j] for a in arrays]) for j in range(3)]
probs=np.where((np.arange(T)//10)%2==0,.04,.25)
ea=ua<probs[None,:,None];eb=ub<.08
phase_rows=[];summary=[];robust=read('robustness_summary.csv');matches=[]
for policy in ['Fixed 0.10','Adaptive audit']:
    for window in [6,12,24]:
        nh=np.zeros((N,T),int);eh=nh.copy();means=np.zeros((N,T));safe=np.zeros((N,T),bool)
        values=np.zeros((N,T));changes=np.zeros((N,T));prev=None
        for t in range(T):
            n=nh[:,max(0,t-window):t].sum(1);e=eh[:,max(0,t-window):t].sum(1)
            a=1+e;b=9+n-e;means[:,t]=a/(a+b);safe[:,t]=means[:,t]>.14
            risk=1-betainc(a,b,.14)
            q=np.full(N,.1) if policy=='Fixed 0.10' else np.where((risk>.1)&(risk<.9),.4,.025)
            audit=uq[:,t]<q[:,None];nh[:,t]=audit.sum(1);eh[:,t]=(audit&ea[:,t]).sum(1)
            if prev is not None:changes[:,t]=prev!=q
            errors=np.where(safe[:,t,None],eb[:,t],ea[:,t]).sum(1)
            values[:,t]=np.where(safe[:,t],.82,1)-3*errors/B-.2*nh[:,t]/B-.5*changes[:,t]/B
            prev=q
        expected=float(next(r['mean'] for r in robust if r['scenario']=='Fast shifts' and r['policy']==policy and int(r['window'])==window))
        assert abs(values.mean()-expected)<1e-12
        matches.append({'policy':policy,'window':window,'absolute_error':abs(values.mean()-expected)})
        use=np.arange(T)>=40;low=use&(probs==.04);high=use&(probs==.25)
        wrong_low=float(safe[:,low].mean());wrong_high=float((~safe[:,high]).mean())
        # Conditional expected production value, averaged over realized routing choices.
        route_loss=.5*(.30*wrong_low+.33*wrong_high)
        audit_cost=float(.2*nh[:,use].mean()/B);switch_cost=float(.5*changes[:,use].mean()/B)
        summary.append(dict(policy=policy,window=window,low_regime_wrong_B=wrong_low,high_regime_wrong_A=wrong_high,route_regret=route_loss,audit_cost=audit_cost,switch_cost=switch_cost,expected_net_value=.73-route_loss-audit_cost-switch_cost,realized_net_value=float(values[:,use].mean()),all_period_value=float(values.mean())))
        for phase in range(20):
            mask=use&(np.arange(T)%20==phase)
            per_seed=means[:,mask].mean(1);b_seed=safe[:,mask].mean(1)
            phase_rows.append(dict(policy=policy,window=window,phase=phase,true_p=float(probs[phase]),posterior_mean=float(per_seed.mean()),posterior_ci95_halfwidth=float(1.96*per_seed.std(ddof=1)/N**.5),B_share=float(b_seed.mean()),B_ci95_halfwidth=float(1.96*b_seed.std(ddof=1)/N**.5)))
write('phase_diagnostics.csv',phase_rows);write('phase_summary.csv',summary)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(7,3.1),layout='constrained')
for ax in axs:ax.axvspan(9.5,19.5,color='#e8edf2',zorder=0)
for i,window in enumerate([6,12,24]):
    r=[r for r in phase_rows if r['policy']=='Fixed 0.10' and r['window']==window]
    x=np.arange(20);color=['#0072B2','#D55E00','#009E73'][i]
    for ax,key,ci in [(axs[0],'posterior_mean','posterior_ci95_halfwidth'),(axs[1],'B_share','B_ci95_halfwidth')]:
        y=np.array([a[key] for a in r]);h=np.array([a[ci] for a in r])
        ax.plot(x,y,color=color,ls=['-','--',':'][i],label=f'Window {window}');ax.fill_between(x,y-h,y+h,color=color,alpha=.12,linewidth=0)
axs[0].axhline(.14,color='#555555',ls='-.',lw=.8,label='Routing threshold')
axs[0].set(title='a  Lagged error estimate',ylabel='Mean posterior error probability',ylim=(0,.28))
axs[1].set(title='b  Route choice by regime phase',ylabel='Fraction choosing route B',ylim=(0,1))
for ax in axs:ax.set(xlabel='Phase in 20-period cycle',xlim=(-.5,19.5),xticks=[0,5,10,15,19])
axs[0].legend(frameon=False,fontsize=7,loc='upper right')
for ext in ['pdf','svg','png']:fig.savefig(ROOT/'figures'/f'fig6_phase.{ext}',dpi=180)
plt.close(fig)
report={'analysis_status':'post-hoc descriptive diagnostic','seeds':N,'discarded_initial_periods':40,'cycles_per_seed':6,'regression_checks':matches,'summary':summary,'paired_contrasts':contrasts}
(ROOT/'qa/review_analysis_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
