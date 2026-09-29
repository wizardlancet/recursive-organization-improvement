"""All data here are synthetic. No human data or trained language model is used.
Reproduce figures and tables: python scripts/experiments.py
"""
from pathlib import Path
import sys,json,csv,platform
ROOT=Path(__file__).resolve().parents[1]
runtime=ROOT.parent/'.paper_runtime'
if runtime.exists():sys.path.insert(0,str(runtime))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.special import betainc
from contracts import checks
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.titleweight':'bold','axes.labelcolor':'#263746','text.color':'#263746','savefig.facecolor':'white'})
COL=['#0072B2','#D55E00','#009E73','#CC79A7','#666666']
for d in ['figures','results','qa']:(ROOT/d).mkdir(exist_ok=True)
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=180)
    plt.close(fig)
def csvout(name,rows):
    with (ROOT/'results'/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

# Probe 1. Exact steady-state fluid network; capacities in accepted items / period.
def throughput(ai,wf,demand=20,review=5):
    return min(demand,4*(1+ai),review*(1+wf),9)
fac=[]
for demand in [3,6,20]:
    for review in [3,5,8]:
        cells=[throughput(a,w,demand,review) for a,w in [(0,0),(1,0),(0,1),(1,1)]]
        fac.append(dict(demand=demand,review=review,y00=cells[0],y10=cells[1],y01=cells[2],y11=cells[3],interaction=cells[3]-cells[1]-cells[2]+cells[0]))
csvout('factorial.csv',fac)
assert throughput(1,0)==5 and throughput(1,1)==8
assert throughput(1,1,3)-throughput(1,0,3)-throughput(0,1,3)+throughput(0,0,3)==0
fig,axs=plt.subplots(1,2,figsize=(7,2.7),layout='constrained')
cells=[throughput(a,w) for a,w in [(0,0),(1,0),(0,1),(1,1)]]
axs[0].bar(range(4),cells,color=[COL[4],COL[0],COL[1],COL[2]],width=.65)
axs[0].set(xticks=range(4),xticklabels=['Neither','AI only','Workflow\nonly','Both'],ylim=(0,9),ylabel='Accepted items / period',title='a  Bottleneck complementarity')
for i,y in enumerate(cells):axs[0].text(i,y+.12,str(y),ha='center')
for i,demand in enumerate([3,6,20]):
    rs=np.linspace(2,10,81)
    z=[throughput(1,1,demand,r)-throughput(1,0,demand,r)-throughput(0,1,demand,r)+throughput(0,0,demand,r) for r in rs]
    axs[1].plot(rs,z,color=COL[i],linestyle=['-','--',':'][i],label=f'Demand = {demand}')
axs[1].set(xlabel='Initial review capacity',ylabel='Factorial interaction (items / period)',title='b  Complementarity has boundaries',ylim=(-.15,4.2))
axs[1].legend(frameon=False,fontsize=8)
save(fig,'fig3_capacity')

# Probe 2. Exact variance of unbiased errors and a cost-accounted breadth crossover.
rows=[]
fig,axs=plt.subplots(1,3,figsize=(7,2.9),layout='constrained')
for rho,ls,c in [(0,'-',COL[0]),(.3,'--',COL[1]),(.7,':',COL[2])]:
    n=np.arange(1,13);v=rho+(1-rho)/n
    axs[0].plot(n,v,ls,color=c,label=f'ρ = {rho}')
    for nn,vv in zip(n,v):rows.append(dict(module='ensemble',x=float(nn),parameter=rho,y=float(vv)))
axs[0].set(xlabel='Number of judgments',ylabel='Variance / individual variance',title='a  Correlation',ylim=(0,1.05));axs[0].legend(frameon=False,fontsize=7)
# Human variance=1, agent variance=.64, covariance=0; include optimal private baseline.
alpha=np.linspace(0,1,101)
blind=.25*(1+.64)
order=((1-alpha)/2)**2+((1+alpha)/2)**2*.64
axs[1].plot(alpha,order,color=COL[1],label='Exposed, equal aggregate')
axs[1].axhline(blind,color=COL[0],ls='--',label='Private, equal weights')
optimal=.64/1.64
axs[1].axhline(optimal,color=COL[2],ls=':',lw=1.6,label='Private, optimal weights')
assert abs((((1-(1-.64)/(1+.64))/2)**2+((1+(1-.64)/(1+.64))/2)**2*.64)-optimal)<1e-12
for name,value in [('private_equal',blind),('private_optimal',optimal)]:
    rows.append(dict(module=name,x=0.,parameter=.64,y=value))
axs[1].set(xlabel='Advice weight α',ylabel='Aggregate mean squared error',title='b  Evidence reuse',ylim=(.3,.85));axs[1].legend(frameon=False,fontsize=6.3,loc='upper left')
for aa,yy in zip(alpha,order):rows.append(dict(module='evidence_reuse',x=float(aa),parameter=.64,y=float(yy)))
h=np.linspace(0,1.2,101)
for i,m in enumerate([1,2,3]):
    gain=m*h-(.35+.10*m)
    axs[2].plot(h,gain,color=COL[i],linestyle=['-','--',':'][i],label=f'{m} handoff(s) avoided')
    for hh,gg in zip(h,gain):rows.append(dict(module='breadth',x=float(hh),parameter=m,y=float(gg)))
axs[2].axhline(0,color='#777777',lw=.7)
axs[2].set(xlabel='Cost per avoided handoff',ylabel='Net gain (normalized value)',title='c  Functional breadth');axs[2].legend(frameon=False,fontsize=7.2)
csvout('mechanisms.csv',rows);save(fig,'fig4_mechanisms')
assert abs(order[-1]-.64)<1e-12

# Probe 3. Diagnostic auditing changes evidence available to the routing learner.
N=400;T=160;B=32;WINDOW=12;QLOW=.025;QHIGH=.4
policies=['Fixed sparse','Fixed dense','Periodic dense','Adaptive audit','No audit','Fixed 0.05','Fixed 0.10','Fixed 0.20','Fixed 0.60']
scenarios=['Stationary','Regime shifts']
allrows=[];times=[];trace=[]
for scenario in scenarios:
    probs=np.full(T,.04)
    if scenario=='Regime shifts':probs[40:80]=.25;probs[120:160]=.25
    for policy in policies:
        scores=[];qs=[];safes=[]
        for seed in range(N):
            rng=np.random.default_rng(20260928+seed)
            # Common random numbers across policies and scenarios.
            ua=rng.random((T,B));ub=rng.random((T,B));uq=rng.random((T,B))
            errsA=ua<probs[:,None];errsB=ub<.08
            history=[];qprev=None;v=[];qseq=[];bseq=[];audit_total=0;switch_total=0;error_total=0
            for t in range(T):
                prior=history[max(0,t-WINDOW):t]
                n=sum(k[0] for k in prior);e=sum(k[1] for k in prior)
                a=1+e;b=9+n-e # common Beta(1,9), prior mean .1
                mean=a/(a+b);risk=1-betainc(a,b,.14)
                safe=mean>.14
                if policy=='Fixed sparse':q=QLOW
                elif policy=='Fixed dense':q=QHIGH
                elif policy=='Periodic dense':q=QHIGH if t%12<3 else QLOW
                elif policy=='Adaptive audit':q=QHIGH if .1<risk<.9 else QLOW
                elif policy.startswith('Fixed 0.'):q=float(policy.split()[-1])
                else:q=0.
                audit=uq[t]<q;na=int(audit.sum());ea=int((audit&errsA[t]).sum())
                history.append((na,ea))
                switchcost=.5 if qprev is not None and q!=qprev else 0.
                outputerror=errsB[t] if safe else errsA[t]
                value=((.82 if safe else 1.)*B-3*int(outputerror.sum())-.2*na-switchcost)/B
                v.append(value);qseq.append(q);bseq.append(float(safe));qprev=q
                audit_total+=na;switch_total+=switchcost/.5;error_total+=int(outputerror.sum())
                if seed==0:trace.append(dict(scenario=scenario,policy=policy,period=t,posterior_mean=mean,prob_above_threshold=risk,audit_probability=q,audited=na,observed_errors=ea,route='B' if safe else 'A',net_value=value))
            scores.append(v);qs.append(qseq);safes.append(bseq)
            allrows.append(dict(scenario=scenario,policy=policy,seed=20260928+seed,value=float(np.mean(v)),audit_rate=float(np.mean(qseq)),safe_route_share=float(np.mean(bseq)),audited=audit_total,switches=switch_total,errors=error_total))
        scores=np.array(scores);qs=np.array(qs);safes=np.array(safes)
        for t in range(T):times.append(dict(scenario=scenario,policy=policy,period=t,value=float(scores[:,t].mean()),se=float(scores[:,t].std(ddof=1)/np.sqrt(N)),audit_rate=float(qs[:,t].mean()),safe_route_share=float(safes[:,t].mean())))
csvout('longitudinal_runs.csv',allrows);csvout('longitudinal_periods.csv',times);csvout('policy_trace_seed0.csv',trace)
summary=[]
for sc in scenarios:
    vals={p:np.array([r['value'] for r in allrows if r['scenario']==sc and r['policy']==p]) for p in policies}
    for p in policies:
        x=vals[p];delta=x-vals['Fixed dense']
        summary.append(dict(scenario=sc,policy=p,mean=float(x.mean()),ci95_halfwidth=float(1.96*x.std(ddof=1)/np.sqrt(N)),difference_vs_dense=float(delta.mean()),paired_ci95_halfwidth=float(1.96*delta.std(ddof=1)/np.sqrt(N))))
csvout('longitudinal_summary.csv',summary)
fig,axs=plt.subplots(1,2,figsize=(7,3.05),layout='constrained')
for i,sc in enumerate(scenarios):
    for j,p in enumerate(policies[:5]):
        z=[r for r in times if r['scenario']==sc and r['policy']==p]
        # Nonoverlapping eight-period means, explicitly disclosed.
        y=np.array([r['value'] for r in z]).reshape(20,8).mean(1)
        axs[i].plot(np.arange(20)*8+3.5,y,color=COL[j],ls=['-','--',':','-.','-'][j],lw=1.3,label=p,marker='.' if j==4 else None)
    if i==1:
        for l,r in [(40,80),(120,160)]:axs[i].axvspan(l,r,color='#e8edf2',zorder=0)
    axs[i].set(title=f'{chr(97+i)}  {sc}',xlabel='Period (32 tasks each)',ylabel='Net value per task',ylim=(.12,.94),xlim=(0,160))
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
save(fig,'fig5_longitudinal')

# Accounting sensitivity: same traces, varied audit cost, no policy retuning.
sensitivity=[]
for sc in scenarios:
    for cost in [0,.05,.1,.2,.4,.8]:
        for p in policies:
            z=[r['value']+(.2-cost)*r['audited']/(T*B) for r in allrows if r['scenario']==sc and r['policy']==p]
            sensitivity.append(dict(scenario=sc,audit_cost=cost,policy=p,mean=float(np.mean(z))))
csvout('audit_cost_sensitivity.csv',sensitivity)

# Conceptual diagrams: native vector output with editable text.
def box(ax,x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.02',facecolor=color,edgecolor='#bdcbd5',linewidth=.8))
    ax.text(x+w/2,y+h*.72,title,ha='center',va='center',fontsize=10,fontweight='bold')
    ax.text(x+w/2,y+h*.34,body,ha='center',va='center',fontsize=8,linespacing=1.45)
def arrow(ax,a,b,label=None):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#486779',lw=1.1))
    if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.023,label,ha='center',fontsize=7.5)
fig,ax=plt.subplots(figsize=(7,3.85));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.015,.57,.29,.31,'Organization state','Tasks · actors · dependencies\nInformation · authority\nInteraction protocol', '#e9f2f8')
box(ax,.355,.57,.29,.31,'Versioned event trace','Commit → reveal → evaluate\nObservation and provenance\nTime and resource cost','#e8f4ef')
box(ax,.695,.57,.29,.31,'Change procedure','Propose · test · select\nRetain · monitor · roll back\nEvidence acquisition rule','#fff0e6')
arrow(ax,(.305,.73),(.35,.73));arrow(ax,(.645,.73),(.69,.73))
box(ax,.12,.11,.33,.22,'Workflow patch','Changes task allocation,\ninteraction, or authority','#f0f3f5')
box(ax,.56,.11,.33,.22,'Procedure patch','Revises one or more parts\nof the change procedure','#f4ecf5')
arrow(ax,(.76,.56),(.72,.34));arrow(ax,(.69,.58),(.45,.30));arrow(ax,(.28,.34),(.16,.56));arrow(ax,(.88,.34),(.88,.56))
ax.text(.67,.43,'proposes',fontsize=7.5,ha='center');ax.text(.94,.45,'revises',fontsize=7.5,ha='center')
ax.text(.50,.98,'A declared boundary makes the level of change inspectable',ha='center',va='top',fontsize=10,fontweight='bold')
ax.text(.50,.015,'External criterion, authority limits, and evaluation horizon are declared separately.',ha='center',fontsize=8)
fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.02);save(fig,'fig1_framework')
fig,ax=plt.subplots(figsize=(7,3.6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def wfnode(x,y,w,label,color):
    ax.add_patch(FancyBboxPatch((x,y),w,.13,boxstyle='round,pad=.006,rounding_size=.015',facecolor=color,edgecolor='#bdcbd5',lw=.8))
    ax.text(x+w/2,y+.065,label,ha='center',va='center',fontsize=8.5,fontweight='bold')
ax.text(.015,.94,'Execution',fontsize=9,fontweight='bold')
for i,b in enumerate(['Requirement','Agent drafts','Review + test','Release approval']):
    x=.02+i*.25;wfnode(x,.76,.21,b,'#e9f2f8')
    if i<3:arrow(ax,(x+.214,.825),(x+.247,.825))
ax.text(.015,.65,'Workflow change',fontsize=9,fontweight='bold')
for x,y,w,label in [(.02,.33,.18,'Risk triage'),(.28,.44,.23,'Automated tests'),(.28,.23,.23,'Human review*'),(.59,.33,.17,'Merge evidence'),(.81,.33,.175,'Release approval')]:wfnode(x,y,w,label,'#e8f4ef')
for a,b in [((.20,.40),(.275,.505)),((.20,.39),(.275,.295)),((.515,.505),(.585,.40)),((.515,.295),(.585,.39)),((.765,.395),(.805,.395))]:arrow(ax,a,b)
ax.text(.50,.135,'*Human review is conditional; tests and selected reviews can run concurrently.',ha='center',fontsize=8)
ax.text(.50,.045,'Procedure change: audit unflagged work and record coverage when evaluating workflow changes.',ha='center',fontsize=7.8)
fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.02);save(fig,'fig2_workflow')

checks_result=checks()
checks_result.update({'factorial_exact_values':cells,'synthetic_replicates':N,'advice_variance_endpoint':float(order[-1]),'python':platform.python_version(),'numpy':np.__version__,'scipy':__import__('scipy').__version__,'matplotlib':matplotlib.__version__})
(ROOT/'qa/computation_checks.json').write_text(json.dumps(checks_result,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
