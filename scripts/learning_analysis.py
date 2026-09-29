"""Prespecified summaries plus explicitly exploratory acquisition/timing contrasts."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT.parent/'.paper_runtime').exists():sys.path.insert(0,str(ROOT.parent/'.paper_runtime'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def read(n):return list(csv.DictReader((ROOT/'results'/n).open(encoding='utf-8')))
def write(n,rr):
    with (ROOT/'results'/n).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
base=read('learning_runs.csv');summary=read('learning_summary.csv');control=read('learning_controls_runs.csv');sens=read('learning_sensitivity_runs.csv')
def values(rows,env,mode,arm,field='net',**extra):
    rr=[r for r in rows if r['environment']==env and r['memory']==mode and r['arm']==arm and all(r[k]==str(v) for k,v in extra.items())]
    return {int(r['replicate']):float(r[field]) for r in rr}
def contrast(a,b,**labels):
    assert a.keys()==b.keys() and a
    d=np.array([a[i]-b[i] for i in sorted(a)]);return dict(**labels,n=len(d),delta=float(d.mean()),ci95=float(1.96*d.std(ddof=1)/len(d)**.5))
out=[]
for env in ['Stationary harm','Workflow reversal','Uniform gain']:
    for mode in ['Reset','Cumulative','Window 8']:
        a=values(base,env,mode,'Repeated discovery')
        for ref in ['Balanced','Successive rejects','Discover once']:
            out.append(contrast(a,values(base,env,mode,ref),study='core',environment=env,memory=mode,comparison='Repeated - '+ref))
        vv=[values(base,env,mode,arm) for arm in ['Biased','Balanced','Neyman']]
        mixture={i:sum(v[i] for v in vv)/3 for i in a}
        out.append(contrast(a,mixture,study='core',environment=env,memory=mode,comparison='Repeated - uniform fixed mixture'))
    for arm in ['Balanced','Repeated discovery','Discover once']:
        for mode in ['Cumulative','Window 8']:
            out.append(contrast(values(base,env,mode,arm),values(base,env,'Reset',arm),study='core',environment=env,memory=mode,comparison=arm+' - Reset'))
for env in dict.fromkeys(r['environment'] for r in control):
    for mode in ['Reset','Cumulative','Window 8']:
        for ref in ['Balanced','Trial-matched Balanced','Discover once']:
            out.append(contrast(values(control,env,mode,'Repeated discovery'),values(control,env,mode,ref),study='exploratory control',environment=env,memory=mode,comparison='Repeated - '+ref))
write('learning_contrasts.csv',out)
selected=[r for r in out if (r['study']=='core' and r['environment']!='Uniform gain' and (r['comparison'] in ['Repeated - Balanced','Repeated - Discover once','Balanced - Reset'])) or (r['study']=='exploratory control' and r['memory']=='Window 8' and r['comparison']=='Repeated - Trial-matched Balanced')]
print(json.dumps(selected,indent=2))
lines=[]
for env in ['Stationary harm','Workflow reversal','Uniform gain']:
    for mode in ['Reset','Cumulative','Window 8']:
        cells=[]
        for arm in ['Biased','Balanced','Neyman','Successive rejects','Discover once','Repeated discovery']:
            r=next(r for r in summary if r['environment']==env and r['memory']==mode and r['arm']==arm)
            cells.append(f"{float(r['net']):.4f}")
        cells.append(f"{sum(float(x) for x in [next(r['net'] for r in summary if r['environment']==env and r['memory']==mode and r['arm']==arm) for arm in ['Biased','Balanced','Neyman']])/3:.4f}")
        lines.append(env+' & '+mode+' & '+' & '.join(cells)+r' \\')
(ROOT/'sections/learning_base_rows.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
lines=[]
for share in [.2,.4,.6]:
    for trials in [1,3,6]:
        cells=[]
        for env,mode in [('Stationary harm','Cumulative'),('Workflow reversal','Window 8')]:
            a=values(sens,env,mode,'Repeated discovery',meta_share=share,trials=trials,pool=True,program_memory=True)
            b=values(sens,env,mode,'Discover once',meta_share=share,trials=trials,pool=True,program_memory=True)
            c=contrast(a,b);cells.append(f"{c['delta']:+.4f} $\\pm$ {c['ci95']:.4f}")
        lines.append(f'{share:.1f} & {trials} & '+' & '.join(cells)+r' \\')
(ROOT/'sections/learning_grid_rows.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
lines=[]
conditional=read('learning_conditional.csv')
for env,mode,short in [('Stationary harm','Cumulative','Stationary'),('Workflow reversal','Window 8','Reversal'),('Uniform gain','Cumulative','Uniform gain')]:
    for review in [1,9,17,25,33,41]:
        rr=[r for r in conditional if r['environment']==env and r['memory']==mode and r['arm']=='Repeated discovery' and int(r['review_round'])==review]
        cells=[next(r for r in rr if r['program']==p) for p in ['Biased','Balanced','Neyman']]
        balanced=cells[1]
        vals=[f"{100*float(r['share']):.1f}" for r in cells]
        vals += [str(balanced['n']),f"{float(balanced['net']):.4f}",f"{100*float(balanced['harm']):.2f}"]
        lines.append(f'{short} & {review} & '+' & '.join(vals)+r' \\')
(ROOT/'sections/learning_program_rows.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
arms=['Biased','Balanced','Neyman','Successive rejects','Discover once','Repeated discovery'];colors=['#D55E00','#0072B2','#009E73','#444444','#E69F00','#CC79A7'];markers=['o','^','D','x','s','P']
fig,axs=plt.subplots(1,3,figsize=(8,3.6),layout='constrained')
for a,env in zip(axs,['Stationary harm','Workflow reversal','Uniform gain']):
    for j,arm in enumerate(arms):
        rr=[next(r for r in summary if r['environment']==env and r['memory']==m and r['arm']==arm) for m in ['Reset','Cumulative','Window 8']]
        a.errorbar(np.arange(3)+(j-2.5)*.11,[float(r['net']) for r in rr],yerr=[float(r['net_ci95']) for r in rr],fmt=markers[j],color=colors[j],ms=4,capsize=2,label=arm,ls='none')
    a.set(title=env,xticks=range(3),xticklabels=['Reset','Cumul.','Window 8'],ylabel='Net value / task')
fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8,frameon=False)
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=180)
    plt.close(fig)
save(fig,'fig9_memory')
periods=read('learning_periods.csv')
fig,axs=plt.subplots(1,2,figsize=(8,3.6),layout='constrained')
for mode,color,style in [('Reset','#777777',':'),('Cumulative','#0072B2','--'),('Window 8','#009E73','-')]:
    rr=[r for r in periods if r['environment']=='Workflow reversal' and r['memory']==mode and r['arm']=='Balanced']
    axs[0].plot([int(r['round']) for r in rr],[float(r['specialized_share']) for r in rr],label=mode,color=color,ls=style)
axs[0].axvline(24.5,color='#999999',lw=.8);axs[0].set(title='a  Reusing and retiring evidence',xlabel='Round',ylabel='Fraction selecting specialized',ylim=(-.02,1.02));axs[0].legend(frameon=False,fontsize=8)
for j,(comp,label,color,marker) in enumerate([('Repeated - Balanced','Ordinary Balanced','#0072B2','o'),('Repeated - Trial-matched Balanced','Trial-matched Balanced','#D55E00','s')]):
    rr=[next(r for r in out if r['study']=='exploratory control' and r['environment']==env and r['memory']=='Window 8' and r['comparison']==comp) for env in ['Workflow reversal 20','Workflow reversal','Workflow reversal 28']]
    axs[1].errorbar(np.arange(3)+(j-.5)*.1,[r['delta'] for r in rr],yerr=[r['ci95'] for r in rr],color=color,fmt=marker,capsize=3,label=label)
axs[1].axhline(0,color='#777777',lw=.8);axs[1].set(title='b  What does reassessment add?',xticks=range(3),xticklabels=['21','25','29'],xlabel='First round after reversal',ylabel='Repeated minus comparator');axs[1].legend(frameon=False,fontsize=7)
axs[1].set_ylim(-.002,.009)
axs[1].set_yticks([-.002,0,.002,.004,.006,.008])
save(fig,'fig10_attribution')
(ROOT/'qa/learning_analysis.json').write_text(json.dumps({'contrasts':out,'main_figure_policy':'All six arms and all three memory conditions shown for every core environment','exploratory_controls_reported_in_full':True},indent=2),encoding='utf-8')
