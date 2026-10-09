"""Render Figure 2 from saved summary rows, with consistent display terminology.

No experiment module is imported and no outcomes are sampled. Machine-readable
arm names in the archived CSV remain unchanged.
"""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'results/learning_summary.csv') if (ROOT/'results').is_dir() else (ROOT/'reproducibility/core_results/learning_summary.csv')
rows=list(csv.DictReader(source.open(encoding='utf-8')))
arms=['Biased','Balanced','Neyman','Successive rejects','Discover once','Repeated discovery']
colors=['#D55E00','#0072B2','#009E73','#444444','#E69F00','#CC79A7']
markers=['o','^','D','x','s','P']
environments=['Stationary harm','Workflow reversal','Uniform gain']
memories=['Reset','Cumulative','Window 8']
data=[]
with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False}):
    fig,axs=plt.subplots(1,3,figsize=(8,3.6),layout='constrained')
    for ax,environment in zip(axs,environments):
        for j,arm in enumerate(arms):
            rr=[next(r for r in rows if r['environment']==environment and r['memory']==m and r['arm']==arm) for m in memories]
            label='Successive rejection' if arm=='Successive rejects' else arm
            ax.errorbar(np.arange(3)+(j-2.5)*.11,[float(r['net']) for r in rr],yerr=[float(r['net_ci95']) for r in rr],fmt=markers[j],color=colors[j],ms=4,capsize=2,label=label,ls='none')
            data.extend({'environment':environment,'arm_id':arm,'memory':r['memory'],'net':float(r['net']),'net_ci95':float(r['net_ci95'])} for r in rr)
        ax.set(title=environment,xticks=range(3),xticklabels=['Reset','Cumul.','Window 8'],ylabel='Net value / task')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8,frameon=False)
    for ext in ('pdf','svg','png'):fig.savefig(ROOT/f'figures/fig9_memory.{ext}',dpi=180)
    plt.close(fig)
svg=ROOT/'figures/fig9_memory.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
record={'source':'results/learning_summary.csv','local_snapshot':source.relative_to(ROOT).as_posix(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'purpose':'Figure 2 display label: Successive rejects -> Successive rejection','new_experiments_run':False,'data':data,'output_sha256':hashlib.sha256((ROOT/'figures/fig9_memory.pdf').read_bytes()).hexdigest()}
(ROOT/'figures/fig9_memory_provenance.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'plotted_saved_points':len(data),'new_experiments_run':False}))
