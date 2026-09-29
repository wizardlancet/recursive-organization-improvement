"""Deterministic contrasts, accounting identities and manuscript table rows."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT.parent/'.paper_runtime').exists():sys.path.insert(0,str(ROOT.parent/'.paper_runtime'))
import numpy as np
from contracts import checks
def read(name):return list(csv.DictReader((ROOT/'results'/name).open(encoding='utf-8')))
rows=read('coverage_runs.csv');contrasts=[]
for sc in ['Blind harm','Moving harm','Uniform gain']:
    rr=[r for r in rows if r['scenario']==sc]
    by={arm:{int(r['seed']):r for r in rr if r['arm']==arm} for arm in {r['arm'] for r in rr}}
    for comparator in ['Biased search','Balanced search','Neyman search','Freeze after first revision']:
        a=by['Procedure revision'];b=by[comparator];d=np.array([float(a[s]['net_value'])-float(b[s]['net_value']) for s in sorted(a)])
        contrasts.append(dict(scenario=sc,comparator=comparator,delta=float(d.mean()),ci95=float(1.96*d.std(ddof=1)/len(d)**.5)))
with (ROOT/'results/coverage_contrasts.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(contrasts[0]));w.writeheader();w.writerows(contrasts)
for r in rows:
    best=.76 if r['scenario']=='Uniform gain' else .52
    assert abs(float(r['net_value'])-(best-.02-float(r['routing_regret'])-float(r['evaluation_expense'])))<1e-12
    assert 0<=float(r['harmful_adoption'])<=1
    assert float(r['procedure_expense'])<=float(r['evaluation_expense'])+1e-12
    if r['arm']=='Fixed workflow':assert abs(float(r['net_value'])-.38)<1e-12
summary=read('coverage_summary.csv');lines=[]
for sc in ['Blind harm','Moving harm','Uniform gain']:
    for arm in ['Fixed workflow','Biased search','Balanced search','Neyman search','Procedure revision','Freeze after first revision']:
        r=next(r for r in summary if r['scenario']==sc and r['arm']==arm)
        label='Freeze after first' if arm=='Freeze after first revision' else arm
        lines.append(f"{sc} & {label} & {float(r['net_value']):.4f} $\\pm$ {float(r['ci95']):.4f} & {100*float(r['harmful_adoption']):.2f} & {float(r['evaluation_expense']):.4f} \\\\")
(ROOT/'sections/v5_base_rows.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
sens=read('coverage_sensitivity_summary.csv');lines=[]
for cond in dict.fromkeys(r['condition'] for r in sens):
    vals=[]
    for sc in ['Blind harm','Moving harm']:
        r=next(r for r in sens if r['condition']==cond and r['scenario']==sc and r['arm']=='Procedure revision')
        vals.append(f"{float(r['difference_vs_balanced']):+.4f} $\\pm$ {float(r['paired_ci95']):.4f}")
    lines.append(cond.replace('_',r'\_')+' & '+' & '.join(vals)+r' \\')
(ROOT/'sections/v5_sensitivity_rows.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
ext=read('audit_extensions_summary.csv');lines=[]
for prior in ['1:1','1:9','1:19']:
    for theta in [.14,.16]:
        vals=[]
        for sc in ['Stationary','Regime shifts']:
            for arm in ['Fixed 0.10','Adaptive audit','EVSI']:
                r=next(r for r in ext if r['prior']==prior and float(r['threshold'])==theta and float(r['candidate_generation_cost'])==0 and r['scenario']==sc and r['policy']==arm)
                vals.append(f"{float(r['mean']):.4f}")
        lines.append(prior+f' & {theta:.2f} & '+' & '.join(vals)+r' \\')
(ROOT/'sections/audit_rows.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
report={'accounting_identities_checked':len(rows),'contracts':checks(),'contrasts':contrasts,'sensitivity_revision_vs_balanced_range':[min(float(r['difference_vs_balanced']) for r in sens if r['arm']=='Procedure revision'),max(float(r['difference_vs_balanced']) for r in sens if r['arm']=='Procedure revision')]}
(ROOT/'qa/v5_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
