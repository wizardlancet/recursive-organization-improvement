"""Finite-catalog exact benchmarks; no Monte Carlo noise in phase diagrams."""
from pathlib import Path
from itertools import product
import csv,json
import numpy as np
from model import catalog,exact,exact_catalog,Topology
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())

def write_csv(path,rows):
    rows=iter(rows);first=next(rows)
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(first));w.writeheader();w.writerow(first);w.writerows(rows)

def run():
    out=ROOT/'results';out.mkdir(exist_ok=True)
    cfg=P['phase'];ts=catalog(cfg['max_reviewers'])
    keys=['p_h','p_a','rho','eta','human_agent_price_ratio','blind_overhead']
    grid=np.array(list(product(*(cfg[k] for k in keys))))
    ph,pa,rho,eta,ratio,overhead=grid.T
    vals=exact_catalog(ts,ph,pa,rho,eta,eta,1/ratio,overhead)
    records=[]
    for budget,regime in product(cfg['budget'],cfg['budget_regimes']):
        feasible=vals['hard_cost' if regime=='hard' else 'cost']<=budget+1e-10
        risks=np.where(feasible,vals['error'],np.inf)
        minrisk=risks.min(-1)
        tied=np.abs(risks-minrisk[:,None])<1e-12
        best=np.where(tied,vals['cost'],np.inf).argmin(-1)
        dl=3*vals['error']+.02*vals['cost']
        best_dl=np.where(feasible,dl,np.inf).argmin(-1)
        for i,j in enumerate(best):
            t=ts[j];row=dict(zip(keys,grid[i]));row.update(budget=budget,regime=regime,winner=t.name,
                family=t.family,source=t.source,reviewers=t.k,error_tie_count=int(tied[i].sum()),
                deployment_winner=ts[best_dl[i]].name,deployment_loss=float(dl[i,best_dl[i]]))
            row.update({k:float(v[i,j]) for k,v in vals.items()});records.append(row)
    write_csv(out/'phase.csv',records)
    fixed=P['fixed_roster'];keys2=['p_h','p_a','rho','eta']
    fgrid=np.array(list(product(*(fixed[k] for k in keys2))))
    ph,pa,rho,eta=fgrid.T;records=[]
    for s,k in product(fixed['sources'],fixed['reviewer_counts']):
        for t in [x for x in ts if x.source==s and x.k==k]:
            v=exact(t,ph,pa,rho,eta,eta)
            for i in range(len(fgrid)):
                row=dict(zip(keys2,fgrid[i]));row.update(topology=t.name,family=t.family,source=s,reviewers=k)
                row.update({key:float(x[i]) for key,x in v.items()});records.append(row)
    write_csv(out/'fixed_roster.csv',records)
    # Source identity sensitivity, orthogonal to private dependence.
    records=[]
    for eh,ea,rho in product([0,.2,.5,.8,1],[0,.2,.5,.8,1],[0,.5,.9]):
        v=exact_catalog(ts,.2,.1,rho,eh,ea)
        for i,t in enumerate(ts):
            row=dict(eta_h=eh,eta_a=ea,rho=rho,topology=t.name)
            row.update({key:float(x[i]) for key,x in v.items()});records.append(row)
    write_csv(out/'source_asymmetry.csv',records)
    summary={'phase_cells':len(grid)*6,'phase_candidates':len(ts),'fixed_roster_rows':len(fgrid)*2*7*6,'source_asymmetry_rows':len(records)}
    (out/'oracle_summary.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':run()
