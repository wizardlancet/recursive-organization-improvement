"""Exploratory timing and acquisition-matched controls, declared separately."""
import json
from learning_experiment import ROOT,simulate,bank,write,aggregate
p=json.loads((ROOT/'examples/learning_controls.json').read_text())
rows=[];periods=[]
for env in p['environments']:
    for memory in p['memories']:
        for arm in p['arms']:
            rr,pp,*_=simulate(env,memory,arm,size=p['replicates'],seed=p['stream_seed']);rows+=rr;periods+=pp
    bank.cache_clear();print('Control complete:',env,flush=True)
write('learning_controls_runs.csv',rows);write('learning_controls_summary.csv',aggregate(rows));write('learning_controls_periods.csv',periods)
print('Control trajectories:',len(rows))
