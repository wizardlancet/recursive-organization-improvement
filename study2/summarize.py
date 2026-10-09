"""Summarize saved Study 2 trajectories; no simulation is executed.

Default: verify the archived summaries. --write: rebuild the two summary CSVs.
"""
from pathlib import Path
import argparse, csv, json
import numpy as np

ROOT=Path(__file__).resolve().parent
METRICS=('error','net','regret','hit','acquisition_cost','production_cost')
PAIRS=[('Full_trace','Ignore_echo','cumulative','cumulative'),
       ('Full_trace','Randomized_exposure','cumulative','cumulative'),
       ('Full_trace','Full_trace','window8','cumulative'),
       ('Randomized_exposure','Randomized_exposure','window8','cumulative'),
       ('Full_trace','blind_vote:A:k3','cumulative','none')]

def ci(values):
    return float(values.mean()),float(1.96*values.std(ddof=1)/np.sqrt(len(values)))

def summarize(directory):
    meta=json.loads((directory/'learning_metadata.json').read_text())
    data=np.load(directory/'learning_raw.npz')
    summaries=[];contrasts=[]
    for wi,world in enumerate(meta['worlds']):
        for pi,(policy,memory) in enumerate(meta['policies']):
            row=dict(world=world,policy=policy,memory=memory,replicates=meta['replicates'])
            for metric in METRICS:
                for suffix,sl in [('',slice(None)),('_late',slice(-16,None))]:
                    avg,half=ci(data[metric][wi,pi,:,sl].mean(-1))
                    row[metric+suffix]=avg;row[metric+suffix+'_ci95']=half
            summaries.append(row)
        for a,b,ma,mb in PAIRS:
            ai=meta['policies'].index([a,ma]);bi=meta['policies'].index([b,mb])
            row=dict(world=world,policy_a=a,memory_a=ma,policy_b=b,memory_b=mb)
            for metric in ('error','net','regret'):
                avg,half=ci((data[metric][wi,ai]-data[metric][wi,bi]).mean(-1))
                row[metric+'_difference']=avg;row[metric+'_ci95']=half
            contrasts.append(row)
    return {'learning_summary.csv':summaries,'paired_contrasts.csv':contrasts}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',help='overwrite summaries from saved trajectories')
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    for filename,records in summarize(args.results).items():
        path=args.results/filename
        if args.write:
            with path.open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        else:
            with path.open(encoding='utf-8',newline='') as f:stored=list(csv.DictReader(f))
            assert len(stored)==len(records),(filename,len(stored),len(records))
            for got,want in zip(stored,records):
                assert set(got)==set(want)
                for k,v in want.items():
                    if isinstance(v,(int,float)):assert abs(float(got[k])-v)<1e-12,(filename,k)
                    else:assert got[k]==v,(filename,k)
        print(f'{filename}: {len(records)} rows '+('written' if args.write else 'verified'))

if __name__=='__main__':main()
