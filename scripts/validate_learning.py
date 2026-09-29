"""Independent scalar reconstruction for fixed Balanced and targeted semantic checks."""
import json
import numpy as np
from learning_experiment import ROOT,bank,simulate,screen,risk_matrix,memory_at,T,POP
cases=0
for env in ['Stationary harm','Workflow reversal','Uniform gain']:
    for mode in ['Reset','Cumulative','Window 8']:
        rows,*_=simulate(env,mode,'Balanced',size=4,seed=999001)
        for i in range(4):
            observed=[];values=[]
            for t in range(T):
                current=bank(env,t,0,0,0,999001,4)[i,:,:,:57].sum(-1)
                prior=[(tt,e) for tt,e in observed if mode=='Cumulative' or (mode=='Window 8' and tt>=t-8)]
                e=current+sum((ee for _,ee in prior),np.zeros((3,2)))
                n=57*(1+len(prior))
                estimate=((1+e)/(2+n)).mean(1);selected=int(estimate.argmin())
                risk=risk_matrix(env,t).mean(1)[selected]
                values.append(1-3*risk-.02-.22*342/POP);observed.append((t,current))
            assert abs(np.mean(values)-rows[i]['net'])<1e-12
            cases+=1
zero=np.zeros((2,3,2));labels=np.ones((2,3,2,256),bool);labels[:,0]=False
chosen,e,n=screen(labels,342,'Successive rejects',zero,zero)
assert (chosen==0).all() and (n.sum((1,2))==336).all()
entries=[(1,np.ones((1,3,2)),np.ones((1,3,2))),(2,np.full((1,3,2),2),np.full((1,3,2),2)),(9,np.full((1,3,2),9),np.full((1,3,2),9))]
e,n=memory_at(entries,10,'Window 8',(1,3,2));assert (e==11).all()
print('Independent scalar cases:',cases)
(ROOT/'qa/learning_validation.json').write_text(json.dumps({'independent_scalar_cases':cases,'absolute_tolerance':1e-12,'successive_rejection_budget_and_selection':True,'lookback_boundary':True},indent=2),encoding='utf-8')
