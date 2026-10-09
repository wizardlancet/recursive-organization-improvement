"""Independent bit-state enumeration, stochastic checks and learning accounting."""
from pathlib import Path
from itertools import product
import json
import numpy as np
from model import Topology,catalog,exact,execute,draw,trace_stats,fit_trace,FAMILIES,BLIND,GATES
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())

def enumerate_bits(t,ph,pa,rho,eh,ea):
    """Independent brute force over actor bits and copy decisions, no exact kernels."""
    answer=dict(error=0.,cost=0.,agreement=0.,silent_wrong=0.,manufactured_agreement=0.,escalation=0.)
    for up in (0,1):
        for a in product((0,1),repeat=t.k):
            vals=(up,)+a if t.source=='A' else a
            independent=np.prod([(1-pa) if x else pa for x in vals])
            shared=((1-pa) if vals[0] else pa) if len(set(vals))==1 else 0.
            weight=(1-rho)*independent+rho*shared
            if t.source=='H':weight*=1-ph if up else ph
            for copy in product((0,1),repeat=t.k):
                report=[];w=weight
                for j in range(t.k):
                    eta=0. if t.family in BLIND else (eh if t.source=='H' and (t.family!='serial_last' or j==0) else ea)
                    w*=eta if copy[j] else 1-eta
                    upstream=report[-1] if t.family=='serial_last' and j else up
                    report.append(upstream if copy[j] else a[j])
                agree=all(x==up for x in report);preagree=all(x==up for x in a)
                call=0.;base=(1 if t.source=='H' else .25)+t.k*.25+(.05*t.k if t.family in BLIND else 0)
                if t.family in GATES:
                    call=(.25 if t.family=='audit_gate' else 0) if agree else 1.
                    error=(1-call)*(1-up)+call*ph
                elif t.family=='serial_last':error=1-report[-1]
                else:
                    count=up+sum(report);error=float(2*count<t.k+1)+.5*float(2*count==t.k+1)
                values=dict(error=error,cost=base+call,agreement=agree,silent_wrong=(1-up)*agree,
                            manufactured_agreement=agree and not preagree,escalation=call)
                for k,v in values.items():answer[k]+=w*v
    return answer

def run():
    qa=ROOT/'qa';qa.mkdir(exist_ok=True)
    environments=[(.2,.1,0.,0.,0.),(.2,.1,.3,.8,.4),(.1,.25,.9,1.,1.),(.3,.05,1.,.7,.2)]
    maximum=0.;count=0
    for e,t in product(environments,[t for t in catalog(3) if t.k]):
        slow=enumerate_bits(t,*e);fast=exact(t,*e)
        for k in slow:maximum=max(maximum,abs(slow[k]-float(fast[k])))
        count+=1
    assert maximum<1e-11,(maximum,count)
    rng=np.random.default_rng(P['validation_seed']);mc=[];n=150000
    for e in environments:
        panel=draw(rng,(n,),*e[:3],7)
        for t in catalog(7):
            fast=exact(t,*e);real=execute(t,panel,*e[3:])
            for metric in ('error','cost','agreement','silent_wrong','manufactured_agreement','escalation'):
                x=real[metric];mean=x.mean();se=x.std()/np.sqrt(n)
                assert abs(mean-fast[metric])<=7*se+2e-5,(e,t,metric,mean,fast[metric],se)
            mc.append(dict(env=e,topology=t.name,error_mc=float(real['error'].mean()),error_exact=float(fast['error'])))
    # Closed-form confounding of intrinsic dependence and copying for one reviewer.
    t=Topology('exposed_vote','A',1)
    a=exact(t,.2,.1,.8,0,0);b=exact(t,.2,.1,0,.8,.8)
    for key in ('error','agreement','silent_wrong'):assert abs(a[key]-b[key])<1e-12
    # eta=1 suppresses every star disagreement, eta=0 equals the private vote.
    for s,k in product(('H','A'),range(1,8)):
        a=exact(Topology('echo_gate',s,k),.2,.1,.3,1,1)
        assert abs(a['escalation'])<1e-12
        u=.2 if s=='H' else .1
        assert abs(a['error']-u)<1e-12
        a=exact(Topology('exposed_vote',s,k),.2,.1,.3,0,0)
        b=exact(Topology('blind_vote',s,k),.2,.1,.3,.9,.9)
        assert abs(a['error']-b['error'])<1e-12
    panel=draw(rng,(1,300000),.2,.1,.35,5)
    estimates=[float(x[0]) for x in fit_trace(trace_stats(panel,.8,.3))]
    assert np.max(np.abs(np.array(estimates)-[.2,.1,.35,.8,.3]))<.015
    from learning_experiment import Memory
    for mode,want in [('reset',10),('cumulative',55),('window8',54)]:
        m=Memory(mode)
        for i in range(1,11):v=m.add(np.array(float(i)))
        assert v==want
    result={'passed':True,'enumeration_cases':count,'maximum_absolute_enumeration_error':maximum,
            'monte_carlo_cases':len(mc),'tasks_per_environment':n,'independent_environment_panels':len(environments),
            'monte_carlo_rule':'7 SE + 0.00002 tolerance across dependent diagnostic comparisons',
            'fitted_parameters_large_sample':estimates,'learning_accounting':'pending full run'}
    path=ROOT/'results/learning_raw.npz'
    if path.exists():
        d=np.load(path);C=P['learning'];acq=d['acquisition_cost']
        assert acq.max()<=C['acquisition_budget_per_round']+1e-8
        assert acq.reshape(*acq.shape[:-1],6,8).sum(-1).max()<=C['block_budget']+1e-8
        expected=1-3*d['error']-.02*d['production_cost']-(acq+.5*d['changed'])/4096
        assert np.max(np.abs(expected-d['net']))<1e-12
        assert d['regret'].min()>-1e-10
        assert np.all((d['error']>=0)&(d['error']<=1))
        assert d['production_cost'].max()<=3+1e-10
        result['learning_accounting']='passed';result['maximum_acquisition_per_round']=float(acq.max())
        result['learning_trajectories']=int(np.prod(acq.shape[:-1]))
    (qa/'validation.json').write_text(json.dumps(result,indent=2))
    (qa/'monte_carlo_checks.json').write_text(json.dumps(mc,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':run()
