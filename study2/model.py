"""Synthetic proposer/reviewer organizations. No manuscript imports or writes.

All bits below encode correctness, exploiting binary truth symmetry. An agent
proposer belongs to the SAME correlated private-signal group as its reviewers.
Echo is subsequent causal copying, not a change in private ability/correlation.
"""
from dataclasses import dataclass
from math import comb
import numpy as np

FAMILIES=('exposed_vote','blind_vote','echo_gate','blind_gate','audit_gate','serial_last')
GATES=('echo_gate','blind_gate','audit_gate')
BLIND=('blind_vote','blind_gate')

@dataclass(frozen=True)
class Topology:
    family: str
    source: str
    k: int
    @property
    def name(self): return f'{self.family}:{self.source}:k{self.k}'

def catalog(max_reviewers=7):
    return [Topology('alone',s,0) for s in ('H','A')]+[
        Topology(f,s,k) for s in ('H','A') for k in range(1,max_reviewers+1) for f in FAMILIES]

def hard_cost(t,c_a=.25,blind_overhead=.05):
    return (1. if t.source=='H' else c_a)+t.k*c_a+(t.family in GATES)+t.k*blind_overhead*(t.family in BLIND)

def binomial(n,p):
    p=np.asarray(p,float)
    return np.stack([comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)],axis=-1)

def joint_private(t,p_h,p_a,rho):
    """P(upstream correctness u, reviewer correct count c), including dependence."""
    p_h,p_a,rho=np.broadcast_arrays(p_h,p_a,rho)
    base=binomial(t.k,1-p_a)
    out=np.empty(p_h.shape+(2,t.k+1))
    if t.source=='H':
        reviewers=(1-rho[...,None])*base
        reviewers[...,0]+=rho*p_a;reviewers[...,-1]+=rho*(1-p_a)
        out[...,0,:]=p_h[...,None]*reviewers
        out[...,1,:]=(1-p_h[...,None])*reviewers
    else:
        out[...,0,:]=p_a[...,None]*(1-rho[...,None])*base
        out[...,1,:]=(1-p_a[...,None])*(1-rho[...,None])*base
        out[...,0,0]+=rho*p_a;out[...,1,-1]+=rho*(1-p_a)
    return out

def maj_error(correct,n): return float(2*correct<n)+.5*float(2*correct==n)

def exact(t,p_h,p_a,rho,eta_h,eta_a,c_a=.25,blind_overhead=.05):
    p_h,p_a,rho,eta_h,eta_a,c_a,blind_overhead=np.broadcast_arrays(
        *map(lambda x:np.asarray(x,float),(p_h,p_a,rho,eta_h,eta_a,c_a,blind_overhead)))
    pu=p_h if t.source=='H' else p_a
    zeros=np.zeros_like(pu)
    if t.family=='alone':
        cost=np.ones_like(pu) if t.source=='H' else c_a
        return dict(error=pu,cost=cost,hard_cost=cost,agreement=np.ones_like(pu),
                    silent_wrong=pu,wrong_up_approval=np.ones_like(pu),manufactured_agreement=zeros,
                    escalation=zeros,unaudited_wrong=pu)
    eta=(eta_h if t.source=='H' else eta_a) if t.family not in BLIND else zeros
    joint=joint_private(t,p_h,p_a,rho)
    err=zeros.copy();agree=zeros.copy();silent=zeros.copy();preagree=zeros.copy()
    for u in (0,1):
        for c in range(t.k+1):
            w=joint[...,u,c]
            d=t.k-c if u else c
            if t.family=='serial_last' and d:
                a=((t.k-d)/t.k)*eta_a**d+(d/t.k)*eta*eta_a**(d-1)
            else: a=eta**d
            agree+=w*a;silent+=w*a*(1-u)
            if d==0: preagree+=w
            if t.family in ('exposed_vote','blind_vote'):
                for j in range(d+1):
                    post=c+j if u else c-j
                    err+=w*comb(d,j)*eta**j*(1-eta)**(d-j)*maj_error(u+post,t.k+1)
    escalation=zeros.copy();unaudited=silent.copy()
    if t.family in GATES:
        audit=.25 if t.family=='audit_gate' else 0.
        escalation=1-(1-audit)*agree
        unaudited=(1-audit)*silent
        err=unaudited+escalation*p_h
    elif t.family=='serial_last':
        first=(1-eta)*p_a+eta*pu
        err=p_a+eta_a**(t.k-1)*(first-p_a)
    cost=(1. if t.source=='H' else c_a)+t.k*c_a+t.k*blind_overhead*(t.family in BLIND)+escalation
    return dict(error=err,cost=cost,hard_cost=np.broadcast_to(hard_cost(t,c_a,blind_overhead),pu.shape),
                agreement=agree,silent_wrong=silent,wrong_up_approval=silent/pu,
                manufactured_agreement=agree-preagree,escalation=escalation,unaudited_wrong=unaudited)

def exact_catalog(ts,p_h,p_a,rho,eta_h,eta_a,c_a=.25,blind_overhead=.05):
    rows=[exact(t,p_h,p_a,rho,eta_h,eta_a,c_a,blind_overhead) for t in ts]
    return {key:np.stack([r[key] for r in rows],axis=-1) for key in rows[0]}

def draw(rng,shape,p_h,p_a,rho,max_reviewers=7):
    a=rng.random(shape+(max_reviewers+1,))>=p_a
    shared=rng.random(shape)>=p_a;on=rng.random(shape)<rho
    a=np.where(on[...,None],shared[...,None],a)
    return dict(h=rng.random(shape+(2,))>=p_h,a=a,
                copy_h=rng.random(shape+(max_reviewers,)),copy_a=rng.random(shape+(max_reviewers,)),
                audit=rng.random(shape),tie=rng.random(shape))

def reports(t,panel,eta_h,eta_a):
    up=panel['h'][...,0] if t.source=='H' else panel['a'][...,0]
    private=panel['a'][...,1:t.k+1]
    coins=panel['copy_h' if t.source=='H' else 'copy_a'][...,:t.k]
    eta=eta_h if t.source=='H' else eta_a
    if t.family in BLIND: post=private
    elif t.family=='serial_last':
        previous=up;rs=[]
        for j in range(t.k):
            previous=np.where(coins[...,j]<(eta if j==0 else eta_a),previous,private[...,j]);rs.append(previous)
        post=np.stack(rs,axis=-1)
    else: post=np.where(coins<eta,up[...,None],private)
    return up,private,post

def execute(t,panel,eta_h,eta_a,c_a=.25,blind_overhead=.05):
    up=panel['h'][...,0] if t.source=='H' else panel['a'][...,0]
    base=(1. if t.source=='H' else c_a)+t.k*c_a+t.k*blind_overhead*(t.family in BLIND)
    if t.family=='alone':
        return dict(error=(~up).astype(float),cost=np.full(up.shape,base),agreement=np.ones(up.shape),
                    silent_wrong=(~up).astype(float),manufactured_agreement=np.zeros(up.shape),escalation=np.zeros(up.shape))
    up,private,post=reports(t,panel,eta_h,eta_a)
    agree=np.all(post==up[...,None],axis=-1)
    preagree=np.all(private==up[...,None],axis=-1)
    escalation=np.zeros(up.shape,dtype=bool)
    if t.family in GATES:
        escalation=~agree
        if t.family=='audit_gate': escalation=escalation|(panel['audit']<.25)
        final=np.where(escalation,panel['h'][...,1],up)
    elif t.family=='serial_last':final=post[...,-1]
    else:
        count=up.astype(int)+post.sum(-1);n=t.k+1
        final=(2*count>n)|((2*count==n)&(panel['tie']<.5))
    return dict(error=(~final).astype(float),cost=base+escalation,
                agreement=agree.astype(float),silent_wrong=((~up)&agree).astype(float),
                manufactured_agreement=(agree&(~preagree)).astype(float),escalation=escalation.astype(float))

def private_stats(h,a):
    """Returns 4 sufficient counts per independent batch (penultimate sample axis)."""
    n=h.shape[-2];ae=(~a).sum(-1)
    return np.stack([np.full(h.shape[:-2],n),(~h).sum((-2,-1)),ae.sum(-1),(ae*(ae-1)/2).sum(-1)],axis=-1)

def fit_private(stats,max_a=6):
    n,he,ae,pairs=np.moveaxis(stats,-1,0)
    ph=np.clip((he+1)/(2*n+2),.001,.499)
    pa=np.clip((ae+1)/(max_a*n+2),.001,.499)
    raw=np.divide(ae,max_a*n,out=np.zeros_like(ae),where=n>0)
    pair=np.divide(pairs,comb(max_a,2)*n,out=np.zeros_like(ae),where=n>0)
    denom=raw*(1-raw)
    rho=np.clip(np.divide(pair-raw**2,denom,out=np.zeros_like(raw),where=denom>1e-12),0,1)
    return ph,pa,rho

def trace_stats(panel,eta_h,eta_a):
    stats=[private_stats(panel['h'],panel['a'])]
    for s in ('H','A'):
        up,private,post=reports(Topology('exposed_vote',s,5),panel,eta_h,eta_a)
        discord=private!=up[...,None]
        stats.append(np.stack([discord.sum((-2,-1)),(discord&(post==up[...,None])).sum((-2,-1))],axis=-1))
    return np.concatenate(stats,axis=-1)

def fit_trace(stats,ignore_echo=False):
    ph,pa,rho=fit_private(stats[...,:4])
    eh=(stats[...,5]+.5)/(stats[...,4]+1);ea=(stats[...,7]+.5)/(stats[...,6]+1)
    if ignore_echo: eh=np.zeros_like(eh);ea=np.zeros_like(ea)
    return ph,pa,rho,eh,ea
