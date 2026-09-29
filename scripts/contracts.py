"""Minimal executable contracts, not an organization simulator or security sandbox."""
from dataclasses import dataclass, replace
from typing import FrozenSet

@dataclass(frozen=True)
class Event:
    id: str
    time: int
    actor: str
    kind: str
    inputs: FrozenSet[str]
    outputs: FrozenSet[str]
    recipients: FrozenSet[str]
    version: int

@dataclass(frozen=True)
class Patch:
    id: str
    target: str
    expected_version: int
    actor: str
    evidence: FrozenSet[str]
    reversible: bool
    cost: float

@dataclass(frozen=True)
class State:
    version: int
    permissions: FrozenSet[tuple]
    budget: float

def visible_before(events, actor, time):
    return frozenset(x for e in events if e.time < time and actor in e.recipients for x in e.outputs)

def validate_trace(events, initial):
    seen=set(); produced=set(initial)
    for e in sorted(events,key=lambda x:x.time):
        assert e.id not in seen, 'duplicate event'
        assert e.outputs.isdisjoint(produced), 'artifact identifiers must be versioned'
        assert e.inputs <= (visible_before(events,e.actor,e.time) | frozenset(initial)), 'future or hidden input'
        seen.add(e.id); produced.update(e.outputs)
    return True

def apply_patch(state, patch, observed, protected=frozenset({'external_criterion'})):
    if patch.target in protected:raise ValueError('protected criterion')
    if patch.expected_version!=state.version:raise ValueError('stale version')
    if (patch.actor,patch.target) not in state.permissions:raise ValueError('unauthorized target')
    if not patch.evidence <= observed:raise ValueError('unavailable evidence')
    if patch.cost<0 or patch.cost>state.budget:raise ValueError('resource violation')
    return replace(state,version=state.version+1,budget=state.budget-patch.cost)

def boundary_checks():
    """Constructed authority example, not inferred permissions of a real repository."""
    state=State(7,frozenset({('release_owner','routing'),('review_board','audit_rule')}),2)
    evidence=frozenset({'coverage_report'})
    routing=Patch('routing_patch','routing',7,'release_owner',evidence,True,1)
    after=apply_patch(state,routing,evidence)
    assert after.version==8 and after.budget==1
    audit=replace(routing,id='audit_patch',target='audit_rule')
    try:apply_patch(state,audit,evidence)
    except ValueError as error:assert str(error)=='unauthorized target'
    else:raise AssertionError('audit change admitted without target authority')
    authorized=apply_patch(state,replace(audit,actor='review_board'),evidence)
    assert authorized.version==8 and authorized.budget==1
    return {'routing_authorized':True,'audit_rejected_for_release_owner':True,
            'audit_authorized_for_review_board':True,'retained_budget':1,
            'permissions_source':'constructed illustration'}

def checks():
    a=Event('e1',1,'human','commit',frozenset({'task'}),frozenset({'h1'}),frozenset({'human'}),0)
    b=Event('e2',2,'agent','commit',frozenset({'task'}),frozenset({'a1'}),frozenset({'agent'}),0)
    c=Event('e3',3,'human','reveal',frozenset({'h1'}),frozenset({'h1_shared'}),frozenset({'agent','human'}),0)
    assert validate_trace([a,b,c],{'task'})
    try:validate_trace([a,replace(b,inputs=frozenset({'h1'}))],{'task'})
    except AssertionError:pass
    else:raise AssertionError('leakage not rejected')
    s=State(0,frozenset({('chair','workflow'),('chair','audit_rule')}),2)
    p=Patch('p1','audit_rule',0,'chair',frozenset({'finding'}),True,1)
    assert apply_patch(s,p,frozenset({'finding'})).version==1
    bad=[replace(p,target='external_criterion'),replace(p,expected_version=1),replace(p,actor='agent'),replace(p,evidence=frozenset({'hidden'})),replace(p,cost=3)]
    for q in bad:
        try:apply_patch(s,q,frozenset({'finding'}))
        except ValueError:pass
        else:raise AssertionError('invalid patch admitted')
    return {'valid_trace':True,'hidden_input_rejected':True,'valid_patch':True,'invalid_patch_cases_rejected':len(bad),'boundary_example':boundary_checks()}

if __name__=='__main__':print(checks())
