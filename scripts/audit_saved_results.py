"""Audit saved results and their manuscript displays without executing experiments.

Only qa/ (public repo) or review/ (local review workspace) reports are written.
No simulator is imported and no random numbers are generated.
"""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, re
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=(ROOT/'study2').exists()
CORE=ROOT/('results' if PUBLIC else 'reproducibility/core_results')
ACTOR=ROOT/('study2' if PUBLIC else 'reproducibility/actor_review')
OUT=ROOT/('qa' if PUBLIC else 'review'); OUT.mkdir(exist_ok=True)
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def text(p):return p.read_text(encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def select(rr,**keys):
    found=[r for r in rr if all(str(r[k])==str(v) for k,v in keys.items())]
    assert len(found)==1,(keys,len(found));return found[0]
def stats(a):
    a=np.asarray(a,dtype=float);return float(a.mean()),float(1.96*a.std(ddof=1)/np.sqrt(len(a)))
COUNTS=defaultdict(int); MAXIMUM=defaultdict(float)
def close(a,b,category,tol=1e-12):
    delta=abs(float(a)-float(b));assert delta<tol,(category,a,b,delta)
    COUNTS[category]+=1;MAXIMUM[category]=max(MAXIMUM[category],delta)
def grouped(rr,keys):
    d=defaultdict(list)
    for r in rr:d[tuple(r[k] for k in keys)].append(r)
    return d

# Study 1 full summaries, including every sensitivity and exploratory cell.
datasets={}
for stem in ('learning','learning_sensitivity','learning_controls'):
    raw=rows(CORE/f'{stem}_runs.csv');summ=rows(CORE/f'{stem}_summary.csv');datasets[stem]=(raw,summ)
    keys=('environment','memory','arm','meta_share','trials','pool','program_memory')
    groups=grouped(raw,keys)
    for r in summ:
        g=groups[tuple(r[k] for k in keys)];assert len(g)==int(r['n'])
        for metric in ('net','harm','regret','expense','meta_expense','late_net'):
            for col,v in zip((metric,metric+'_ci95'),stats([x[metric] for x in g])):close(r[col],v,stem)
core=datasets['learning'][1]
contrasts=rows(CORE/'learning_contrasts.csv')
def trajectory(rr,env,mem,arm,**extra):
    return {int(r['replicate']):float(r['net']) for r in rr if r['environment']==env and r['memory']==mem and r['arm']==arm and all(r[k]==str(v) for k,v in extra.items())}
for r in contrasts:
    raw=datasets['learning' if r['study']=='core' else 'learning_controls'][0]
    e,m,c=r['environment'],r['memory'],r['comparison']
    if c.endswith(' - Reset'):
        arm=c.removesuffix(' - Reset');a=trajectory(raw,e,m,arm);b=trajectory(raw,e,'Reset',arm)
    else:
        a=trajectory(raw,e,m,'Repeated discovery');arm=c.removeprefix('Repeated - ')
        if arm=='uniform fixed mixture':
            sources=[trajectory(raw,e,m,x) for x in ('Biased','Balanced','Neyman')]
            b={i:sum(v[i] for v in sources)/3 for i in a}
        else:b=trajectory(raw,e,m,arm)
    assert a.keys()==b.keys() and len(a)==int(r['n'])
    for col,v in zip(('delta','ci95'),stats([a[i]-b[i] for i in a])):close(r[col],v,'study1_contrasts')

# Supplementary audit models: marginal and paired summaries from stored trajectories.
for stem,keys,ci,diff,pair in [
 ('longitudinal',('scenario','policy'),'ci95_halfwidth','difference_vs_dense','paired_ci95_halfwidth'),
 ('robustness',('scenario','window','policy'),'ci95_halfwidth','difference_vs_fixed010','paired_ci95_halfwidth'),
 ('audit_extensions',('prior','threshold','candidate_generation_cost','scenario','policy'),'ci95','difference_vs_fixed010','paired_ci95')]:
    raw=rows(CORE/f'{stem}_runs.csv');summ=rows(CORE/f'{stem}_summary.csv');groups=grouped(raw,keys)
    for r in summ:
        group=groups[tuple(r[k] for k in keys)];vals={int(x['seed']):float(x['value']) for x in group}
        refkey=tuple(r[k] if k!='policy' else ('Fixed dense' if stem=='longitudinal' else 'Fixed 0.10') for k in keys)
        ref={int(x['seed']):float(x['value']) for x in groups[refkey]}
        for col,v in zip(('mean',ci),stats(list(vals.values()))):close(r[col],v,stem)
        for col,v in zip((diff,pair),stats([vals[i]-ref[i] for i in vals])):close(r[col],v,stem)

# Study 2: full and late summaries, paired contrasts, and exact cost identities.
data=np.load(ACTOR/'results/learning_raw.npz')
meta=json.loads(text(ACTOR/'results/learning_metadata.json'))
actor=rows(ACTOR/'results/learning_summary.csv')
apairs=rows(ACTOR/'results/paired_contrasts.csv')
for r in actor:
    wi=meta['worlds'].index(r['world']);pi=meta['policies'].index([r['policy'],r['memory']])
    for metric in ('error','net','regret','hit','acquisition_cost','production_cost'):
        for suffix,sl in [('',slice(None)),('_late',slice(-16,None))]:
            vals=data[metric][wi,pi,:,sl].mean(-1)
            for col,v in zip((metric+suffix,metric+suffix+'_ci95'),stats(vals)):close(r[col],v,'study2_summaries')
for r in apairs:
    wi=meta['worlds'].index(r['world']);a=meta['policies'].index([r['policy_a'],r['memory_a']]);b=meta['policies'].index([r['policy_b'],r['memory_b']])
    for metric in ('error','net','regret'):
        for col,v in zip((metric+'_difference',metric+'_ci95'),stats((data[metric][wi,a]-data[metric][wi,b]).mean(-1))):close(r[col],v,'study2_contrasts')
assert np.max(data['acquisition_cost'])<=240+1e-10
assert np.max(data['acquisition_cost'].reshape(6,14,128,6,8).sum(-1))<=1920+1e-10
expected=1-3*data['error']-.02*data['production_cost']-(data['acquisition_cost']+.5*data['changed'])/4096
close(np.max(np.abs(expected-data['net'])),0,'study2_accounting')
assert data['error'].shape==(6,14,128,48)

# Direct audit of original main-paper numeric tables (including all unmodified rows).
def table_lines(name):return [l for l in text(ROOT/'sections'/name).splitlines() if ' & ' in l]
def table_numbers(s):return [float(x) for x in re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?',s)]
def rounded_matches(actual,expected,digits,category):
    assert len(actual)==len(expected),(category,actual,expected)
    for a,b in zip(actual,expected):close(a,round(float(b),digits),category,1e-10)
for line in table_lines('learning_base_rows.tex'):
    e,m,*cells=line.replace(r'\\','').split(' & ')
    vals=[float(select(core,environment=e,memory=m,arm=arm)['net']) for arm in ('Biased','Balanced','Neyman','Successive rejects','Discover once','Repeated discovery')]
    rounded_matches(list(map(float,cells)),vals+[sum(vals[:3])/3],4,'table6')
for line in table_lines('learning_grid_rows.tex'):
    cells=line.split(' & ');f=float(cells[0]);q=int(cells[1]);expected=[]
    for e,m in [('Stationary harm','Cumulative'),('Workflow reversal','Window 8')]:
        raw=datasets['learning_sensitivity'][0];extra=dict(meta_share=f,trials=q,pool=True,program_memory=True)
        a=trajectory(raw,e,m,'Repeated discovery',**extra);b=trajectory(raw,e,m,'Discover once',**extra)
        expected.extend(stats([a[i]-b[i] for i in a]))
    rounded_matches(table_numbers(' & '.join(cells[2:])),expected,4,'table7')
conditional=rows(CORE/'learning_conditional.csv')
for line in table_lines('learning_program_rows.tex'):
    short,rd,*cells=line.replace(r'\\','').split(' & ')
    e,m={'Stationary':('Stationary harm','Cumulative'),'Reversal':('Workflow reversal','Window 8'),'Uniform gain':('Uniform gain','Cumulative')}[short]
    rr=[select(conditional,environment=e,memory=m,arm='Repeated discovery',review_round=rd,program=p) for p in ('Biased','Balanced','Neyman')]
    rounded_matches([float(x) for x in cells[:3]],[100*float(r['share']) for r in rr],1,'table8')
    close(cells[3],rr[1]['n'],'table8');close(cells[4],round(float(rr[1]['net']),4),'table8');close(cells[5],round(100*float(rr[1]['harm']),2),'table8')
audit=rows(CORE/'audit_extensions_summary.csv')
for line in table_lines('audit_rows.tex'):
    prior,threshold,*cells=line.replace(r'\\','').split(' & ')
    vals=[select(audit,prior=prior,threshold=threshold,candidate_generation_cost=0,scenario=e,policy=p)['mean'] for e in ('Stationary','Regime shifts') for p in ('Fixed 0.10','Adaptive audit','EVSI')]
    rounded_matches([float(x) for x in cells],vals,4,'table9')
protocol=json.loads(text(ACTOR/'protocol.json'))
assert sha(ACTOR/'protocol.json')==json.loads(text(ACTOR/'qa/pre_run_record.json'))['protocol_sha256']
assert len(rows(ACTOR/'results/phase.csv'))==16200
assert len(rows(ACTOR/'results/fixed_roster.csv'))==25200
assert len(rows(ACTOR/'results/source_asymmetry.csv'))==6450
assert len(meta['topologies'])==58
assert len(conditional)==189 and len(actor)==84 and len(apairs)==30
assert [len(datasets[x][0]) for x in datasets]==[6912,4608,6144]

WORLDS=dict(no_echo='No echo',high_echo='High echo',both_dependences='Both dependences',echo_shift='Echo shift',source_asymmetry='Source asymmetry',strong_humans='Stronger humans')
POLICIES=dict(Full_trace='Paired',Randomized_exposure='Randomized',Ignore_echo='Omit echo',Final_uniform='Final only',**{'exposed_vote:A:k3':'Fixed exposed','blind_vote:A:k3':'Fixed blind'})
expected=[]
for w in WORLDS:
    for p in POLICIES:
        r=select(actor,world=w,policy=p,memory='none' if p.startswith(('exposed_','blind_')) else 'cumulative');expected.append(r)
lines=[l for l in table_lines('actor_results.tex') if r'$' in l and r'\pm' in l or re.match(r'(?:Fixed| & Fixed|.* & Fixed)',l)]
# Data rows always end with a TeX line break and have exactly four ampersands.
lines=[l for l in text(ROOT/'sections/actor_results.tex').splitlines() if l.count(' & ')==4 and any((' & '+v+' & ') in l for v in POLICIES.values())]
assert len(lines)==36
for line,r in zip(lines,expected):
    cells=line.split(' & ')
    assert cells[1]==POLICIES[r['policy']]
    er=table_numbers(cells[2]);nv=table_numbers(cells[3]);hit=table_numbers(cells[4])
    rounded_matches(er,[100*float(r['error'])]+([] if r['memory']=='none' else [100*float(r['error_ci95'])]),3,'table12')
    rounded_matches(nv,[float(r['net'])]+([] if r['memory']=='none' else [float(r['net_ci95'])]),5,'table12')
    rounded_matches(hit,[100*float(r['hit'])],2,'table12')

# Claim ledger: full-precision source values, display rounding, and scope.
claims=[]
def claim(label,value,display,source,scope='',digits=None):
    if digits is not None:close(float(display),round(float(value),digits),'prose_claims',1e-10)
    claims.append(dict(claim=label,source_value=value,manuscript_display=display,source=source,scope=scope))
def cs(e,m,a,metric='net'):return float(select(core,environment=e,memory=m,arm=a)[metric])
def ca(w,p='Full_trace',m='cumulative',metric='net'):return float(select(actor,world=w,policy=p,memory=m)[metric])
for e,m,a,metric,display in [
 ('Stationary harm','Reset','Balanced','net','.45224'),('Stationary harm','Cumulative','Balanced','net','.48007'),
 ('Stationary harm','Cumulative','Balanced','late_net','.48163'),('Stationary harm','Reset','Balanced','regret','.02939'),
 ('Stationary harm','Cumulative','Balanced','regret','.00156'),('Stationary harm','Reset','Balanced','expense','.01837'),
 ('Stationary harm','Cumulative','Discover once','net','.47928'),('Stationary harm','Cumulative','Repeated discovery','net','.48000'),
 ('Stationary harm','Cumulative','Successive rejects','net','.48059'),('Workflow reversal','Cumulative','Balanced','net','.48026'),
 ('Workflow reversal','Window 8','Balanced','net','.57583'),('Workflow reversal','Reset','Balanced','net','.58280'),
 ('Workflow reversal','Window 8','Balanced','late_net','.72163'),('Workflow reversal','Reset','Balanced','late_net','.71237'),
 ('Workflow reversal','Cumulative','Balanced','late_net','.48280')]:
    claim(f'{e}/{m}/{a}/{metric}',cs(e,m,a,metric),display,'results/learning_summary.csv',digits=5)
for study,e,m,c,displays in [
 ('core','Stationary harm','Cumulative','Balanced - Reset',('.02783','.02659','.02907')),
 ('core','Stationary harm','Reset','Repeated - Balanced',('-.01702','-.01964','-.01440')),
 ('core','Stationary harm','Cumulative','Repeated - Balanced',('-.00007','-.00100','.00085')),
 ('core','Workflow reversal','Window 8','Repeated - Discover once',('.00975','.00709','.01242')),
 ('core','Workflow reversal','Window 8','Repeated - Balanced',('.00623',)),
 ('exploratory control','Workflow reversal','Window 8','Repeated - Balanced',('.00607','.00390','.00825')),
 ('exploratory control','Workflow reversal','Window 8','Repeated - Trial-matched Balanced',('.00191','-.00016','.00398')),
 ('exploratory control','Workflow reversal 20','Window 8','Repeated - Trial-matched Balanced',('-.00063',)),
 ('exploratory control','Workflow reversal 28','Window 8','Repeated - Trial-matched Balanced',('-.00057',)),
 ('exploratory control','Workflow reversal 20','Window 8','Repeated - Balanced',('.00031',)),
 ('exploratory control','Workflow reversal 28','Window 8','Repeated - Balanced',('.00004',))]:
    r=select(contrasts,study=study,environment=e,memory=m,comparison=c);v=float(r['delta']);h=float(r['ci95'])
    for value,display,part in zip((v,v-h,v+h),displays,('mean','lower','upper')):claim('/'.join((study,e,m,c,part)),value,display,'results/learning_contrasts.csv',digits=5)
for m,display,gain in [('Reset','.43426','.00096'),('Cumulative','.47657','.00343')]:
    mix=sum(cs('Stationary harm',m,a) for a in ('Biased','Balanced','Neyman'))/3
    claim('Uniform fixed mixture / '+m,mix,display,'results/learning_summary.csv',digits=5)
    claim('Repeated minus mixture / '+m,cs('Stationary harm',m,'Repeated discovery')-mix,gain,'results/learning_summary.csv',digits=5)
sens=datasets['learning_sensitivity'][1]
for e,m,pool,display in [('Stationary harm','Cumulative',True,'.48043'),('Stationary harm','Cumulative',False,'.47751'),('Workflow reversal','Window 8',True,'.58214'),('Workflow reversal','Window 8',False,'.57392')]:
    r=select(sens,environment=e,memory=m,arm='Repeated discovery',meta_share=.2,trials=1,pool=pool,program_memory=True)
    claim('Pooling / '+e+' / '+str(pool),float(r['net']),display,'results/learning_sensitivity_summary.csv',digits=5)
for w,p,m,metric,display,digits,factor in [
 ('high_echo','Full_trace','cumulative','error','1.773',3,100),('high_echo','Full_trace','cumulative','net','.86077',5,1),
 ('high_echo','Ignore_echo','cumulative','error','9.825',3,100),('high_echo','Ignore_echo','cumulative','net','.62323',5,1),
 ('high_echo','Randomized_exposure','cumulative','error','1.782',3,100),('high_echo','Randomized_exposure','cumulative','net','.86078',5,1),
 ('high_echo','blind_vote:A:k3','none','net','.87140',5,1),('high_echo','blind_vote:A:k3','none','error','3.520',3,100),
 ('strong_humans','Full_trace','cumulative','net','.77306',5,1),('strong_humans','blind_vote:A:k3','none','net','.60740',5,1)]:
    claim('/'.join((w,p,m,metric)),factor*ca(w,p,m,metric),display,'study2/results/learning_summary.csv',digits=digits)
for p,nets,late in [('Full_trace',['.85798','.86255','.86274'],[None,'1.7704','1.7704']),('Randomized_exposure',['.85081','.85206','.85659'],['2.3212','1.8372','1.7704'])]:
    for m,v,er in zip(('reset','cumulative','window8'),nets,late):
        claim('Echo shift / '+p+' / '+m,ca('echo_shift',p,m),v,'study2/results/learning_summary.csv',digits=5)
        if er:claim('Echo shift late error / '+p+' / '+m,100*ca('echo_shift',p,m,'error_late'),er,'study2/results/learning_summary.csv',digits=4)
pair=select(apairs,world='high_echo',policy_a='Full_trace',memory_a='cumulative',policy_b='Ignore_echo',memory_b='cumulative')
claim('High echo paired-minus-omit net',float(pair['net_difference']),'.23755','study2/results/paired_contrasts.csv',digits=5)
claim('High echo paired-minus-omit halfwidth',float(pair['net_ci95']),'.00004','study2/results/paired_contrasts.csv',digits=5)
curve=rows(ACTOR/'results/illustrative_curve.csv')
for eta,metric,display,digits in [(0.,'agreement','69.06',2),(.8,'agreement','91.20',2),(0.,'error','3.52',2),(.8,'error','9.64',2),(0.,'wrong_up_approval','10.09',2),(.8,'wrong_up_approval','59.62',2)]:
    r=next(r for r in curve if r['topology']=='exposed_vote:A:k3' and abs(float(r['eta'])-eta)<1e-12)
    claim(f'Copying={eta}/{metric}',100*float(r[metric]),display,'study2/results/illustrative_curve.csv',digits=digits)
# Equal accuracy need not imply equal deployment loss.
for p,m,metric,display,digits,factor in [
 ('exposed_vote:A:k3','none','net','.87440',5,1),
 ('blind_vote:A:k3','none','net','.87140',5,1),
 ('Randomized_exposure','cumulative','hit','56.67',2,100),
 ('Randomized_exposure','cumulative','error','1.7722',4,100),
 ('Full_trace','cumulative','error','1.7722',4,100)]:
    claim('No echo / '+p+' / '+metric,factor*ca('no_echo',p,m,metric),display,'study2/results/learning_summary.csv',digits=digits)
close(ca('no_echo','Randomized_exposure','cumulative','error'),ca('no_echo','Full_trace','cumulative','error'),'no_echo_equal_error')
assert ca('no_echo','Randomized_exposure','cumulative','production_cost')>ca('no_echo','Full_trace','cumulative','production_cost')
close(ca('no_echo','exposed_vote:A:k3','none')-ca('no_echo','blind_vote:A:k3','none'),.02*(3*.05)+.5/(48*4096),'no_echo_blind_cost')

# Abstract values are derived independently at their coarser precision.
for display,v,digits in [('0.452',cs('Stationary harm','Reset','Balanced'),3),('0.480',cs('Stationary harm','Cumulative','Balanced'),3),
 ('0.006',float(select(contrasts,study='exploratory control',environment='Workflow reversal',memory='Window 8',comparison='Repeated - Balanced')['delta']),3),
 ('0.002',float(select(contrasts,study='exploratory control',environment='Workflow reversal',memory='Window 8',comparison='Repeated - Trial-matched Balanced')['delta']),3)]:
    assert display in text(ROOT/'main.tex').split(r'\begin{abstract}',1)[1].split(r'\end{abstract}',1)[0]
    claim('Abstract '+display,v,display,'same source as corresponding full-precision body claim',digits=digits)

cost_comparisons=[]
for w in WORLDS:
    a=select(actor,world=w,policy='Full_trace',memory='cumulative');b=select(actor,world=w,policy='blind_vote:A:k3',memory='none')
    deployment_gain=3*(float(b['error'])-float(a['error']))+.02*(float(b['production_cost'])-float(a['production_cost']))
    acquisition=float(a['acquisition_cost'])/4096
    net_delta=float(a['net'])-float(b['net'])
    cost_comparisons.append(dict(world=w,error_reduction=float(b['error'])-float(a['error']),deployment_gain=deployment_gain,acquisition_per_task=acquisition,net_difference=net_delta,switch_difference_per_task=deployment_gain-acquisition-net_delta))
assert all(x['error_reduction']>0 for x in cost_comparisons)
assert sum(x['net_difference']<0 for x in cost_comparisons)==5
assert sum(x['deployment_gain']<x['acquisition_per_task'] for x in cost_comparisons)==5
claim('Paired cumulative below fixed blind net',5,'5 of 6','study2/results/learning_summary.csv','Full horizon, paired observation, cumulative memory, fixed blind A+3A comparator; not every policy or memory')
claim('Paired acquisition amortization',233.6/4096,'.05703','study2/protocol.json; study2/results/learning_summary.csv',digits=5)
strict_best=[];near=[]
for (e,m),rr in grouped(core,('environment','memory')).items():
    best=max(float(r['net']) for r in rr);v=cs(e,m,'Successive rejects')
    if v==best:strict_best.append([e,m])
    elif e=='Uniform gain':near.append(best-v)
assert len(strict_best)==5 and len(near)==2 and max(near)<.00004
claim('SR highest unrounded mean',5,'five of nine','results/learning_summary.csv','Descriptive comparison; two further Uniform gain retained-memory cells are within .00004, not exact ties')

longitudinal=rows(CORE/'longitudinal_summary.csv')
for policy,display in [('Fixed sparse','.86089'),('Fixed 0.05','.86395'),('Fixed 0.10','.85841'),('No audit','.87976')]:
    claim('Appendix D stationary / '+policy,float(select(longitudinal,scenario='Stationary',policy=policy)['mean']),display,'results/longitudinal_summary.csv',digits=5)
audit_pair=select(rows(CORE/'paired_contrasts.csv'),scenario='Regime shifts',comparator='Fixed 0.10')
for key,display in [('difference','.00857'),('ci95_low','.00676'),('ci95_high','.01038')]:
    claim('Appendix D adaptive minus fixed .10 / '+key,float(audit_pair[key]),display,'results/paired_contrasts.csv',digits=5)
for program,display in [('Biased','38.3'),('Balanced','31.2'),('Neyman','30.5')]:
    r=select(conditional,environment='Stationary harm',memory='Cumulative',arm='Repeated discovery',review_round=41,program=program)
    claim('Last stationary retained program / '+program,100*float(r['share']),display,'results/learning_conditional.csv',digits=1)
    if program=='Balanced':
        assert int(r['n'])==40 and float(r['harm'])==0
        claim('Last stationary Balanced conditional net',float(r['net']),'.4819','results/learning_conditional.csv',digits=4)
public_rows=rows(CORE/'public_trace_summary.csv')
assert [sum(int(r[k]) for r in public_rows) for k in ('commit_artifacts','review_records','head_check_records','mapped_events')]==[4,8,42,56]
for row in public_rows:
    expected=' & '.join(row[k] for k in ('commit_artifacts','review_records','head_check_records','mapped_events'))
    assert expected in text(ROOT/'sections/appendices.tex')
    COUNTS['table5']+=4
phase=rows(ACTOR/'results/phase.csv')
phase_filter=dict(p_h=.2,p_a=.1,human_agent_price_ratio=4,blind_overhead=.05,budget=3,regime='hard')
display_grid=rows(ROOT/'figures/fig4_phase.csv')
assert len(display_grid)==25
for r in display_grid:
    source=next(x for x in phase if all(x[k]==r[k] for k in r))
    assert all(source[k]==str(v) or (k!='regime' and float(source[k])==float(v)) for k,v in phase_filter.items())
assert next(r['winner'] for r in display_grid if float(r['rho'])==0 and float(r['eta'])==.5)=='blind_vote:A:k6'
assert next(r['winner'] for r in display_grid if float(r['rho'])==1 and float(r['eta'])==.5)=='blind_gate:H:k1'
for world in protocol['learning']['worlds']:
    expected=' & '.join(f"{float(world[k]):g}".removeprefix('0') if float(world[k])!=0 else '0' for k in ('p_h','p_a','rho','eta_h','eta_a'))
    assert expected in text(ROOT/'sections/actor_review.tex'),expected
    COUNTS['table11']+=5

# Non-result numerals are protocol settings, mathematical constants, or provenance.
# Inventory every numeral-bearing source line rather than silently excluding prose.
inventory=[]
files=[]
def visit(p):
    if p in files:return
    files.append(p)
    for child in re.findall(r'\\(?:input|tableinput)\{([^}]+)\}',text(p)):
        q=ROOT/child
        if not q.suffix:q=q.with_suffix('.tex')
        assert q.is_file(),q
        visit(q)
visit(ROOT/'main.tex')
for p in files:
    for n,line in enumerate(text(p).splitlines(),1):
        if not re.search(r'\d',line) or line.startswith('%'):continue
        name=p.name
        if name.endswith('_rows.tex') or name=='actor_results.tex':kind='numeric table checked against identified CSV rows';sources=['results/learning_*.csv','results/audit_extensions_summary.csv','study2/results/learning_summary.csv']
        elif name=='supplementary_tables.tex':kind='supplementary table or setting checked against saved inputs';sources=['results/learning_summary.csv','results/learning_contrasts.csv','results/learning_conditional.csv','results/robustness_summary.csv','results/audit_extensions_summary.csv','study2/results/learning_summary.csv','study2/results/paired_contrasts.csv']
        elif name=='study.tex':kind='Study 1 settings or result claim';sources=['examples/learning_protocol.json','examples/learning_controls.json','results/learning_*.csv']
        elif name in ('study2.tex','actor_review.tex'):kind='Study 2 settings, result claim, analytical law, or validation record';sources=['study2/protocol.json','study2/results/','study2/model.py','study2/qa/validation.json','qa/protocol_provenance.json']
        elif name=='learning_protocol.tex':kind='Study 1 methods or settings';sources=['examples/learning_protocol.json','examples/learning_controls.json','scripts/learning_experiment.py','qa/learning_validation.json']
        elif name=='appendices.tex':kind='public snapshot, analytical illustration, or audit setting/result';sources=['results/public_trace_summary.csv','results/longitudinal_summary.csv','results/paired_contrasts.csv','results/audit_extensions_summary.csv','scripts/experiments.py','scripts/audit_extensions.py','qa/audit_extensions_checks.json']
        elif name=='specification.tex':kind='formal definition or public snapshot count';sources=['scripts/contracts.py','results/public_trace_summary.csv']
        elif name=='introduction.tex':kind='study identifier or explicit bottleneck illustration';sources=['analytic calculation: min(4,5,9)=4; min(8,5,9)=5; min(8,8,9)=8; interaction=3']
        elif name=='main.tex':kind='abstract result, typesetting, or date';sources=['numeric claim ledger; LaTeX settings']
        else:kind='study/table identifier, disclosure, or cross-study synthesis';sources=['numeric claim ledger; protocol provenance; release metadata']
        inventory.append(dict(file=p.relative_to(ROOT).as_posix(),line=n,text=line,kind=kind,sources=sources))

joined='\n'.join(text(p) for p in files)
labels=re.findall(r'\\label\{([^}]+)\}',joined);assert len(labels)==len(set(labels))
refs={x.strip() for group in re.findall(r'\\(?:[cC]ref|ref|eqref)\{([^}]+)\}',joined) for x in group.split(',')};assert refs<=set(labels)
cites={x.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',joined) for x in group.split(',')}
bib=set(re.findall(r'@\w+\s*\{([^,]+),',text(ROOT/'bibliography.bib')));assert cites<=bib and len(cites)==53
assert not re.search(r'actor-level extension|this revision(?! loop)|reader-facing|dated local|template-level mechanism (?:study|experiment)|the main study|Frozen core',joined,re.I)
supp=json.loads(text(ROOT/'supplement/table_inventory.json'))
assert sum(p['rows'] for p in supp['panels'])==879 and len(supp['panels'])==11
for source_list in supp['sources'].values():
    for source in source_list:
        actual=ACTOR/source.removeprefix('study2/') if source.startswith('study2/') else CORE/source.removeprefix('results/')
        assert actual.is_file()

report=dict(passed=True,new_experiments_run=False,method='Saved-trajectory arithmetic and CSV-to-TeX checks only; no simulation module imported',checks=dict(COUNTS),maximum_absolute_differences=dict(MAXIMUM),claim_count=len(claims),numeric_source_lines=len(inventory),study2_cost_comparisons=cost_comparisons,sr_strict_best_cells=strict_best,sr_uniform_gain_gaps=near,citations=len(cites),supplement_rows=879,study2_protocol_sha256=sha(ACTOR/'protocol.json'))
(OUT/'saved_results_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
(OUT/'numeric_claim_ledger.json').write_text(json.dumps(claims,indent=2)+'\n',encoding='utf-8',newline='\n')
(OUT/'numeric_source_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:report[k] for k in ('passed','new_experiments_run','checks','claim_count','numeric_source_lines','citations','supplement_rows')},indent=2))
