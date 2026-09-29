"""Fetch a small public PR snapshot (--fetch), or map the frozen snapshot offline.
No credentials, private content, comment bodies, or email addresses are collected.
Selection: first three merged PRs before 2026-09-30 in the 100 closed PRs
returned in descending creation order by the stated endpoint on retrieval.
This bounded convenience sample supports schema feasibility, not inference.
"""
from pathlib import Path
import json, csv, sys, urllib.request, hashlib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'examples/public_trace'
DATA.mkdir(parents=True, exist_ok=True)
API = 'https://api.github.com'
REPO = '/repos/pallets/flask'
SELECT = REPO + '/pulls?state=closed&sort=created&direction=desc&per_page=100'

def pseudonymize(snapshot):
    """Remove direct account identifiers; public PR provenance remains linkable."""
    ids=set()
    for p in snapshot['pull_requests']:
        ids.update([p['opener']['id'],p['merged_by']['id']])
        ids.update(r['account_id'] for r in p['reviews'])
    if all(isinstance(i,str) and i.startswith('actor-') for i in ids):return snapshot
    mapping={i:f'actor-{j+1:03d}' for j,i in enumerate(sorted(ids,key=str))}
    for p in snapshot['pull_requests']:
        for field in ['opener','merged_by']:p[field]['id']=mapping[p[field]['id']]
        for r in p['reviews']:r['account_id']=mapping[r['account_id']]
    snapshot['privacy']='Snapshot-scoped actor pseudonyms; no account IDs or usernames. Public PR provenance remains linkable.'
    return snapshot

def get(path):
    req = urllib.request.Request(API + path, headers={'User-Agent':'ROI-schema-feasibility', 'Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(req, timeout=45) as response:
        raw = response.read()
        return json.loads(raw), {'url':API+path, 'retrieved_utc':datetime.now(timezone.utc).isoformat(),
            'response_sha256':hashlib.sha256(raw).hexdigest(), 'next_page': 'rel="next"' in response.headers.get('Link','')}

def fetch():
    candidates, receipt = get(SELECT)
    selected = [p for p in candidates if p['merged_at'] and p['merged_at'] < '2026-09-30'][:3]
    assert len(selected) == 3
    out = {'repository':'pallets/flask','selection_receipt':receipt,
        'selection_rule':__doc__.split('Selection: ')[1].split('This bounded')[0].strip(),
        'selection_candidates':[{'number':p['number'],'created_at':p['created_at'],'merged_at':p['merged_at']} for p in candidates], 'pull_requests':[]}
    for selected_pr in selected:
        number = selected_pr['number']
        p, pr_receipt = get(REPO + f'/pulls/{number}')
        head = p['head']['sha']
        paths = [REPO+f'/pulls/{number}/commits?per_page=100', REPO+f'/pulls/{number}/reviews?per_page=100', REPO+f'/commits/{head}/check-runs?per_page=100']
        with ThreadPoolExecutor(max_workers=3) as pool:
            (commits, cr),(reviews, rr),(checks, kr) = list(pool.map(get, paths))
        assert not any(r['next_page'] for r in [cr,rr,kr]), 'paginate before using this sample'
        out['pull_requests'].append({
            'number':number,'url':p['html_url'],'created_at':p['created_at'],'merged_at':p['merged_at'],
            'head_sha':head,'merge_sha':p['merge_commit_sha'],
            'opener':{'id':p['user']['id'],'type':p['user']['type']},
            'merged_by':{'id':p['merged_by']['id'],'type':p['merged_by']['type']},
            'commits':[{'sha':c['sha'],'parent_shas':[x['sha'] for x in c['parents']], 'committer_timestamp':c['commit']['committer']['date']} for c in commits],
            'reviews':[{'id':r['id'],'account_id':r['user']['id'],'account_type':r['user']['type'],'state':r['state'], 'submitted_at':r.get('submitted_at'),'commit_id':r['commit_id']} for r in reviews],
            'check_runs':[{'id':c['id'],'name':c['name'],'head_sha':c['head_sha'],'status':c['status'],'conclusion':c['conclusion'], 'started_at':c['started_at'],'completed_at':c['completed_at'],'app_id':c['app']['id'],'url':c['html_url']} for c in checks['check_runs']],
            'check_total_count':checks['total_count'],'receipts':[pr_receipt,cr,rr,kr]})
    (DATA/'snapshot.json').write_text(json.dumps(pseudonymize(out),indent=2),encoding='utf-8')

def map_snapshot():
    snapshot=pseudonymize(json.loads((DATA/'snapshot.json').read_text(encoding='utf-8')))
    traces=[]; rows=[]
    for p in snapshot['pull_requests']:
        events=[{'source_id':f"pr:{p['number']}:open",'kind':'open','timestamp':p['created_at'],'account_id':p['opener']['id']},
                {'source_id':f"pr:{p['number']}:merge",'kind':'merge','timestamp':p['merged_at'],'account_id':p['merged_by']['id'],'output_sha':p['merge_sha']}]
        for r in p['reviews']:
            events.append({'source_id':f"review:{r['id']}",'kind':'review','timestamp':r['submitted_at'],'account_id':r['account_id'],'reported_commit_sha':r['commit_id'],'reported_state':r['state']})
        for c in p['check_runs']:
            events.append({'source_id':f"check:{c['id']}",'kind':'check','timestamp':c['completed_at'] or c['started_at'],'reported_commit_sha':c['head_sha'],'reported_conclusion':c['conclusion']})
        for e in events:
            e.update({'mapping_status':'adapter_mapping','source_status':'public_record','visible_inputs':None,'recipients':None,'organization_version':None,'effort_cost':None})
        events.sort(key=lambda e:(e['timestamp'] or '',e['source_id']))
        assert len({e['source_id'] for e in events})==len(events)
        assert all(e['visible_inputs'] is None for e in events)
        traces.append({'pr_number':p['number'],'events':events,'commit_artifacts':p['commits'],
            'unknown':['private_exposure','historical_permissions','historical_workflow_rules','active_effort','AI_usage','output_correctness','counterfactual'],
            'conformance_status':'incomplete; actor-visibility and authority conformance cannot be certified'})
        rows.append({'pr':p['number'],'commit_artifacts':len(p['commits']),'review_records':len(p['reviews']),'head_check_records':len(p['check_runs']),'mapped_events':len(events),'merge_recorded':1})
    (DATA/'mapped_traces.json').write_text(json.dumps(traces,indent=2),encoding='utf-8')
    with (ROOT/'results/public_trace_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'records':rows,'unique_event_ids':True,'unknown_fields_preserved':True,
        'scope':'Current public head-SHA check records; missing records do not imply no historical CI.',
        'snapshot_sha256':hashlib.sha256((DATA/'snapshot.json').read_bytes()).hexdigest()}
    (ROOT/'qa/public_trace_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    if '--fetch' in sys.argv: fetch()
    map_snapshot()
