"""Freeze grouped v2 splits and run paired native modes using the existing worker."""
import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'experiments/datasets/private/benchmark_v2_full330'
OUT=ROOT/'experiments/results/private/benchmark-v2-baselines-v2-nonthinking'
HERE=Path(__file__).parent
SEED='benchmark-v2-20260924'
MODEL='deepseek-v4-flash-0731'
# Document versions, implementing rules and overlapping institutional frameworks.
POLICY_LINKS=[
 ['79197','79198','80755','80756'],
 ['79260','79261','79274','83198'],
 ['79226','79294'], ['79220','83188'],
 ['79321','79322','79329','80763','83215'],
 ['79300','80762','80782','80796'],
 ['79308','79315','80808','81055','81069'],
 ['79221','79231','80805','80920','81106'],
 ['80806','80821','80940'], ['80815','80966'],
 ['79234','83177'], ['79200','80847'], ['79297','80854'],
 ['80811','80813','79237'],
]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,v):Path(path).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def readl(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def write_l(path,rows):Path(path).write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in rows))
def key(qid):return hashlib.sha256((SEED+qid).encode()).hexdigest()

def policy_groups(rows,links):
    parents={q['question_id']:q['question_id'] for q in rows};seen={}
    def find(x):
        while parents[x]!=x: parents[x]=parents[parents[x]];x=parents[x]
        return x
    def union(a,b):parents[find(a)]=find(b)
    for q in rows:
        tags=['s:'+s for s in q['source_group_seed']]+['f:'+q['question_family_id']]
        for t in tags:
            if t in seen:union(q['question_id'],seen[t])
            else:seen[t]=q['question_id']
    for sources in links:
        ids=[seen['s:'+s] for s in sources if 's:'+s in seen]
        for qid in ids[1:]:union(ids[0],qid)
    groups=collections.defaultdict(list)
    for qid in parents:groups[find(qid)].append(qid)
    return sorted([sorted(g) for g in groups.values()],key=lambda g:g[0])

def split_groups(groups,rows,test_target=90):
    assignment={}; size=0
    for g in sorted(groups,key=lambda g:key(g[0])):
        pilot=any(x.startswith('mb2-') and x[4:].isdigit() and int(x[4:])<=60 for x in g)
        use_test=not pilot and abs(size+len(g)-test_target)<abs(size-test_target)
        for qid in g:assignment[qid]='test' if use_test else 'dev'
        if use_test:size+=len(g)
    return assignment

def runtime_questions(rows,ids):
    byid={q['question_id']:q for q in rows};out=[]
    for qid in ids:
        if qid not in byid:raise ValueError('missing_question')
        if byid[qid].get('split')!='dev':raise ValueError('test_not_allowed_in_development_run')
        out.append({'question_id':qid,'question':byid[qid]['question']})
    return out

def remaining_ids(planned,attempted,records):
    if len(attempted)!=len(set(attempted)) or len(records)!=len({r['question_id'] for r in records}):raise ValueError('duplicate_attempt')
    if set(attempted)!={r['question_id'] for r in records}:raise ValueError('incomplete_prior_attempt')
    if any(r.get('ledger',{}).get('pending_calls') for r in records):raise ValueError('unknown_usage_requires_inspection')
    return [qid for qid in planned if qid not in attempted]

def choose_stratified(dev,n_each=10):
    # Round-robin source groups avoids filling a stratum with one long regulation.
    chosen=[]
    for level in ['Easy','Medium','Hard']:
        buckets=collections.defaultdict(list)
        for q in dev:
            if q['difficulty']==level:buckets[q['policy_group_id']].append(q)
        order=sorted(buckets,key=key)
        for g in buckets:buckets[g].sort(key=lambda q:key(q['question_id']))
        rows=[]
        while order and len(rows)<n_each:
            for g in list(order):
                rows.append(buckets[g].pop(0))
                if not buckets[g]:order.remove(g)
                if len(rows)==n_each:break
        chosen.extend(rows)
    # Interleave difficulty, so a budget stop doesn't execute only one stratum.
    bylevel={d:[q for q in chosen if q['difficulty']==d] for d in ['Easy','Medium','Hard']}
    return [bylevel[d][i]['question_id'] for i in range(n_each) for d in bylevel if i<len(bylevel[d])]

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'frozen.json').exists():raise FileExistsError('already_frozen')
    core=readl(DATA/'benchmark_v2_core300.jsonl');boundary=readl(DATA/'benchmark_v2_boundary30.jsonl')
    groups=policy_groups(core,POLICY_LINKS);assign=split_groups(groups,core)
    gm={qid:f'policy-{i:03d}' for i,g in enumerate(groups,1) for qid in g}
    for q in core:q.update(split=assign[q['question_id']],policy_group_id=gm[q['question_id']])
    dev=[q for q in core if q['split']=='dev'];test=[q for q in core if q['split']=='test']
    for q in boundary:
        ps={assign[x] for x in q['parent_question_ids']}
        q['split']='boundary_'+ps.pop() if len(ps)==1 else 'boundary_cross_split_diagnostic_only'
    selected=choose_stratified(dev); high=selected[:9]
    assert len(selected)==30 and len(high)==9
    for name,rows in [('dev',dev),('test',test),('boundary',boundary)]:write_l(OUT/f'{name}.jsonl',rows)
    audit={'policy_links':POLICY_LINKS,'groups':[{'id':f'policy-{i:03d}','questions':g,'split':assign[g[0]]} for i,g in enumerate(groups,1)],
        'dev_count':len(dev),'test_count':len(test),'dev_difficulty':dict(collections.Counter(q['difficulty'] for q in dev)),
        'test_difficulty':dict(collections.Counter(q['difficulty'] for q in test)),
        'boundary_assignment':dict(collections.Counter(q['split'] for q in boundary)),
        'note':'Current questions were authored/reviewed but test system outputs remain withheld; grouping is conservative manual reconciliation, not a proof of zero semantic overlap.'}
    dump(OUT/'split_audit.json',audit)
    frozen={'seed':SEED,'model':MODEL,'dev30':selected,'high9':high,'all_dev':[q['question_id'] for q in dev],
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in [DATA/'benchmark_v2_core300.jsonl',DATA/'benchmark_v2_boundary30.jsonl',OUT/'dev.jsonl',OUT/'test.jsonl',OUT/'boundary.jsonl']},
        'evaluation_contract':'required-point completeness and evidence support separate; unknown citation pending; invalid labels applied uniformly; no reference or difficulty in runtime',
        'limits':{'rag':{'run_tokens':180000,'question_tokens':35000,'question_calls':2,'query_embedding_calls':4},
                  'low':{'run_tokens':420000,'question_tokens':80000,'question_calls':14,'query_embedding_calls':24},
                  'high':{'run_tokens':300000,'question_tokens':100000,'question_calls':18,'query_embedding_calls':48}},
        'timeouts':{'rag':60,'low':90,'high':90},
        'scope':'Stage1 dev30 RAG/Low and high9; full-dev expansion depends on observed costs/technical status. Overall maximum generation commitment 900k tokens; no ingestion.'}
    dump(OUT/'frozen.json',frozen)
    base=json.loads((ROOT/'experiments/results/private/native-agent-high-compose-fix-v1-full-review/config.json').read_text())
    live=json.loads((OUT/'live_snapshot.json').read_text())
    assert live['index_snapshot']==base['index_snapshot'],'index snapshot changed'
    assert live['runtime_code_hashes']==base['runtime_code_hashes'],'installed runtime changed'
    assert len(base['review']['entries'])==3205
    for arm,reasoning in [('rag',0),('low',1),('high',3)]:
        c=copy.deepcopy(base);c.update(live)
        ids=high if arm=='high' else selected
        c.update(model=MODEL,runtime_model_override=True,reasoning=reasoning,native_patch=None,questions=runtime_questions(dev,ids),question_ids=ids,
            limits=frozen['limits'][arm],question_timeout_seconds=frozen['timeouts'][arm],
            container_ledger=f'/tmp/benchmark-v2-baselines-v1-{arm}/calls.jsonl',
            configuration_changes=['uniform_outbound_sanitization','isolated_runtime_model_override','is_tools_true','cold_json_cache','provider_max_output_1200','physical_request_budget','bounded_wall_time'],
            stage_note='v2 paired native baseline; no compose/evidence algorithm patches; development only')
        for k in ['technical_retry','local_review_summary','historical_native_report']:c.pop(k,None)
        c['sources']={name:(HERE/f'{name}.py').read_text() for name in ['native_outbound','native_agent_guard','native_agent_transport']}
        c['input_sha256']={str(p.relative_to(ROOT)):digest(p) for p in [HERE/'native_agent_worker.py',Path(__file__),OUT/'frozen.json',OUT/'dev.jsonl',*(HERE/f'{n}.py' for n in c['sources'])]}
        d=OUT/arm;d.mkdir(exist_ok=True);dump(d/'config.json',c)
    print(json.dumps({**{k:v for k,v in audit.items() if k not in ['policy_links','groups']},'dev30':selected,'high9':high,'model':MODEL},ensure_ascii=False,indent=2))

def validate_ready_ledger(ready, records):
    expected=records[-1]['ledger'] if records else {'provider_calls':0,'provider_total_tokens':0,'provider_not_sent_calls':0,'pending_calls':[]}
    fields=['provider_calls','provider_total_tokens','provider_not_sent_calls','pending_calls']
    if any(ready.get(k)!=expected.get(k) for k in fields):raise ValueError('remote_ledger_does_not_match_local_history')
    if ready.get('pending_calls'):raise ValueError('unknown_remote_usage')

def validate_result(qid,row):
    if row.get('question_id')!=qid:raise ValueError('completed_question_identity_mismatch')

def receive_line(lines,deadline):
    import queue,time
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('host_deadline')
    try:return lines.get(timeout=remaining)
    except queue.Empty:raise TimeoutError('host_deadline') from None

# The host supervisor verifies the unique initial environment token and private
# process group before terminating the actual container worker, never the service.
STOP_WORKER = r'''
import json,os,signal,sys,time
from pathlib import Path
p=json.loads(sys.stdin.read()); tag=('BENCHMARK_RUN_TOKEN='+p['token']).encode()
found=[]
for path in Path('/proc').iterdir():
 if not path.name.isdigit() or int(path.name)<=1:continue
 try:
  if tag not in (path/'environ').read_bytes().split(b'\0'):continue
  pid=int(path.name)
  if os.getpgid(pid)!=pid:raise RuntimeError('worker_group_not_private')
  os.killpg(pid,signal.SIGKILL);found.append(pid)
 except ProcessLookupError:pass
 except FileNotFoundError:pass
ledger=Path(p['ledger'])
print(json.dumps({'terminated_pids':found,'ledger_events':[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []}))
'''

def run(arm,maximum=None,dry=False):
    if arm == 'high':
        raise ValueError('high_disabled_by_user')
    import queue,threading,time,uuid
    from experiments.evaluation.native_baseline import DOCKER
    from experiments.evaluation.native_agent_guard import CallLedger
    d=OUT/arm;config=json.loads((d/'config.json').read_text())
    for path,want in config['input_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('frozen_input_changed:'+path)
    if dry:
        d=OUT/(arm+'-dry');d.mkdir(exist_ok=True);config['dry_run']=True
        config['container_ledger']=f'/tmp/{OUT.name}-{arm}-dry/calls.jsonl'
    attempts=d/'attempts.jsonl';records_path=d/'records.jsonl'
    records=readl(records_path) if records_path.exists() else []
    attempted=[x['question_id'] for x in readl(attempts)] if attempts.exists() else []
    ids=remaining_ids(config['question_ids'],attempted,records)
    if records and records[-1].get('status')=='host_timeout':raise ValueError('host_timeout_requires_inspection')
    if maximum is not None:ids=ids[:maximum]
    qmap={q['question_id']:q for q in config['questions']}
    def save(p,row):
        with p.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    def read_stdout(stream,lines):
        try:
            for line in stream:lines.put(line)
        finally:lines.put(None)
    index=0
    with (d/'stderr.log').open('a') as err,(d/'noise.log').open('a') as noise:
        while index<len(ids):
            token=uuid.uuid4().hex;config['worker_token']=token
            cmd=DOCKER[:3]+['-e','BENCHMARK_RUN_TOKEN='+token]+DOCKER[3:]+[(HERE/'native_agent_worker.py').read_text()]
            with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1) as worker:
                lines=queue.Queue();threading.Thread(target=read_stdout,args=(worker.stdout,lines),daemon=True).start()
                worker.stdin.write(json.dumps(config,ensure_ascii=False)+'\n');worker.stdin.flush()
                deadline=time.monotonic()+120;sent=False;started=None
                def dispatch():
                    nonlocal sent,started,deadline
                    qid=ids[index];save(attempts,{'question_id':qid});sent=True;started=time.monotonic()
                    deadline=started+float(config['question_timeout_seconds'])
                    worker.stdin.write(json.dumps(qmap[qid],ensure_ascii=False)+'\n');worker.stdin.flush()
                try:
                    while True:
                        line=receive_line(lines,deadline)
                        if line is None:raise RuntimeError('worker_exited_before_result; inspect private logs')
                        if not line.startswith('AGENT_RESULT:'):noise.write(line);noise.flush();continue
                        row=json.loads(line[len('AGENT_RESULT:'):]);save(d/'events.jsonl',row)
                        if row.get('event')=='ready':
                            if row.get('worker_token')!=token:raise ValueError('worker_identity_mismatch')
                            validate_ready_ledger(row['ledger'],records);dispatch()
                        elif row.get('event')=='question_complete':
                            done=row['record'];validate_result(ids[index],done);save(records_path,done);records.append(done)
                            print(json.dumps({'arm':arm,'question_id':ids[index],'status':done['status'],'latency_ms':done['latency_ms'],'ledger':done['ledger']},ensure_ascii=False),flush=True)
                            sent=False;index+=1
                            terminal=(done['ledger']['pending_calls'] or done['ledger']['fatal_reason'] in ['run_token_budget','provider_connect_failed','provider_usage_unknown'] or done['status'] not in ['ok','budget_censored','timeout'])
                            if terminal or index==len(ids):worker.stdin.close();worker.wait(timeout=30);return
                            if done['status']!='ok':worker.stdin.close();worker.wait(timeout=30);break
                            dispatch()
                except TimeoutError:
                    result=subprocess.run(DOCKER+[STOP_WORKER],input=json.dumps({'token':token,'ledger':config['container_ledger']}),text=True,capture_output=True,timeout=15,check=True)
                    snapshot=json.loads(result.stdout)
                    save(d/'host_deadlines.jsonl',{'question_id':ids[index] if sent else None,'token':token,**snapshot})
                    if sent:
                        audit=d/f'{ids[index]}.deadline-ledger.jsonl';write_l(audit,snapshot['ledger_events'])
                        lim=config['limits'];ledger=CallLedger(audit,run_limit=lim['run_tokens'],question_limit=lim['question_tokens'],call_limit=lim['question_calls'])
                        calls=[r for r in ledger.calls.values() if r['question_id']==ids[index]]
                        done={'question_id':ids[index],'question':qmap[ids[index]]['question'],'status':'host_timeout','answer':'','latency_ms':round((time.monotonic()-started)*1000),'ledger':ledger.summary(),'calls':calls,'error':'host_deadline_exceeded; provider/embedding usage may be incomplete; no replay'}
                        save(records_path,done);print(json.dumps({k:done[k] for k in ['question_id','status','latency_ms','ledger']}),flush=True)
                    if worker.stdin and not worker.stdin.closed:worker.stdin.close()
                    worker.wait(timeout=10)
                    return
                finally:
                    if worker.stdin and not worker.stdin.closed:worker.stdin.close()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--arm',choices=['rag','low']);a.add_argument('--max-questions',type=int);a.add_argument('--dry',action='store_true');args=a.parse_args()
    prepare() if args.action=='prepare' else run(args.arm,args.max_questions,args.dry)
