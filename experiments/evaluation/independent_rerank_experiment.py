"""Prepare a bounded privacy-gated rerank batch and evaluate saved responses."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

from experiments.evaluation.context_completion import Candidate, select_context
from experiments.evaluation.context_experiment import DEV_SHA, TEST_PATH, TEST_SHA, _aggregate, _alternatives, _paired_delta, digest, evaluate_context, jsonlines, read_verified, write_json, write_jsonl
from experiments.evaluation.fusion_experiment import evidence_prefix_rank, project_saved
from experiments.evaluation.metrics import retrieval_metrics
from experiments.retrieval.independent_rerank import eligible_source, reorder_slots, sanitize_outbound, validate_scores
from experiments.retrieval.rerank_worker import ENDPOINT, make_payload, validate_config


CONFIG = Path('experiments/configs/independent_rerank_v1.json')
ROOT = Path('experiments/datasets/private/independent_rerank_v1')
CODE = [Path(p) for p in [
    'experiments/retrieval/independent_rerank.py', 'experiments/retrieval/rerank_worker.py',
    'experiments/retrieval/test_independent_rerank.py', 'experiments/evaluation/independent_rerank_experiment.py',
    'experiments/evaluation/context_completion.py', 'experiments/evaluation/context_experiment.py',
    'experiments/evaluation/fusion_experiment.py', 'experiments/evaluation/metrics.py',
]]


def inputs(config):
    for path, sha in config['input_sha256'].items():
        read_verified(path, sha)
    if digest(TEST_PATH) != TEST_SHA or config['input_sha256'][config['benchmark']] != DEV_SHA:
        raise ValueError('Only the frozen dev experiment is allowed')
    if config['rerank_prefix'] != 32 or config['candidate_count'] != 64 or config['max_actual_calls'] != 60 or config['endpoint'] != ENDPOINT:
        raise ValueError('Unexpected request scope')
    snapshot = json.loads(Path(config['snapshot']).read_text())
    chunks = {c['id']: c for c in snapshot['chunks']}
    dense = jsonlines(Path(config['dense_rankings']).read_text())
    if len(dense) != 51 or len({r['question_id'] for r in dense}) != 51 or any(r['status'] != 'ok' or len(r['retrieved_chunks']) != 64 for r in dense):
        raise ValueError('Expected the frozen Dense dev51 candidate lists')
    for row in dense:
        project_saved(row['retrieved_chunks'], chunks)
    return chunks, dense


def build_requests(config):
    chunks, dense = inputs(config)
    raw = {str(d['系统ID']): d for d in jsonlines(Path(config['source']).read_text())}
    review = json.loads(Path(config['prior_privacy_review']).read_text())
    hold = json.loads(Path(config['privacy_hold']).read_text())
    blocked = set(review['blocked_source_ids']) | {hold['source_document_id']}
    requests, review_records = [], {}
    for row in dense:
        slots, documents, ids, held = [], [], [], []
        for index, hit in enumerate(row['retrieved_chunks'][:32]):
            c = chunks[hit['id']]; sid = c['source_document_id']
            prior = review['chunks'].get(c['id'])
            previously_reviewed = bool(prior and prior['source_document_id'] == sid and prior['raw_sha256'] == c['content_sha256'])
            source_eligible = eligible_source(sid, raw[sid], blocked)
            text = sanitize_outbound(c['content'])
            allowed = sid not in blocked and (source_eligible or previously_reviewed) and len(re.sub(r'^#.*\n?', '', text)) >= 40
            if not allowed:
                held.append({'rank': index+1, 'chunk_id': c['id'], 'reason': 'privacy_or_insufficient_outbound_text'})
                continue
            if re.search(r'(?<!\d)(?:1[3-9]\d{9}|\d{17}[\dXx])(?!\d)|https?://|联系人|[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}', text):
                raise ValueError('Residual outbound privacy marker')
            slots.append(index);documents.append(text);ids.append(c['id'])
            review_records[c['id']] = {'source_document_id': sid, 'raw_sha256': c['content_sha256'], 'outbound_sha256': hashlib.sha256(text.encode()).hexdigest(), 'basis': 'prior_chunk_hash_review' if previously_reviewed else 'public_readable_policy_title_and_contact_screen', 'outbound_chars': len(text), 'clipped_at_1800_chars': len(sanitize_outbound(c['content'])) == 1800 and len(c['content']) > 1800}
        payload = make_payload(row['question'], documents)
        requests.append({'request_id': row['question_id'], 'query': row['question'], 'documents': documents, 'slots': slots, 'sent_chunk_ids': ids, 'held_slots': held, 'payload_sha256': hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()})
    return requests, {'reviewer': 'assistant', 'scope': 'rule-based source metadata and directed contact review; not exhaustive human privacy certification', 'blocked_source_ids': sorted(blocked), 'chunks': review_records, 'unique_outbound_chunks': len(review_records), 'candidate_occurrences_sent': sum(len(r['slots']) for r in requests), 'candidate_occurrences_held': sum(len(r['held_slots']) for r in requests)}


def prepare():
    config = json.loads(CONFIG.read_text())
    requests, review = build_requests(config)
    ROOT.mkdir(parents=True, exist_ok=False)
    write_jsonl(ROOT/'requests.jsonl', requests)
    write_json(ROOT/'outbound_review.json', review)
    paths = [CONFIG, Path(config['protocol']), *CODE, ROOT/'requests.jsonl', ROOT/'outbound_review.json']
    frozen = {'created_at': datetime.now(timezone.utc).isoformat(), 'files': {**config['input_sha256'], **{str(p): digest(p) for p in paths}, str(TEST_PATH): TEST_SHA}, 'question_count': 51, 'max_actual_calls':60, 'test_queries':0, 'model_calls':0}
    write_json(ROOT/'pre_call_freeze.json', frozen)
    for p in [CONFIG, Path(config['protocol']), *CODE]:
        target = ROOT/'code'/p
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    print(json.dumps({'prepared':str(ROOT),'questions':51,'unique_outbound_chunks':review['unique_outbound_chunks'],'pairs':review['candidate_occurrences_sent'],'held':review['candidate_occurrences_held'],'outbound_model':'gte-rerank-v2'}))


def verified():
    frozen = json.loads((ROOT/'pre_call_freeze.json').read_text())
    for p, sha in frozen['files'].items():
        if digest(p) != sha:
            raise ValueError(f'Frozen file changed: {p}')
    config = json.loads(CONFIG.read_text())
    requests = jsonlines((ROOT/'requests.jsonl').read_text())
    rebuilt, review = build_requests(config)
    if requests != rebuilt or review != json.loads((ROOT/'outbound_review.json').read_text()):
        raise ValueError('Outbound request or privacy review changed')
    return config, requests, frozen


def run(output, allow_model_api):
    if not allow_model_api:
        raise ValueError('Real rerank requires explicit --allow-model-api')
    config, requests, frozen = verified()
    validate_config({'endpoint': config['endpoint'], 'model_id': config['model_id'], 'max_calls': 60})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'run_manifest.json', {'pre_call_freeze_sha256':digest(ROOT/'pre_call_freeze.json'), 'model':'gte-rerank-v2','endpoint':ENDPOINT,'max_actual_calls':60,'question_count':51,'retrieval_calls':0,'embedding_calls':0,'generation_calls':0,'test_queries':0})
    proc = subprocess.Popen(['docker','exec','-e','POLICY_APPROVED_RERANK_MODEL_ID','-i','docker-ragflow-cpu-1','/ragflow/.venv/bin/python','-u','-c',Path('experiments/retrieval/rerank_worker.py').read_text()],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
    calls, consecutive, stopped = 0, 0, False
    responses = []
    try:
        proc.stdin.write(json.dumps({'endpoint':ENDPOINT,'model_id':config['model_id'],'max_calls':60})+'\n');proc.stdin.flush()
        ready = json.loads(proc.stdout.readline())
        if ready.get('status') != 'ready':
            write_json(output/'init_error.json',ready)
            raise RuntimeError('Rerank initialization failed before requests')
        with (output/'attempts.jsonl').open('x') as attempts, (output/'responses.jsonl').open('x') as stream:
            for ordinal, request in enumerate(requests,1):
                final={'request_id':request['request_id'],'status':'not_run','attempts':0,'scores':[]}
                if not stopped:
                    for attempt in [1,2]:
                        if calls >= 60:
                            stopped=True;break
                        proc.stdin.write(json.dumps({k:request[k] for k in ['request_id','query','documents']},ensure_ascii=False)+'\n');proc.stdin.flush()
                        line=proc.stdout.readline()
                        if not line:
                            result={'request_id':request['request_id'],'status':'error','attempted':True,'error_category':'worker_protocol','error_type':'UnexpectedEOF'}
                        else:
                            result=json.loads(line)
                        if result.get('request_id') != request['request_id']:
                            raise RuntimeError('Worker response identity mismatch')
                        calls+=int(result.get('attempted',False))
                        record={**result,'attempt':attempt,'payload_sha256':request['payload_sha256']}
                        attempts.write(json.dumps(record)+'\n');attempts.flush()
                        final={**record,'attempts':attempt}
                        if result['status']=='ok':
                            validate_scores(result['scores'],len(request['slots']))
                            break
                        if result.get('error_category')!='transport_or_service':
                            stopped=True;break
                    consecutive=0 if final['status']=='ok' else consecutive+1
                    if consecutive>=2:
                        stopped=True
                responses.append(final)
                stream.write(json.dumps(final)+'\n');stream.flush()
                print(json.dumps({'progress':f'{ordinal}/51','question_id':request['request_id'],'status':final['status'],'actual_calls':calls,'stopped':stopped}),flush=True)
    finally:
        if proc.stdin and not proc.stdin.closed:proc.stdin.close()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.terminate()
    verified()
    all_attempts=jsonlines((output/'attempts.jsonl').read_text())
    usage=[a.get('usage',{}).get('total_tokens') for a in all_attempts]
    write_json(output/'response_freeze.json',{'responses_sha256':digest(output/'responses.jsonl'),'attempts_sha256':digest(output/'attempts.jsonl'),'actual_calls':calls,'successful_queries':sum(r['status']=='ok' for r in responses),'reported_total_tokens':sum(v for v in usage if isinstance(v,(int,float))),'attempts_with_unknown_usage':sum(a.get('attempted',False) and v is None for a,v in zip(all_attempts,usage)),'model':'gte-rerank-v2','stopped_early':stopped})


def evaluate(output):
    config, requests, frozen = verified()
    response_freeze=json.loads((output/'response_freeze.json').read_text())
    responses=jsonlines(read_verified(output/'responses.jsonl',response_freeze['responses_sha256']))
    read_verified(output/'attempts.jsonl',response_freeze['attempts_sha256'])
    if [r['request_id'] for r in responses] != [r['request_id'] for r in requests]:
        raise ValueError('Response question set differs')
    chunks,dense=inputs(config)
    cache,rankings,selections={},{},[]
    serialized=[]
    for row, request, response in zip(dense,requests,responses):
        qid=row['question_id'];original=[r['id'] for r in row['retrieved_chunks']]
        for method in ['dense','dense_partial_rerank']:
            status='ok' if method=='dense' else response['status']
            ordered=original if method=='dense' else reorder_slots(original,request['slots'],validate_scores(response['scores'],len(request['slots']))) if status=='ok' else []
            candidates=[Candidate(cid,chunks[cid]['document_id'],chunks[cid]['source_document_id'],i,chunks[cid]['order'],chunks[cid]['content'],chunks[cid]['tokens']) for i,cid in enumerate(ordered,1)]
            cache[method,qid]=candidates;rankings[method,qid]=ordered
            serialized.append({'method':method,'question_id':qid,'status':status,'ordered_ids':ordered})
            for policy in ['ranked','boundary']:
                selected=select_context(candidates,policy,2560,3)
                selections.append({'method':method,'question_id':qid,'policy':policy,'saved_status':status,'selected_ids':[c.id for c in selected],'selected_ranks':[c.rank for c in selected]})
    write_jsonl(output/'rankings.jsonl',serialized)
    write_jsonl(output/'selections.jsonl',selections)
    benchmark={q['question_id']:q for q in jsonlines(Path(config['benchmark']).read_text())}
    if list(benchmark)!=[r['question_id'] for r in dense] or any(q['split']!='dev' for q in benchmark.values()):
        raise ValueError('Wrong scoring benchmark')
    qrels=json.loads(Path(config['qrels']).read_text());reviews=jsonlines(Path(config['reviews']).read_text())
    judgments={(r['question_id'],r['candidate_source_document_id']):r['judgment'] for r in reviews}
    alternatives=_alternatives(qrels,benchmark);evaluated=[];ranking_metrics=[]
    for row in selections:
        qid=row['question_id'];lookup={c.id:c for c in cache[row['method'],qid]}
        chosen=[lookup[cid] for cid in row['selected_ids']]
        evaluated.append({**row,**evaluate_context(benchmark[qid],chosen,alternatives[qid],set(qrels['relevant_document_ids'][qid]),judgments)})
    for (method,qid),candidates in cache.items():
        prefix=evidence_prefix_rank(benchmark[qid]['evidence'],candidates)
        ranking_metrics.append({'method':method,'question_id':qid,'complete_evidence_prefix_rank':prefix,'metrics':retrieval_metrics([asdict(c) for c in candidates],qrels['relevant_document_ids'][qid])})
    results={method:{policy:_aggregate([r for r in evaluated if r['method']==method and r['policy']==policy]) for policy in ['ranked','boundary']} for method in ['dense','dense_partial_rerank']}
    groups={qid:q['policy_group_id'] for qid,q in benchmark.items()}
    paired={policy:_paired_delta({r['question_id']:r for r in evaluated if r['method']=='dense' and r['policy']==policy},{r['question_id']:r for r in evaluated if r['method']=='dense_partial_rerank' and r['policy']==policy},groups,config) for policy in ['ranked','boundary']}
    succeeded={r['request_id'] for r in responses if r['status']=='ok'}
    success_diagnostic={method:{policy:_aggregate([r for r in evaluated if r['method']==method and r['policy']==policy and r['question_id'] in succeeded]) if succeeded else None for policy in ['ranked','boundary']} for method in ['dense','dense_partial_rerank']}
    doc_summary={method:{k:statistics.mean(r['metrics'][k] for r in ranking_metrics if r['method']==method) for k in ranking_metrics[0]['metrics']} for method in ['dense','dense_partial_rerank']}
    write_jsonl(output/'per_query.jsonl',evaluated);write_jsonl(output/'ranking_metrics.jsonl',ranking_metrics)
    write_json(output/'summary.json',{'results':results,'paired_vs_dense':paired,'ranking_metrics':doc_summary,'paired_success_diagnostic':success_diagnostic,'calls':response_freeze,'generation':'Not Run','test_queries':0,'scope':'privacy-gated partial reranking within original top32; full64 candidate identities preserved on success'})
    verified()
    write_json(output/'evaluation_manifest.json',{'input_sha256':{**frozen['files'],str(output/'response_freeze.json'):digest(output/'response_freeze.json')},'output_sha256':{p.name:digest(p) for p in output.iterdir() if p.is_file()},'selections_saved_before_labels':True,'question_count':51,'model_calls_during_evaluation':0,'test_queries':0})
    print(json.dumps({'complete_counts':{m:{p:r['original_complete_count'] for p,r in v.items()} for m,v in results.items()},'paired':paired,'calls':response_freeze},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','evaluate']);parser.add_argument('--output',type=Path);parser.add_argument('--allow-model-api',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif not args.output:parser.error('--output required')
    elif args.action=='run':run(args.output,args.allow_model_api)
    else:evaluate(args.output)


if __name__=='__main__':
    main()
