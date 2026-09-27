"""Freeze, retrieve local BM25, then replay rank fusion without model calls."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

from experiments.evaluation.context_completion import Candidate, evidence_coverage, select_context
from experiments.evaluation.context_experiment import DEV_SHA, TEST_PATH, TEST_SHA, _aggregate, _alternatives, _paired_delta, digest, evaluate_context, jsonlines, read_verified, write_json, write_jsonl
from experiments.evaluation.metrics import retrieval_metrics
from experiments.retrieval.fusion import Hit, fuse


CONFIG = Path('experiments/configs/rrf_fusion_v1.json')
ROOT = Path('experiments/datasets/private/rrf_fusion_v1')
FREEZE = ROOT / 'pre_retrieval_freeze.json'
SCRIPTS = [Path(p) for p in [
    'experiments/retrieval/fusion.py', 'experiments/retrieval/bm25_worker.py',
    'experiments/evaluation/fusion_experiment.py', 'experiments/evaluation/context_completion.py',
    'experiments/evaluation/context_experiment.py', 'experiments/evaluation/metrics.py',
    'experiments/retrieval/test_fusion.py', 'experiments/evaluation/test_fusion_experiment.py',
]]


def project_saved(rows, snapshot):
    projected = []
    for rank, row in enumerate(rows, 1):
        cached = snapshot[row['id']]
        if row['rank'] != rank or any(row[k] != cached[k] for k in ['document_id', 'source_document_id']):
            raise ValueError('Saved ranking identity mismatch')
        projected.append({'id': row['id'], 'score': row['similarity']})
    return projected


def validate_candidates(rows, expected_ids, snapshot):
    if [r['question_id'] for r in rows] != expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Candidate question identities changed')
    for row in rows:
        if row['status'] not in {'ok', 'error'} or (row['status'] != 'ok' and row['bm25']):
            raise ValueError('Invalid BM25 failure row')
        for route in ['bm25', 'dense']:
            hits = row[route]
            if len(hits) > 64 or any(h['id'] not in snapshot for h in hits):
                raise ValueError('Out-of-snapshot or oversized candidate list')
            fuse([Hit(h['id'], h['score']) for h in hits], [], 'rrf')  # validates scores/order/duplicates


def evidence_prefix_rank(evidence, candidates):
    if not candidates or not all(evidence_coverage(evidence, candidates)):
        return None
    low, high = 1, len(candidates)
    while low < high:
        mid = (low + high) // 2
        if all(evidence_coverage(evidence, candidates[:mid])):
            high = mid
        else:
            low = mid + 1
    return low


def _load():
    frozen = json.loads(FREEZE.read_text())
    for path, sha in frozen['files'].items():
        if digest(path) != sha:
            raise ValueError(f'Frozen input changed: {path}')
    config = json.loads(CONFIG.read_text())
    if config['benchmark_sha256'] != DEV_SHA or digest(TEST_PATH) != TEST_SHA or config['cloud_calls_allowed'] or config['test_queries_allowed']:
        raise ValueError('Only the frozen development experiment is permitted')
    snapshot = json.loads(Path(config['snapshot']).read_text())
    chunks = {c['id']: c for c in snapshot['chunks']}
    if len(chunks) != 3205 or any(hashlib.sha256(c['content'].encode()).hexdigest() != c['content_sha256'] for c in chunks.values()):
        raise ValueError('Chunk snapshot integrity failure')
    dense = jsonlines(Path(config['rankings']['dense']['path']).read_text())
    if len(dense) != 51 or any(r['status'] != 'ok' or len(r['retrieved_chunks']) != 64 for r in dense):
        raise ValueError('Expected the fully saved Dense dev51 run')
    return config, frozen, snapshot, chunks, dense


def prepare():
    config = json.loads(CONFIG.read_text())
    paths = [CONFIG, Path(config['protocol']), *SCRIPTS, TEST_PATH]
    paths += [Path(config[k]) for k in ['snapshot', 'document_map', 'benchmark', 'qrels', 'reviews']]
    paths += [Path(r['path']) for r in config['rankings'].values()]
    for key in ['benchmark', 'qrels', 'snapshot', 'document_map']:
        read_verified(config[key], config[key + '_sha256'])
    for spec in config['rankings'].values():
        read_verified(spec['path'], spec['sha256'])
    if digest(TEST_PATH) != TEST_SHA:
        raise ValueError('Held-out file changed')
    ROOT.mkdir(parents=True, exist_ok=False)
    write_json(FREEZE, {'created_at': datetime.now(timezone.utc).isoformat(), 'experiment': config['experiment'], 'files': {str(p): digest(p) for p in paths}, 'model_calls': 0, 'test_queries': 0, 'scores_not_yet_evaluated': True})
    code = ROOT / 'code'
    for path in [CONFIG, Path(config['protocol']), *SCRIPTS]:
        destination = code / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    _load()
    print(json.dumps({'freeze': str(FREEZE), 'files': len(paths)}))


def retrieve():
    config, frozen, snapshot, chunks, dense = _load()
    output = ROOT / 'candidates.json'
    if output.exists():
        raise ValueError('Refusing to repeat or overwrite retrieval')
    mapping = json.loads(Path(config['document_map']).read_text())
    request = {'mapping': mapping, 'expected_index_state': snapshot['metadata']['after'], 'queries': [{'question_id': r['question_id'], 'question': r['question']} for r in dense]}
    script = Path('experiments/retrieval/bm25_worker.py').read_text()
    result = subprocess.run(['docker', 'exec', '-i', 'docker-ragflow-cpu-1', '/ragflow/.venv/bin/python', '-c', script], input=json.dumps(request, ensure_ascii=False), text=True, capture_output=True, timeout=600)
    if result.returncode:
        raise RuntimeError('Local BM25 worker failed; sensitive stderr suppressed')
    response = json.loads(result.stdout)
    if response.get('status') != 'ok':
        write_json(ROOT / 'retrieval_initialization_error.json', response)
        raise RuntimeError('BM25 initialization/integrity failed; see sanitized record')
    rows = []
    if [r['question_id'] for r in response['queries']] != [r['question_id'] for r in dense]:
        raise ValueError('Worker query identities changed')
    for lexical, semantic in zip(response['queries'], dense):
        for hit in lexical['hits']:
            if chunks[hit['id']]['document_id'] != hit['document_id']:
                raise ValueError('BM25 identity differs from snapshot')
        rows.append({'question_id': semantic['question_id'], 'question': semantic['question'], 'status': lexical['status'], 'bm25': [{'id': h['id'], 'score': h['score']} for h in lexical['hits']], 'dense': project_saved(semantic['retrieved_chunks'], chunks), 'tokenized_query': lexical.get('tokenized_query'), 'local_bm25_ms': lexical['local_bm25_ms'], 'error_type': lexical.get('error_type')})
    validate_candidates(rows, [r['question_id'] for r in dense], chunks)
    _load()  # detect local changes during the service read
    write_json(output, {'metadata': response['metadata'], 'queries': rows})
    write_json(ROOT / 'candidate_freeze.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'pre_retrieval_freeze_sha256': digest(FREEZE), 'candidates_sha256': digest(output), 'query_count': len(rows), 'successful_bm25': sum(r['status'] == 'ok' for r in rows), 'model_calls': 0, 'index_writes': 0, 'test_queries': 0})
    print(json.dumps({'output': str(output), 'queries': len(rows), 'successful': sum(r['status'] == 'ok' for r in rows), 'model_calls': 0}))


def evaluate(output):
    config, frozen, snapshot, chunks, dense = _load()
    candidate_freeze = json.loads((ROOT / 'candidate_freeze.json').read_text())
    if digest(FREEZE) != candidate_freeze['pre_retrieval_freeze_sha256']:
        raise ValueError('Candidate protocol changed')
    inputs = json.loads(read_verified(ROOT / 'candidates.json', candidate_freeze['candidates_sha256']))
    expected_ids = [r['question_id'] for r in dense]
    validate_candidates(inputs['queries'], expected_ids, chunks)
    native = jsonlines(Path(config['rankings']['native_hybrid']['path']).read_text())
    if [r['question_id'] for r in native] != expected_ids or any(r['status'] != 'ok' for r in native):
        raise ValueError('Native reference run differs')
    native = {r['question_id']: project_saved(r['retrieved_chunks'], chunks) for r in native}
    rankings, selections, cache, unions, timings = [], [], {}, {}, []
    for row in inputs['queries']:
        qid = row['question_id']
        routes = {name: [Hit(h['id'], h['score']) for h in row[name]] for name in ['bm25', 'dense']}
        unions[qid] = list(dict.fromkeys(h.id for name in routes for h in routes[name]))
        for method in config['methods']:
            started = time.perf_counter()
            if method in ['weighted', 'rrf']:
                ranked = fuse(routes['bm25'], routes['dense'], method, config['rrf_k'], config['dense_weight'], config['candidate_limit']) if row['status'] == 'ok' else []
            else:
                ranked = native[qid] if method == 'native_hybrid' else row[method]
            timings.append({'question_id': qid, 'method': method, 'pure_ranking_ms': (time.perf_counter() - started) * 1000})
            candidates = [Candidate(cid, chunks[cid]['document_id'], chunks[cid]['source_document_id'], rank, chunks[cid]['order'], chunks[cid]['content'], chunks[cid]['tokens']) for rank, hit in enumerate(ranked, 1) for cid in [hit['id']]]
            cache[(method, qid)] = candidates
            status = row['status'] if method in ['bm25', 'weighted', 'rrf'] else 'ok'
            rankings.append({'method': method, 'question_id': qid, 'status': status, 'chunks': ranked})
            for policy in config['policies']:
                chosen = select_context(candidates, policy, config['budget_tokens'], config['seed_top_k'])
                selections.append({'method': method, 'policy': policy, 'question_id': qid, 'saved_status': status, 'selected_ids': [c.id for c in chosen], 'selected_ranks': [c.rank for c in chosen]})
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / 'rankings.jsonl', rankings)
    write_jsonl(output / 'selections.jsonl', selections)
    write_jsonl(output / 'timings.jsonl', timings)
    # No reference answers or relevance labels enter the above selection phase.
    benchmark = {r['question_id']: r for r in jsonlines(read_verified(config['benchmark'], DEV_SHA))}
    if list(benchmark) != expected_ids or any(r['split'] != 'dev' or r['review_status'] != 'human_approved' for r in benchmark.values()):
        raise ValueError('Wrong benchmark')
    qrels = json.loads(Path(config['qrels']).read_text())
    reviews = jsonlines(Path(config['reviews']).read_text())
    judgments = {(r['question_id'], r['candidate_source_document_id']): r['judgment'] for r in reviews}
    alternatives = _alternatives(qrels, benchmark)
    evaluated, rank_metrics = [], []
    for row in selections:
        qid = row['question_id']
        lookup = {c.id: c for c in cache[(row['method'], qid)]}
        selected = [lookup[cid] for cid in row['selected_ids']]
        evaluated.append({**row, **evaluate_context(benchmark[qid], selected, alternatives[qid], set(qrels['relevant_document_ids'][qid]), judgments)})
    for row in rankings:
        qid = row['question_id']
        candidates = cache[(row['method'], qid)]
        prefix = evidence_prefix_rank(benchmark[qid]['evidence'], candidates)
        rank_metrics.append({'question_id': qid, 'method': row['method'], 'complete_evidence_prefix_rank': prefix, 'complete_evidence_reciprocal_rank': 1 / prefix if prefix else 0, 'metrics': retrieval_metrics([asdict(c) for c in candidates], qrels['relevant_document_ids'][qid])})
    groups = {qid: r['policy_group_id'] for qid, r in benchmark.items()}
    results, document_metrics, comparisons = {}, {}, {}
    for method in config['methods']:
        results[method] = {policy: _aggregate([r for r in evaluated if r['method'] == method and r['policy'] == policy]) for policy in config['policies']}
        rows = [r for r in rank_metrics if r['method'] == method]
        document_metrics[method] = {k: statistics.mean(r['metrics'][k] for r in rows) for k in rows[0]['metrics']}
        document_metrics[method].update(mean_complete_evidence_reciprocal_rank=statistics.mean(r['complete_evidence_reciprocal_rank'] for r in rows), candidate_complete_count=sum(r['complete_evidence_prefix_rank'] is not None for r in rows))
    for base, treatment in [('weighted','rrf'), ('dense','weighted'), ('dense','rrf'), ('dense','bm25')]:
        comparisons[f'{treatment}_vs_{base}'] = {policy: _paired_delta({r['question_id']:r for r in evaluated if r['method']==base and r['policy']==policy}, {r['question_id']:r for r in evaluated if r['method']==treatment and r['policy']==policy}, groups, config) for policy in config['policies']}
    union_ceiling = {}
    for qid, ids in unions.items():
        candidates = [Candidate(cid, chunks[cid]['document_id'], chunks[cid]['source_document_id'], i, chunks[cid]['order'], chunks[cid]['content'], chunks[cid]['tokens']) for i, cid in enumerate(ids, 1)]
        union_ceiling[qid] = {'chunks': len(ids), 'complete': all(evidence_coverage(benchmark[qid]['evidence'], candidates))}
    summary = {'results': results, 'ranking_metrics': document_metrics, 'paired_comparisons': comparisons, 'union_candidate_ceiling': union_ceiling, 'model_calls': 0, 'test_queries': 0, 'generation': 'Not Run'}
    write_jsonl(output / 'per_query.jsonl', evaluated)
    write_jsonl(output / 'ranking_metrics.jsonl', rank_metrics)
    write_json(output / 'summary.json', summary)
    _load()
    if digest(ROOT / 'candidates.json') != candidate_freeze['candidates_sha256']:
        raise ValueError('Candidates changed during evaluation')
    write_json(output / 'manifest.json', {'experiment': config['experiment'], 'input_sha256': {**frozen['files'], str(FREEZE): digest(FREEZE), str(ROOT / 'candidate_freeze.json'): digest(ROOT / 'candidate_freeze.json'), str(ROOT / 'candidates.json'): digest(ROOT / 'candidates.json')}, 'output_sha256': {p.name: digest(p) for p in sorted(output.iterdir())}, 'model_calls': 0, 'test_queries': 0, 'question_count': 51, 'policy_groups': len(set(groups.values())), 'selections_saved_before_loading_labels': True, 'nondeterministic_outputs': ['timings.jsonl']})
    print(json.dumps({'output': str(output), 'complete_counts': {m: {p: r['original_complete_count'] for p, r in v.items()} for m, v in results.items()}, 'union_ceiling': sum(r['complete'] for r in union_ceiling.values()), 'paired_rrf_vs_weighted': comparisons['rrf_vs_weighted']}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'retrieve', 'evaluate'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'retrieve':
        retrieve()
    elif args.output:
        evaluate(args.output)
    else:
        parser.error('evaluate requires --output')


if __name__ == '__main__':
    main()
