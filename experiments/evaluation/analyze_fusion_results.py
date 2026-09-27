"""Post-run audit using full-document character intervals and independent scores.

Does not import the fusion implementation, context selector or coverage scorer.
Never changes a ranking, hyperparameter or relevance label.
"""

import argparse
from collections import defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics


def _json(path):
    return json.loads(path.read_text())


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(text):
    return ''.join(re.sub('<[^>]+>', '', html.unescape(text)).split())


def audit(output, replay):
    cfg = _json(Path('experiments/configs/rrf_fusion_v1.json'))
    snapshot = _json(Path(cfg['snapshot']))
    chunks = {c['id']:c for c in snapshot['chunks']}
    inputs = _json(Path('experiments/datasets/private/rrf_fusion_v1/candidates.json'))
    benchmark = {q['question_id']:q for q in _rows(Path(cfg['benchmark']))}
    qrels = _json(Path(cfg['qrels']))['relevant_document_ids']
    summary = _json(output / 'summary.json')
    rankings = {(r['method'],r['question_id']):r for r in _rows(output / 'rankings.jsonl')}
    evaluated = _rows(output / 'per_query.jsonl')
    rank_metrics = {(r['method'],r['question_id']):r for r in _rows(output / 'ranking_metrics.jsonl')}
    full_docs = defaultdict(list)
    for c in chunks.values():
        full_docs[c['document_id']].append(c)
    source_docs = defaultdict(list)
    for doc_chunks in full_docs.values():
        text, intervals = '', []
        for c in sorted(doc_chunks, key=lambda c:c['order']):
            start = len(text)
            text += _normalized(c['content'])
            intervals.append((start,len(text),c['id']))
        source_docs[doc_chunks[0]['source_document_id']].append((text,intervals))
    # A reference occurrence is represented by the complete chunks whose
    # character intervals overlap it; all of these chunks must be selected.
    span_options = {}
    for qid, q in benchmark.items():
        options = []
        for ref in q['evidence']:
            needle = _normalized(ref['text'])
            assert needle
            occurrences = []
            for text, intervals in source_docs[ref['source_document_id']]:
                pos = text.find(needle)
                while pos >= 0:
                    occurrences.append({cid for start,end,cid in intervals if start < pos+len(needle) and end > pos})
                    pos = text.find(needle,pos+1)
            options.append(occurrences)
        span_options[qid] = options

    def coverage(qid, ids):
        selected = set(ids)
        return [any(needed <= selected for needed in occurrences) for occurrences in span_options[qid]]

    background_union_positions = {}
    fusion_checks = 0
    for row in inputs['queries']:
        assert row['status']=='ok'
        maps = {route:{h['id']:(rank,h['score']) for rank,h in enumerate(row[route],1)} for route in ['bm25','dense']}
        union = set(maps['bm25']) | set(maps['dense'])
        assert summary['union_candidate_ceiling'][row['question_id']] == {'chunks':len(union),'complete':all(coverage(row['question_id'],union))}
        for method in ['rrf','weighted']:
            scores = {}
            for cid in union:
                value = 0
                for route in maps:
                    if cid not in maps[route]:
                        continue
                    rank, score = maps[route][cid]
                    if method=='rrf':
                        value += 1/(60+rank)
                    else:
                        values = [s for _,s in maps[route].values()]
                        low,high = min(values),max(values)
                        value += .5*((score-low)/(high-low) if high>low else 1)
                scores[cid]=value
            ordered = sorted(scores,key=lambda cid:(-scores[cid],cid))
            saved = rankings[method,row['question_id']]['chunks']
            assert [r['id'] for r in saved]==ordered[:64]
            assert all(math.isclose(r['score'],scores[r['id']],rel_tol=1e-12,abs_tol=1e-15) for r in saved)
            fusion_checks += len(scores)
            if row['question_id']=='med-017':
                background_union_positions[method]=ordered.index('cd5872b854398460')+1
    span_checks=0
    for row in evaluated:
        ids=row['selected_ids'];qid=row['question_id']
        candidate_ids={r['id'] for r in rankings[row['method'],qid]['chunks']}
        assert len(ids)==len(set(ids)) and set(ids)<=candidate_ids
        assert row['tokens']==sum(chunks[cid]['tokens'] for cid in ids)<=2560
        expected=coverage(qid,ids)
        assert row['evidence_covered']==expected and row['original_complete']==all(expected)
        span_checks+=len(expected)
    for key,row in rankings.items():
        qid=row['question_id'];ids=[r['id'] for r in row['chunks']]
        docs=list(dict.fromkeys(chunks[cid]['source_document_id'] for cid in ids))
        metrics=rank_metrics[key]['metrics']
        assert metrics['recall@5']==len(set(docs[:5])&set(qrels[qid]))/len(qrels[qid])
        assert metrics['mrr@10']==next((1/r for r,d in enumerate(docs[:10],1) if d in qrels[qid]),0)
        first=next((rank for rank in range(1,len(ids)+1) if all(coverage(qid,ids[:rank]))),None)
        assert first==rank_metrics[key]['complete_evidence_prefix_rank']
    # Unchanged control pipelines must exactly reproduce the earlier experiment.
    prior={(r['method'],r['policy'],r['question_id']):r for r in _rows(Path('experiments/results/private/context_completion-dev51-v1/per_query.jsonl'))}
    controls=0
    for row in evaluated:
        if row['method'] not in ['dense','native_hybrid']:
            continue
        previous=prior['hybrid' if row['method']=='native_hybrid' else 'dense',row['policy'],row['question_id']]
        assert all(row[k]==previous[k] for k in ['selected_ids','selected_ranks','tokens','evidence_covered','original_complete'])
        controls+=1
    replay_hashes={}
    for name in ['rankings.jsonl','selections.jsonl','per_query.jsonl','ranking_metrics.jsonl','summary.json']:
        assert (output/name).read_bytes()==(replay/name).read_bytes()
        replay_hashes[name]=_digest(output/name)
    manifest=_json(output/'manifest.json')
    assert all(_digest(Path(p))==sha for p,sha in manifest['input_sha256'].items())
    assert all(_digest(output/p)==sha for p,sha in manifest['output_sha256'].items())
    validation={'fusion_union_scores_verified':fusion_checks,'rankings_verified':len(rankings),'context_rows_verified':len(evaluated),'independent_reference_interval_checks':span_checks,'unchanged_control_rows_verified':controls,'replay_byte_identical':replay_hashes,'manifest_hashes_match':True,'model_calls':0,'test_queries':0,'index_state_unchanged':inputs['metadata']['before']==inputs['metadata']['after']==snapshot['metadata']['after']}
    prefix_changes={}
    for base,treatment in [('weighted','rrf'),('dense','rrf')]:
        changes=[]
        for qid in benchmark:
            a=rank_metrics[base,qid]['complete_evidence_prefix_rank'];b=rank_metrics[treatment,qid]['complete_evidence_prefix_rank']
            if a!=b:
                changes.append({'question_id':qid,'baseline_prefix':a,'treatment_prefix':b,'direction':'improved' if b is not None and (a is None or b<a) else 'regressed'})
        prefix_changes[f'{treatment}_vs_{base}']=changes
    failures=[]
    for row in evaluated:
        if row['original_complete']:
            continue
        qid=row['question_id'];method=row['method'];ranked=rankings[method,qid]['chunks']
        candidate_coverage=coverage(qid,[r['id'] for r in ranked])
        missing=[i for i,covered in enumerate(row['evidence_covered']) if not covered]
        failure_stage='context_selection_or_budget' if all(candidate_coverage) else ('fusion_top64_truncation' if method in ['rrf','weighted'] and summary['union_candidate_ceiling'][qid]['complete'] else 'returned_candidates_missing_reference')
        failures.append({'method':method,'policy':row['policy'],'question_id':qid,'missing_reference_indices':missing,'candidate_evidence_covered':candidate_coverage,'diagnostic_stage':failure_stage,'generation':'Not Run'})
    analysis={'kind':'post-run descriptive diagnosis; no parameter selection','evidence_prefix_changes':prefix_changes,'med017_background_full_union_rank':background_union_positions,'all_incomplete_contexts':failures,'timing_note':'BM25 includes tokenization/first-call warmup; saved Dense timing is not concurrent; pure fusion timing excludes retrieval and generation','local_bm25_mean_ms':statistics.mean(r['local_bm25_ms'] for r in inputs['queries']),'local_bm25_median_ms':statistics.median(r['local_bm25_ms'] for r in inputs['queries'])}
    for name,value in [('validation.json',validation),('analysis.json',analysis)]:
        with (output/name).open('x') as f:
            json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'validation':validation,'prefix_changes':prefix_changes,'incomplete_contexts':len(failures)},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay',type=Path,required=True)
    args=parser.parse_args()
    audit(args.output,args.replay)


if __name__=='__main__':
    main()
