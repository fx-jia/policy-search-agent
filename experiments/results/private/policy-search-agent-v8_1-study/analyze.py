"""Offline content annotations and cost/trace diagnostics for frozen v8 arms."""
import argparse
import collections
import copy
import importlib.util
import json
from pathlib import Path
import re
import statistics
from experiments.evaluation.search_agent_report import diagnostics, paired_changes, readl

OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'benchmark-v2-baselines-v2-nonthinking'
spec=importlib.util.spec_from_file_location('fixed_metrics',OUT.parent/'policy-search-agent-v7-dev60-serial/analyze.py')
fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)


def main(require_complete=False):
    manifest=json.loads((OUT/'manifest.json').read_text())
    qs={q['question_id']:q for q in readl(BASE/'dev.jsonl')}
    notes=json.loads((OUT/'reviews.json').read_text()) if (OUT/'reviews.json').exists() else {}
    overrides=json.loads((BASE/'evaluation_overrides.json').read_text())['questions']
    metrics={}; all_diagnostics=[]; alternative=[]
    answers=['# v8.1 A/B实验：问题、参考原文、回答及初评','','助手逐题初评；未知替代来源不自动判错。技术回归6题与新增55题分别统计。','']
    for cohort,arms in manifest['arms'].items():
        ids=manifest['selection'][cohort];table={};score_maps={};row_maps={}
        for arm in arms:
            folder=OUT/cohort/arm;rows=readl(folder/'records.jsonl');row_maps[arm]={r['question_id']:r for r in rows}
            assert len(rows)==len(row_maps[arm]) and set(row_maps[arm])<=set(ids)
            annotations=notes.get(cohort,{}).get(arm,{})
            scores={}
            for row in rows:
                q=row['question_id']
                if q in annotations:scores[q]={'question_id':q,'arm':arm,'reviewer':'assistant_local_reading',**annotations[q]}
                elif row['status']!='ok':scores[q]={'question_id':q,'arm':arm,'outcome':'technical_error','points':[None]*len(qs[q]['required_points']),'note':'Technical status '+row['status']+' / '+row.get('stop_reason','')}
            for q,s in scores.items():
                assert len(s['points'])==len(qs[q]['required_points'])
                if s['outcome']=='complete':assert all(x is True for x in s['points'])
            score_maps[arm]=scores
            (folder/'assessments.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in scores.values()))
            m=table[arm]=fixed.summarize(ids,rows,scores)
            m['required_points_passed']=sum(p is True for s in scores.values() for i,p in enumerate(s['points']) if not (s['question_id']=='mb2-052' and i==2))
            m['required_points_total']=sum(len(qs[q]['required_points'])-(q=='mb2-052') for q in ids)
            m['required_points_pending']=sum(p is None for s in scores.values() for p in s['points'])
            m['by_difficulty']={d:fixed.summarize([q for q in ids if qs[q]['difficulty']==d],rows,scores) for d in ['Easy','Medium','Hard']}
            m['latency_stdev_seconds']=round(statistics.stdev(r['latency_ms']/1000 for r in rows),2) if len(rows)>1 else None
            m['answered_without_citation_ids']=[r['question_id'] for r in rows if r['status']=='ok' and r.get('answer','').strip() and not re.search(r'\[ID:\d+\]',r['answer'])]
            m['stop_reasons']=dict(collections.Counter(r.get('stop_reason') for r in rows))
            m['tool_actions']=dict(collections.Counter(t['action']['type'] for r in rows for t in r.get('trace',[]) if t.get('event')=='tool'))
            m['action_override_questions']=sum(any(t.get('event')=='action_override' for t in r.get('trace',[])) for r in rows)
            calls=[c for r in rows for c in r.get('calls',[])]
            m['judge_tokens']=sum(c.get('usage',{}).get('total_tokens',0) for c in calls if c.get('request',{}).get('response_format'))
            m['final_tokens']=m['generation_tokens']-m['judge_tokens']
            m['citation_embedding_calls']=sum(r.get('citation_embedding_calls',0) for r in rows)
            m['generation_prompt_tokens']=sum(c.get('usage',{}).get('prompt_tokens',0) for c in calls)
            m['generation_completion_tokens']=sum(c.get('usage',{}).get('completion_tokens',0) for c in calls)
            m['generation_provider_cached_prompt_tokens']=sum(c.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens',0) for c in calls)
            m['generation_not_sent_attempts']=sum(c.get('transport_status')=='not_sent' for c in calls)
            query_events=[[e for e in r.get('events',[]) if e.get('event')=='query_embedding' or (e.get('event')=='runtime_embedding' and e.get('purpose') in ('encode_queries','navigation_query'))] for r in rows]
            m['query_embedding_observed_successes']=sum(map(len,query_events))
            m['query_embedding_confirmed_not_sent_attempts']=sum(r.get('embedding_physical',{}).get('query_not_sent_calls',0) for r in rows) if arm not in ('rag','low') else None
            m['query_embedding_unknown_usage_questions']=[r['question_id'] for r in rows if r.get('embedding_physical',{}).get('unknown_embedding_usage')] if arm not in ('rag','low') else None
            m['query_embedding_duplicate_observed_successes']=sum(len(es)-len({e['query_sha256'] for e in es}) for es in query_events if es and all('query_sha256' in e for e in es)) if arm in ('rag','low') else None
            m['query_embedding_known_tokens']=sum(r.get('query_embedding_tokens',0) for r in rows) if arm not in ('rag','low') else sum(e.get('tokens',0) for r in rows for e in r.get('events',[]) if e.get('event')=='runtime_embedding' and e.get('purpose') in ('encode_queries','navigation_query'))
            if arm in ('rag','low'):m['embedding_cache_hits']=None
            m['extra_claim_flag_questions']=[q for q,s in scores.items() if s.get('extra_claims')]
            m['core_accepted_range']=[m['accepted'],m['accepted']+m['outcomes'].get('pending_review',0)+m['unassessed']]
            m['query_embedding_failed_request_questions']=[r['question_id'] for r in rows if type(r.get('query_embedding_calls')) is int and r.get('embedding_physical',{}).get('query_physical_calls',0)>r['query_embedding_calls'] and r['status']!='ok']
            config=json.loads((folder/'config.json').read_text())
            cohort_diag=[]
            for row in rows:
                record=copy.deepcopy(row);q=row['question_id']
                if arm in ('rag','low'):
                    info=row.get('ordinary_context',{}) if arm=='rag' else row.get('compose_input',{}).get('kbinfos',{})
                    record['selected']=[{'chunk_id':c.get('chunk_id',c.get('id')),'document_id':c['doc_id']} for c in info.get('chunks',[])]
                diagnostic=diagnostics(qs[q],record,config['document_map']['documents'],overrides.get(q,{}))
                cohort_diag.append(diagnostic);all_diagnostics.append({'cohort':cohort,'arm':arm,'question_id':q,**diagnostic})
                refs=row.get('reference',{}).get('chunks',[])
                for idx in diagnostic['additional_citation_ids_pending']:
                    ref=refs[idx];cid=ref.get('id',ref.get('chunk_id'))
                    source=row.get('seen_chunks',{}).get(cid,{})
                    alternative.append({'cohort':cohort,'arm':arm,'question_id':q,'citation_id':idx,'chunk_id':cid,'source':source,'answer':row.get('answer',''),'status':'unlabelled_not_automatically_wrong'})
            m['candidate_literal_all_spans']=sum(all(d['candidate_literal_evidence']) for d in cohort_diag)
            m['final_literal_all_spans']=sum(all(d['final_literal_evidence']) for d in cohort_diag)
            if require_complete:assert set(scores)==set(row_maps[arm])==set(ids),(cohort,arm,'incomplete')
        for baseline in arms:
            if baseline=='ab':continue
            a={q:s['outcome'] for q,s in score_maps['ab'].items()};b={q:s['outcome'] for q,s in score_maps[baseline].items()}
            table['ab_vs_'+baseline]=paired_changes(ids,b,a)
            valid=[q for q in ids if q in row_maps['ab'] and q in row_maps[baseline] and row_maps['ab'][q]['status']==row_maps[baseline][q]['status']=='ok']
            table['content_ab_vs_'+baseline]=paired_changes(valid,b,a)
        metrics[cohort]=table
        for q in ids:
            info=qs[q];answers += [f'## {cohort} / {q} / {info["difficulty"]}', '',info['question'],'','参考答案：'+info['reference_answer'],'','必答点：'+'；'.join(p['text'] for p in info['required_points']),'']
            for e in info['evidence']:answers += ['参考原文 '+e['source_document_id']+'：',e['text'],'']
            for arm in arms:
                row=row_maps[arm].get(q,{});s=score_maps[arm].get(q,{})
                answers += ['### '+arm,'',s.get('outcome','unassessed')+'：'+s.get('note',''),'',row.get('answer','未产出'), '']
    (OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    (OUT/'all_answers.md').write_text('\n'.join(answers))
    (OUT/'diagnostics.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in all_diagnostics))
    (OUT/'alternative_citations.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in alternative))
    print(json.dumps({cohort:{a:{k:table[a][k] for k in ['attempted','accepted','unassessed','generation_tokens']} for a in manifest['arms'][cohort]} for cohort,table in metrics.items()},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');main(parser.parse_args().require_complete)
