"""Post-run interval audit, descriptive analysis and report; no model calls."""

import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import statistics


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalized(text):return ''.join(re.sub(r'<[^>]+>','',html.unescape(text)).split())


def analyze(root,replay):
    config=read('experiments/configs/independent_rerank_v1.json')
    chunks={r['id']:r for r in read(config['snapshot'])['chunks']}
    benchmark={r['question_id']:r for r in rows(config['benchmark'])}
    requests={r['request_id']:r for r in rows('experiments/datasets/private/independent_rerank_v1/requests.jsonl')}
    responses={r['request_id']:r for r in rows(root/'responses.jsonl')}
    ranked={(r['method'],r['question_id']):r for r in rows(root/'rankings.jsonl')}
    metrics={(r['method'],r['question_id']):r for r in rows(root/'ranking_metrics.jsonl')}
    contexts=rows(root/'per_query.jsonl');summary=read(root/'summary.json')
    qrels=read(config['qrels'])['relevant_document_ids']
    bydoc=defaultdict(list)
    for c in chunks.values():bydoc[c['document_id']].append(c)
    bysource=defaultdict(list)
    for cs in bydoc.values():
        text='';intervals=[]
        for c in sorted(cs,key=lambda c:c['order']):
            start=len(text);text+=normalized(c['content']);intervals.append((start,len(text),c['id']))
        bysource[cs[0]['source_document_id']].append((text,intervals))
    spans={}
    for qid,q in benchmark.items():
        spans[qid]=[]
        for ref in q['evidence']:
            needle=normalized(ref['text']);options=[]
            for text,intervals in bysource[ref['source_document_id']]:
                pos=text.find(needle)
                while pos>=0:
                    options.append({cid for a,b,cid in intervals if a<pos+len(needle) and b>pos})
                    pos=text.find(needle,pos+1)
            spans[qid].append(options)
    def cover(qid,ids):
        selected=set(ids)
        return [any(s<=selected for s in options) for options in spans[qid]]
    scores_checked=0
    for qid,request in requests.items():
        original=ranked['dense',qid]['ordered_ids'];response=responses[qid]
        actual=ranked['dense_partial_rerank',qid]['ordered_ids']
        assert [original[i] for i in request['slots']]==request['sent_chunk_ids']
        assert all(chunks[cid]['source_document_id']!='79870' for cid in request['sent_chunk_ids'])
        if response['status']=='ok':
            score_by_index={r['index']:r['relevance_score'] for r in response['scores']}
            assert set(score_by_index)==set(range(len(request['slots'])))
            assert all(0<=v<=1 for v in score_by_index.values())
            sorted_ids=[request['sent_chunk_ids'][i] for i in sorted(score_by_index,key=lambda i:(-score_by_index[i],request['slots'][i]))]
            expected=original[:]
            for slot,cid in zip(request['slots'],sorted_ids):expected[slot]=cid
            assert actual==expected and set(actual)==set(original) and actual[32:]==original[32:]
            scores_checked+=len(score_by_index)
        else:assert actual==[]
    span_checks=0;controls=0
    old={(r['policy'],r['question_id']):r for r in rows('experiments/results/private/context_completion-dev51-v1/per_query.jsonl') if r['method']=='dense'}
    for c in contexts:
        ids=c['selected_ids'];qid=c['question_id']
        assert set(ids)<=set(ranked[c['method'],qid]['ordered_ids']) and len(ids)==len(set(ids))
        assert c['tokens']==sum(chunks[cid]['tokens'] for cid in ids)<=2560
        expected=cover(qid,ids)
        assert c['evidence_covered']==expected and c['original_complete']==all(expected)
        span_checks+=len(expected)
        if c['method']=='dense':
            baseline=old[c['policy'],qid]
            assert all(c[k]==baseline[k] for k in ['selected_ids','selected_ranks','tokens','evidence_covered'])
            controls+=1
    for key,r in ranked.items():
        qid=r['question_id'];ids=r['ordered_ids'];m=metrics[key]
        docids=list(dict.fromkeys(chunks[cid]['source_document_id'] for cid in ids))
        assert m['metrics']['recall@5']==len(set(docids[:5])&set(qrels[qid]))/len(qrels[qid])
        assert m['metrics']['mrr@10']==next((1/i for i,d in enumerate(docids[:10],1) if d in qrels[qid]),0)
        assert m['complete_evidence_prefix_rank']==next((i for i in range(1,len(ids)+1) if all(cover(qid,ids[:i]))),None)
    equal={}
    for name in ['rankings.jsonl','selections.jsonl','per_query.jsonl','ranking_metrics.jsonl','summary.json']:
        assert (root/name).read_bytes()==(replay/name).read_bytes();equal[name]=sha(root/name)
    manifest=read(root/'evaluation_manifest.json')
    assert all(sha(p)==h for p,h in manifest['input_sha256'].items())
    assert all(sha(root/p)==h for p,h in manifest['output_sha256'].items())
    validation={'provider_scores_checked':scores_checked,'rankings_checked':len(ranked),'contexts_checked':len(contexts),'independent_character_interval_checks':span_checks,'unchanged_dense_control_rows':controls,'replay_byte_identical':equal,'frozen_input_and_output_hashes_match':True,'test_queries':0,'model_calls_during_audit':0}
    failures=[{'question_id':r['question_id'],'method':r['method'],'policy':r['policy'],'missing_reference_indices':[i for i,x in enumerate(r['evidence_covered']) if not x],'selected_ids':r['selected_ids'],'all64_reference_complete':all(cover(r['question_id'],ranked[r['method'],r['question_id']]['ordered_ids']))} for r in contexts if not r['original_complete']]
    changed=[]
    for qid in benchmark:
        a=metrics['dense',qid]['complete_evidence_prefix_rank'];b=metrics['dense_partial_rerank',qid]['complete_evidence_prefix_rank']
        if a!=b:changed.append({'question_id':qid,'dense_prefix':a,'partial_rerank_prefix':b})
    attempts=rows(root/'attempts.jsonl');latencies=[r['latency_ms'] for r in attempts if 'latency_ms' in r]
    semantic=[
        {'question_id':'med-026','assessment':'精确资金来源证据变弱','basis':'原始目标段包含医疗救助基金全额资助；选入FAQ只提供财政资金全额资助，且高排片段包含长期护理保险。不能将一般财政表述自动标为原通知精确资金来源的完整替代。','reviewer':'assistant','human_approved':False},
        {'question_id':'med-029','assessment':'替代证据可推算答案','basis':'boundary仍包含2022年340元，以及2024年440元含20元长护险；可得到不含长护险420元。严格原始参考段覆盖退步不等于答案不可得。','sources':['79273','79321'],'reviewer':'assistant','human_approved':False},
        {'question_id':'med-115','assessment':'具体费率依据缺失','basis':'10%、11%或6%、个人不缴生育保险费所在原始块由第1降至第9，未选入；新上下文以封面总则和其他年度规则为主，泛指沿用原费率不足以提供全部数值。','reviewer':'assistant','human_approved':False},
        {'question_id':'med-116','assessment':'多子问证据不完整','basis':'一般职工待遇支付基金的目标块由第1降至第11；保留待遇不变说明与灵活就业生育费用基金来源，但未找到一般职工待遇资金来源的同等明确表述。','reviewer':'assistant','human_approved':False},
    ]
    analysis={'kind':'post-run analysis; no labels or parameters modified','all_incomplete_contexts':failures,'all_evidence_prefix_changes':changed,'focused_semantic_review':semantic,'latency_ms':{'mean':statistics.mean(latencies),'median':statistics.median(latencies)},'scope_limit':'partial reranking of screened slots in original top32, not full64 reranking','semantic_notes_not_generation_accuracy':True}
    for name,value in [('validation.json',validation),('analysis.json',analysis)]:
        with (root/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    report=render(summary,analysis,validation)
    with (root/'report.md').open('x') as f:f.write(report)
    print(json.dumps({'validation':validation,'failed_contexts':len(failures),'semantic_alternative':'med-029; no qrels modification'},ensure_ascii=False))


def render(summary,analysis,validation):
    lines=['# Dense固定候选的独立局部重排结果','',
           '**当前决策：保留Dense＋边界补全，不采用本轮gte-rerank-v2局部重排配置。** 51题全部调用成功，0次重试；主要严格证据指标下降，不能再将本轮差异归因于此前原生Rerank的TLS失败。', '',
           '## 实现与控制条件','',
           '复用Dense每题64个候选，只对原始前32位置中的出站白名单文本调用已配置的gte-rerank-v2。每题实际11～32块，共1463个问题—片段配对、300个唯一片段；169个受限或清理后过短的位置不出站并保留原位。后32位置也保持不变。按模型分数直接排序回填可移动位置，不混入Dense/BM25分数。', '',
           '两组本地候选身份、正文、Token计数、2560参考Token预算、ranked/boundary选择规则完全相同。出站仅发送经过联系方式、标识符及链接清理的文字，长块截到1800字符，只有一个唯一块触及该长度上限。来源79870未发送。该白名单规则和定向检查不是穷尽隐私审核。', '',
           '这是**有隐私门槛和输入表示约束的局部重排**，不是全部64块的无约束Rerank；不能把结果推广到其他窗口、模型或完整RAGFlow重排实现。协议和执行代码已在调用前冻结，参见仓库docs/independent_rerank_protocol_v1.md。', '',
           '## 结果','',
           '完整覆盖要求全部原始参考片段出现，可拼接同来源连续块；不是生成答案正确率。所有51题计入分母，已审核替代证据计分没有改变本轮完整数。', '',
           '| 方案 | ranked完整 / 51 | boundary完整 / 51 | boundary平均Token | 文档Recall@5 | 文档MRR@10 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for method,v in summary['results'].items():
        m=summary['ranking_metrics'][method]
        lines.append(f"| {method} | {v['ranked']['original_complete_count']} | {v['boundary']['original_complete_count']} | {v['boundary']['mean_tokens']:.1f} | {m['recall@5']:.2%} | {m['mrr@10']:.4f} |")
    lines+=['','文档MRR略升但完整证据下降：找对文档不代表把其中的答案段排到了前面。成功子集与主表均为同一51题，无失败筛选差异。原生RAGFlow Hybrid的历史同预算结果为43/51、加边界补全46/51，只作历史系统参考；本轮严格对照是上述两行。','','| 上下文规则 | 新增完整题 | 退步题 | 配对变化 | 6组bootstrap探索区间 |','| --- | --- | --- | ---: | --- |']
    for policy,r in summary['paired_vs_dense'].items():
        lo,hi=r['exploratory_cluster_bootstrap_95pct']
        lines.append(f"| {policy} | {', '.join(r['improved']) or '无'} | {', '.join(r['regressed']) or '无'} | {100*r['complete_rate_delta']:+.2f} pp | [{100*lo:+.2f}, {100*hi:+.2f}] pp |")
    lines+=['','## 全部新增退步题的解释','','| 题目 | 观察到的变化 | 判断边界 |','| --- | --- | --- |',
            '| med-026 | 包含医疗救助基金全额资助的段落从第4降到第8，未进入上下文；长护险和居民医保FAQ排在前面 | 精确资金来源变弱；一般财政资助说法不能自动替代指定通知的基金来源 |',
            '| med-029 | 2024目标段从第2降到第5；ranked仍覆盖，boundary扩展其他来源后挤掉该段 | 仍有2022年340元和2024年440元含20元长护险，可算出420元；这是严格参考段损失，不宜认定核心答案缺失 |',
            '| med-115 | 费率答案段由第1降到第9；封面由第6升到第1 | 当前上下文缺少10%、11%或6%及个人不缴费等完整明确依据 |',
            '| med-116 | 待遇/支付来源段由第1降到第11；封面由第6升到第1 | 保留待遇不变和灵活就业部分，普通职工待遇资金来源仍缺少同等明确支持 |', '',
            '以上语义判断由助手本地复核，未独立人工确认，也没有运行生成模型。尤其med-029不能记成“一道新答错题”。没有擅自补改qrels或原始参考答案，后续如正式采用该替代证据，应由用户复核。全部27条不完整上下文及所有证据前缀变化保存在analysis.json。', '',
            '重排实际把部分封面、总则、宣传说明排到了具体金额或待遇段之前；这是可由返回分数和排名核验的现象。是否由年份、标题词匹配造成只能作为进一步假设，不能从分数直接推断模型内部机制。med-029还显示：边界规则在换排序后会额外占用预算，因此原先的无退步结论不能跨检索配置套用。', '',
            '## 调用、核验与局限','',
            f"实际调用{summary['calls']['actual_calls']}次，成功{summary['calls']['successful_queries']}题，提供方报告总用量{summary['calls']['reported_total_tokens']:,} Token，用量未知的尝试为{summary['calls']['attempts_with_unknown_usage']}次。平均单次重排{analysis['latency_ms']['mean']:.1f}ms，中位数{analysis['latency_ms']['median']:.1f}ms；这是新增重排请求耗时，不是整个问答链路延迟。没有新Embedding、文档解析、生成或测试集调用。", '',
            f"19项针对性测试通过；独立核验{validation['provider_scores_checked']}个提供方分数映射、{validation['rankings_checked']}条排名、{validation['contexts_checked']}条上下文、{validation['independent_character_interval_checks']}次字符区间覆盖。{validation['unchanged_dense_control_rows']}条Dense控制记录与历史结果完全一致。新进程重放的排名、选择、逐题指标及汇总逐字节一致。", '',
            '开发集只有6个政策组，且已用于多轮实验，bootstrap只供探索。没有测全部候选重排、多种模型或多种窗口；本轮不能得出“所有Rerank都无效”。qrels不完整，原始段落严格覆盖也会漏掉等价替代依据。最终选择仍需一次冻结后的独立测试。', '',
            '服务接口与限制参见[官方Rerank协议](https://www.alibabacloud.com/help/en/model-studio/text-rerank-api)和[模型说明](https://www.alibabacloud.com/help/en/model-studio/rerank)。本机已配置SDK地址为DashScope原有北京端点，已真实验证51次成功；没有迁移提供方或更换模型。', '',
            '## 复现与下一步','',
            '配置为experiments/configs/independent_rerank_v1.json，调用前冻结与出站请求位于experiments/datasets/private/independent_rerank_v1。responses.jsonl与attempts.jsonl保留最终结果及每次尝试；只需复制这两份文件和response_freeze.json到一个新目录即可重放，不要重新执行付费run：', '',
            '```bash','python3 -m experiments.evaluation.independent_rerank_experiment evaluate \\','  --output /path/to/new-directory-containing-saved-responses','```','',
            '下一阶段把Dense＋boundary＋短编号引用收敛成可运行问答链路，检查多子问和引用回归，冻结配置后准备99题正式测试。本轮不追加重排模型或参数搜索，不将未获收益的配置替换进主方案。', '',
            '面试追问：', '',
            '1. 为什么文档MRR提高、上下文完整证据却下降？',
            '2. 如何保证重排前后的差异来自排序，而不是召回候选、接口失败或Token预算变化？',
            '3. 隐私门槛、局部窗口和长片段截断如何限制结论？',
            '4. med-029为什么严格覆盖失败却仍可能答对，应该如何处理替代证据标注？']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);a=p.parse_args();analyze(a.results,a.replay)


if __name__=='__main__':main()
