"""Render the frozen experiment tables and recorded post-run diagnosis."""

import argparse
import json
from pathlib import Path


def render(root):
    summary=json.loads((root/'summary.json').read_text())
    analysis=json.loads((root/'analysis.json').read_text())
    validation=json.loads((root/'validation.json').read_text())
    labels={'bm25':'正文 BM25','dense':'Dense','weighted':'归一化分数融合','rrf':'RRF','native_hybrid':'原生 Hybrid（系统参考）'}
    lines=['# 相同候选下的 RRF 与分数融合对照', '',
           '2026-09-22。本轮已经执行完成；采用51题开发集、1000文档、3205原有Chunk。51次本地BM25检索全部成功，Dense复用历史候选。0次Embedding、生成或Rerank调用，99题测试集未运行。', '',
           '**结论：RRF改善了一部分证据排名，但没有提高主要指标；暂不替换当前Dense＋边界补全方案。** 两种融合方案在普通上下文下均为44/51，在边界补全下均为47/51；Dense＋边界补全仍为48/51。没有在看到结果后更换权重、RRF参数或候选数。', '',
           '## 做了什么算法对照', '',
           '在同一正文BM25前64和Dense前64候选的并集上，实现两种融合。分数融合先对每路返回分数做min-max归一化，再各占50%；RRF改为按名次累计1/(60+rank)，两路等权。未命中某一路时该路贡献为0，最终统一截取64块。RRF是既有算法，不是本项目提出的新算法。', '',
           '保持同一候选、正文、上下文选择器和2560 cl100k参考Token预算；分别在ranked和boundary规则内部比较。原生Hybrid另列系统参考，不能据此声称仅替换了它的一个融合公式。标题字段在当前索引中使用自定义similarity，本轮标准BM25只查询content_ltks；分词为RAGFlow粗粒度分词，OR匹配至少一个词，不使用全文查询构造器的扩词或词项权重。', '',
           '## 主要结果', '',
           '以下“完整”要求全部原始参考片段出现，允许同来源连续块拼接；它不是模型答案正确率。允许已审核替代证据后，本轮完整题数不变。', '',
           '| 检索方法 | ranked完整 / 51 | boundary完整 / 51 | ranked平均Token | boundary平均Token |',
           '| --- | ---: | ---: | ---: | ---: |']
    for method,policies in summary['results'].items():
        a,b=policies['ranked'],policies['boundary']
        lines.append(f"| {labels[method]} | {a['original_complete_count']} | {b['original_complete_count']} | {a['mean_tokens']:.1f} | {b['mean_tokens']:.1f} |")
    lines += ['', '## 排名指标与候选上限', '',
              '文档Recall和MRR采用已人工补标的已知正例，先对来源文档去重。证据前缀倒数是：需要看到排名前几块才能覆盖全部参考证据，取该名次的倒数；64块内仍不完整计0。它与“第一篇相关文档”的MRR是不同指标。', '',
              '| 检索方法 | 文档Recall@5 | 文档MRR@10 | 完整证据前缀倒数均值 | 全部64块完整 / 51 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for method,r in summary['ranking_metrics'].items():
        lines.append(f"| {labels[method]} | {r['recall@5']:.2%} | {r['mrr@10']:.4f} | {r['mean_complete_evidence_reciprocal_rank']:.4f} | {r['candidate_complete_count']} |")
    sizes=[r['chunks'] for r in summary['union_candidate_ceiling'].values()]
    lines += ['', f"两路并集每题{min(sizes)}～{max(sizes)}块，平均{sum(sizes)/len(sizes):.1f}块；无预算、无截断的严格证据上限仍为49/51，与Dense相同，未补回med-004、045缺失的参考内容。融合截取前64后上限降为48/51。", '',
              'RRF相对分数融合，7题的完整证据前缀提前、1题推后，43题不变；但没有任何题在最终上下文完整覆盖上新增或退步。两种融合相对Dense，普通上下文覆盖集合相同；boundary均只退步med-017，没有新增完整题。', '',
              '## 所有变化和失败的诊断', '',
              '| 题目 / 对照 | 观察 | 对应机制 |',
              '| --- | --- | --- |',
              '| med-017：Dense → 两种融合 | 背景块cd5872b854398460只在Dense第58名，分数融合完整并集第83、RRF第91，均被64截断；核心1350元段仍在首位 | BM25缺席的低排Dense项在融合后被压后，随后无法供boundary补入；严格覆盖退步不等于核心金额答错 |',
              '| med-018：Dense → RRF | 完整证据前缀34→7；boundary两者均完整，ranked两者均不完整 | 排名确有改善，但在既定预算和装块方式下未新增完整上下文 |',
              '| med-026：分数融合 → RRF | 完整证据前缀2→1；两种规则的完整覆盖均不变 | 提前已有证据，未改变最终完整性 |',
              '| med-001：分数融合 → RRF | 已补标两份正例，文档Recall@5从1降到0.5，原始参考覆盖不变 | 已审核替代来源没有进入前5个去重文档，不能漏报这个次要指标退步 |',
              '| med-029：Dense → BM25 | 两个年份的证据需同时出现；2022来源在BM25第24名，未装入上下文，Dense和融合保留两侧 | 词项召回有正确文档仍可能把跨年证据排得过低 |',
              '| med-026：Dense → BM25 boundary | BM25正确块第4名，前三块扩展出的其他来源占用预算 | 边界补全效果依赖上游候选排序，不能假定换检索器后仍无退步 |',
              '| med-004、045：两路并集 | 即使不截断、不设上下文预算仍不满足全部原始参考 | 本轮融合不能创造候选外证据；保留候选缺失结论，不据此宣称整个知识库无答案 |',
              '| med-090：BM25、Dense及两种融合 | 核心截止日期已有证据，但参考中额外背景仍不全；原生Hybrid还缺少另一段原始参考 | 严格长片段指标与核心答案可答性有区别，不能由一组推断所有组 |', '',
              '完整的59条不完整上下文记录、缺失参考索引及候选缺失/融合截断/上下文装入诊断保存在[analysis.json](analysis.json)。该诊断在实验后生成，未反向修改排序或标签。', '',
              '| 完整证据前缀变化：分数融合 → RRF | 原名次 | RRF名次 |',
              '| --- | ---: | ---: |']
    for row in analysis['evidence_prefix_changes']['rrf_vs_weighted']:
        lines.append(f"| {row['question_id']} | {row['baseline_prefix']} | {row['treatment_prefix']} |")
    lines += ['', '## 不确定性、费用和决策', '',
              '51题只有6个政策组，并且此前已多次用于开发。RRF与分数融合每题的完整覆盖结果完全相同，因此配对bootstrap区间退化为[0,0]；这不证明两种算法在其他数据上等效。融合＋boundary相对Dense＋boundary为−1.96个百分点，政策组bootstrap探索区间[−11.11,0]个百分点，不作泛化或显著性结论。', '',
              '加权基线只测试了预定的0.5权重和返回列表min-max，不能推断RRF优于所有分数归一化或加权算法。缺席项没有补算另一路分数。索引的词频统计来自共享索引，返回范围固定在1000篇；该统计范围在两种融合之间保持一致。', '',
              'qrels仍不完整，未审核来源不能作为假阳性。boundary中已审核不支持来源Token占比：Dense 2.50%、分数融合1.48%、RRF1.77%；这仅是有限人工审核下的已知部分，不能宣称全局噪声率下降。', '',
              f"本轮没有云端模型调用。BM25本地平均{analysis['local_bm25_mean_ms']:.1f}ms，中位数{analysis['local_bm25_median_ms']:.1f}ms，平均数包含首次分词预热；融合纯计算另存timings.jsonl。未同期重跑Dense，不能用这些值声称端到端加速或生产延迟改善。", '',
              '**当前决策：保留Dense＋boundary作为上下文主候选，RRF作为已实现、未采纳的实验选项。** 现有短编号引用方案继续保持。下一步如继续比较核心排序算法，应对同一份候选做“无重排 / 独立Rerank”对照，固定上下文规则；随后冻结整体方案并准备正式测试，避免继续在同一开发集扫参。', '',
              '## 复现与核验', '',
              f"24项针对性单元测试通过；另用不导入生产融合/覆盖函数的脚本核验{validation['fusion_union_scores_verified']}个并集融合分数、{validation['rankings_verified']}条排名、{validation['context_rows_verified']}条上下文和{validation['independent_reference_interval_checks']}次原文字符区间覆盖。{validation['unchanged_control_rows_verified']}条Dense/原生Hybrid控制记录与此前实验完全相同。新进程重放的排名、选择、逐题指标和汇总逐字节一致；计时文件仅作运行记录，不要求重放一致。", '',
              '索引UUID、文档数、主分片序列号与旧快照一致。配置、协议及执行代码在BM25请求前冻结，两路候选在评分前冻结；先写出所有选择，再加载金标。', '',
              '从仓库根目录执行，输出目录需不存在；下面的重放不需要Docker或模型密钥：', '',
              '```bash', 'python3 -m experiments.evaluation.fusion_experiment evaluate \\', '  --output /private/tmp/rrf-fusion-new-replay',
              'python3 -m unittest experiments.retrieval.test_fusion \\', '  experiments.evaluation.test_fusion_experiment \\', '  experiments.evaluation.test_context_completion \\', '  experiments.evaluation.test_context_experiment', '```', '',
              '第一次检索的prepare和retrieve已经运行；它们拒绝覆盖冻结文件，复现结果应使用evaluate。详细协议见仓库docs/rrf_fusion_protocol_v1.md。', '',
              '## 面试追问', '',
              '1. 为什么BM25和cosine不能不经处理就直接加权，min-max仍有什么局限？',
              '2. RRF为什么可能奖励两路都找到的片段，又压低只有Dense找到的必要证据？',
              '3. 为什么MRR提高，固定Token预算内的完整证据覆盖仍可能下降？',
              '4. 为什么原生Hybrid只能做系统参考，而分数融合和RRF才是本轮严格对照？', '',
              '面试可陈述：“我冻结BM25和Dense候选，比较分数融合与RRF，并分别验证排名、候选截断和上下文完整性。发现RRF虽改善部分排序，但截断了低排的必要背景证据，最终未超过Dense＋边界补全，因此保留原方案。”不要把本轮写成RRF带来的问答准确率提升。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args()
    with (args.results/'report.md').open('x') as stream:
        stream.write(render(args.results))


if __name__=='__main__':
    main()
