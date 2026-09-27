# 医保政策问答入口与多子问回归

后续进度：[99题检索与上下文测试](<heldout_retrieval_evaluation.md>)已完成，生成评价仍为Not Run。默认入口和以下10题生成结果保持原样；没有因为测试集Hybrid表现更好就更换默认方案。按用户要求不加pass@3。

2026-09-22：已接通可运行的命令行问答流程，并完成一次真实端到端验证。默认方案保持 **Dense64 → 2560参考Token边界补全 → 短编号引用 → qwen3.7-flash → 引用校验**。本轮尝试的逐子问结构化输出未达到预先约定的采用条件，未设为默认。

## 现在如何运行

在仓库根目录、现有Docker服务运行时执行。Python使用标准库，服务依赖和已配置的模型连接在现有RAGFlow容器中读取；不用重新填模型密钥。

```bash
# 真实查询：一次问题向量化流程，再生成一次回答。
# 输出目录必须不存在；正常小规模API调用已获用户授权。
python3 -m experiments.qa \
  --question '徐州从2022年起，普通居民和大学生每年的居民医保个人缴费各是多少？' \
  --output /private/tmp/policy-qa-example \
  --allow-model-api

# 不调用模型：复用一条已存开发排名，重做上下文选择和Prompt准备。
python3 -m experiments.qa \
  --replay-question med-116 --prepare-only \
  --output /private/tmp/policy-qa-prepared-example

# 复用旧检索结果，仅重新生成一次回答。
python3 -m experiments.qa \
  --replay-question med-116 \
  --output /private/tmp/policy-qa-generation-example \
  --allow-model-api
```

真实运行保存 `retrieval.json`、`request.json`、`response.json`、`answer.md`、`manifest.json`，可以追踪候选排名、选中片段、E编号映射、模型原始输出、引文检查与用量。`--prepare-only`不生成答案；如果同时使用`--question`，实时检索仍需要问题Embedding。完全不调用模型时请使用`--replay-question`。

这是基于现有RAGFlow知识库与配置的独立命令行入口。纯Dense仍通过已验证的本地ES KNN适配器实现；没有改动RAGFlow网页聊天界面，也未声称网页现在运行了此流程。

## 本轮具体做了什么

1. **连接原来分散的模块。** 输入问题后，固定1000篇范围召回64个候选；从本地3205块快照补充正文、顺序与Token数；使用原来的boundary规则选择上下文；再生成和检查引用。检索排序与补块规则没有变化。
2. **把引用检查接到回答展示前。** E编号必须属于本次请求，引用原句须在对应脱敏片段中规范化匹配。无法定位或拼接改写的引文会阻止整份回答展示，并保留原始结果。此检查只证明文字存在，不证明语义正确、地区年份适用或没有漏答。
3. **检验一种生成侧候选。** 在同一次生成中，要求模型先从问题列出子问，每项分别输出回答、claims、证据和不足标志，本地按项展示。没有用金标拆题，没有额外Agent、分解调用或修复调用。该候选只保留在冻结实验中用于复现，默认入口继续使用已有短编号格式。

## 固定10题同期对照

与前轮相同的10道开发诊断题、相同45个脱敏片段和顺序、相同模型和解码参数。控制组重新生成；两组只改变系统提示和输出结构。沿用此前冻结的18个评分要点。语义评分由助手逐题复核，尚无独立人工盲审；这是定向开发诊断，不能当作全库或99题测试准确率。

| 指标 | 现有短编号回答 | 逐子问结构化回答 |
| --- | ---: | ---: |
| 主回答全部要点正确 | 9/10 | 8/10 |
| 完整且有上下文及引用支持 | 9/10 | 6/10 |
| 正确答案要点 | 17/18 | 16/18 |
| 回答忠实于上下文 | 10/10 | 8/10 |
| 编号及原句均有效的引用 | 23/23 | 23/28 |
| 因错误引文不能展示的回答 | 0/10 | 2/10 |
| 供应商报告总Token | 18,719 | 20,416 |
| 平均生成耗时 | 3.46秒 | 5.26秒 |

20次请求全部返回有效JSON，无重试。表中“请求成功”不等于“答案通过”。成对实验共39,135个生成Token，没有新增Embedding、Rerank或测试题请求。这里的耗时是模型请求耗时，各10次，不能据此推广为稳定速度差异。

完整答案：med-116改善，med-026、029退步，其余7题不变。完整有据回答：没有新增通过题，med-001、026、029退步，med-116两组都未通过但原因不同。按调用前协议，候选未达到采用条件，保留原模式，不追加调提示来筛选较好的结果。

## 四个需要解释的案例

- **med-116：补回漏答，但引文退步。** 控制组主回答漏掉普通参保职工生育待遇的资金来源，尽管claims中已写出这条事实。主回答仍按2/3要点评分。处理组把三问全答对，但第一项E1引文用中文省略号拼接。另有有效引文也不能抹掉这一错误；按既定的所有引用均须有效口径，两组都未完整有据通过。
- **med-029：引用存在，解释却错误。** 原文420元为不含长护险的医保缴费，440元才是加入20元后的合计。处理组误将420元理解为包含20元，并输出400元，还把440减20写成400。错误引文的文字本身确实存在，因此纯原句检查不能发现这类语义和计算问题。
- **med-026：发布年和缴费年度仍有歧义。** 处理组因问题里的2023年拒绝确认答案，错误描述E4没有年份，实际E4写了2024年缴费标准；四条引文还出现省略号拼接。控制组给出医疗救助基金全额资助的正确答案。本轮没有补充发布时间元数据，因此不能宣布年份问题已解决。
- **med-001：多写出的事实也要配证据。** 两个人群的标准均正确，处理组额外说明2024年标准有调整。所给上下文能够支持这句话，但其claims只覆盖2022年的两项金额，未给额外说明对应的引用归属，因此完整有据未通过。

## 真实串联验证

另外对med-116的问题文本执行一次实时查询，未读取该题金标。问题Embedding报告46 Token；一次生成报告2,051 Token，7条引文全部通过独立字面复核，助手检查三个答案要点均出现且有支持。该次结果独立保存，没有混入上述成对分数。

这次和同期控制使用相同选中片段及相同Prompt，但回答完整性不同。温度0、固定seed和非日期模型名称并不能保证逐字或语义结果完全一致；不将一次冒烟通过当成漏答已修复。实时64候选的排序与旧缓存并非逐位相同，但最终选中上下文相同。索引UUID及分片序列号在实时检索前后均通过冻结快照核验。

查询向量化约1.32秒，ES检索约0.15秒，生成约7.49秒；包括容器导入、客户端初始化、快照校验和写盘在内的命令行冷启动全过程约32.49秒。这是原型入口的单次测量，尚未做常驻服务或并发性能优化。

本阶段合计21次生成，供应商报告41,186个生成Token，另有46个查询Embedding Token；没有文档Embedding、Rerank或正式测试题调用。查询Embedding适配器沿用现有传输实现，记录的是一次向量化流程，不把它冒充底层HTTP尝试数审计。

## 当前边界与下一步

云端生成仍按此前逐块审核的52块清单及原文/脱敏摘要放行，本轮实际使用45块。任何选中块未审核或内容变化都会中止，不会悄悄删块继续生成；来源79870始终禁用。因此，虽然检索范围是1000篇，当前入口的云端回答能力并不覆盖任意检索结果。索引更新也会触发拒绝，需要重新制作并审核快照。这是明确的演示与开发范围限制。

多子问漏答、年份信息歧义和原句存在但理解错误仍存在。99题检索测试现已完成；下一步对Dense＋boundary与Hybrid＋ranked已选上下文完成出站处理，并在生成前冻结答案要点，再按既定方案各生成一次。已有清单还缺399个唯一选中片段的审核，不通过删块或选题绕开。不能仅凭目前10题宣称系统达到90%准确率；未构建真实无答案题集，也不能宣称拒答能力经过系统评测。

## 文件与复现

- [统一入口](<../experiments/qa.py>)与默认配置（本地归档：`experiments/configs/policy_qa_v1.json`，未收入本备份）。
- [调用前协议](<policy_qa_protocol_v1.md>)、冻结输入与代码（本地归档：`experiments/datasets/private/policy_qa_v1/freeze.json`，未收入本备份）。
- 完整逐题报告（本地归档：`experiments/results/private/policy-qa-dev10-v1/evaluation/report.md`，未收入本备份）、原始响应（本地归档：`experiments/results/private/policy-qa-dev10-v1/responses.jsonl`，未收入本备份）、逐题评分理由（本地归档：`experiments/results/private/policy-qa-dev10-v1/assessments.json`，未收入本备份）。
- 独立核验与成对分析（本地归档：`experiments/results/private/policy-qa-dev10-v1/analysis.json`，未收入本备份）。39项针对性测试通过；独立复核51条对照引文、7条实时引文、20份上下文与10份控制Prompt。
- 真实问答示例（本地归档：`experiments/results/private/policy-qa-live-smoke-v1/answer.md`，未收入本备份）与运行清单（本地归档：`experiments/results/private/policy-qa-live-smoke-v1/manifest.json`，未收入本备份）。

离线评分复现不调用模型，输出目录须不存在：

```bash
python3 -m experiments.evaluation.generation_report \
  --result experiments/results/private/policy-qa-dev10-v1 \
  --prepared experiments/datasets/private/policy_qa_v1 \
  --assessments experiments/results/private/policy-qa-dev10-v1/assessments.json \
  --output /private/tmp/policy-qa-report-example

python3 experiments/results/private/policy-qa-dev10-v1/audit_saved_results.py \
  --output /private/tmp/policy-qa-audit-example.json
```

已在新进程重放评分，summary、scored、manifest、report四份文件逐字节一致。

面试追问：

1. 为什么逐子问回答改善一道漏答题，却不值得作为整体默认方案？
2. 为什么“引文在原文中存在”检测不出420元被理解成400元的问题？
3. 为什么同一Prompt在temperature=0下的一次完整回答，不能证明漏答已经修复？
