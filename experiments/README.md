# 医保政策问答与 Search Agent 实验

当前阶段：原v8.1最终95题测试与辅助复核已完成；[Fast RAG效率实验](<../docs/policy_fast_rag.md>)也已完成199道开发题与本地路由验证，观察到提速与质量下降的取舍，暂未上线路由。原交付见[收尾清单](<../docs/policy_search_agent_closeout.md>)。

新增[证据感知路由开发验证](<results/private/evidence-router-v1-dev165/analysis.md>)：复用165道有效配对和相同政策组分折，证据信号改善了相对题干路由的取舍，但尚未证实相对Native RAG保质提速；无新增模型/Embedding调用。

[Fast v2 上下文选择实验](<results/private/fast-rag-v2-dev199/analysis.md>)已完成199题生成与194份新答案评分。词项覆盖重排没有带来正收益；已保留负结果、来源变化审计和15题回归复核材料，不替换Fast v1。

当前系统复用RAGFlow固定1000篇/3205 Chunk、Hybrid检索与原生生成；独立v8.1控制器负责需求证据判断、原文窗口、有限补读/补检和预算控制。生成使用deepseek-v4-flash-0731非思考模式，Embedding沿用qwen3.7-text-embedding。

Benchmark包含300道核心题和30道边界题；核心按政策组划分为205开发题、95测试题。正式基线为Native RAG和Native Low。质量评分由deepseek-v4.1-flash按固定五维量表完成；运行时控制器只接触用户问题和实际证据，不接触参考答案和评测标签。

|入口|内容|
|---|---|
|[控制器说明](<search_agent/README.md>)|架构、单题/批量命令、回放、预算与边界|
|[最终测试](<results/private/policy-search-agent-final-test95/final_summary.md>)|同题配对分差、成本、技术失败与报告索引|
|[辅助复核](<results/private/policy-search-agent-final-test95/assistant-review-v1/report.md>)|53题三方案及8份重跑的逐题建议，原评分保留|
|[收尾清单](<../docs/policy_search_agent_closeout.md>)|项目定位、成果口径、演示和面试准备|
|[演示与验收](<../docs/policy_search_agent_demo.md>)|1题现场问答与2份开发轨迹；答案、引用、补读过程、用量及已知局限|
|简历与面试（已从本仓库移除，保留原文档引用说明）|项目介绍、候选要点、15个主问及6个追问；配套源码导学（已从本仓库移除，保留原文档引用说明）|
|[Benchmark构建](<datasets/private/benchmark_v2_full330/benchmark_overview.md>)|330题问题、原文、答案和难度依据|
|[冻结划分](<results/private/benchmark-v2-baselines-v2-nonthinking/frozen.json>)|当前开发/测试/边界划分及摘要|

v8.1在77对有效测试评分中比Low高13.977分，比RAG高0.016分；不能由此宣称全面优于RAG。成本用三方案均成功生成的同87题对比，v8.1平均耗时9.001秒，Low25.810秒，RAG6.622秒；v8.1生成Token更多。质量分不是正确率，辅助复核不覆盖原始成绩。

源码职责：

- `search_agent/`：不依赖Benchmark标签的控制器、证据处理、RAGFlow适配与调用记录。
- `evaluation/search_agent_eval.py`：单题准备、冻结源码、隔离运行和批量调度。
- `evaluation/benchmark_v2_baselines.py`：原生RAG/Low评测入口。
- `evaluation/answer_quality*.py`：匿名评分、原文摘录验证与报告。
- `datasets/`、`retrieval/`：数据与索引映射、数据划分、检索及上下文实验。
- `results/private/`：原始输出、配置、调用账本、逐题复核与各批报告。

所有新运行使用新目录。已冻结测试策略不因测试表现继续调整；技术失败重跑单列，不挑最好答案。未知调用用量保留预留，不记0。缓存收益只按实际命中统计。

已有Dense/Hybrid、RRF、Rerank和子问复查的研究记录分别见[检索测试](<../docs/heldout_retrieval_evaluation.md>)、[RRF](<../docs/rrf_fusion_evaluation.md>)、[重排](<../docs/independent_rerank_evaluation.md>)和[子问扩验](<../docs/coverage_extension_evaluation.md>)。这些批次的模型、题集与指标不同，不与当前95题成绩混算。旧150题及51/99划分不等于当前330题Benchmark。

当前是本地可评测原型；网页、Router、外网搜索、多轮记忆及跨环境部署验收尚未完成。
