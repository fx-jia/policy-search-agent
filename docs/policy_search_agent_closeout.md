# 医保政策 Search Agent 收尾清单

本文汇总算法实验的交付与证据边界。已有原生基线、独立控制器、开发迭代、冻结测试、逐题轨迹和辅助复核；公开目录保留选定报告、结果表和完整性摘要，完整原始轨迹未公开。

项目定位：**医保政策证据驱动问答系统（RAG + Search Agent）**，目前为本地原型。当前有 Python 单题、批量及多轮入口，已实现医保助手页面/API，另完成Fast RAG和轻量路由离线验证。在线Adaptive路由与外网搜索尚未接入；真实新问答、上海两轮自然更新和跨环境部署尚未验收。运行依赖完整 RAGFlow、知识库索引、来源映射和部署配置，详见[医保助手运行说明](<medical_assistant_frontend.md>)。

|交付项|当前状态|收尾动作|
|---|---|---|
|固定语料与索引|1000篇、3205 Chunk，原 Embedding|保留版本映射；本轮不重新入库|
|Benchmark|330题：300核心题、30边界题；核心205开发/95测试|说明用户浏览认可、定向审核与抽查范围，不宣称330题全部独立人工精标|
|原生基线|Native RAG、Native Low各95题已尝试|保留失败记录；重跑作为单独诊断阶段|
|独立控制器|v8.1 A+B，单题/批量入口及轨迹|冻结策略和源码，不再使用这95题调参|
|离线质量评分|277份首轮评分尝试，255有效、22缺失|原分保留，不把质量分叫正确率|
|辅助复核|53题×3方案及8份重跑，共167份|建议分另存；150题扣分强度保留争议说明|
|运行说明与索引|已更新为最终测试状态|按下面单题演示流程检查|
|演示验收|已完成1题现场问答、2份开发轨迹核验|见演示报告；保留部分拒答案例中的控制器误判，未宣称整体拒答能力已验收|
|Fast与路由探索|Fast v1/v2各尝试199道开发题；路由完成165题政策组交叉验证|保留质量与开销取舍；不采用v2，不宣称Adaptive已保质提速|

核心链路是原问题Hybrid初检，合并的需求提取与证据充分性判断，然后在限定动作内选择同块补读、邻块读取或定向补检，保留可核验原文并进行一次最终回答。底层复用RAGFlow检索、模型接口及引用流程；项目实现有AI辅助，自定义部分集中在证据窗口、搜索行动、停止条件、调用预算及评测隔离。

生成模型为 deepseek-v4-flash-0731，非思考；原索引与查询使用 qwen3.7-text-embedding。离线裁判为 deepseek-v4.1-flash。首版未增加最终答案二次生成或额外答案Judge。Evidence Judge的充分性只是运行控制信号，不是评测标签。

可使用的成果口径：

- 构建有来源原文、必答点、难度依据及政策组划分的Benchmark；核心205题用于开发、95题在冻结策略后测试，边界题单列。
- 实现带原文位置绑定的需求证据选择、同块/邻块补读和有界补检；记录动作、停止原因、引用、物理调用、Token及耗时。
- 冻结测试的同题有效评分中，v8.1相对Native Low平均质量分差为+13.977/100（77对）；相对Native RAG仅+0.016/100（77对，描述性区间跨0）。这是未经人工全面终审的模型评分。
- 三方案均成功生成的同一87题上，平均耗时RAG/v8.1/Low为6.622/9.001/25.810秒，P95为10.323/15.632/32.095秒。v8.1比Low平均耗时低约65.1%，但每题生成Token为8128，对照Low为6079，约高33.7%。
- 同87题查询Embedding观测调用为RAG 1次、v8.1 1次、Low 17次。本批缓存命中为0，调用减少来自流程差异，不能归功于缓存。

本项目的实验结论有明确边界：v8.1相对Low有质量评分和耗时优势，但没有证明相对原生RAG更优；质量与开销需要共同取舍。部分检索或子问复查尝试没有净收益，这些结果同样构成方案选择依据。当前不把它描述为原创检索算法、已上线业务平台，或全库3081篇上的准确率提升。

交付证据：

1. **资料归档**：以本文和两个README为入口，连接测试报告、复核侧表和关键案例。完整冻结配置未公开；原始成绩不变，150题的评分争议继续保留。
2. **单题演示（已完成）**：现有环境中1题真实运行成功，另核验开发题082的邻块补读和043的信息不足回答边界。现场与保存记录分开标注；见[演示与验收](<policy_search_agent_demo.md>)。
3. **业务集成**：多轮会话、上海政策业务链路及页面/API的实现与离线验证见[源码同步说明](<update-2026-10-15.md>)。这些改动晚于原95题实验，不能将旧成绩当作业务链路的验证结果。

后续效率探索与原95题冻结测试分开报告。Fast v1在165道三组共同有效开发题上，相对Native RAG平均快30.4%、生成Token少36.1%，质量低5.25分。证据路由相对题干路由有改善迹象，但仍未证明相对RAG保质提速。Fast v2在174道与v1有效配对题上平均低1.35分，未采用；15道退步中11道丢掉了原答案引用过的来源块，且全部15题均未截断。这些结果说明关键词覆盖不等于关键条款支持；探索性提速尚未成为已上线系统成果。

演示准备命令从仓库根目录执行，输出目录必须不存在：

```bash
python3 -m experiments.evaluation.search_agent_eval prepare \
  --question '按徐州2019年城镇职工医疗保险办法，职工医保的三个层次分别是什么？' \
  --connect-retries 2 \
  --output experiments/results/private/my-agent-demo
```

`prepare`只冻结配置，不调用模型。下列命令需要已部署的完整服务、模型和来源映射，并会产生真实模型调用：

```bash
python3 -m experiments.evaluation.search_agent_eval run \
  --output experiments/results/private/my-agent-demo --stage pilot
```

此题已于2026年9月27日真实运行成功：问答阶段8.235秒、生成7,054 Token、查询Embedding 34 Token；含容器初始化的命令耗时34.056秒。原始运行目录未公开，新运行须使用新的输出目录。此题用于演示链路，不能算新增泛化证据；一次成功演示也不能代替整套部署验收。

证据入口：

- [最终测试汇总](<../experiments/results/private/policy-search-agent-final-test95/final_summary.md>)
- [首轮指标](<../experiments/results/private/policy-search-agent-final-test95/metrics.json>)
- [技术失败单次重跑](<../experiments/results/private/policy-search-agent-final-test95/technical-retry-v1/report.md>)
- [AI辅助复核报告](<../experiments/results/private/policy-search-agent-final-test95/assistant-review-v1/report.md>)
- [冻结及用量核验](<../experiments/results/private/policy-search-agent-final-test95/final_integrity.json>)
- [控制器与运行说明](<../experiments/search_agent/README.md>)
- [三案例演示与验收](<policy_search_agent_demo.md>)
- [Fast v1开发报告](<../experiments/results/private/fast-rag-v1-dev199/report.md>)
- [轻量证据路由分析](<../experiments/results/private/evidence-router-v1-dev165/analysis.md>)
- [Fast v2未采用的原因](<../experiments/results/private/fast-rag-v2-dev199/analysis.md>)

公开报告保留模型评分、缺失评分、失败和负结果。未公开的请求、响应、运行配置和账本在报告中注明；这些摘要不能替代完整原始材料的独立复算。
