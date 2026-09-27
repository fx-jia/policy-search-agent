# 医保 Search Agent

独立、可测试的有界检索控制器。复用已安装 RAGFlow 的 Hybrid 检索与原生 RAG 最终生成，固定1000篇语料。没有改动原生 RAG/Low 基线，也不运行 High。

## 工作方式

原问题初检 → 合并的需求拆分/证据判断 → 最多两次同块补读、邻块读取或定向补检 → 按原文选择上下文 → 一次原生生成。

- `core.py`：`async run(question, backend, AgentLimits())`，不导入 benchmark、模型服务或数据库。
- `evidence.py`：可见原文窗口、片段编号及源偏移绑定、预算选择；每个引用保持文档/版本身份。
- `adapter.py`、`runtime.py`：RAGFlow 检索、3205块审核映射、邻块顺序、生成回放和真实出站上下文核验。
- `cache.py`、`embedding.py`、`transport.py`：请求内复用、物理调用计数、禁重试和未知用量保留。

默认限制：最多两次补充动作、三次决策、一次答案、四次生成HTTP调用、三次查询Embedding，60秒总时限/40秒检索决策，60000 Token/题。生成输出上限1200。普通RAG回退引用可能需要额外的引用Embedding，单独计数，不混入查询调用。

默认模型 `deepseek-v4-flash-0731`，非思考；查询与文档向量均使用原 `qwen3.7-text-embedding`。不得只换查询模型后沿用旧向量索引。

## 使用

从项目根目录运行：

```bash
# 本地无服务测试
python3 -m unittest experiments.search_agent.test_core experiments.search_agent.test_adapter experiments.search_agent.test_runtime experiments.search_agent.test_embedding experiments.search_agent.test_decision_contract experiments.evaluation.test_search_agent_eval experiments.evaluation.test_search_agent_report -q

# 创建不可覆盖的批次；prepare 不调用模型
python3 -m experiments.evaluation.search_agent_eval prepare --output experiments/results/private/my-agent-run

# 真实服务：先6题，链路通过后补齐同版本30题
python3 -m experiments.evaluation.search_agent_eval run --output experiments/results/private/my-agent-run --stage pilot
python3 -m experiments.evaluation.search_agent_eval run --output experiments/results/private/my-agent-run --stage full

# 无模型/无Embedding的原生请求回放
python3 -m experiments.evaluation.search_agent_eval prepare --replay --output experiments/results/private/my-agent-replay
python3 -m experiments.evaluation.search_agent_eval run --output experiments/results/private/my-agent-replay --stage replay

# 单题入口，不读题库评分字段
python3 -m experiments.evaluation.search_agent_eval prepare --question '按库内文件说明某项待遇的适用条件' --output experiments/results/private/my-single-question
python3 -m experiments.evaluation.search_agent_eval run --output experiments/results/private/my-single-question --stage pilot

# 离线报告；内容评分放在该批次assessments.jsonl，缺失评分保持待评
python3 -m experiments.evaluation.search_agent_report --run experiments/results/private/my-agent-run
```

运行需要当前已登录并配置好的本地 RAGFlow 容器，不在命令行或文件中保存模型密钥。所有真实调用均读取已有配置。运行时执行config中冻结且校验过哈希的模块源码；当前目录的源码修改不影响已冻结策略。修改策略需另建版本并保留旧账本，不能选择性重跑取最好答案。仅宿主调度修复可续跑原策略，并逐次记录宿主代码哈希；语料与基线文件仍需逐项校验。报告与评分在独立入口，不向控制器传题号、金标、难度或参考答案。

控制器请求显式使用 JSON 输出模式。全局action位于根层，各需求只选择本轮展示的evidence_ids；代码从原文恢复摘录和位置，不接受模型自由抄写。supported表示足以回答，包括有依据给出否定结论。

非法模型决策保留controller_error，剩余时间至少5秒时可以选择初检原始上下文进行一次原生RAG降级。仍受原调用/Token/时间限制；final发送前检查账本与未知用量，来源或传输异常、预算中止及超时不触发降级。记录fallback_used和是否成功产出答案，不做额外修复生成。`AgentLimits(fallback_on_invalid_decision=False)`可关闭降级。

诊断后另建批次时，给 `prepare` 添加 `--prior-run 旧批次目录`（可重复指定），将所有旧批次已结算生成 Token、未知预留和诊断调用从150k/450k总预算扣除。未知用量不回填为0，旧结果不会拼入新版本成绩。已获准直接跑30题时只校验剩余full预算，不因已完成的pilot预算用尽而阻断；full预算耗尽仍停止。

单题失败后不重跑该题，终止对应隔离进程，宿主扣除已结算用量和全部未知调用预留，再启动下一题。续跑沿用同一结果目录，已尝试题永久跳过。未知调用保持未知，不以0回填；它影响可用预算但不单独阻断其余题。批次预算、源文件校验失败、宿主超时或用量超过预留仍停止整批。6题均已尝试、至少5题正常且唯一异常为生成用量未知时，可扩展30题；算法决策异常仍需先检查。

2026-09-24用户批准有限连接重试，并明确完整30题同题对照目标。新批次可用 `prepare --connect-retries 2 --direct-dev30` 冻结协议，再 `run --stage full`。仅允许httpx连接错误且原因是aiohttp连接器错误，或SSL EOF堆栈证明确实发生在TLS握手阶段的重连；响应读取失败、HTTP错误和不明状态均不重试。最多2次重连，仍纳入每题4次物理生成/3次查询Embedding/60秒上限。默认参数仍为0；旧冻结批次不会自动改协议。

`--ablation no_search` 关闭补充工具动作；`--ablation native_context` 使用原生初检上下文。消融应另建冻结批次并记录额外预算，不自动扩跑。缓存只按实际命中报告收益；避开原生 compiled 扩展属于流程变化，不冒充缓存贡献。

## 已知边界

片段ID消除了抄写差异，但不能保证模型选对证据或理解正确。原文窗口未展示的内容不能被Judge确认为支持；降级也保留原生RAG的上下文限制。发现新但未标注的正确依据时，评测侧复核并统一补标。当前仅单轮政策问题，Router、网页、外网与会话澄清均未实现。

## v8.1：需求窗口和缺口行动

证据预算按内容相关程度共享，短块未用预算可分给长块；按需求选择原文，先保留已核验锚点。控制器使用的整条序列化请求（含提示词、来源及片段元数据）不超过7000 Token，证据上限仍4800；最终证据上限5600、最多8块。

`read_chunk` 从已见块补读尚未展示的原文，保留来源偏移，不调用Embedding；新增原文区间也算进展。相邻块只有尚未被观察时才作为可用方向。两种读取及定向搜索共享最多2次动作。

默认 `AgentLimits(gap_contract=True)`：Judge 按所问答案类型与地区/版本/对象检查充分性；可补救的缺口优先同块、可用邻块、定向检索。缺用户个人情况允许明确停止。`gap_contract=False` 为 A-only 消融，保留新窗口与工具，关闭 B 的追加提示和执行约束；冻结的 v7 对照继续从各实验配置中的源码运行。

`stop_kind` 位于决策JSON根层。控制器在下一次Judge前核算物理调用，包含已确认未发送的连接重试，为最终回答保留一个调用位；不会因此提高原上限。

最终回答仍只生成一次，未加入计算工具、额外答案Judge或Router。开发消融见[开发实验报告](<../results/private/policy-search-agent-v8_1-study/report.md>)，离线条款覆盖改善不能代替问答效果。A和A+B分别冻结评测，不拼接不同版本成绩。

当前v8.1 A+B已完成开发扩验及冻结95题测试。最终同题模型评分相对Native Low为+13.977/100（77对），相对Native RAG为+0.016/100（77对）；后者区间跨0，未证明质量优势。三方案都成功的同87题中，v8.1平均9.001秒、每题生成8128 Token，Low为25.810秒、6079 Token，RAG为6.622秒、3886 Token。质量分不是正确率，成本与质量需共同评估。

首轮三组各95次题目尝试、277份评分尝试，以及8个技术失败组合的单次重跑均已结束。疑问题的AI辅助复核独立存档，未覆盖正式原分。详见[最终汇总](<../results/private/policy-search-agent-final-test95/final_summary.md>)、[复核报告](<../results/private/policy-search-agent-final-test95/assistant-review-v1/report.md>)及[收尾清单](<../../docs/policy_search_agent_closeout.md>)。

2026-09-27完成本机单题演示：1题真实运行成功，并核验2份已有开发轨迹，展示邻块补读及信息不足时不编造总额。后者同时保留控制器误判的局限，不宣称拒答机制普遍可靠。运行数据、引用与逐步讲解见[演示与验收](<../../docs/policy_search_agent_demo.md>)。
