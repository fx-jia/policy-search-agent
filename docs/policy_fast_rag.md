# Fast RAG 首轮开发实验

目的：检验能否通过减少输入上下文，降低回答耗时与 Token，同时为后续本地路由保留可接受的质量。先测两条路径的取舍，再决定是否训练 Router。

首轮已完成：[实验报告](<../experiments/results/private/fast-rag-v1-dev199/report.md>)、[本地路由验证](<../experiments/results/private/fast-rag-v1-dev199/router_report.md>)。165道三组有效配对中，Fast平均3.89秒、2417 Token，比Native RAG约快30%、少36% Token；质量均分92.83，Native RAG98.08。纯问题文本规则/分类器尚未同时达到提速与保持质量，不接入线上。

后续[证据路由验证](<../experiments/results/private/evidence-router-v1-dev165/analysis.md>)也已完成：利用已有初检8块与保留3块的词项/条件覆盖差，在相同165题及政策组5折上复核。阈值0.8时，与旧题干路由同为137题走Fast，质量均分从93.62变为96.62，平均4.57秒；仍低于Native RAG的98.08分。证据特征加分类约0.9毫秒，不增加模型调用；尚未证明保质提速、未上线。所有阈值及不确定性见该报告，不将开发回放当作最终测试。

Fast v2 已完成同一199题实验：[结果与失败分析](<../experiments/results/private/fast-rag-v2-dev199/analysis.md>)。194题生成成功、187份评分有效；四组共同156题中，v2为92.21分/4.24秒，v1为94.82分/3.83秒，Native RAG为98.63分/5.55秒。仍有相对Native RAG的速度优势，但本次重排未产生正收益，不替换现有快路径。它按问题相关性和不同需求的新增覆盖，在原8个候选中选择最多3个来源；整块放不下时保留完整编号条款及其实际所属章/节，普通文本保留命中句及邻句。总预算仍为2000 Token，没有控制器或补检。旧代码和旧结果作为冻结实验依据保留；新选择器为 `experiments/search_agent/fast_context.py`，评测入口为 `experiments.evaluation.fast_context_eval`。主要退步来自替换掉原有有效证据，详细案例及技术失败单列在报告中。

Fast 复用固定1000篇知识库、原向量模型、Hybrid 初检8块、deepseek-v4-flash-0731非思考模式和原生回答/引用模板。仅将最终上下文改为排名前3个去重块、最多2000 Token（计入来源标题和分隔符）。完整块放不下时，保留从块开头到完整句末行的原文前缀；软换行不会单独成为截断依据。没有 Evidence Judge、补检或子问生成。最终一次生成；仅确认未发送的连接失败允许有限重连。

代码入口：`experiments.search_agent.fast.run(question, backend, limits)`。它不读取参考答案和题目难度。真实调用仍经过原有容器隔离、出站脱敏、原文身份验证、实际生成请求检查和用量账本。

本轮固定199道开发题（Easy39、Medium120、Hard40），复用同题Native RAG和v8.1答案；不运行Low/Medium/High，不使用95道已曝光测试题调参。统一使用deepseek-v4.1-flash匿名质量量表：144题旧评分复用，早期55题的RAG/v8.1旧答案重新评分。评分是质量分，不是正确率。技术失败、缺分和裁判争议单列。

结果目录：`experiments/results/private/fast-rag-v1-dev199/`。其中 `generation/records.jsonl` 为实际生成记录，`quality/records.jsonl` 为本次评分，`report.md`、`metrics.json`、`paired_scores.csv`、`all_answers.md`、`review_queue.json` 在评分结束后生成。进度以实际记录为准，`progress.json` 只表示阶段。

```bash
# 新实验须指定新的输出目录；已有冻结目录不能重新 prepare。
python3 -m experiments.evaluation.fast_rag_eval prepare --output experiments/results/private/fast-rag-NEW
python3 -m experiments.evaluation.fast_rag_eval run --output experiments/results/private/fast-rag-NEW
python3 -m experiments.evaluation.fast_rag_report --output experiments/results/private/fast-rag-NEW
```

上述 run 会先在同一冻结批次检查3题，再继续199题和匿名评分；3题结果不会丢弃或另取最好答案。若未知用量、运行中断或评分停止，先检查日志和账本，不直接反复启动。新目录命令仅用于有授权的新批次，不是对本轮的重复执行指令。

路由可行性先看事后上限：假设知道 Fast/v8.1 两份答案的真实质量，仅在 Fast 分数不低于v8.1时走Fast，其余走Agent。该指标使用了事后答案，不能作为上线成绩。只有它显示出有用的质量/成本空间，才在开发题按政策组分折验证无额外模型调用的规则或小型分类器。最终需要新的独立验证，历史基线用时也不等于同期随机对照。
