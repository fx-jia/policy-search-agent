# v8.1 剩余开发题回归

用户于本轮明确指出开发集可以复用，并授权补跑新版本未运行的开发题。核心benchmark为开发205题、测试95题；另有30道边界题（开发18、测试11、跨划分诊断1）。本批不使用测试题或边界题。

冻结的v8.1此前完成6道诊断题和55道扩展题，合计61题。本批按“所有未尝试题”选出144题，Easy39、Medium68、Hard37，不根据旧版本成败筛题。此前55题为Medium52/Hard3，不能单独代表全开发集难度分布。

策略、原文窗口、Gap Contract、生成模型deepseek-v4-flash-0731、原Embedding qwen3.7-text-embedding、1000篇索引和检索参数全部沿用已执行的v8.1冻结源码。每题60秒、最多4次生成物理调用（包括获准的有限连接重试）、最多3次查询Embedding、2次补充动作。串行执行，避免干扰本地检索服务。单题60k Token保护仍保留，整批额度上限按144题预留；实际用量单独记账。不会从失败的多次尝试中挑最好结果。

启动命令：

```bash
PYTHONPATH=. python3 -m experiments.evaluation.search_agent_eval run --output experiments/results/private/policy-search-agent-v8_1-dev144 --stage full
```

执行位置为本机宿主监督进程加本地RAGFlow隔离容器，生成和Embedding使用用户已授权的云端服务，无GPU训练任务。原始输出为records.jsonl，逐次调用及事件留在本目录，不修改原生RAG/Low入口。

历史RAG、Low、v7均有同题答案，可用于开发回归配对。已核对模型、索引、检索配置和共同记录的运行代码哈希。最早30题RAG/Low没有记录Embedding适配器哈希，须保留这一限制；历史结果也不是同一时刻的随机对照。质量评分安排另见answer-quality-dev144-flash，其评分调用不计入Agent的推理开销。

这是已接触开发分布上的回归与版本比较，不是独立泛化证明。新增通过与退步均须保留，未知替代引用进入人工复核。冻结策略后的95道核心测试题继续保留。

本批144次生成已全部完成且status=ok；这仅表示技术成功，不等于内容全对。平均7.65秒，P95为12.46秒，生成Token 1,141,273，查询Embedding实际144次、向量缓存命中0次。每题仅一次正式尝试，所有生成用量已结算；策略、运行输入及95道测试集隔离核验通过，详见integrity_and_costs.json。质量评分已在answer-quality-dev144-flash独立启动。
