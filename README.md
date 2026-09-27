# 江苏医保政策 Search Agent

基于 RAGFlow 的医保政策问答研究项目，包含独立有界 Search Agent、Benchmark、配对评测、Fast RAG 与轻量路由实验。本仓库是 **2026-09-27 整理的私有项目备份**，用于保存代码、实验依据和面试材料。

## 阅读顺序

1. [项目导学](导学-医保Agent.md)：系统职责、RAGFlow 检索与上下文链路、独立 Agent 实现。
2. [核心实现](experiments/search_agent/)：控制器、证据窗口、工具适配、预算、调用记录。
3. [最终 95 题报告](experiments/results/private/policy-search-agent-final-test95/final_summary.md)：主要结论与限制。
4. [Fast RAG 和路由研究](docs/policy_fast_rag.md)：速度与质量取舍。
5. [面试材料](面经-医保Agent.md)、[结论证据账本](docs/policy_search_agent_claims.json)。

## 做了什么

固定 1,000 篇政策文档、3,205 个 Chunk，复用 RAGFlow 的 Hybrid 检索、模型接口及引用处理，自行控制需求分析、证据选择、同块补读、相邻条款读取和定向补检。最多两次补充动作、三次控制决策，最后一次生成完整回答；记录停止原因、实际调用、Token 与耗时。

生成模型为 `deepseek-v4-flash-0731`，非思考模式；索引与查询使用 `qwen3.7-text-embedding`，评分模型为 `deepseek-v4.1-flash`。最终实验保留 Native RAG、Native Low 同题对照，没有运行 Medium/High 作为最终对照。

## 有证据支持的结果

质量分为 0–100 的模型评分，**不是正确率**；只对双方均有有效评分的题目计算分差。

| 最终测试比较 | 有效配对 | 平均质量分差 | 结论 |
| --- | ---: | ---: | --- |
| v8.1 − Native Low | 77 | +13.977 | 本批评分支持优于 Low |
| v8.1 − Native RAG | 77 | +0.016 | 未证明优于 Native RAG |

三方案都成功生成的同 87 题中，RAG / v8.1 / Low 平均耗时为 6.622 / 9.001 / 25.810 秒，每题生成 Token 为 3,886.4 / 8,127.7 / 6,078.7。v8.1 相对 Low 平均耗时下降约 65.1%，但生成 Token 增加约 33.7%。本批没有实际缓存命中，不能把减少调用归功于缓存。

Fast RAG 在开发集观察到速度优势及质量损失；轻量路由完成了离线比较，尚未证明在线系统能稳定保质提速。Fast v2 没有采纳为最终方案。报告保留失败结果、缺失评分和未知用量；技术重跑与首轮成绩分开。

## 目录

| 路径 | 内容 |
| --- | --- |
| `experiments/search_agent/` | 当前 Agent、Fast RAG、路由及离线单测 |
| `experiments/evaluation/` | 运行隔离、评测、评分、报告与预算守卫 |
| `experiments/retrieval/` | 早期检索、融合与重排实验 |
| `experiments/datasets/` | 题库制作、入库与划分辅助程序 |
| `experiments/datasets/private/benchmark_v2_full330/` | 300 道核心题、30 道边界题，问题、答案、原文与难度依据 |
| `experiments/results/private/benchmark-v2-baselines-v2-nonthinking/` | 实际冻结的 205 道开发题、95 道测试题及划分审计 |
| `experiments/results/private/` | 关键批次的报告、数值表、协议与完整性记录 |
| `snapshots/` | 从冻结配置提取的 v7、最终 v8.1、Fast v1/v2 精确源码和安全参数摘要 |
| `docs/` | 方法说明、收尾结论、演示说明与证据账本 |
| `BACKUP_MANIFEST.json` | 导出来源、原件与副本 SHA-256、处理方式及排除范围 |

题库中的 `private` 是原目录名称，不代表其中包含密钥。题库由模型辅助构造，经过用户浏览及针对性复核，不能描述为 330 道全部双人精标。原始制作文件记录的是当时状态；实际开发/测试划分以冻结目录为准。这 95 道测试题已用于最终评测，今后根据其结果设计策略时不能再次声称是未见测试集。

## 离线验证

使用 Python 3.13，在仓库根目录执行：

```bash
python -m pip install -r requirements-offline.txt
python scripts/test_offline.py
python scripts/verify_backup.py
```

离线套件覆盖 Agent 控制、证据来源、上下文预算、缓存、停止条件及主要评测报告逻辑，不访问真实模型或数据库。一个依赖完整本地冻结配置的 Fast 上下文审计测试明确排除，原测试源码仍保留；它不是自包含单元测试。

## 恢复真实实验前必须补齐的内容

本仓库不是整套可直接启动的 RAGFlow 部署。需要另行恢复对应 RAGFlow 容器/服务、同一向量模型生成的知识库索引、文档和 Chunk 映射、原始安全审核材料及本地模型配置。服务凭据通过本地安全配置提供。

`snapshots/*/protocol-summary.json` 保存原生运行文件哈希和冻结参数摘要，可用于核验版本；它不是可直接执行的完整配置。`experiments/configs/` 中的文件属于早期检索示例，不能替代最终冻结协议。

原始请求/响应、完整用量账本、原始全量语料、向量数据库、嵌入原文的运行配置和容器卷留在原机器，未上传。报告脚本依赖这些材料时，不能仅凭精简备份重新计算全部结果。重新生成冻结配置与恢复原批次是不同操作，不应覆盖原实验成绩。

## 备份处理与归属

Python 源码及选中的 JSON/CSV 原件按字节保留；Markdown 副本把本机链接转为仓库内相对链接，未收入的本机材料明确标记。历史记录内部的哈希仍指向原件；副本哈希以备份清单为准。`snapshots/` 仅用于历史审计，不是另一套运行入口。

RAGFlow 是底层开源框架，原生检索和生成能力不能算作本项目自行实现。保留其 [Apache-2.0 许可证](LICENSE)用于所包含的上游代码/摘录；政策原文及第三方材料的权利归相应来源，本备份不额外授权这些材料。来源见 [第三方与版本说明](THIRD_PARTY.md)。
