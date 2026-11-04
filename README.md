# 医保政策 Search Agent

基于 RAGFlow 的医保政策问答研究项目，包含独立有界 Search Agent、Benchmark、配对评测、Fast RAG 与轻量路由实验。本仓库公开当前集成源码和选定实验记录；历史评测与后续业务集成分别说明。

本次同步多轮会话与预算守卫、上海官方医保政策采集及不可变发布、原生调度验收工具，以及 `/medical-assistant` 的 API、页面和登录回跳。API、worker 和前端目录是适配现有 RAGFlow 工程的精选源码，不能单独启动整个系统。详见[本次更新及验证边界](docs/update-2026-10-15.md)、[医保助手运行说明](docs/medical_assistant_frontend.md)。

上海业务链路的完整线上发布、两轮自然更新和真实新问答尚未验收；历史 Search 会话的未知用量状态尚未核销。离线测试和前端构建通过不能替代这些结果。

## 阅读顺序

1. [本次源码更新](docs/update-2026-10-15.md)：当前功能、复用方式与验证边界。
2. [核心实现](experiments/search_agent/)：多轮会话、控制器、证据窗口、工具适配、预算和调用记录。
3. [最终 95 题报告](experiments/results/private/policy-search-agent-final-test95/final_summary.md)：主要结论与限制。
4. [Fast RAG 和路由研究](docs/policy_fast_rag.md)：速度与质量取舍。
5. [最终评测协议](experiments/results/private/policy-search-agent-final-test95/protocol.md)：比较范围、指标与实验约束。

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
| `api/`、`common/`、`rag/`、`deepdoc/` | 上海政策与助手业务模块、必要原生挂钩及 owning 源码；依赖完整 RAGFlow 工程 |
| `web/` | 医保助手页面、请求层、登录回跳和测试；应用到完整 RAGFlow 前端 |
| `conf/`、`docker/`、`test/` | 无凭据示例、部署 overlay 和离线边界测试 |
| `experiments/evaluation/` | 运行隔离、评测、评分、报告与预算守卫 |
| `experiments/retrieval/` | 早期检索、融合与重排实验 |
| `experiments/datasets/` | 题库制作、入库与划分辅助程序 |
| `experiments/datasets/private/benchmark_v2_full330/` | 300 道核心题、30 道边界题，问题、答案、原文与难度依据 |
| `experiments/results/private/benchmark-v2-baselines-v2-nonthinking/` | 实际冻结的 205 道开发题、95 道测试题及划分审计 |
| `experiments/results/private/` | 关键批次的报告、数值表、协议与完整性记录 |
| `docs/` | 方法说明、交付状态、演示说明与实验局限 |
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
