# 技术失败题：原因核查与单次重跑

用户在首轮开始后明确授权技术失败题重跑。此批为诊断性复跑，不是新的独立测试集；原始95题成绩、失败与未知用量全部保留，不将重跑成功说成首轮成功。没有重跑内容错误但技术成功的答案，也没有选择多次中的最好答案。

| 方案 | 题号 | 首轮原因 | 重跑状态 | 重跑质量分 |
|---|---|---|---|---:|
| Native RAG | mb2-061 | question_call_limit | ok | 100.0 |
| Native RAG | mb2-063 | question_wall_time_limit | ok | 100.0 |
| Native RAG | mb2-149 | question_call_limit | ok | 100.0 |
| v8.1 | mb2-068 | provider_usage_unknown | ok | 100.0 |
| v8.1 | mb2-097 | provider_usage_unknown | ok | None |
| v8.1 | mb2-123 | provider_usage_unknown | ok | 100.0 |
| v8.1 | mb2-225 | provider_usage_unknown | ok | 100.0 |
| Native Low | mb2-187 | provider_connect_failed | ok | None |

重跑生成已知Token：47,444；新增未知预留：0。评分另耗已知Token：13,825，未结算预留：0。此前未知用量未清零。

质量分不是正确率；评分缺失不记0。恢复成功只能说明该次重新执行返回了答案，不能单独证明网络故障已经根治，也不证明检索或Agent算法变好。

三种主要首轮故障：连接建立阶段ConnectionResetError被映射为ConnectError并耗尽调用保护；响应读取阶段ReadError/ServerDisconnectedError；生成请求60秒超时。日志无法进一步判定是本机网络、中间链路还是提供商服务导致。

逐题答案见answers.md，原始配置、调用与检查在runs目录。首轮主报告仍在上一级report.md，未被本批覆盖。
