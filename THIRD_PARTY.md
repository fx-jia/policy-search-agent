# 第三方与版本说明

- 底层框架：[infiniflow/ragflow](https://github.com/infiniflow/ragflow)，Apache-2.0。备份保留原仓库 LICENSE；部分补丁、测试夹具和冻结模块包含上游代码片段。
- 本地实验配置中的 RAGFlow 版本标识为 `v0.27.2`；历史镜像标识保存在 `experiments/configs/ragflow_hybrid.json` 的 `ragflow_version` 字段。**应以具体批次保存的运行文件哈希和原机镜像记录核验，不能仅凭版本名称认定源码一致。**
- `snapshots/*/protocol-summary.json` 中的 `runtime_code_hashes` 是冻结原生运行文件指纹，`custom_source_hashes` 是自定义模块输入指纹。
- 本仓库保留自定义模块与必要源码夹具，没有复制完整 RAGFlow 框架，也未备份其依赖包、数据库或镜像。
- 政策材料的来源 URL、标题、引用位置和文本哈希保存在 Benchmark 的 evidence 字段；只保留评测所需摘录，未上传完整原始语料库。
- 生成模型、向量模型和裁判均为外部模型服务；本项目没有训练这些模型。
