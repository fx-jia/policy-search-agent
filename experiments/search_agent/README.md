# 医保 Search Agent

独立、可测试的有界检索控制器，提供单题引擎及多轮会话核心/命令行入口。复用已安装 RAGFlow 的 Hybrid 检索与原生 RAG 最终生成，固定1000篇语料。没有改动原生 RAG/Low 基线，也不运行 High。

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

真实运行示例需要完整 RAGFlow 部署、已审核的来源映射和运行配置；这些原始配置未公开。原生基线另需部署环境中的`POLICY_APPROVED_CHAT_ID`，配置方式见[仓库运行依赖](<../../README.md#运行真实实验的依赖>)。下列无服务单测可在本仓库执行。

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

片段ID消除了抄写差异，但不能保证模型选对证据或理解正确。原文窗口未展示的内容不能被Judge确认为支持；降级也保留原生RAG的上下文限制。发现新但未标注的正确依据时，评测侧复核并统一补标。多轮会话支持独立CLI；上海政策业务入口见下文。[医保助手页面/API](<../../docs/medical_assistant_frontend.md>)已实现并通过离线验证，真实新问答仍未验收，在线Router尚未接入。

## 多轮会话

`conversation.py` 在单题引擎之上保存会话历史和待补条件；`resolution.py` 选择追问所需的用户消息，或返回澄清问题。模型只选择有原话绑定的来源，完整当轮问题由程序保留所选用户消息的全部原文与本轮问题，防止只摘出“职工医保”却丢掉“不包括职工医保”。每轮重新创建backend、检索证据、生成引用。历史回答只帮助理解指代；历史回答中的金额、资格或其他政策结论不能进入展开问题当作事实前提。

例如先问“上海居民医保参保条件是什么？”，再问“那退休人员呢？”；解析器需要继承用户明确的地区和险种。用户说“改成杭州”时以新条件为准。缺用户条件时保存pending，短回复可以补齐原问题；技术失败和取消不会提交新回合或清除pending。明确换主题的成功回合会清除旧话题状态。

命令行在独立worker中串行运行，复用现有已审核模型、语料范围和索引快照。`--config` 提供部署者恢复并审核的Search Agent冻结配置，程序校验后在新的输出目录冻结当前实现；`--max-tokens` 是这次会话运行的生成总预算。原批次完整配置未公开，须将下例路径替换为部署环境中的实际文件。以下命令会在收到问题后调用已配置模型：

```bash
python3 -m experiments.search_agent.cli \
  --config /absolute/path/reviewed-agent-config.json \
  --output experiments/results/private/conversation-2026-10-09 \
  --max-tokens 60000
```

输出目录必须尚不存在。`/new` 创建新会话，`/clear` 清空当前会话，`/exit` 或EOF退出。终端显示展开问题、答案、引用与停止原因。`config.json`/`manifest.json` 保存冻结实现；`attempts.jsonl`、`events.jsonl`、`records.jsonl` 和退出时导出的 `ledger.jsonl`/`ledger_summary.json` 保存逐轮审计。会话状态驻留worker内存，退出后重新开始；已有审计目录保持可读，不自动恢复发送。生成或Embedding用量未知、预算耗尽都会停止运行，并保留已结算用量与未知预留。关闭时缺账本快照则明确标为未核对，保留活动调用记录，不将未知用量当作零。

默认历史最多6个成功回答回合；pending保存尚未解决的原问题及用户补充原话。解析提示词、完整JSON元数据、历史、pending和当前问题合计最多3000 Token，按完整回合裁剪。当前问题或pending本身过长时在发送前返回预算错误，可清空会话后重新提问。历史被裁剪会告知解析器，无法确定指代时应请求澄清。

每轮解析最多一次、最多10秒，计入整轮60秒/检索决策40秒、4次物理生成和60000 Token预算；查询Embedding仍最多3次。解析消耗一个调用后，控制器按实际剩余额度为最终生成预留位置。澄清只产生解析结果，不检索政策或追加答案生成；完整政策回答最后仍只生成一次。

可复用核心：

```python
from experiments.search_agent.conversation import Conversation

conversation = Conversation("local-session")
# 每轮传入新backend；backend实现原单题协议及 async resolve(context)。
result = await conversation.ask("上海居民医保参保条件？", fresh_backend)
# result.status == "needs_clarification" 时可在同一conversation继续补充条件。
```

无模型服务验证：

```bash
python3 -m unittest discover -s experiments/search_agent -p 'test_*.py' -q
```

本次验证覆盖会话核心、解析契约、真实worker协议及受控subprocess命令行交互。真实模型多轮效果和连续问答服务验收仍需另行记录；旧单轮Benchmark成绩保持原版本口径。

## v8.1：需求窗口和缺口行动

证据预算按内容相关程度共享，短块未用预算可分给长块；按需求选择原文，先保留已核验锚点。控制器使用的整条序列化请求（含提示词、来源及片段元数据）不超过7000 Token，证据上限仍4800；最终证据上限5600、最多8块。

`read_chunk` 从已见块补读尚未展示的原文，保留来源偏移，不调用Embedding；新增原文区间也算进展。相邻块只有尚未被观察时才作为可用方向。两种读取及定向搜索共享最多2次动作。

默认 `AgentLimits(gap_contract=True)`：Judge 按所问答案类型与地区/版本/对象检查充分性；可补救的缺口优先同块、可用邻块、定向检索。缺用户个人情况允许明确停止。`gap_contract=False` 为 A-only 消融，保留新窗口与工具，关闭 B 的追加提示和执行约束；冻结的 v7 对照继续从各实验配置中的源码运行。

`stop_kind` 位于决策JSON根层。控制器在下一次Judge前核算物理调用，包含已确认未发送的连接重试，为最终回答保留一个调用位；不会因此提高原上限。

最终回答仍只生成一次，未加入计算工具、额外答案Judge或Router。开发消融见[开发实验报告](<../results/private/policy-search-agent-v8_1-study/report.md>)，离线条款覆盖改善不能代替问答效果。A和A+B分别冻结评测，不拼接不同版本成绩。

当前v8.1 A+B已完成开发扩验及冻结95题测试。最终同题模型评分相对Native Low为+13.977/100（77对），相对Native RAG为+0.016/100（77对）；后者区间跨0，未证明质量优势。三方案都成功的同87题中，v8.1平均9.001秒、每题生成8128 Token，Low为25.810秒、6079 Token，RAG为6.622秒、3886 Token。质量分不是正确率，成本与质量需共同评估。

首轮三组各95次题目尝试、277份评分尝试，以及8个技术失败组合的单次重跑均已结束。疑问题的AI辅助复核独立存档，未覆盖正式原分。详见[最终汇总](<../results/private/policy-search-agent-final-test95/final_summary.md>)、[复核报告](<../results/private/policy-search-agent-final-test95/assistant-review-v1/report.md>)及[收尾清单](<../../docs/policy_search_agent_closeout.md>)。

2026-09-27完成本机单题演示：1题真实运行成功，并核验2份已有开发轨迹，展示邻块补读及信息不足时不编造总额。后者同时保留控制器误判的局限，不宣称拒答机制普遍可靠。运行数据、引用与逐步讲解见[演示与验收](<../../docs/policy_search_agent_demo.md>)。

## 上海医保政策定时更新

`ShanghaiMedicalConnector → SyncBase → 原生上传/解析队列 → PolicyPublicationService → PolicyBackend → Conversation` 是新增业务路径。同步 worker 每24小时扫描上海医保局三个栏目并重新读取已采用的政策；正文和附件分别检测变化。归档失败、缺附件、解析或索引未就绪时保留候选及旧发布版本。独立维护协程每60秒检查发布和恢复；失败重试间隔为1、5、15分钟，耗尽后仍保留下一次日常扫描。

每轮最多接纳40份新政策、解析100份资产，总覆盖最多200份政策；发现但延后的候选单独保留。已发布且正文、附件、解析构建均未改变的政策继续留存观察记录，不重复占用解析额度。延期、修改和废止等通用通知也会有界读取：引用已覆盖文号时保留原文关联证据，模糊关联进入待审，程序不推断法律替代关系。

当前官网列表还引用`/zxzc/`和`/zxgkzfxx/`历史详情。仅在官网政策条目的结构中发现这些精确路径时，记录标题、链接和来源页，标为范围外待审；不抓取正文、不消耗新政策解析额度，也不扩展详情或重定向允许范围。列表遍历完成仍保留这些覆盖缺口。

知识库必须专用、归当前租户所有，并采用原生解析器。`parse_build_key`包含实际解析配置、embedding模型配置和构建代次；修改配置后重新`init`。已发布版本的文档和chunk不能原地覆盖；需要重建时增加`--parse-generation`。文档embedding用量属于原生入库流程，不包含在问答ledger里。

原生解析的两种执行模式都在模型调用前核对租户、连接器、文档绑定和实际解析/embedding配置。发布时验证真实索引向量维度；查询时核对发布模型指纹与查询向量维度，配置不匹配会停止。缺少构建证明的旧绑定也会停止，需增加构建代次重新入库。

在已经安装完整项目依赖、加载正确服务配置的部署环境中执行下列命令。运行中的容器需要包含本次源码；宿主修改不会自动进入未挂载源码的容器。先通过RAGFlow管理界面创建一个专用知识库，并使用它的真实租户和知识库ID：

```bash
python3 -m experiments.search_agent.policy_cli check-source
python3 -m experiments.search_agent.policy_cli init --tenant-id TENANT_ID --kb-id KB_ID
python3 -m experiments.search_agent.policy_cli sync-now --tenant-id TENANT_ID --kb-id KB_ID
python3 -m experiments.search_agent.policy_cli status --tenant-id TENANT_ID --kb-id KB_ID
python3 -m experiments.search_agent.policy_cli ask --tenant-id TENANT_ID --kb-id KB_ID \
  --question '上海职工医保综合减负的适用对象是什么？' --as-of 2026-10-10 \
  --output experiments/results/private/shanghai-policy-session
```

`check-source`每栏目检查首页和发现的下一页，另读取最多三个已核实样本及附件；不调用模型、不写知识库。`init`创建或更新原生连接器、绑定知识库、建立三张政策表；日常定时工作由现有`rag/svr/sync_data_source.py`进程完成。`sync-now`只排入原生队列；`status`区分发现、已收录、已发布与pending数量。省略`--question`进入多轮模式，`/clear`清空用户会话条件，保留已用预算和未知usage阻断状态，`/exit`退出。

扫描状态分为列表发现完成、候选观察归档完成和全部政策抓取完成。政策额度、待审文件或已记录的附件失败会保留不完整覆盖与错误信息；观察任务正常结束后，原生调度仍安排下一轮。DNS安全校验、归档、列表遍历或上传失败会让任务失败。只有全部政策抓取完成才推进`last_scan_success_at`，部分覆盖不会被标记为完整成功。维护协程使用线程内有界等待，避免阻塞原生调度事件循环。

`docker/docker-compose.shanghai-policy.yml`通过逐文件只读挂载部署源码；其DNS映射及`conf/policy_sources/shanghai.docker.yaml`中的本机代理属于部署示例，使用前须核对目标环境。SOCKS5使用本地解析，官网请求仍经过公网DNS检查、地址固定和TLS验证；`socks5h`、HTTP代理、带凭据代理和未获准的端点均不接受。代理需要持续可用；官网IP变化时重新核实并更新专用Compose映射，预检失败时不会绕过检查。

```bash
cd docker
docker compose -f docker-compose.yml -f docker-compose.shanghai-policy.yml --profile cpu up -d --no-deps ragflow-cpu
docker compose -f docker-compose.yml -f docker-compose.shanghai-policy.yml exec -w /ragflow ragflow-cpu \
  python -m experiments.search_agent.policy_cli check-source \
  --config conf/policy_sources/shanghai.docker.yaml
```

专用知识库可以在`parser_config.llm_id`中选择该租户所属的可审计chat模型；embedding使用知识库自己的模型引用。两者均沿用原生模型解析服务，不修改租户默认配置。

首轮仅有三个官方种子和按主题发现的候选，不声称已建立上海基础政策的完整覆盖。所有政策默认`unknown`时效；发布时间不代表生效日。核对原文后，可给指定修订写入有原文证据的时效确认，再发布新的元数据清单。例如以下内容来自[综合减负办法通知](https://ybj.sh.gov.cn/gfxwj/20260911/f57a8b26aa414a15a57d1040c1024600.html)，实际`evidence`必须逐字匹配归档文本：

```json
{"status":"confirmed","effective_from":"2026-09-22","effective_to":"2031-09-21","evidence":"本通知自2026年9月22日起施行，有效期至2031年9月21日。"}
```

```bash
python3 -m experiments.search_agent.policy_cli review --tenant-id TENANT_ID --kb-id KB_ID \
  --revision-id REVISION_ID --metadata /absolute/path/temporal-review.json
python3 -m experiments.search_agent.policy_cli publish --tenant-id TENANT_ID --kb-id KB_ID
# 为已发布版本重新解析，构建代次递增后再次排队
python3 -m experiments.search_agent.policy_cli init --tenant-id TENANT_ID --kb-id KB_ID --parse-generation 1
python3 -m experiments.search_agent.policy_cli sync-now --tenant-id TENANT_ID --kb-id KB_ID
```

业务Search Agent每轮解析用户原话后绑定一次不可变release，下一轮可以读取新release。检索固定tenant/KB/doc_ids并检查chunk SHA256；空范围直接返回无证据。无可靠位置的片段不开放邻块读取。unknown时效不能单独支持“当前适用”，历史问题与指定时点冲突先澄清。输出包含引用、release_id、资料截至时间、coverage、pending、stale及实际调用ledger。

中文完整日期与ISO日期都参与冲突检查，包括解析器继承的用户条件。当前问题涉及的政策有新观察待处理、相关通知待核对，或超过48小时没有完整成功扫描时，会降低适用性结论并强制附上更新滞后说明；不把无关政策的待处理记录直接当作所问政策的冲突。

每轮最多4次物理生成、3次查询embedding、60秒；会话总预算最多60000 Token，最终回答仅生成一次。使用请求独立的RAGFlow模型实例和HTTP client，禁用隐式重试；provider缺少真实usage时保留未知预留并禁止此会话继续。首版支持可审计的OpenAI SDK兼容chat/embedding路径，其他provider在发送前明确拒绝；不能靠估算usage绕过。输出目录属于固定tenant/KB/session/budget，记录`records.jsonl`、`events.jsonl`及逐轮`ledgers/`。CLI创建新session时需使用新的输出目录。

离线验收与真实上线验收分别记录。真实官网样本能读取并不代表20–40份政策已经解析验收，也不代表定时任务已经启用；真实数据库、对象存储、索引、模型及两轮自动更新仍需在部署环境留证。公开目录保留历史benchmark的结果和限制；共享adapter/evidence源码有新增来源字段，新实验需重新冻结源码，不能把旧评分当作这个业务链路的效果。

### 历史部署状态与验收边界

2026-10-11 的本机记录表明，部分上海政策已完成入库和发布，但覆盖仍不完整，仍有待审核和解析中的资产；两轮自然更新验收通过数为0。部分来源任务和问答尝试失败或中断，技术修复与离线测试通过不能作为线上验收通过的证据。

历史问答曾返回技术成功的答案，但存在政策生效日期和适用范围问题，质量尚未验收；后续调用又发生连接错误，供应商用量无法确认。原会话继续受未知用量阻断，不能把未知预留记为0，也不能通过新建会话绕过。原生入库的完整供应商用量同样尚未核清。

2026-10-13 已验证助手页面可恢复历史记录并展示引用，新的提问仍被用量状态阻断。页面/API已实现，真实新问答和完整线上验收仍未完成，详见[医保助手运行说明](<../../docs/medical_assistant_frontend.md>)。具体部署身份、容器、任务与会话标识及原始操作日志不在公开文档中保留。

专用两轮加速验收脚本`policy_acceptance`要求部署者独立设置`POLICY_APPROVED_TENANT_ID`和`POLICY_APPROVED_KB_ID`；命令行传入的`--tenant-id`、`--kb-id`必须与该批准范围一致，不能仅凭命令行参数授权自己。两个变量都须为32位小写十六进制标识，无内置部署默认值；真实值通过未跟踪的本地配置提供。

脚本只调整该知识库唯一连接器的频率，不排队、不改任务时间戳、不重置重试。它要求初始正常完成并发布至少20份政策，再临时改为5分钟；两轮必须保留自然创建的原始SCHEDULE记录、实际到期开始时间、来源完成时冻结的精确修订/构建/资产以及后续完整发布清单。解析任务、模型计数与文档成功token差异按observer_windows全局观察窗报告，可能跨来源轮次，不能归因单轮provider成本。回拨时间戳的任务会被拒绝；失败、中断或成功均恢复1440分钟并核对回读。初始入库或重试不计入这两轮，部分覆盖与待审记录会继续显示。

```bash
python3 -m pytest -c /dev/null -p no:cacheprovider \
  test/unit_test/common/test_shanghai_medical_connector.py \
  test/unit_test/api/db/services/test_policy_catalog.py \
  test/unit_test/api/db/services/test_policy_publication.py test/unit_test/policy/
python3 -m unittest discover -s experiments/search_agent -p 'test_*.py'
```
