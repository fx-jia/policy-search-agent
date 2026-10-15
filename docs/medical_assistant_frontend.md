# 医保政策助手：运行与验收

首版在现有 RAGFlow `web/` 工程新增独立入口 `/medical-assistant`，复用登录、主题、基础组件和请求封装。页面包含文本提问、多轮追问与澄清、Markdown 答案、逐轮原文引用、来源限制、实际检索过程及新话题。桌面显示右侧依据栏，手机使用抽屉。

代码已接入上海动态政策业务服务。2026-10-12 完成离线界面、HTTP 单元测试和业务回归；2026-10-13 启动本机服务并确认浏览器恢复真实会话。原上海会话的 provider usage 仍待核对，新提问禁用，真实新问答尚未验收；没有另建会话或修改原账本。

## 2026-10-13 本机启动记录

前端 `http://127.0.0.1:5174/medical-assistant` 与 Python API `http://127.0.0.1:9380` 已运行。Docker Desktop 启动后恢复原数据库、索引、对象存储和 Redis 容器；原上海同步和解析任务继续运行。

已通过数据库只读核对配置用户是该租户 owner，绑定原 `search-session/session.json`。由于原容器有正在执行的同步和解析，本次将两份新医保模块与配置部署到运行容器，仅重启 API。API 使用 `MEDICAL_ASSISTANT_CONFIG=/ragflow/conf/medical_assistant.local.json`，日志为 `docker/ragflow-logs/medical-assistant-api.log`。

已验证前端代理的 language 接口正常、新医保接口未登录返回 HTTP 401、已登录浏览器恢复四条真实历史记录，并点击历史答案引用打开原文面板。页面真实显示 `unresolved_provider_usage`，新提问禁用；这不是一次新问答验收。原 session、records 与四份 ledgers 的哈希未改变，记录见 `.superpowers/sdd/2026-10-12-search-agent-frontend/startup-2026-10-13.json`。

新增 `docker/docker-compose.medical-assistant.yml` 保存医保模块/配置的只读挂载及环境变量，不覆盖原容器命令。当前运行容器没有整体重建。待现有同步/解析自然结束后，后续完整部署应把该 overlay 一同应用；单纯再次启动旧容器不会继承本次手工 API 进程的环境变量：

```bash
docker compose -f docker/docker-compose.yml \
  -f docker/docker-compose.shanghai-policy.yml \
  -f docker/docker-compose.medical-assistant.yml up -d --no-deps ragflow-cpu
```

本次 `conf/medical_assistant.local.json` 使用容器内路径，不能直接用于宿主机 `uv` API；宿主机部署须填写指向同一原会话的宿主绝对目录。

## 启动

在 `web/` 下运行：

```bash
npm ci
API_PROXY_SCHEME=python npm run dev -- --host 127.0.0.1
```

打开 Vite 输出地址的 `/medical-assistant`。本次开发服务为 `http://127.0.0.1:5174/medical-assistant`，默认 5173 已被占用。未登录时会进入 RAGFlow 登录页，成功后返回助手；只允许站内助手地址作为回跳目标。

开发代理将 `/api` 与 `/v1` 转发到 Python API `127.0.0.1:9380`。本工程开发环境默认可选择 Go，启动时明确设置 `API_PROXY_SCHEME=python`。Go 模式显示能力未接入提示。

前端没有生产样例开关。服务未配置或资料未就绪时显示不可查询，输入框禁用。

## 绑定已有上海政策会话

复制 `conf/medical_assistant.example.json` 为部署配置，填写已授权用户、租户、上海政策知识库及已有运行会话。`output_dir` 必须是 API 进程能访问的绝对路径。目录里的 `session.json` 必须已经存在，并与配置的 `tenant_id`、`kb_id`、`session_id`、`budget` 四项完全一致。

`owner_user_id` 是登录用户 ID；该用户还必须具有配置知识库的访问权限，知识库必须属于配置租户。预算沿用原会话，不能增大或重置；配置合法范围为 1 至 60000。

在已配好依赖、数据库、索引服务及模型的 Python API 环境中，从仓库根目录启动单个 API 进程：

```bash
export MEDICAL_ASSISTANT_CONFIG="$PWD/conf/medical_assistant.local.json"
export PYTHONPATH="$PWD"
uv run python api/ragflow_server.py
```

配置示例含占位符，不可直接作为真实部署配置。服务不会根据这个示例自动创建新会话。如果此前运行资料只有诊断文件，需要先由会话管理流程核对并接续原身份与账本，不能通过空目录或新 session ID 绕过旧的用量待核对状态。

本期支持单个 Python API 实例。会话目录需要持久保存 `session.json`、`records.jsonl`、`ledgers/`、`web_owner.json`、`web_owner.lock` 和 `web_commands.jsonl`；进程持有目录内的 owner 锁。该目录不能同时由 CLI 或其他实例执行。容器部署须挂载配置文件和同一个持久会话目录，并传入配置的容器内绝对路径。

## 调用路径与恢复

```text
/medical-assistant
  → hook → service → next-request
  → /api/v1/medical-assistant/*
  → MedicalAssistantService
  → PolicySearchService
  → Conversation → Search Agent core / PolicyBackend
  → 上海政策发布快照及 RAGFlow 检索、模型能力
```

只有三个操作：

| 操作 | 参数 | 行为 |
| --- | --- | --- |
| GET `/api/v1/medical-assistant/session` | 无 | 读取原会话、已完成结果、资料范围和可查询状态 |
| POST `/api/v1/medical-assistant/turn` | `request_id`、`question` | 发送当轮原话；服务负责多轮条件和澄清 |
| POST `/api/v1/medical-assistant/clear` | `request_id` | 开始新话题，保留会话身份、用量和阻断状态 |

`request_id` 使用 UUID。命令先持久记录，再执行；同 ID 重复请求返回原结果，同 ID 不同正文拒绝，执行中拒绝另一个新问题。HTTP 连接断开后服务器任务继续，页面通过 GET 确认状态，不自动重发 POST。服务重启时未完成的命令保留为“待确认”，禁止自动重放。

首版等待完整 JSON 答案。检索过程只显示真实显式事件；补读或补检来自 `tool.action`，证据判断中的行动建议不当作已执行活动。页面不模拟流式内容或工具进度。

引用保留本轮 `reference.chunks` 顺序：`##0$$` 或 `[ID:0]` 对应页面 `[1]`。来源标题、文号、发布日期、生效日期和原文按返回字段展示；缺元数据保持未知。技术成功与政策覆盖、适用时点、扫描滞后分别显示，不能用成功状态推断资料完整。

## 2026-10-12 离线交付验收记录

完整记录在 `.superpowers/sdd/2026-10-12-search-agent-frontend/`。

| 验证 | 结果与证据 |
| --- | --- |
| 前端功能及登录回归 | 61/61 通过，`frontend-verified.log`；认证测试依赖修正后另跑 3/3，`auth-final.log` |
| HTTP / 会话 owner / 政策服务 | 53/53 通过，`backend-verified.log`；无数据库或模型调用 |
| 已有多轮核心回归 | 20/20 通过，`conversation-verified.log` |
| 触及文件 lint / format | 后端 Ruff 通过；前端 oxlint 无错误，保留登录页原有一处 console 警告；新增业务及认证文件 oxfmt 通过 |
| Vite production build | 通过，`build-verified.log`；保留原工程的大包、pdfjs eval 等构建警告 |
| 全仓 TypeScript | 未通过，215 项原有诊断；新增助手文件无诊断，`type-check-verified.log` / `type-check-comparison.json` |
| 独立代码审查 | 完整路径审查发现六处问题，修复后六项均已关闭，`final-review.md` / `fix-review.md` |
| 浏览器界面 | 1440×960 桌面及 390×844 手机，离线 HTTP fixtures；真实工具 DTO 展示、澄清、定向搜索、补读、无依据、时效未知/滞后、用量阻断、刷新不 POST 重放通过；长标题/文号及第八条引用定位均无面板横向溢出，`visual-check.json` |
| 真实网页问答 | 未执行、未验收；本机依赖服务未启动，已有运行记录中的未知用量状态保留 |

首次 TypeScript 运行与前端实现并行，223 项诊断中包含 8 项当时尚未完成的 feature 模块错误；这 8 项已消除。最终其余 215 项的文件路径与错误码在首次记录中均已存在。没有为通过全仓检查修改无关页面。

页面预览均为明确标注的离线样例：

![桌面离线页面预览](<assets/medical-assistant/desktop-offline.png>)

![手机离线引用抽屉预览](<assets/medical-assistant/mobile-offline.png>)

验证命令：

```bash
# 仓库根目录；需有 pytest / Quart / httpx 等项目依赖
python -m pytest -c /dev/null -p no:cacheprovider \
  test/unit_test/api/apps/test_medical_assistant_api.py \
  test/unit_test/policy/test_medical_assistant_service.py \
  test/unit_test/policy/test_policy_search_service.py -q
python -m unittest experiments.search_agent.test_conversation -q

# web/
npx jest --runInBand --no-cache --coverage=false \
  src/pages/medical-assistant src/pages/login-next/index.test.tsx \
  src/utils/__tests__/login-return.test.ts \
  src/wrappers/__tests__/medical-assistant-auth.test.tsx
npm run type-check
npm run build
```

浏览器验收通过隔离页面拦截 HTTP 响应，样例答案及截图标明“离线界面验收”。样例没有实际医保待遇或资格结论，也不能证明当前政策库完整性、真实模型答复质量或引用准确率。生产代码中不包含浏览器验收用的伪造登录或 API 响应。

## 主要文件

- 页面与业务组件：`web/src/pages/medical-assistant/`
- 登录保护与回跳：`web/src/wrappers/medical-assistant-auth.tsx`、`web/src/utils/login-return.ts`
- 数据请求：`web/src/hooks/use-medical-assistant-request.ts`、`web/src/services/medical-assistant-service.ts`
- HTTP 入口：`api/apps/restful_apis/medical_assistant_api.py`
- 会话 owner 与公开 DTO：`api/db/services/medical_assistant_service.py`
- 上海业务及只读恢复：`api/db/services/policy_search_service.py`

本期没有增加知识库管理、文件上传、模型配置、工作流、同步控制或历史会话管理页面。
