"""Render paired, explicitly annotated results; requires every planned row."""
import csv
import json
from pathlib import Path

from analyze import main, fixed, readl

P = Path(__file__).resolve().parent
ARMS = ('rag', 'low', 'v7', 'a', 'ab')
NAMES = dict(rag='Native RAG', low='Native Low', v7='v7', a='v8.1 A', ab='v8.1 A+B')


def build():
    main(True)
    metrics = json.loads((P / 'metrics.json').read_text())['extension']
    manifest = json.loads((P / 'manifest.json').read_text())
    ids = manifest['selection']['extension']
    qs = {q['question_id']: q for q in readl(P.parent / 'benchmark-v2-baselines-v2-nonthinking/dev.jsonl')}
    scores = {a: {r['question_id']: r for r in readl(P / 'extension' / a / 'assessments.jsonl')} for a in ARMS}
    rows = {a: {r['question_id']: r for r in readl(P / 'extension' / a / 'records.jsonl')} for a in ARMS}
    lines = ['# v8.1：需求窗口与缺口行动的配对验证', '',
             '**六题人工复核已合入，正确率与回答质量分开解释。** A+B为54/55核心通过、1题技术失败；v7为55/55。双方正常完成的54对中，二元核心内容判定均通过。A+B的Judge异常从4题降为0，082获得更直接的指定政策依据，用户明确认为质量更好；但多维质量评分尚未运行，不能用二元分数否定或证明这个质量差异。', '',
             'Low为42题通过、13题不完整；双方正常完成的54对中，A+B新增核心通过13题、内容退步0题。另有1题A+B技术失败、Low通过，必须单列。多数优势v7已经具备，不能当作本次A+B新增收益。Native RAG为53/55通过、2题技术失败，其正常完成的答案核心内容均通过；不能把传输失败造成的差距解释成Agent算法更准。', '',
             '固定1000篇、3205块，原 qwen3.7-text-embedding 索引，deepseek-v4-flash-0731 非思考模式；RAG/v7/A/A+B初检Hybrid权重0.3、top_n=8，独立Agent复用RAG的8192上下文模板；Low保留自身编译检索与生成路径。统一出站脱敏、串行执行、每题一次正式尝试。', '',
             '评分由助手逐题初评，再合入用户对024/082/086/110/112/165六题的统一裁定，其余题仍是助手初评。原始用户输入保留在review_queue.md，结构化修订见human_adjudications.json，修订前结果在archive/before_human_review_20260926。同题答案若仅空白或引用编号不同，可复用正文判定，引用独立记录。核心通过不等于全文和引用质量满分，未知替代依据不直接判错。55题均为此前未运行开发题，但政策组已见；不是冻结测试成绩。', '',
             '## 实际改动', '',
             '- A：取消每块平均分配，按问题/需求给原文共享预算，短块腾出的空间给长块；已验证摘录优先保留。新增 read_chunk 读取已见块隐藏条款，记录真实偏移，不调用Embedding；相邻块只展示尚未见过的方向。',
             '- B：Judge 要判断证据是否足以回答所问数额、期限、对象与条件；可补救缺口不能随意结束。执行器依次尝试同块、邻块、定向检索。A-only保留窗口和工具，关闭B追加提示与动作覆盖。',
             '- A包含窗口、读原文工具和基础Judge协议调整，是模块组消融，不是只改变一个窗口数字；B比较在同一A底座上追加的提示与行动约束。',
             '- v8.1工程修复：停止类型统一放在JSON根层；连接重试占用物理调用后，控制器为最终回答保留一个调用位。两次补充动作、三次Judge、四次物理生成、一次最终回答和60秒上限均未增加。',
             '- 控制器完整序列化请求≤7000 Token，证据窗口≤4800；最终原文≤5600、最多8块。最终请求核验所选原文未丢失。没有加入计算工具、额外答案Judge或Router。', '',
             '## 验证与协议', '',
             '67项容器离线测试通过；另13项评测入口测试通过。覆盖共享预算、锚点溢出、真实原文偏移、重复补读、不可用邻块、JSON字段和重试预留最终调用。真实分词复验242/247关键条款可见，完整Judge输入6825/6956 Token。', '',
             'v8旧20题A/AB诊断原样保留，未混入本批。v8.1先6题技术回归，全部正常产出、4通过2不完整；随后运行预先冻结的全部55剩余开发题，5组配对共275次题目尝试。95道测试题未使用；根据55题结果未再改策略。', '',
             '旧错题回归尚不完整：v7两批新增60题的11道内容失败中，最终v8.1只重测242/247/095/059；前两题本次通过，后两题仍未通过。020/052是原通过对照。045/046/048/060/091/276/281尚未用最终v8.1重测；55道新题不能替代这些旧错题回归。详见[逐题对照](old_failure_regression.md)。', '',
             '冻结预算并不在所有框架间相同：RAG为35k Token/2次生成/4次查询/60秒；Low为80k/14次生成/24次查询/90秒；v7、A、A+B均为60k/4次物理生成/3次物理查询/60秒。A/B消融预算完全相同，原生基线沿用既有上限。实际请求中RAG/v7/A/A+B温度均为0；Low各阶段使用0/0.1/0.3，最终生成0.3，沿用原生实现。因此与Low是系统级对照，不能将所有差异解释为单一算法因素。连接失败触顶单列，不能据此声称内容算法获益。', '',
             '## 新增55题结果', '',
             '| 方案 | 核心通过 | 待复核 | 内容未通过 | 技术失败 | 平均秒 | P95秒 | 生成Token |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        m = metrics[a]; o = m['outcomes']
        lines.append(f'| {NAMES[a]} | {m["accepted"]}/55 | {o.get("pending_review",0)} | {o.get("incomplete",0)} | {o.get("technical_error",0)} | {m["mean_seconds"]} | {m["p95_seconds"]} | {m["generation_tokens"]:,} |')
    lines += ['', '耗时包含技术失败，不扣除慢请求。生成Token为实际已结算输入+输出，未结算预留另列。供应商缓存Token不等于本地查询缓存收益。双方均正常完成的配对成本另见 paired_success_costs.json，避免失败题提前退出降低均值。', '',
              '按双方运行正常的54题比较：v7平均6.47秒、395,024 Token，A+B平均7.70秒、446,554 Token，约增加19%耗时、13%生成Token。对Low的54题则从22.82秒降到7.70秒，约缩短66%，但生成Token从342,410增至446,554，约增加30%。因此本轮不能声称Agent同时减少了生成Token。', '',
              '本轮v8.1新增55题五组生成总用量1,804,823 Token，加6题技术回归73,260，共1,878,083。前序v8的两组20题诊断另耗381,061，总计2,259,144生成Token；版本成绩分开，开销全部记账。本批生成未结算预留为0，查询Embedding用量独立记录，未并入生成Token。', '',
              '## A+B配对变化', '',
              '| 对照 | 已判定配对 | 新增通过 | 退步 | 未判定配对 | 双方运行正常时：新增/退步 |',
              '|---|---:|---:|---:|---:|---:|']
    intervals = {}; paired_costs = {}
    for a in ('rag', 'low', 'v7', 'a'):
        m = metrics['ab_vs_' + a]; c = metrics['content_ab_vs_' + a]
        lines.append(f'| {NAMES[a]} | {m["paired"]} | {len(m["improved"])} | {len(m["regressed"])} | {len(m["unpaired"])} | {len(c["improved"])}/{len(c["regressed"])}（{c["paired"]}对） |')
        valid = [q for q in ids if q not in c['unpaired'] and rows[a][q]['status'] == rows['ab'][q]['status'] == 'ok']
        cost_ids=[q for q in ids if rows[a][q]['status']==rows['ab'][q]['status']=='ok']
        if cost_ids:
            paired_costs[a]={'questions':len(cost_ids),'arms':{k:{'mean_seconds':round(sum(rows[k][q]['latency_ms']/1000 for q in cost_ids)/len(cost_ids),2),'generation_tokens':sum(c.get('usage',{}).get('total_tokens',0) for q in cost_ids for c in rows[k][q].get('calls',[]))} for k in (a,'ab')}}
        if valid:
            intervals[a] = fixed.cluster_interval(valid, qs, scores[a], scores['ab'])
    lines += ['', '新增与退步是单次配对观察，不是保证修复。包含技术失败的变化不能全归因于检索策略。按政策组的描述性bootstrap结果见 intervals.json；开发集上的小样本区间不替代测试集验证。', '',
              '## 开销与诊断', '',
              '| 方案 | 生成调用 | 查询Embedding应用成功次数 | 查询Embedding物理次数 | 本地缓存命中 | 补充动作题数 | Judge异常/降级题数 | 全参考片段候选覆盖 | 全参考片段最终覆盖 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        m = metrics[a]
        lines.append(f'| {NAMES[a]} | {m["generation_calls"]} | {m["query_embedding_observed_successes"]} | {m["query_embedding_physical_calls"]} | {m["embedding_cache_hits"]} | {m["supplement_questions"]} | {m["controller_failures"]}/{m["fallback_questions"]} | {m["candidate_literal_all_spans"]}/55 | {m["final_literal_all_spans"]}/55 |')
    lines += ['', 'None表示未取得对应遥测，不解释为0。确认未发送的Embedding连接失败按0用量记账、物理尝试单列；状态不明保留为未知。Native Low未接入本地缓存，不能把省去其扩展流程的调用计成缓存收益。参考片段覆盖严格按来源+单块内完整原文字面匹配，不拼接跨块条款，只是诊断下界，不等于所有可替代证据覆盖或答案准确率。', '',
              'A+B在7/55题（12.7%）触发13次邻块读取，没有执行新查询或同块补读；A在15/55题（27.3%）触发20次查询补检。因此本批能观察邻块读取的行为，不能单独证明新read_chunk工具的总体收益。自己的三个Agent组查询缓存命中均为0。Low记录935次查询Embedding应用成功，其中825次在同题内输入哈希重复；这是重复调用观察，尚不能当作已经节省的实际收费。', '',
              '技术失败共5次：RAG的079/214在确认未发送的连接失败后触及原2次生成尝试上限；A的023查询Embedding连接失败、287生成连接失败；A+B的036生成连接失败。原始状态分别保留在raw_table.csv。没有发现Token总量、60/90秒超时导致的中止；不将传输失败包装成内容错误或通过重跑替换。', '',
              '## 必答点与引用检查', '',
              '| 方案 | 已覆盖必答点 / 100 | 尚未确认的点（含技术失败） | 有未预标引用的题数 |',
              '|---|---:|---:|---:|']
    alternative = readl(P / 'alternative_citations.jsonl')
    for a in ARMS:
        m = metrics[a]
        extra = {r['question_id'] for r in alternative if r['cohort'] == 'extension' and r['arm'] == a}
        lines.append(f'| {NAMES[a]} | {m["required_points_passed"]} | {m["required_points_pending"]} | {len(extra)} |')
    lines += ['', '点覆盖与整题通过是两个指标；版本归属错误也可能导致整题待复核，即使必答点本身正确。所有运行的引用编号均在返回来源范围内，但这只验证映射，不验证条款是否支持结论。未预标来源先入队，不自动扣分。Low的280结论正确，却使用动态调价/支付标准等间接材料论证“不再议价”，其引用语义支持仍需复核；165则无引用地答错审批主体。', '',
              '## 难度分组', '', '| 方案 | Medium（52） | Hard（3） |', '|---|---:|---:|']
    for a in ARMS:
        m = metrics[a]['by_difficulty']
        lines.append(f'| {NAMES[a]} | {m["Medium"]["accepted"]}/52 | {m["Hard"]["accepted"]}/3 |')
    lines += ['', '本批无Easy题。Hard只有3题，不据此推断Hard总体水平。样本是同一语料、已见政策组下的新问题，不能据此声称跨政策泛化或上线准确率。', '',
              '## 可追溯的局部效果', '',
              '新增165：Low的候选原文包含省级医保/财政部门批准条款，最终生成上下文却只保留文件导言，模型随后无引用地答成省级人民政府批准。属于已有依据未进入最终上下文，再叠加无依据生成。', '',
              '新增079：Low最终生成请求将原题改成一般受理条件，删除甲乙丙、4个月、3个月和35分情境，答案未回应评分能否绕过受理限制。独立Agent与Native RAG均保留原问题，详见 question_trace_audit.json。文字改写本身不是错误，具体必答点丢失才计错。', '',
              '新增053：Low未检得尼洛替尼已纳入和特药保留原流程这两条关键规则，最终无法回答；RAG/v7/A/A+B均通过。Low约22.99秒、17次查询Embedding应用成功调用、6732生成Token。该优势属于独立检索流程相对Low的表现，不能归因于v8.1增量，因为v7也通过。', '',
              '新增082：RAG/v7/A均答通常7天，却称指定细则未明确；A+B补读同一79261号文的前后邻块，从7378c4761cee2fd6找到第九条公示时间7日，最终明确回答7日与20工作日。查询Embedding仍只有1次，但用了3次Judge+1次最终生成，共14791 Token。用户裁定四组二元标准均通过、Low不通过，同时认为A+B回答质量更高；因此它是证据质量改善案例，不计为相对RAG/v7新增正确题。', '',
              '技术回归242/247：长文件关键条款在新窗口中可见，并被正确回答；这是局部证据保留的验证。新增211 A组：一次确认未发送的连接失败后，预留机制将物理调用控制在4次，并成功产出最终答案。', '',
              '## 尚未解决的问题', '',
              '新增029在A组用了两次额外检索、最终回答通过，而v7无需补检已通过；增加调用未带来该题收益。诊断095明确暴露B的边界：题目问徐州，但执行器因词项相似优先补读苏州通知的前后块，甚至覆盖了Judge提出的定向检索。动作有执行不代表动作有效。059仍漏选点数量和异地结算条件。45/276/281的数字条件应用也不是本次窗口与补检机制能够保证解决的。', '',
              '通过内容评阅也不自动代表每个引用在语义上正确。编号/来源映射由运行核验，未预标引用保留在 alternative_citations.jsonl；需结合实际政策版本和条款语义复核。', '',
              '## 方案取舍', '',
              '六题裁定已完成。二元通过率不足以区分全部质量差异，后续可对同一批原始答案增加固定量表的匿名评分，并保留本报告通过率和成本。用户对082的偏好是已知开发观察，质量量表在看到结果后新增，因此本批质量分析属于探索性分析，须先冻结量表再用于95题测试。Native RAG和v7继续作为强对照，v8.1保留为质量候选，主方案取舍需结合质量和额外成本。若继续改Agent，优先研究地区与版本一致的工具选择，避免095错误地区邻块占满行动预算；本轮未实施。', '',
              '冻结95题尚未使用。后续应在方案和评分口径最终冻结后做一次RAG/Low/选定Agent的配对测试；不根据测试结果继续调参。本轮没有运行Medium/High、没有重建索引、没有增加Router。', '',
              '## 逐题原始表', '', '| 题号 | 难度 | RAG | Low | v7 | A | A+B |', '|---|---|---|---|---|---|---|']
    labels = {'complete': '通过', 'incomplete': '未通过', 'pending_review': '待复核', 'technical_error': '技术失败'}
    flat = []
    for q in ids:
        lines.append('| '+q+' | '+qs[q]['difficulty']+' | '+' | '.join(labels[scores[a][q]['outcome']] for a in ARMS)+' |')
        for a in ARMS:
            r = rows[a][q]; s = scores[a][q]
            flat.append({'question_id': q, 'difficulty': qs[q]['difficulty'], 'policy_group_id': qs[q]['policy_group_id'],
                         'arm': a, 'outcome': s['outcome'], 'status': r['status'], 'stop_reason': r.get('stop_reason', ''),
                         'latency_ms': r['latency_ms'], 'generation_tokens': sum(c.get('usage', {}).get('total_tokens', 0) for c in r.get('calls', [])),
                         'note': s.get('note', '')})
    review = ['# v8.1 统一复核队列', '', '每个问题的判定应统一应用到全部方案；不要仅调整Agent。原始运行记录不修改。', '']
    pending = [q for q in ids if any(scores[a][q]['outcome'] == 'pending_review' for a in ARMS)]
    for q in pending:
        review += ['## '+q, '', qs[q]['question'], '', '必答点：'+'；'.join(p['text'] for p in qs[q]['required_points']), '', '**你的裁定：待填写**', '']
        for a in ARMS:
            review += ['### '+NAMES[a], '', scores[a][q]['outcome']+'：'+scores[a][q].get('note',''), '', rows[a][q]['answer'], '']
        for e in qs[q]['evidence']:
            review += ['参考原文 '+e['source_document_id']+'：', '', e['text'], '']
    lines += ['', '附件：[逐题问题、原文和全部答案](all_answers.md)、[统一复核队列](review_queue.md)、[指标](metrics.json)、[完整性核验](integrity.json)、[原始对比表](raw_table.csv)、[运行配置冻结](manifest.json)。']
    (P / 'report.md').write_text('\n'.join(lines)+'\n')
    if not (P / 'review_queue.md').exists():
        (P / 'review_queue.md').write_text('\n'.join(review)+'\n')
    (P / 'paired_success_costs.json').write_text(json.dumps(paired_costs,ensure_ascii=False,indent=2)+'\n')
    (P / 'intervals.json').write_text(json.dumps(intervals, ensure_ascii=False, indent=2)+'\n')
    with (P / 'raw_table.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)


if __name__ == '__main__':
    build()
