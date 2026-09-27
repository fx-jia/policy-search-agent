"""Build the expanded review release using only local files; no model or network calls."""
import copy
from evidence import *
from author_expansion import items as authored
from author_boundary import items as boundary_authored

PILOT = OUT.parent / 'benchmark_v2_pilot60' / 'benchmark_v2_pilot60.jsonl'
pilot = [json.loads(line) for line in PILOT.read_text().splitlines()]
protected = [RAW, FROZEN, REVIEW, OLD] + sorted(p for p in PILOT.parent.iterdir() if p.is_file())
lock_path = OUT / 'input_lock.json'
if not lock_path.exists():
    dump('input_lock.json', {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in protected})
for name, digest in json.loads(lock_path.read_text()).items():
    assert sha((ROOT/name).read_bytes()) == digest, f'Input modified: {name}'
new=[]; issues=[]
for a in authored:
    q=copy.deepcopy({k:v for k,v in a.items() if k!='evidence_specs'})
    ev=[bind_evidence(s,i+1) for i,s in enumerate(a['evidence_specs'])]
    sids=sorted({e['source_document_id'] for e in ev})
    q.update(dataset_version='benchmark_v2_full330_candidate_v1', evidence=ev,
        source_dataset_sha256=sha(RAW.read_bytes()), known_relevant_document_ids=sids,
        known_relevant_chunk_ids=sorted({c['chunk_id'] for e in ev for c in e['indexed_safe_chunk_links']}),
        answerable=True, answerability_scope='fixed_corpus1000_local_snapshot_with_unified_sanitized_text',
        review_status='pending_human_review', reviewer=None, difficulty_status='provisional_structural_label',
        split='unassigned_candidate', eligible_for_unseen_test=False,
        qrels_scope='known_positive_evidence_bundles_only_not_exhaustive',
        unlabelled_citation_rule='record_for_review; do not mark incorrect solely because absent from annotations',
        creation_method='assistant_local_source_reading_and_scenario_authoring_no_model_api',
        source_group_seed=sids, group_split_status='not_frozen; join source versions, duplicate policies, shared families before split',
        dedup_status='exact_question_checked; lexical_neighbors_only; semantic_review_pending',
        historical_scope='仅按题干文件与年度，不代表现行有效政策。',
        scope_note='候选题待人工审核；难度为结构标签，未以模型成败反推。')
    for e in ev:
        if e['safe_text_alignment']['coverage']<1: issues.append({'question_id':q['question_id'],'evidence_id':e['evidence_id'],**e['safe_text_alignment']})
    new.append(q)
dump('alignment_issues.json',issues)
assert not issues, 'Resolve exact evidence visibility first.'
assert len(new)==240
core=pilot+new
byid={q['question_id']:q for q in core}
boundary=[]
for a in boundary_authored:
    q=copy.deepcopy(a); evs={}
    for pid in q['parent_question_ids']:
        for e in byid[pid]['evidence']:
            evs[(e['source_document_id'],e['start'],e['end'])]=copy.deepcopy(e)
    for spec in q.pop('extra_evidence_specs',[]):
        e=bind_evidence(spec,1)
        assert e['safe_text_alignment']['coverage']==1
        evs[(e['source_document_id'],e['start'],e['end'])]=e
    ev=list(evs.values())
    for i,e in enumerate(ev): e['evidence_id']=f'e{i+1}'
    q.update(evidence=ev, dataset_version='benchmark_v2_boundary30_candidate_v1',
        task_type='clarification_and_evidence_scope', answerable=None,
        answerability_scope='not_a_proven_corpus_unanswerable_label',
        required_behavior=['识别会改变结论的缺失条件或证据边界','可先提供有依据的条件化说明，必要时澄清','不得作题内列出的无依据推断'],
        review_status='pending_human_review', reviewer=None, difficulty_status='provisional_structural_label',
        split='separate_boundary_challenge', eligible_for_unseen_test=False,
        parent_family_ids=sorted({byid[x]['question_family_id'] for x in q['parent_question_ids']}),
        known_relevant_document_ids=sorted({e['source_document_id'] for e in ev}),
        qrels_scope='illustrative_policy_evidence_not_exhaustive',
        unlabelled_citation_rule='record_for_review; do not mark incorrect solely because absent from annotations')
    boundary.append(q)
assert len(boundary)==30
allq=core+boundary
assert len({q['question_id'] for q in allq})==330
assert len({norm(q['question']) for q in allq})==330
assert not ({norm(q['question']) for q in new+boundary} & {norm(q['question']) for q in old})
for q in core:
    ids={e['evidence_id'] for e in q['evidence']}
    for p in q['required_points']:
        assert all(bundle and set(bundle)<=ids for bundle in p['support_sets'])

# Literal bigram overlap is a review queue, never a proof of semantic independence.
def grams(t):
    t=norm(t); return {t[i:i+2] for i in range(len(t)-1)}
def score(a,b): return len(a&b)/len(a|b) if a|b else 0
pool=[(q['question_id'],q['question'],grams(q['question'])) for q in old+core]
neighbors=[]
for q in new+boundary:
    g=grams(q['question']); ranked=sorted(((score(g,h),qid,t) for qid,t,h in pool if qid!=q['question_id']),reverse=True)[:3]
    neighbors.append({'question_id':q['question_id'],'method':'normalized_character_bigram_Jaccard','neighbors':[{'question_id':qid,'similarity':round(s,4),'question':t} for s,qid,t in ranked]})
dump('lexical_neighbors.json',neighbors)
flags=[{'question_id':r['question_id'],**n} for r in neighbors for n in r['neighbors'] if n['similarity']>=.6]
dump('lexical_review_queue.json',flags)

# Conservative connected components join shared original documents and named families.
# Policy versions or duplicate documents with different IDs still need human joining.
parents={q['question_id']:q['question_id'] for q in allq}
def find(x):
    while parents[x]!=x: parents[x]=parents[parents[x]]; x=parents[x]
    return x
def union(a,b): parents[find(a)]=find(b)
seen={}
for q in allq:
    keys=['source:'+e['source_document_id'] for e in q['evidence']]
    keys+=['family:'+f for f in ([q['question_family_id']] if 'question_family_id' in q else q['parent_family_ids'])]
    for key in keys:
        if key in seen: union(q['question_id'],seen[key])
        else: seen[key]=q['question_id']
groups=collections.defaultdict(list)
for qid in parents: groups[find(qid)].append(qid)
old_sids={str(sid) for q in old for sid in q.get('relevant_document_ids',[q['source_document_id']])}
qidmap={q['question_id']:q for q in allq}; group_rows=[]
for i,ids in enumerate(sorted(groups.values(),key=lambda v:min(v)),1):
    sids=sorted({e['source_document_id'] for qid in ids for e in qidmap[qid]['evidence']})
    group_rows.append({'group_id':f'conservative-{i:03d}','question_ids':sorted(ids),'source_ids':sids,
        'contains_pilot_dev':any(x in {p['question_id'] for p in pilot} for x in ids),
        'shared_source_with_old150':sorted(set(sids)&old_sids),'split':'unassigned; manual version/family reconciliation required'})
dump('source_group_candidates.json',group_rows)

def write_jsonl(name,rows):
    (OUT/name).write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in rows))
write_jsonl('benchmark_v2_core300.jsonl',core)
write_jsonl('benchmark_v2_new240.jsonl',new)
write_jsonl('benchmark_v2_boundary30.jsonl',boundary)
dump('author_specs_new240.json',authored)
dump('author_specs_boundary30.json',boundary_authored)
selected=sorted({e['source_document_id'] for q in core for e in q['evidence']})
manifest={'version':'candidate_v1','created_date':'2026-09-24','core_count':300,'preserved_pilot_count':60,'new_core_count':240,
    'boundary_count':30,'difficulty_counts':dict(collections.Counter(q['difficulty'] for q in core)),
    'new_difficulty_counts':dict(collections.Counter(q['difficulty'] for q in new)),
    'boundary_difficulty_counts':dict(collections.Counter(q['difficulty'] for q in boundary)),
    'role_counts':dict(collections.Counter(q['user_role'] for q in core)),
    'selected_source_documents':len(selected),'new_question_source_documents':len({e['source_document_id'] for q in new for e in q['evidence']}),
    'core_question_families':len({q['question_family_id'] for q in core}),
    'core_evidence_count':sum(len(q['evidence']) for q in core),
    'conservative_group_count_including_boundary':len(group_rows),
    'corpus_documents':len(frozen_map),'corpus_chunks':sum(len(c) for c in safe_chunks.values()),
    'human_approved_in_this_release':0,'unseen_test_count':0,'review_status':'pending_human_review',
    'generation_api_calls':0,'embedding_api_calls':0,'retrieval_api_calls':0,'system_evaluation_calls':0,
    'all_quote_visibility_passed':True,'semantic_dedup_complete':False,'lexical_flag_pairs':len(flags),
    'protected_input_lock':'input_lock.json','scope':'historical-file-scoped diagnostic candidates; not traffic-representative; not a held-out evaluation result'}
dump('manifest.json',manifest)
dump('source_inventory.json',[{'source_document_id':sid,'title':records[sid]['文件名称'],'body_sha256':sha(records[sid]['政策正文']),'source_url':records[sid].get('数据来源路径',{}).get('原始文件路径'),'frozen_index_entry':frozen_map[sid]} for sid in selected])
review_path=OUT/'review_template.json'
if not review_path.exists():
    dump('review_template.json',[{'question_id':q['question_id'],'status':'待审核','question_ok':None,'answer_supported':None,'evidence_sufficient':None,'difficulty_agreed':None,'suggested_difficulty':None,'additional_support_sources':[],'notes':''} for q in allq])
else:
    assert {q['question_id'] for q in json.loads(review_path.read_text())}=={q['question_id'] for q in allq}

def support(p): return ' 或 '.join(' + '.join(b) for b in p['support_sets'])
def card_md(q):
    lines=[f"## {q['question_id']} · {q['difficulty']} · 待人工审核",'',f"**问题：** {q['question']}",'',f"**参考答案／预期行为：** {q['reference_answer']}",'',f"**难度依据：** {q['difficulty_reason']}",'']
    if 'required_points' in q:
        lines += [f"**角色 / 考点组：** {q['user_role']} / `{q['question_family_id']}`",'', '**必答点：**','']
        lines += [f"- {p['point_id']}：{p['text']}（支持：{support(p)}）" for p in q['required_points']]
    else:
        lines += ['**应识别的缺失条件：**','']+[f'- {s}' for s in q['missing_conditions']]
        lines += ['','**不得直接推断：**','']+[f'- {s}' for s in q['prohibited_inferences']]
        lines += ['',f"关联主集题：{', '.join(q['parent_question_ids'])}。本题不声称全库无答案，也不要求把全部澄清一次问完。"]
    lines += ['','**参考原文：**','']
    for e in q['evidence']:
        lines += [f"### {e['evidence_id']} · {e['source_title']}（{e['source_document_id']}）",'',f"[原始页面]({e['source_url']})；元数据日期：{e['metadata_publication_date']}。以正文时间为准。",'', '\n'.join('> '+s for s in e['text'].splitlines()),'', f"本地原文：第 {e['jsonl_line']} 条记录，政策正文字段字符 [{e['start']}, {e['end']})；SHA256 `{e['quote_sha256']}`。",'',f"索引核验：{len(e['indexed_safe_chunk_links'])} 个已有 Chunk 的规范化字符匹配覆盖 100%。这不等于系统已召回或语义审核通过。",'']
    lines += ['**人工审核：** 待审核','', '- 结论（通过 / 部分支持 / 不通过）：','- 修订建议 / 其他正确支持来源：','']
    return '\n'.join(lines)
volumes=[]
for first in range(0,300,60):
    name=f'report_{first+1:03d}_{first+60:03d}.md'; volumes.append(name)
    intro=f'# 江苏医保 Benchmark v2 · {first+1:03d}–{first+60:03d}\n\n每题包含问题、参考答案、必答点、原文及难度依据。按题干历史文件评分；本卷为候选题，尚未逐题人工通过。未标注的替代证据应记录复核，不直接判错。\n\n'
    (OUT/name).write_text(intro+'\n'.join(card_md(q) for q in core[first:first+60]))
(OUT/'boundary_review_report.md').write_text('# 30道澄清与证据边界挑战题\n\n本集单列，不进入300道可回答题的准确率分母。考查条件澄清、范围限定与避免臆造；不把缺少用户条件标成“全库没有答案”。给出有依据的条件化说明并澄清关键条件即可，不要求一次索取所有信息。\n\n'+'\n'.join(card_md(q) for q in boundary))

esc=html.escape
cards=[]
for q in allq:
    kind='主集' if 'required_points' in q else '边界'
    evidence_html=''.join(f"<details><summary>{esc(e['evidence_id']+' · '+e['source_title'])}</summary><p>来源编号 {e['source_document_id']} · 原始日期字段 {esc(str(e['metadata_publication_date']))}</p><blockquote>{esc(e['text'])}</blockquote></details>" for e in q['evidence'])
    points=''.join('<li>'+esc(p['text'])+' <small>('+esc(support(p))+')</small></li>' for p in q.get('required_points',[]))
    if kind=='边界': points=''.join('<li>需确认：'+esc(s)+'</li>' for s in q['missing_conditions'])+''.join('<li>不得推断：'+esc(s)+'</li>' for s in q['prohibited_inferences'])
    cards.append(f"<article data-difficulty='{q['difficulty']}' data-kind='{kind}'><h2>{q['question_id']} · {q['difficulty']} · {kind}</h2><h3>{esc(q['question'])}</h3><p><b>参考答案／预期行为：</b>{esc(q['reference_answer'])}</p><p><b>难度依据：</b>{esc(q['difficulty_reason'])}</p><ul>{points}</ul>{evidence_html}<p class='status'>待人工审核 · 本页为只读浏览；请在分册或 review_template.json 记录结论。</p></article>")
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>江苏医保 Benchmark v2 · 300 + 30</title><style>body{font-family:system-ui,sans-serif;background:#f4f6f8;color:#182532;line-height:1.75;max-width:1080px;margin:auto;padding:24px}header{background:#fff;padding:24px;border-radius:12px}nav{position:sticky;top:0;background:#e5edf3;padding:12px;display:flex;gap:10px;flex-wrap:wrap}input{flex:1;min-width:200px}input,select{padding:10px;font:inherit}article{background:white;border:1px solid #dce3e9;border-radius:12px;padding:24px;margin:20px 0}h2{font-size:18px;color:#265c76}h3{font-size:20px}blockquote{white-space:pre-wrap;background:#f6f8fa;padding:16px;margin:8px 0;overflow-wrap:anywhere}details{margin:12px 0}summary{cursor:pointer;color:#215a78}small,.status{color:#617283;font-size:13px}[hidden]{display:none!important}@media(max-width:600px){body{padding:12px}article{padding:16px}}</style><header><h1>江苏医保 Benchmark v2</h1><p>300 道主集候选题（原60 + 新240）及单列30道边界挑战题。所有题待人工审核，暂无独立测试集；本轮未调用 Embedding 或配置的生成 API。</p><p>引用按本地历史文档快照，不代表现行政策。字符核验不能替代语义复核。未标注引用先记录复核，不直接判错。页面为只读浏览，不保存审核结论。</p></header><nav><input id="search" placeholder="搜索题号、问题、答案或原文" aria-label="搜索"><select id="difficulty" aria-label="难度"><option value="">全部难度</option><option>Easy</option><option>Medium</option><option>Hard</option></select><select id="kind" aria-label="题集"><option value="">全部题集</option><option>主集</option><option>边界</option></select><span id="count"></span></nav>'''+''.join(cards)+'''<script>
const cards = [...document.querySelectorAll('article')];
const search = document.querySelector('#search'), difficulty = document.querySelector('#difficulty'), kind = document.querySelector('#kind');
function update(){let n=0;for(const card of cards){card.hidden=!(card.textContent.toLowerCase().includes(search.value.trim().toLowerCase())&&(!difficulty.value||card.dataset.difficulty===difficulty.value)&&(!kind.value||card.dataset.kind===kind.value));if(!card.hidden)n++;}document.querySelector('#count').textContent=n+' / 330 题';}
[search,difficulty,kind].forEach(el=>el.addEventListener('input',update));update();
</script></html>'''
(OUT/'benchmark_review_report.html').write_text(page)
counts=manifest['difficulty_counts']
overview=f'''# 江苏医保 Benchmark v2 扩充报告

已制作 **300 道可回答候选题 + 单列 30 道澄清/边界题**：保留首批60题，新增240题主集。所有题仍待逐题人工审核；用户的大致质量认可未转换为批量“通过”。

## 本轮范围与用量

- 固定语料：1000篇文档、3205个已解析Chunk；未扩库或重新入库。
- 本轮Embedding、检索接口、配置的生成模型API和High/Low系统评测调用均为 **0**。问题由助手阅读本地原文编写；“0 API”不表示助手编写与独立复核没有使用模型推理。
- 仅制作数据与报告，未运行系统，不能据此宣称检索或问答效果提升。
- 主集来源覆盖 **{len(selected)}篇**，已知支持引用 **{manifest['core_evidence_count']}段**。不是遍历1000篇后的均匀抽样，也不代表覆盖全库所有主题。

## 难度、构造方法与用途

| 难度 | 主集题数 | 依据 |
|---|---:|---|
| Easy | {counts.get('Easy',0)} | 条件明确，单条直接查找 |
| Medium | {counts.get('Medium',0)} | 整理多个要素、步骤或明确限制；独立子问不自动算Hard |
| Hard | {counts.get('Hard',0)} | 多条件分支、时间/版本切换、先后支付计算或相互依赖证据 |

难度为预先给出的结构标签，不按某个模型答错与否反推，也不强凑三档等量。需后续人工复核并用模型结果校准，但保留版本记录。边界30题的难度单列：{json.dumps(manifest['boundary_difficulty_counts'],ensure_ascii=False)}。

采用“读原文→选独立考点或条件组合→写问题与最小必答点→绑定精确原文→核验索引可见性→人工审核”的过程。每个必答点标明支持组合；数组内为共同需要，多个数组表示可替代支持。计算题中的假设金额与比例必须与政策实际标准分开。

主集沿用明确文件和年度的问法，降低政策适用歧义。包含待遇、参保、长护、异地、救助、经办、采购及机构管理；因此是医保政策诊断集，**不等于纯居民咨询流量样本**。具体角色分布见manifest，角色也是待审核标签；机构题应与居民题分层汇报。较明确的文件提示可能使检索更容易，后续真实问法可在同考点组增加改写，但不能把改写当独立政策事实。

## 审核分册

'''
for i,name in enumerate(volumes): overview+=f'- [{i*60+1:03d}–{i*60+60:03d} 题：'+('首批保留' if i==0 else '新增')+f']({OUT/name})\n'
overview+=f'''- [边界挑战30题]({OUT/'boundary_review_report.md'})
- [可搜索、按难度筛选的只读HTML]({OUT/'benchmark_review_report.html'})
- [结构化审核模板]({OUT/'review_template.json'})

建议从061开始审核新题：检查“题意是否明确、答案是否被原文支持、必答点有没有扩大题意、难度是否合理”。可在分册每题尾部写结论；结构化模板为另一个入口，**两者不自动同步，复核时需指定实际填写的文件**。部分支持请直接记缺失条件或要删改的必答点。HTML仅便于浏览，不保存结论。

## 评价契约与数据划分

回答正确性与引用支持分开评。已知证据不是穷尽标签：模型使用未标注的文档时进入替代证据复核，不能单凭文档ID不在gold中判错。文档级MRR/Recall不能证明有效Chunk或答案必答点已覆盖，应另评上下文证据与最终回答。

边界题允许先给有依据的条件化说明、再问关键条件，不强制僵硬拒答或一次性索取所有信息。题干缺条件不等于全库无答案，也不将当前快照没有验证未来政策记为事实性“政策不存在”。它们单独评澄清、范围限定和无依据承诺，不进主集300题准确率分母。

旧150题文件原样保留。新主集中的首批60已曝光，仍为开发校准；新240暂未分开发/测试。分组表先连结共同来源和考点（连同边界关联共{len(group_rows)}个保守连通组），还须人工补齐跨ID的同政策、版本和近义题关系，之后才能冻结划分。当前 **0道题被声明为未见测试集**。

300是题目条目数，不是300个相互独立的政策事实。已做规范化精确重复检查和字符双字组近邻检索，近邻≥0.60候选共{len(flags)}条方向性配对，供人工查看；这不是Embedding语义去重，也不证明低相似度题无语义重复。见 [近邻]({OUT/'lexical_neighbors.json'}) 和 [保守分组]({OUT/'source_group_candidates.json'})。

## 核验、局限与交付

每段引文保留原始JSONL记录位置、字符起止、原文与引文SHA256、固定语料文档ID及已审脱敏Chunk链接。规范化字符匹配覆盖100%说明文本可在现有索引快照找到，不等于语义正确、相关来源穷尽或实际系统已经召回。

人工内容结论尚未完成；独立助手复核也不替代用户逐题审核。历史文档的元数据日期可能不同于正文签发/生效日，答案按题干和正文限定；本轮未进行现行法规有效性核验。

HTML只做静态结构及JavaScript语法检查；本环境浏览器预览受限，未声称完成视觉和交互实测。Markdown分册是主要审阅材料。

- [300题主集JSONL]({OUT/'benchmark_v2_core300.jsonl'})
- [仅新增240题JSONL]({OUT/'benchmark_v2_new240.jsonl'})
- [30题边界集JSONL]({OUT/'benchmark_v2_boundary30.jsonl'})
- [数据清单与用量]({OUT/'manifest.json'})
- [完整性验证结果]({OUT/'validation.json'})
- [独立内容复核与修订记录]({OUT/'quality_review.md'})

用户已大致浏览认可，允许先评测再对失败和疑似标注问题定向复核；不再要求全部逐题审核后才能启动。后续顺序：按政策/来源/题族冻结划分与当前标签版本 → 在固定语料、相同模型和统一出站脱敏下跑Native RAG、Low Agent及预算受限High对照 → 据失败类型验证Adaptive/Search Agent组件。真实检索阶段通常需要为问题计算Embedding，本次编题阶段没有。
'''
(OUT/'benchmark_overview.md').write_text(overview)
(OUT/'README.md').write_text('# Benchmark v2：300 + 30 候选题\n\n入口：[扩充报告](benchmark_overview.md)。保留首批60，新增240；30边界单列。均待人工审核。\n\n重建：`python3 experiments/datasets/private/benchmark_v2_full330/build_artifacts.py`\n\n验证：`python3 experiments/datasets/private/benchmark_v2_full330/verify_artifacts.py`\n\n重建会覆盖生成分册，若已填写分册审核意见，请先保留副本再重建。review_template.json存在时保持不改。\n')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
