"""Offline evaluation only. Labels and reference answers never enter the Agent."""
import argparse
import collections
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/results/private/benchmark-v2-baselines-v2-nonthinking"


def readl(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def summarize(ids, rows, assessments):
    selected = [r for r in rows if r["question_id"] in ids]
    times = sorted(r["latency_ms"] / 1000 for r in selected)
    physical_calls = [r.get("embedding_physical", {}).get("query_physical_calls") for r in selected]
    return {"planned": len(ids), "attempted": len(selected),
            "accepted": sum(assessments.get(q, {}).get("outcome") == "complete" for q in ids),
            "unassessed": sum(q not in assessments for q in ids),
            "outcomes": dict(collections.Counter(assessments.get(q, {}).get("outcome", "unassessed") for q in ids)),
            "statuses": dict(collections.Counter(r["status"] for r in selected)),
            "mean_seconds": round(statistics.mean(times), 2) if times else None,
            "p95_seconds": round(times[math.ceil(.95 * len(times)) - 1], 2) if times else None,
            "generation_calls": sum(len(r.get("calls", [])) for r in selected),
            "generation_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for r in selected for c in r.get("calls", [])),
            "unknown_generation_reservation": sum(c["reserve_tokens"] for r in selected for c in r.get("calls", []) if "usage" not in c),
            "query_embedding_calls": sum(r.get("query_embedding_calls", 0) for r in selected),
            "query_embedding_physical_calls": sum(physical_calls) if physical_calls and all(type(n) is int and n >= 0 for n in physical_calls) else None,
            "embedding_cache_hits": sum(r.get("embedding_cache_hits", 0) for r in selected),
            "controller_failures": sum(bool(r.get("controller_error")) for r in selected),
            "controller_telemetry_incomplete": sum(r.get("control_trace_complete") is False for r in selected),
            "fallback_questions": sum(bool(r.get("fallback_used")) for r in selected),
            "fallback_answered_questions": sum(bool(r.get("fallback_used")) and r["status"] == "ok" and bool(r.get("answer")) for r in selected),
            "supplement_questions": sum(any(t.get("event") == "tool" for t in r.get("trace", [])) for r in selected)}


def paired_changes(ids, baseline, agent):
    pending = {"pending_review", "unassessed"}
    valid = [q for q in ids if q in baseline and q in agent and baseline[q] not in pending and agent[q] not in pending]
    return {"paired": len(valid), "improved": [q for q in valid if baseline[q] != "complete" and agent[q] == "complete"],
            "regressed": [q for q in valid if baseline[q] == "complete" and agent[q] != "complete"],
            "unpaired": [q for q in ids if q not in valid]}


def norm(value):
    return re.sub(r"\W", "", value)


def diagnostics(question, record, document_map, override):
    seen = record.get("seen_chunks", {})
    body = record.get("final_request") or {}
    attempts = [c for c in record.get("calls", []) if c.get("request") == body and c.get("transport_status") != "not_sent"]
    delivery = "confirmed" if any(c.get("usage", {}).get("total_tokens", 0) > 0 for c in attempts) else "unknown" if attempts else "not_sent"
    # The request is captured before reservation. A budget-blocked body never
    # reached the model; unsettled delivery cannot prove context coverage either.
    request_text = "\n".join(m.get("content", "") for m in body.get("messages", []) if isinstance(m.get("content"), str)) if delivery == "confirmed" else ""
    candidate_covered, final_covered = [], []
    # Associate literal text with its actual rendered source block, not any text
    # anywhere in the request (different policy versions may repeat a sentence).
    headers = list(re.finditer(r"(?m)^ID: (\d+)\n├── Title: ([^\n]*)\n└── Content:\n", request_text))
    selected = record.get("selected", [])
    final_by_source = collections.defaultdict(list)
    for n, match in enumerate(headers):
        idx = int(match[1])
        if idx >= len(selected):
            continue
        item = selected[idx]
        original = seen.get(item["chunk_id"], {})
        if match[2] != original.get("docnm_kwd"):
            continue
        text = request_text[match.end():headers[n + 1].start() if n + 1 < len(headers) else len(request_text)]
        final_by_source[document_map.get(item["document_id"])].append(text)
    # Strict literal support is a diagnostic lower bound, not semantic grading.
    for evidence in question["evidence"]:
        needle = norm(evidence["text"])
        source = evidence["source_document_id"]
        relevant = [c["content_with_weight"] for c in seen.values() if document_map.get(c["doc_id"]) == source]
        candidate_covered.append(any(needle in norm(text) for text in relevant))
        final_covered.append(any(needle in norm(text) for text in final_by_source[source]))
    refs = record.get("reference", {}).get("chunks", [])
    cited = sorted(set(map(int, re.findall(r"\[ID:(\d+)\]", record.get("answer", "")))))
    accepted = set(question["known_relevant_chunk_ids"]) | set(override.get("accepted_alternative_chunk_ids", []))
    unknown = [i for i in cited if i < len(refs) and refs[i].get("id", refs[i].get("chunk_id")) not in accepted]
    return {"candidate_literal_evidence": candidate_covered, "final_literal_evidence": final_covered,
            "final_request_delivery": delivery,
            "reference_spans": len(candidate_covered), "citation_ids": cited, "unmapped_citation_ids": [i for i in cited if i >= len(refs)],
            "additional_citation_ids_pending": unknown,
            "label": "literal reference-span diagnostics only; unknown citations not counted as incorrect"}


def build(out):
    manifest = json.loads((out / "manifest.json").read_text())
    config = json.loads((out / "config.json").read_text())
    ids = manifest["planned"]
    rows = readl(out / "records.jsonl")
    assessments = {r["question_id"]: r for r in readl(out / "assessments.jsonl")}
    native_assessments = readl(BASE / "assessments.jsonl")
    questions = {q["question_id"]: q for q in readl(BASE / "dev.jsonl")}
    overrides = json.loads((BASE / "evaluation_overrides.json").read_text())["questions"]
    metrics = {"agent": summarize(ids, rows, assessments)}
    native_rows = {arm: readl(BASE / arm / "records.jsonl") for arm in ("rag", "low")}
    for arm in ("rag", "low"):
        scores = {r["question_id"]: r for r in native_assessments if r["arm"] == arm}
        metrics[arm] = summarize(ids, native_rows[arm], scores)
        metrics["vs_" + arm] = paired_changes(ids, {q: r["outcome"] for q, r in scores.items()}, {q: r["outcome"] for q, r in assessments.items()})
    metrics["by_difficulty"] = {level: summarize([q for q in ids if questions[q]["difficulty"] == level], rows, assessments) for level in ("Easy", "Medium", "Hard")}
    metrics["grading"] = "Assistant provisional; baseline six-case human adjudications and shared overrides applied. Not held-out test accuracy."
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    details, diagnostics_rows = [], []
    for r in rows:
        qid = r["question_id"]
        q = questions[qid]
        diag = diagnostics(q, r, config["document_map"]["documents"], overrides.get(qid, {}))
        diagnostics_rows.append({"question_id": qid, **diag})
        score = assessments.get(qid, {"outcome": "unassessed", "note": "待内容评阅"})
        details += [f'## {qid} · {q["difficulty"]}', '', q["question"], '', '**参考答案**：' + q["reference_answer"], '',
                    '**必答点**：' + '；'.join(p["text"] for p in q["required_points"]), '']
        if qid in overrides:
            details += ['**统一评分修订**：' + json.dumps(overrides[qid], ensure_ascii=False), '']
        details += ['**Agent回答**：' + (r.get("answer") or '（无最终回答）'), '',
                    f'**当前评定**：{score["outcome"]}；{score.get("note", "")}', '',
                    f'**运行**：{r["status"]}；停止：{r.get("stop_reason", "")}；{r["latency_ms"]/1000:.2f}秒', '']
        for t in r.get("trace", []):
            if t.get("event") == "tool":
                details += ['**工具动作**：' + json.dumps(t, ensure_ascii=False), '']
        for e in q["evidence"]:
            details += [f'**参考原文：{e["source_title"]}**', '', e["text"], '']
        for i in diag["citation_ids"]:
            refs = r.get("reference", {}).get("chunks", [])
            if i < len(refs):
                ref = refs[i]
                details += [f'**实际引用 ID:{i} · {ref.get("document_name", ref.get("docnm_kwd", ""))}**', '', ref.get("content", ref.get("content_with_weight", "")), '']
    (out / "all_answers.md").write_text('# Search Agent开发题回答与依据\n\n未标注但可能正确的引用先待复核，不自动计错。\n\n' + '\n'.join(details))
    (out / "diagnostics.jsonl").write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in diagnostics_rows))
    lines = ['# 独立 Search Agent 首版开发实验', '', '固定1000篇、原索引模型、统一出站脱敏；仅开发题。RAG/Low为原生固定对照。', '',
             '内容分数为助手初评，沿用基线六题人工裁定及统一评分修订；不能表述为最终测试准确率。未评阅与未执行项保留，不当成已确认内容错误。', '',
             '| 指标 | RAG | Low | Search Agent |', '|---|---:|---:|---:|']
    if metrics['agent']['unassessed'] == len(ids):
        lines[2:2] = ['**尚无Agent质量结论。** 本批没有已评分答案；“内容通过0”不是0%的准确率，不能与基线质量排名。请先查看运行状态及实现说明。', '']
    for title, key in [('计划题数', 'planned'), ('已执行', 'attempted'), ('内容通过', 'accepted'), ('未评阅', 'unassessed'),
                       ('平均秒数', 'mean_seconds'), ('P95秒数', 'p95_seconds'), ('生成Token', 'generation_tokens'), ('生成调用', 'generation_calls')]:
        lines.append('| ' + title + ' | ' + ' | '.join(str(metrics[a][key]) for a in ('rag', 'low', 'agent')) + ' |')
    lines += ['', '平均/P95只描述已执行请求，技术失败不删除；未完成时不能用整批基线与部分Agent的耗时作总体比较。', '',
              'Agent查询向量物理调用：' + str(metrics['agent']['query_embedding_physical_calls']) + '；缓存命中：' + str(metrics['agent']['embedding_cache_hits']) + '。基线原报表为应用层成功调用，不能未经核实视为物理请求次数。', '']
    lines += [f'已记录控制器异常：{metrics["agent"]["controller_failures"]}题；选择原生降级：{metrics["agent"]["fallback_questions"]}题，其中正常产出答案{metrics["agent"]["fallback_answered_questions"]}题。降级回答不等于控制器正常。',
              f'控制事件记录不完整：{metrics["agent"]["controller_telemetry_incomplete"]}题；这类记录不能据缺少事件判定控制器未出错。', '']
    for arm in ('rag', 'low'):
        change = metrics['vs_' + arm]
        lines += [f'相对{arm}：新增通过{len(change["improved"])}题，退步{len(change["regressed"])}题；未形成已评分配对{len(change["unpaired"])}题。',
                  '新增：' + ('、'.join(change['improved']) or '无') + '；退步：' + ('、'.join(change['regressed']) or '无'), '']
    lines += ['## 分难度', '', '| 难度 | 通过/计划 | 已执行 |', '|---|---:|---:|']
    for d, m in metrics['by_difficulty'].items():
        lines.append(f'| {d} | {m["accepted"]}/{m["planned"]} | {m["attempted"]} |')
    lines += ['', '候选/最终参考原文字面覆盖仅作诊断，不等于语义支持或问答准确率。替代依据按统一规则补标。', '',
              '附件：[逐题答案与原文](all_answers.md)、[指标](metrics.json)、[诊断](diagnostics.jsonl)、[运行冻结](manifest.json)。']
    (out / "report.md").write_text('\n'.join(lines) + '\n')
    print(json.dumps(metrics['agent'], ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    build(parser.parse_args().run.resolve())
