"""Paired development quality/cost analysis; oracle results are explicitly hindsight."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import random
from statistics import mean, median

from experiments.evaluation.fast_rag_eval import BASE, DUMP, OUT, PRIVATE, READ, historical, unique_rows, verify


def scores(records):
    result = {}
    seen = set()
    for row in records:
        key = row["arm"], row["question_id"]
        if key in seen:
            raise ValueError("duplicate_grade")
        seen.add(key)
        if row["status"] == "ok" and row["grade"]["total"] is not None:
            result.setdefault(row["arm"], {})[row["question_id"]] = row["grade"]["total"]
    return result


def cost(row):
    known = reserve = unknown = 0
    for call in row.get("calls", []):
        if "usage" in call:
            known += call["usage"]["total_tokens"]
        elif call.get("status") in ("not_sent", "provider_not_sent"):
            continue
        else:
            unknown += 1
            value = call.get("reserve_tokens")
            if type(value) is not int or value <= 0:
                raise ValueError("unknown_usage_without_reserve")
            reserve += value
    return {"known_tokens": known, "unknown_reserve": reserve, "unknown_calls": unknown}


def distribution(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    ordered = sorted(values)
    return {"n": len(values), "mean": mean(values), "median": median(values), "p95": ordered[math.ceil(.95 * len(values)) - 1]}


def oracle(ids, quality, rows):
    # Zero score loss per question is an explicit feasibility criterion, not a learned threshold.
    routes = {qid: "fast" if quality["fast"][qid] >= quality["ab"][qid] else "ab" for qid in ids}
    if not ids:
        return {"n": 0}
    chosen = [rows[arm][qid] for qid, arm in routes.items()]
    if any(cost(row)["unknown_calls"] for row in chosen):
        raise ValueError("oracle_requires_known_usage")
    return {"n": len(ids), "fast_questions": sum(a == "fast" for a in routes.values()),
            "quality_mean": mean(quality[a][q] for q, a in routes.items()),
            "latency_mean_ms": mean(r["latency_ms"] for r in chosen),
            "latency_p95_ms": distribution([r["latency_ms"] for r in chosen])["p95"],
            "generation_mean_tokens": mean(cost(r)["known_tokens"] for r in chosen),
            "routes": routes, "implementation": False,
            "limit": "Hindsight grades are unavailable at serving time; historical durations and zero router overhead make this optimistic."}


def _paired(ids, left, right, quality, questions):
    delta = {qid: quality[left][qid] - quality[right][qid] for qid in ids}
    if not ids:
        return {"n": 0}
    groups = defaultdict(list)
    for qid in ids:
        groups[questions[qid]["policy_group_id"]].append(delta[qid])
    rng = random.Random(20260927)
    grouped = list(groups.values())
    boot = sorted(mean(v for values in rng.choices(grouped, k=len(grouped)) for v in values) for _ in range(2000))
    return {"n": len(ids), "left_mean": mean(quality[left][q] for q in ids),
            "right_mean": mean(quality[right][q] for q in ids), "delta_mean": mean(delta.values()),
            "wins": sum(d > 0 for d in delta.values()), "ties": sum(d == 0 for d in delta.values()),
            "losses": sum(d < 0 for d in delta.values()), "policy_groups": len(grouped),
            "cluster_bootstrap_95": [boot[49], boot[1949]], "question_ids": ids}


def write_report(out=OUT):
    manifest = verify(out)
    ids = manifest["question_ids"]
    questions = {q["question_id"]: q for q in READ(BASE / "dev.jsonl")}
    rows = historical()
    rows["fast"] = unique_rows([READ(out / "generation/records.jsonl")])
    current_grades = READ(out / "quality/records.jsonl")
    grading_jobs = json.loads((out / "quality/jobs.json").read_text())
    grading_complete = {r["id"] for r in current_grades} == {j["id"] for j in grading_jobs}
    historical_grades = [r for r in READ(PRIVATE / "answer-quality-dev144-flash/records.jsonl") if r["arm"] in ("rag", "ab")]
    grading = current_grades + historical_grades
    quality = scores(grading)
    arms = ("fast", "rag", "ab")
    for arm in arms:
        quality.setdefault(arm, {})
    data = {"scope": "199 development questions; scores are quality points, not accuracy",
            "grading_complete": grading_complete, "grading_attempts": len(current_grades), "grading_planned": len(grading_jobs),
            "generation_status": {}, "grade_status": {}, "cost_all_attempts": {}, "pairs": {}, "difficulty": {}, "timings": {}}
    for arm in arms:
        data["generation_status"][arm] = dict(Counter(rows[arm][q]["status"] for q in ids if q in rows[arm]))
        data["grade_status"][arm] = dict(Counter(r["status"] for r in grading if r["arm"] == arm))
        costs = [cost(rows[arm][q]) for q in ids if q in rows[arm]]
        data["cost_all_attempts"][arm] = {key: sum(c[key] for c in costs) for key in ("known_tokens", "unknown_reserve", "unknown_calls")}
    for left, right in (("fast", "rag"), ("fast", "ab"), ("ab", "rag")):
        common = [q for q in ids if q in quality[left] and q in quality[right]]
        data["pairs"][left + "-" + right] = _paired(common, left, right, quality, questions)
    common = [q for q in ids if all(q in quality[a] for a in arms)]
    data["three_way_questions"] = common
    data["three_way_quality"] = {a: mean(quality[a][q] for q in common) if common else None for a in arms}
    for difficulty in ("Easy", "Medium", "Hard"):
        group = [q for q in common if questions[q]["difficulty"] == difficulty]
        data["difficulty"][difficulty] = {"n": len(group), **{a: mean(quality[a][q] for q in group) if group else None for a in arms}}
    for arm in arms:
        selected = [rows[arm][q] for q in common]
        data["timings"][arm] = {"latency_ms": distribution([r["latency_ms"] for r in selected]),
                                "generation_tokens": distribution([cost(r)["known_tokens"] for r in selected if not cost(r)["unknown_calls"]]),
                                "prompt_tokens": distribution([sum(c["usage"]["prompt_tokens"] for c in r.get("calls", []))
                                                               for r in selected if not cost(r)["unknown_calls"]]),
                                "completion_tokens": distribution([sum(c["usage"]["completion_tokens"] for c in r.get("calls", []))
                                                                   for r in selected if not cost(r)["unknown_calls"]]),
                                "query_embedding_calls": distribution([r["query_embedding_calls"] for r in selected])}
    data["fast_phases"] = {k: distribution([r["usage"][k] for r in rows["fast"].values() if k in r.get("usage", {})])
                           for k in ("retrieval_ms", "selection_ms", "generation_ms", "context_tokens")}
    data["fast_context"] = {"selected_chunk_counts": dict(Counter(len(r.get("selected", [])) for r in rows["fast"].values())),
                            "truncated_questions": sum(any(e["truncated"] for e in r.get("selected", [])) for r in rows["fast"].values())}
    known = [q for q in common if all(not cost(rows[a][q])["unknown_calls"] for a in arms)]
    data["hindsight_oracle"] = oracle(known, quality, rows)
    identical, differences = [], []
    for qid, r in rows["fast"].items():
        initial = next((t["initial_ids"] for t in r.get("trace", []) if t.get("event") == "fast_context"), None)
        old = rows["rag"][qid].get("ordinary_context", {}).get("chunks")
        if initial is not None and old is not None:
            previous = [c.get("chunk_id", c.get("id")) for c in old]
            identical.append(initial == previous)
            if initial != previous:
                differences.append({"question_id": qid, "same_first_three": initial[:3] == previous[:3],
                                    "fast_initial": initial, "historical_initial": previous})
    data["retrieval_order_audit"] = {"compared": len(identical), "identical": sum(identical), "differences": differences}
    queue = []
    for r in grading:
        if r["status"] != "ok" or r.get("grade", {}).get("review_needed") or r.get("grade", {}).get("critical_error"):
            queue.append({"question_id": r["question_id"], "arm": r["arm"], "status": r["status"],
                          "reason": r.get("validation_error"), "grade": r.get("grade")})
    data["review_count"] = len(queue)
    data["judge_usage_this_run"] = {"known_tokens": sum(r.get("ledger", {}).get("provider_total_tokens", 0) for r in current_grades),
         "unknown_or_failed_records": sum(bool(r.get("usage_unknown") or r.get("ledger", {}).get("pending_calls") or r["status"] in ("host_timeout", "worker_error")) for r in current_grades)}
    DUMP(out / "metrics.json", data)
    DUMP(out / "review_queue.json", queue)
    with (out / "paired_scores.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["question_id", "difficulty", "fast", "rag", "ab", "fast_minus_rag", "fast_minus_ab"])
        for qid in ids:
            s = {a: quality[a].get(qid) for a in arms}
            writer.writerow([qid, questions[qid]["difficulty"], s["fast"], s["rag"], s["ab"],
                             s["fast"] - s["rag"] if s["fast"] is not None and s["rag"] is not None else "",
                             s["fast"] - s["ab"] if s["fast"] is not None and s["ab"] is not None else ""])
    state = "全部评分尝试已结束" if grading_complete else f"评分尚未完成：{len(current_grades)}/{len(grading_jobs)}，以下仅为过程数据"
    lines = ["# Fast RAG v1 开发集实验", "", "**" + state + "。**", "", "固定 1000 篇语料，199 道开发题；检索/模型/回答模板不变。Fast 最终最多三块、2000 Token 原文，无 Judge、一次生成。",
             "", "评分为匿名质量分（满分100），不是正确率。三组使用 deepseek-v4.1-flash 及同一量表；144题旧分复用，55题旧答案重新评分。历史用时不属于同时随机对照。", "",
             "| 比较（前者减后者） | 有效配对 | 平均分差 | 按政策组重采样95%区间 | 胜/平/负 |",
             "|---|---:|---:|---|---|" ]
    for name, pair in data["pairs"].items():
        if pair["n"]:
            lines.append(f"| {name} | {pair['n']} | {pair['delta_mean']:+.2f} | {pair['cluster_bootstrap_95'][0]:+.2f}～{pair['cluster_bootstrap_95'][1]:+.2f} | {pair['wins']}/{pair['ties']}/{pair['losses']} |")
    lines += ["", f"三组共同有效评分 {len(common)}/199；以下成本均按这组相同题目计算。技术失败、非法评分和缺分单列，不记零分。", "",
              "| 方案 | 质量均分 | 平均耗时(s) | P95(s) | 平均生成Token | 平均查询Embedding次数 |", "|---|---:|---:|---:|---:|---:|"]
    for a, label in (("fast", "Fast RAG"), ("rag", "Native RAG"), ("ab", "Search Agent v8.1")):
        t = data["timings"][a]
        if common:
            lines.append(f"| {label} | {data['three_way_quality'][a]:.2f} | {t['latency_ms']['mean']/1000:.2f} | {t['latency_ms']['p95']/1000:.2f} | {t['generation_tokens']['mean']:.0f} | {t['query_embedding_calls']['mean']:.2f} |")
    lines += ["", "| 难度 | 三组有效配对 | Fast | RAG | v8.1 |", "|---|---:|---:|---:|---:|"]
    for d, v in data["difficulty"].items():
        if v["n"]:
            lines.append(f"| {d} | {v['n']} | {v['fast']:.2f} | {v['rag']:.2f} | {v['ab']:.2f} |")
    if common:
        lines += ["", "同题平均生成输入/输出Token：" + "；".join(
            f"{a} {data['timings'][a]['prompt_tokens']['mean']:.0f}/{data['timings'][a]['completion_tokens']['mean']:.0f}" for a in arms) + "。",
            "耗时为每题已启动运行环境内的墙钟时间，不含容器/模型客户端的批次启动；生成Token含该题所有生成请求，未发送失败按账本计。引用支持评分沿用完整被引来源块；不把完整块包含证据等同于截取的生成上下文已覆盖全部证据。"]
    lines += ["", f"Fast 选块数分布：{data['fast_context']['selected_chunk_counts']}；触发段落裁剪 {data['fast_context']['truncated_questions']} 题。"]
    lines += ["", f"初检顺序复核：{sum(identical)}/{len(identical)} 与历史Native RAG完全一致；其余题号、前三块是否相同详见metrics.json。配置相同不保证不同批次的全部实际排序完全一致。"]
    o = data["hindsight_oracle"]
    if o["n"]:
        lines += ["", f"事后理想路由：知道两路质量分后，仅当 Fast 分数不低于 v8.1 时选 Fast，选中 {o['fast_questions']}/{o['n']} 题；质量均分 {o['quality_mean']:.2f}，平均 {o['latency_mean_ms']/1000:.2f}s，平均生成 {o['generation_mean_tokens']:.0f} Token。",
                  "这只是可行性上限，不是可上线分类器的结果；未计路由成本，沿用不同批次观测时间，不能据此宣称 Adaptive RAG 已实现收益。"]
    lines += ["", "生成状态：`" + json.dumps(data["generation_status"], ensure_ascii=False) + "`。",
              "", "评分状态：`" + json.dumps(data["grade_status"], ensure_ascii=False) + "`。",
              "", f"裁判争议/非法评分 {len(queue)} 条，详见 review_queue.json。保留既有人工审查材料；新报告展示统一裁判原始分，尚未把新争议当作已确认错误。",
              "", "全部尝试用量（含失败；未知用量保留预留）：`" + json.dumps(data["cost_all_attempts"], ensure_ascii=False) + "`。",
              "", "限制：开发题已反复用于方法选择；95题旧测试已曝光。本批结论用于可行性诊断。若继续训练路由，须按政策组分折，仅用开发集；最终收益需要新的独立验证。"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    answers = ["# Fast RAG 开发集答案（可与 paired_scores.csv / review_queue.json 对照）", ""]
    for qid in ids:
        answers += [f"## {qid} · {questions[qid]['difficulty']}", "", questions[qid]["question"], ""]
        for arm in arms:
            row = rows[arm].get(qid, {})
            answers += [f"### {arm} · {row.get('status', 'unattempted')} · 质量分 {quality[arm].get(qid, '缺失')}", "", row.get("answer", ""), ""]
    (out / "all_answers.md").write_text("\n".join(answers))
    print(json.dumps({"report": str(out / "report.md"), "three_way_pairs": len(common), "review_count": len(queue)}), flush=True)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=type(OUT), default=OUT)
    write_report(parser.parse_args().output)
