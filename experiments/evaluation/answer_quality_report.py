"""Report saved judge scores with paired comparisons and explicit missingness."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

from experiments.evaluation.answer_quality import WEIGHTS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/results/private/benchmark-v2-baselines-v2-nonthinking"
LABELS = {"rag": "Native RAG", "low": "Native Low", "v7": "v7", "a": "v8.1 A", "ab": "v8.1 A+B"}


def readl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def mean(values):
    return round(statistics.mean(values), 3) if values else None


def quantile(values, p):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (pos - lo), 3)


def paired(a, b, questions):
    ids = sorted(set(a) & set(b))
    delta = {q: a[q] - b[q] for q in ids}
    groups = {}
    for q, value in delta.items():
        groups.setdefault(questions[q]["policy_group_id"], []).append(value)
    rng = random.Random(20260926)
    boot = []
    if groups:
        keys = sorted(groups)
        for _ in range(5000):
            sample = [v for group in rng.choices(keys, k=len(keys)) for v in groups[group]]
            boot.append(statistics.mean(sample))
    return {"n": len(ids), "delta_mean": mean(list(delta.values())),
            "wins": sum(v > 0 for v in delta.values()), "ties": sum(v == 0 for v in delta.values()),
            "losses": sum(v < 0 for v in delta.values()), "policy_groups": len(groups),
            "descriptive_cluster_interval": [quantile(boot, .025), quantile(boot, .975)],
            "by_difficulty": {d: {"n": sum(questions[q]["difficulty"] == d for q in ids),
                "delta_mean": mean([v for q, v in delta.items() if questions[q]["difficulty"] == d])}
                for d in ("Easy", "Medium", "Hard")},
            "deltas": delta}


def report(out):
    manifest = json.loads((out / "manifest.json").read_text())
    config = json.loads((out / "config.json").read_text())
    jobs = json.loads((out / "jobs.json").read_text())
    rows = readl(out / "records.jsonl")
    assert len(rows) == len({r["id"] for r in rows}), "duplicate_result"
    jobmap = {j["id"]: j for j in jobs}
    assert {r["id"] for r in rows} <= set(jobmap)
    qmap = {q["question_id"]: q for q in readl(BASE / "dev.jsonl")}
    ids = manifest["question_ids"]
    arms = [a for a in LABELS if any(j["arm"] == a for j in jobs)]
    count = len(ids)
    source_digest = hashlib.sha256(json.dumps({"rows": rows, "jobs": jobs}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    review_file = f"review_cases-{source_digest[:12]}.md"
    raw = []
    rmap = {r["id"]: r for r in rows}
    for job in jobs:
        row = rmap.get(job["id"], {})
        valid = row.get("status") == "ok"
        grade = row.get("grade", {}) if valid else {}
        raw.append({"question_id": job["question_id"], "arm": job["arm"],
            "difficulty": qmap[job["question_id"]]["difficulty"], "stage": job["stage"],
            "status": row.get("status", "not_attempted"), "validation_error": row.get("validation_error", ""),
            "total": grade.get("total"), **{k: grade.get("scores", {}).get(k) for k in WEIGHTS},
            "review_needed": grade.get("review_needed"), "critical_error_flag_unreviewed": grade.get("critical_error"),
            "judge_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in row["calls"]) if "calls" in row else None,
            "judge_latency_seconds": row["latency_ms"] / 1000 if "latency_ms" in row else None,
            "judge_physical_calls": len(row["calls"]) if "calls" in row else None})
    with (out / "raw_scores.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    table = {}
    scores = {}
    for arm in arms:
        rr = [r for r in raw if r["arm"] == arm]
        valid = [r for r in rr if r["total"] is not None]
        scores[arm] = {r["question_id"]: r["total"] for r in valid}
        technical = sum(e["arm"] == arm for e in manifest["technical_exclusions"])
        missing = len(rr) - len(valid)
        total = sum(r["total"] for r in valid)
        table[arm] = {"original_questions": count, "original_technical_failures": technical,
            "answers_for_grading": len(rr), "numeric_grades": len(valid), "missing_grades": missing,
            "quality_mean": mean([r["total"] for r in valid]),
            "quality_stdev": round(statistics.stdev(r["total"] for r in valid), 3) if len(valid) > 1 else None,
            "dimension_means_0_to_4": {k: mean([r[k] for r in rr if r[k] is not None]) for k in WEIGHTS},
            "all_question_utility_bounds": [round(total / count, 3), round((total + missing * 100) / count, 3)],
            "judge_tokens_known": sum(r["judge_tokens"] for r in rr if r["judge_tokens"] is not None),
            "judge_latency_mean": mean([r["judge_latency_seconds"] for r in rr if r["judge_latency_seconds"] is not None]),
            "judge_latency_p95": quantile([r["judge_latency_seconds"] for r in rr if r["judge_latency_seconds"] is not None], .95),
            "review_flag_count": sum(r["review_needed"] is True for r in rr),
            "critical_flags_unreviewed": sum(r["critical_error_flag_unreviewed"] is True for r in rr),
            "by_difficulty": {d: {"n": sum(r["difficulty"] == d for r in valid),
                 "mean": mean([r["total"] for r in valid if r["difficulty"] == d])} for d in ("Easy", "Medium", "Hard")}}
    comparisons = {"ab_vs_" + a: paired(scores["ab"], scores[a], qmap) for a in arms if a != "ab"}
    calls = [c for row in rows for c in row.get("calls", [])]
    stats = {"model": config["model"], "rubric": manifest["rubric"], "questions": count,
        "planned_grade_jobs": len(jobs), "completed_grade_jobs": len(rows),
        "judge_tokens_known": sum(c.get("usage", {}).get("total_tokens", 0) for c in calls),
        "judge_physical_calls": len(calls), "confirmed_not_sent_calls": sum(c.get("transport_status") == "not_sent" for c in calls),
        "unsettled_call_reservations": sum(c.get("reserve_tokens", 0) for c in calls if "usage" not in c),
        "jobs_without_call_telemetry": sum("calls" not in r for r in rows),
        "unknown_usage_jobs": sum(bool("calls" not in r or r.get("usage_unknown") or r.get("ledger", {}).get("pending_calls")
                                        or any("usage" not in c for c in r.get("calls", []))) for r in rows),
        "table": table, "paired": comparisons}
    (out / "metrics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 已有开发答案的匿名质量评分", "",
        f"裁判：{config['model']}；量表：{manifest['rubric']}。已处理{len(rows)}/{len(jobs)}份已有答案，覆盖{count}题。",
        "这是探索性开发评分，结构与原文摘录校验通过不等于裁判语义判断已获人工确认。原二元内容通过率保持独立。", "",
        "## 质量均分与缺失", "", "总分0—100；维度为准确性35、完整性25、证据支持25、确定性10、清晰度5。",
        "单组均分的有效题目可能不同，结论优先看下方同题配对；评分失败不记0分。", "",
        "| 方案 | 有效数值评分 | 评分缺失 | 原系统技术失败 | 均分 | 标准差 | 全题业务效用范围 |",
        "|---|---:|---:|---:|---:|---:|---|" ]
    for arm, m in table.items():
        lines.append(f"| {LABELS[arm]} | {m['numeric_grades']} | {m['missing_grades']} | {m['original_technical_failures']} | {m['quality_mean']} | {m['quality_stdev']} | {m['all_question_utility_bounds']} |")
    lines += ["", "业务效用范围把原系统技术失败记为0效用，评分缺失视为0—100未知；这不等于把技术失败认定为内容质量0。", "",
        "## v8.1 A+B的同题配对", "", "| 对照 | 双方有效题数 | 平均分差 | 胜/平/负 | 按政策组重采样的描述性95%区间 |",
        "|---|---:|---:|---|---|" ]
    for key, m in comparisons.items():
        lines.append(f"| {LABELS[key.removeprefix('ab_vs_')]} | {m['n']} | {m['delta_mean']} | {m['wins']}/{m['ties']}/{m['losses']} | {m['descriptive_cluster_interval']} |")
    lines += ["", "上述区间只描述当前开发样本不确定性；不是独立测试集上的提升证明。缺失与题型、引用长度可能相关，完整配对仍有选择偏差。", "",
        "## 限制与复核", "",
        "- 单一裁判，与答案生成模型属于同一模型家族；temperature=0也不保证评分稳定。没有逐维人工数值金标，因此不能报告完整人机评分一致率。",
        "- critical_error只是裁判待审标记。尤其要区分‘本次检索材料不足’与‘整个政策库没有规定’，不能因为裁判的参考答案有依据就把合理拒答认定为事实错误。",
        "- 所有未知但可能适用的替代引用进入复核，不按是否出现在金标中判错；精确摘录检查不能验证全部语义支持关系。",
        f"- 本批{count}题使用同一冻结量表，均为开发分析，95道测试题未执行。", "",
        "## 用量", "",
        f"本批已知评分Token：{stats['judge_tokens_known']:,}；可见物理尝试{len(calls)}次，其中明确未发送{stats['confirmed_not_sent_calls']}次；可见未结算预留{stats['unsettled_call_reservations']}。用量未知任务{stats['unknown_usage_jobs']}个，其中缺少调用遥测{stats['jobs_without_call_telemetry']}个；这些任务不能按零开销理解。",
        "评分耗时与生成答案耗时分开记录，不能加进Agent推理开销。初始Qwen四次额度拒绝记录在answer-quality-v1，不与本批成绩混算，其usage未知，不按零计费。", "",
        f"逐题分数见raw_scores.csv，配对差值和分难度结果见metrics.json；需核查的原文与答案见[{review_file}]({review_file})。复核材料按输入快照保存，更新结果不会覆盖旧快照中的人工意见。", ""]
    (out / "report.md").write_text("\n".join(lines))
    review = []
    rng = random.Random(20260926)
    clean = [r for r in raw if r["total"] is not None and not r["review_needed"]]
    sampled = {(r["question_id"], r["arm"]) for r in rng.sample(clean, min(8, len(clean)))}
    for r in raw:
        if r["total"] is None or r["review_needed"] or (r["question_id"], r["arm"]) in sampled:
            review.append(r)
    (out / "review_queue.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    detail = ["# 评分器复核材料", "", "复核对象是裁判结论；不覆盖已有人工二元审核。包含无效评分、模型待审标记与固定种子随机8份有效评分。", ""]
    lookup = {(j["question_id"], j["arm"]): j for j in jobs}
    for r in review:
        j = lookup[r["question_id"], r["arm"]]
        row = rmap.get(j["id"], {})
        packet = j["packet"]
        detail += [f"## {r['question_id']} / {LABELS[r['arm']]}", "", f"状态：{r['status']}；分数：{r['total']}；验证错误：{r['validation_error']}", "",
                   "问题：" + packet["question"], "", "参考答案：" + packet["reference_answer"], "", "范围：" + packet["scope"], "", "被评答案：", packet["answer"], ""]
        for source in packet["reference_evidence"]:
            detail += ["参考原文：" + source["title"], source["content"], ""]
        for source in packet["citations"]:
            detail += [f"实际引用ID:{source['citation_id']}：{source['title']}", source["content"], ""]
        detail += ["裁判原输出：", "```json", row.get("output", row.get("reason", "未完成")), "```", "", "人工复核意见：待填写", ""]
    if not (out / review_file).exists():
        (out / review_file).write_text("\n".join(detail))
    (out / "review_artifact.json").write_text(json.dumps({"file": review_file, "input_sha256": source_digest}, indent=2) + "\n")
    print(json.dumps({"completed": len(rows), "planned": len(jobs), "known_tokens": stats["judge_tokens_known"],
                      "means": {a: m["quality_mean"] for a, m in table.items()},
                      "paired_deltas": {a: m["delta_mean"] for a, m in comparisons.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)
