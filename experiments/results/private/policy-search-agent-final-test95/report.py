"""Read-only analysis of the frozen final test; never supplies inference input."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from experiments.evaluation.answer_quality import validate_result
from experiments.evaluation.answer_quality_report import mean, paired
from experiments.evaluation.search_agent_eval import digest, dump, readl

OUT = Path(__file__).resolve().parent
BASE = OUT.parent / "benchmark-v2-baselines-v2-nonthinking"
ROOT = OUT.parents[3]
LABELS = {"rag": "Native RAG", "low": "Native Low", "ab": "v8.1 A+B"}


def cost(rows):
    calls = [c for r in rows for c in r.get("calls", [])]
    times = sorted(r["latency_ms"] / 1000 for r in rows)
    embeddings = [r.get("query_embedding_calls") for r in rows]
    return {
        "n": len(rows), "latency_mean_seconds": mean(times),
        "latency_p95_nearest_rank": times[math.ceil(.95 * len(times)) - 1] if times else None,
        "known_generation_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in calls),
        "mean_generation_tokens": mean([sum(c["usage"]["total_tokens"] for c in r.get("calls", []))
                                        for r in rows]) if all("usage" in c for c in calls) else None,
        "physical_attempts": len(calls),
        "confirmed_unsent_attempts": sum(c.get("transport_status") == "not_sent" for c in calls),
        "unsettled_reservations": sum(c.get("reserve_tokens", 0) for c in calls if "usage" not in c),
        "rows_without_call_telemetry": sum("calls" not in r for r in rows),
        "mean_observed_query_embedding_calls": mean(embeddings) if all(v is not None for v in embeddings) else None,
        "query_embedding_cache_hits": sum(r.get("embedding_cache_hits", 0) for r in rows),
    }


def report():
    manifest = json.loads((OUT / "manifest.json").read_text())
    for name, expected in manifest["protected_files"].items():
        assert digest(ROOT / name) == expected, name
    qs = {q["question_id"]: q for q in readl(BASE / "test.jsonl")}
    ids = manifest["question_ids"]
    rows = {a: {r["question_id"]: r for r in readl(OUT / a / "records.jsonl")} for a in LABELS}
    assert all(set(rr) == set(ids) for rr in rows.values())
    jobs = json.loads((OUT / "quality/jobs.json").read_text())
    grades = readl(OUT / "quality/records.jsonl")
    assert len(grades) == len({r["id"] for r in grades})
    jobmap = {j["id"]: j for j in jobs}
    assert {r["id"] for r in grades} <= set(jobmap)
    gmap, scores = {}, {a: {} for a in LABELS}
    for row in grades:
        job = jobmap[row["id"]]
        assert (row["question_id"], row["arm"]) == (job["question_id"], job["arm"])
        gmap[row["question_id"], row["arm"]] = row
        if row["status"] == "ok":
            assert validate_result(json.loads(row["output"]), job["packet"]) == row["grade"]
            if row["grade"]["total"] is not None:
                scores[row["arm"]][row["question_id"]] = row["grade"]["total"]
    common = sorted(set.intersection(*[{q for q, r in rr.items() if r["status"] == "ok"} for rr in rows.values()]))
    full_costs = {a: cost(list(rr.values())) for a, rr in rows.items()}
    costs = {a: cost([rows[a][q] for q in common]) for a in LABELS}
    comparisons = {a: paired(scores["ab"], scores[a], qs) for a in ("rag", "low")}
    table = {}
    for arm in LABELS:
        failures = sum(r["status"] != "ok" for r in rows[arm].values())
        missing = 95 - failures - len(scores[arm])
        total = sum(scores[arm].values())
        table[arm] = {"numeric_grades": len(scores[arm]), "mean": mean(list(scores[arm].values())),
            "stdev": round(statistics.stdev(scores[arm].values()), 3) if len(scores[arm]) > 1 else None,
            "generation_failures": failures, "grading_missing": missing,
            "status_counts": dict(Counter(r["status"] for r in rows[arm].values())),
            "all_question_utility_bounds": [round(total / 95, 3), round((total + 100 * missing) / 95, 3)],
            "by_difficulty": {d: {"n": sum(qs[q]["difficulty"] == d for q in scores[arm]),
                "mean": mean([s for q, s in scores[arm].items() if qs[q]["difficulty"] == d])}
                for d in ("Easy", "Medium", "Hard")}}
    calls = [c for r in grades for c in r.get("calls", [])]
    judge_cost = {"jobs": len(jobs), "completed": len(grades),
        "known_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in calls),
        "physical_attempts": len(calls),
        "confirmed_unsent_attempts": sum(c.get("transport_status") == "not_sent" for c in calls),
        "unsettled_reservations": sum(c.get("reserve_tokens", 0) for c in calls if "usage" not in c),
        "statuses": dict(Counter(r["status"] for r in grades))}
    snapshot = hashlib.sha256(json.dumps({"generations": rows, "grades": grades}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    metrics = {"split": "heldout_test95", "rubric": manifest["rubric"], "unmodified_uncalibrated_judge": True,
        "quality": table, "paired": comparisons, "same_success_cost_question_ids": common,
        "same_success_costs": costs, "all_generation_costs": full_costs,
        "judge_cost": judge_cost, "source_snapshot_sha256": snapshot}
    dump(OUT / "metrics.json", metrics)
    flat, blocks, review_ids = [], {}, set()
    for qid in ids:
        q = qs[qid]
        block = [f"## {qid} / {q['difficulty']}", "", q["question"], "", "参考答案：" + q["reference_answer"], ""]
        for e in q["evidence"]:
            block += ["参考原文：" + e["source_title"], e["text"], ""]
        for arm in LABELS:
            row, grade = rows[arm][qid], gmap.get((qid, arm), {})
            numeric = scores[arm].get(qid)
            flat.append({"question_id": qid, "difficulty": q["difficulty"], "arm": arm,
                         "generation_status": row["status"], "grading_status": grade.get("status", "not_graded"),
                         "quality_total": numeric, "latency_ms": row["latency_ms"]})
            block += ["### " + LABELS[arm], "", f"生成：{row['status']}；评分：{grade.get('status', 'not_graded')}；分数：{numeric}", "",
                      row.get("answer") or "未产出有效答案", ""]
            job = next((j for j in jobs if j["question_id"] == qid and j["arm"] == arm), None)
            if job:
                for citation in job["packet"]["citations"]:
                    block += [f"实际引用ID:{citation['citation_id']}：{citation['title']}", citation["content"], ""]
            if grade.get("status") == "ok":
                block += ["裁判理由：" + json.dumps(grade["grade"]["reasons"], ensure_ascii=False), ""]
            if numeric is None or numeric < 100 or grade.get("grade", {}).get("review_needed"):
                review_ids.add(qid)
        blocks[qid] = block
    pool = sorted(set(ids) - review_ids)
    random_ids = random.Random(20260926).sample(pool, min(8, len(pool)))
    review_ids.update(random_ids)
    with (OUT / "raw_results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (OUT / "all_answers.md").write_text("\n".join(["# 冻结测试95题全部答案与依据", ""] + [line for q in ids for line in blocks[q]]))
    review_path = OUT / f"review_queue-{snapshot[:12]}.md"
    if not review_path.exists():
        review_path.write_text("\n".join(["# 最终测试人工复核", "", "包含非满分、待审标记、评分缺失/技术失败的所有题和额外随机8道。每题展示三方案，未将裁判标记自动当成人工结论。", ""] +
            [line for q in sorted(review_ids) for line in blocks[q] + ["人工复核意见：待填写", ""]]))
    dump(OUT / "review_artifact.json", {"file": review_path.name, "question_ids": sorted(review_ids),
         "random_clean_question_ids": random_ids, "source_snapshot_sha256": snapshot})
    lines = ["# Native RAG / Native Low / v8.1：冻结测试95题", "",
        "生成采用deepseek-v4-flash-0731非思考、固定1000篇语料及原Embedding。生成全局串行、三个方案各95题一次正式尝试。裁判deepseek-v4.1-flash，用户决定跳过进一步校准，沿用policy-answer-quality-v1。", "",
        "以下为未人工终审的模型评分，不是整题正确率。测试输出未用于修改当前系统；原始分数、缺失、真实退步和技术失败全部保留。", "",
        "| 方案 | 有效评分 | 生成失败 | 评分缺失 | 质量均分 |", "|---|---:|---:|---:|---:|"]
    for arm, m in table.items():
        lines.append(f"| {LABELS[arm]} | {m['numeric_grades']}/95 | {m['generation_failures']} | {m['grading_missing']} | {m['mean']} |")
    lines += ["", "单组有效题可能不同，主要比较下方同题配对。评分失败为未知，不记0；另行业务效用范围见metrics.json。", "",
              "| v8.1对照 | 配对题数 | 平均分差 | 胜/平/负 | 政策组重采样描述性95%区间 |", "|---|---:|---:|---|---|"]
    for arm, m in comparisons.items():
        lines.append(f"| {LABELS[arm]} | {m['n']} | {m['delta_mean']} | {m['wins']}/{m['ties']}/{m['losses']} | {m['descriptive_cluster_interval']} |")
    lines += ["", "| 难度 | vs RAG | vs Low |", "|---|---|---|"]
    for d in ("Easy", "Medium", "Hard"):
        lines.append("| " + d + " | " + " | ".join(f"{comparisons[a]['by_difficulty'][d]['delta_mean']}（n={comparisons[a]['by_difficulty'][d]['n']}）" for a in ("rag", "low")) + " |")
    lines += ["", f"生成成本以下使用三方案都成功的同一{len(common)}题；全量成本与未知预留另外保存在JSON。P95采用最近秩。", "",
              "| 方案 | 平均秒 | P95秒 | 每题生成Token | 每题观测查询Embedding |", "|---|---:|---:|---:|---:|"]
    for a, c in costs.items():
        lines.append(f"| {LABELS[a]} | {c['latency_mean_seconds']} | {c['latency_p95_nearest_rank']} | {c['mean_generation_tokens']} | {c['mean_observed_query_embedding_calls']} |")
    lines += ["", f"评分完成{len(grades)}/{len(jobs)}份，另耗已知Token {judge_cost['known_tokens']:,}；评分成本不计入Agent推理。",
        "", "限制：单一、未经进一步数值校准的裁判，且与生成模型同属DeepSeek家族；开发阶段已知存在漏评和近似答案评分不一致。bootstrap区间不覆盖裁判系统性偏差，不能只凭微小分差宣称稳定优势。运行顺序与服务负载影响耗时，查询向量遥测不等于总成本或缓存收益。", "",
        f"人工复核材料：[打开]({review_path.name})；包括{len(review_ids)}道题和三方案答案。替代来源先核验支持性，不能因未预标而直接判引用错误。人工裁决另存，修订统一适用于同类答案，不能把评估修订称作算法改善。", "",
        "逐项分数见raw_results.csv，全部问题、原文、答案见all_answers.md。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"grades": len(grades), "paired": {a: m['delta_mean'] for a, m in comparisons.items()},
                      "review_questions": len(review_ids)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    report()
