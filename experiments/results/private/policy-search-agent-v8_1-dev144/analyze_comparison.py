"""Offline quality, cost and grader consistency analysis on the same dev144."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.evaluation.answer_quality import validate_result
from experiments.evaluation.answer_quality_report import paired

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
GRADE = OUT.parent / "answer-quality-dev144-flash"
LABEL = {"rag": "Native RAG", "low": "Native Low", "v7": "v7", "ab": "v8.1 A+B"}


def readl(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    manifest = json.loads((GRADE / "manifest.json").read_text())
    for name, expected in manifest["input_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for name in ("config", "jobs"):
        assert hashlib.sha256((GRADE / (name + ".json")).read_bytes()).hexdigest() == manifest[name + "_sha256"]
    ids = manifest["question_ids"]
    audit = json.loads((GRADE / "comparison_audit.json").read_text())
    rows = {arm: {} for arm in LABEL}
    for source in audit:
        for r in readl(ROOT / source["source"] / "records.jsonl"):
            if r["question_id"] in ids:
                assert r["question_id"] not in rows[source["arm"]]
                rows[source["arm"]][r["question_id"]] = r
    assert all(set(rows[a]) == set(ids) for a in LABEL)
    jobs = {j["id"]: j for j in json.loads((GRADE / "jobs.json").read_text())}
    grades = readl(GRADE / "records.jsonl")
    assert len(grades) == len({r["id"] for r in grades}) == len(jobs) == 570
    assert {a["id"] for a in readl(GRADE / "attempts.jsonl")} == set(jobs)
    grade_map, valid = {}, {a: {} for a in LABEL}
    for r in grades:
        job = jobs[r["id"]]
        assert (job["question_id"], job["arm"]) == (r["question_id"], r["arm"])
        assert not r.get("ledger", {}).get("pending_calls")
        grade_map[r["question_id"], r["arm"]] = r
        if r["status"] == "ok":
            assert validate_result(json.loads(r["output"]), job["packet"]) == r["grade"]
            if r["grade"]["total"] is not None:
                valid[r["arm"]][r["question_id"]] = r["grade"]["total"]
    qs = {q["question_id"]: q for q in readl(OUT.parent / "benchmark-v2-baselines-v2-nonthinking/dev.jsonl")}
    common = set.intersection(*[{q for q, r in rr.items() if r["status"] == "ok"} for rr in rows.values()])
    costs = {}
    for arm, rr in rows.items():
        same = [rr[q] for q in sorted(common)]
        times = sorted(r["latency_ms"] / 1000 for r in same)
        assert all("usage" in c for r in same for c in r.get("calls", []))
        all_calls = [c for r in rr.values() for c in r.get("calls", [])]
        costs[arm] = {
            "successful_common_questions": len(same),
            "mean_latency_seconds": round(statistics.mean(times), 3),
            "p95_latency_seconds_nearest_rank": times[math.ceil(.95 * len(times)) - 1],
            "mean_generation_tokens": round(statistics.mean(sum(c["usage"]["total_tokens"] for c in r.get("calls", [])) for r in same), 3),
            "mean_observed_query_embedding_calls": statistics.mean(r["query_embedding_calls"] for r in same),
            "all144_generation_known_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in all_calls),
            "all144_generation_unsettled_reservation": sum(c.get("reserve_tokens", 0) for c in all_calls if "usage" not in c),
        }
    quality = json.loads((GRADE / "metrics.json").read_text())
    for arm in LABEL:
        assert round(statistics.mean(valid[arm].values()), 3) == quality["table"][arm]["quality_mean"]
    comparisons = {a: paired(valid["ab"], valid[a], qs) for a in ("rag", "low", "v7")}
    assert all(comparisons[a] == quality["paired"]["ab_vs_" + a] for a in comparisons)
    unresolved_rubric_cases = {
        "mb2-171": "v7与v8.1引用全文相同，答案几乎一致；裁判给100与76.25，对相同额外新生儿政策说明采用不同扣分口径。",
        "mb2-276": "v7和v8.1均有‘100元低于100元标准’的同类错误，裁判却给100与88.75；RAG此处的解释正确。",
        "mb2-091": "RAG与v7都只复述条件和题干、没有计算缺口，引用相同；裁判给51.25与32.5。v8.1实际补齐计算，但收益分值大小需校准。",
        "mb2-281": "v7回答额度余量0万元，参考原文和算式应为2万元；裁判准确性理由写2万元、完整性理由又写0万元，仍给满分。v8.1已答对2万元，这次改善被漏评。",
    }
    sensitivity = {a: paired({q: s for q, s in valid["ab"].items() if q not in unresolved_rubric_cases},
                             {q: s for q, s in valid[a].items() if q not in unresolved_rubric_cases}, qs)
                   for a in comparisons}
    dump(OUT / "paired_costs_and_quality.json", {"costs": costs, "paired_quality": comparisons,
        "posthoc_sensitivity_excluding_four_judge_disagreements": sensitivity,
        "sensitivity_note": "Posthoc diagnostic only; these exclusions are applied to every arm and do not replace primary scores.",
        "judge_consistency_cases": unresolved_rubric_cases})
    flat = []
    all_answers = ["# 144道开发题：同题答案与质量评分", "", "原始模型评分保留，非人工终审。评分失败是评分缺失，不等于原答案错误。", ""]
    focus = ["# 优先复核：裁判一致性与真实回归", "", "不直接修改任何模型分数。171/276/091/281用于统一评分口径；049用于核对真实退步及缺失评分影响。", ""]
    for qid in ids:
        q = qs[qid]
        block = [f"## {qid} / {q['difficulty']}", "", q["question"], "", "参考答案：" + q["reference_answer"], ""]
        for e in q["evidence"]:
            block += ["参考原文：" + e["source_title"], e["text"], ""]
        for arm in LABEL:
            r = rows[arm][qid]
            g = grade_map.get((qid, arm), {})
            numeric = g.get("grade", {}).get("total") if g.get("status") == "ok" else None
            flat.append({"question_id": qid, "difficulty": q["difficulty"], "arm": arm, "generation_status": r["status"],
                         "grading_status": g.get("status", "no_valid_generated_answer"), "quality_total": numeric,
                         "generation_latency_ms": r["latency_ms"],
                         "generation_known_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in r.get("calls", []))})
            block += ["### " + LABEL[arm], "", f"生成状态：{r['status']}；评分状态：{g.get('status', '未评分')}；模型分数：{numeric}", "",
                      r.get("answer", "未产出"), ""]
            if g.get("status") == "ok":
                block += ["裁判理由：" + json.dumps(g["grade"]["reasons"], ensure_ascii=False), ""]
        all_answers += block
        if qid == "mb2-281" and not (OUT / "judge_case_281.md").exists():
            (OUT / "judge_case_281.md").write_text("\n".join(["# 281：裁判漏评的新增复核材料", "",
                unresolved_rubric_cases[qid], "", "原始模型分数保留，未据此修饰整体均分。", ""] + block + ["人工复核意见：待填写", ""]))
        if qid in unresolved_rubric_cases or qid == "mb2-049":
            focus += ["核查点：" + unresolved_rubric_cases.get(qid, "v8.1把72岁退休职工的90%比例错用为87%，这是实际数值错误；v7该题评分无效，配对均值暂不能包含它。"), ""] + block
            focus += ["人工评分口径与复核意见：待填写", ""]
    with (OUT / "raw_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (OUT / "all_answers.md").write_text("\n".join(all_answers))
    if not (OUT / "priority_score_review.md").exists():
        (OUT / "priority_score_review.md").write_text("\n".join(focus))
    lines = ["# v8.1在144道已使用开发题上的回归与评分比较", "",
        "结论：在当前裁判下，v8.1相对Native Low有较大质量优势且更快；相对Native RAG和v7整体分差仍很小，尚未证明稳定的总体质量提升。Hard子集有值得复核的积极信号，但不能把开发子集分数写成测试集收益。", "",
        "## 实验口径", "",
        "本批是先前150道开发题中，最终v8.1尚未尝试的144题，Easy39/Medium68/Hard37；另6道诊断题与55道扩展题不混入本批。v8.1现已覆盖全部205道核心开发题，95道测试题未执行。生成沿用deepseek-v4-flash-0731、原Embedding及固定1000篇语料；RAG/Low/v7使用已经保存的同题历史答案。历史对照的模型、索引、检索参数及共同记录的运行哈希相同，但不是同期随机对照；最早30题缺少Embedding适配器代码哈希。", "",
        "裁判统一为deepseek-v4.1-flash，单份匿名答案评分；准确性35、完整性25、证据支持25、确定性10、清晰度5，各维0—4，总分折算100。没有人工逐维数值校准，裁判与生成器同属DeepSeek模型家族。此前55题使用DeepSeek Pro裁判，其分数不拼接到本批。", "",
        "四组共576个题目—方案组合，6个历史结果未产出有效答案（RAG预算中止3、Low保护中止1、v7保护中止2）；570次评分中542份有效、20份摘录/结构不合格、8份连接失败。28份评分缺失不按0分或答错处理。v8.1的144次生成均技术成功，不等于内容全部正确。", "",
        "## 质量结果", "", "| 方案 | 有效评分 | 均分 | 准确性 | 完整性 | 证据支持 | 确定性 | 清晰度 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in LABEL:
        m = quality["table"][arm]
        dims = m["dimension_means_0_to_4"]
        lines.append(f"| {LABEL[arm]} | {m['numeric_grades']}/144 | {m['quality_mean']:.2f} | " + " | ".join(f"{dims[k]:.2f}" for k in ("accuracy", "completeness", "grounding", "calibration", "clarity")) + " |")
    lines += ["", "各维满分4。单组有效题目不同，以下同题配对是主要比较口径。", "", "| 对照 | 双方有效题数 | v8.1平均分差 | 胜/平/负 | 描述性95%区间 |", "|---|---:|---:|---|---|"]
    for arm, m in comparisons.items():
        lines.append(f"| {LABEL[arm]} | {m['n']} | {m['delta_mean']:+.3f} | {m['wins']}/{m['ties']}/{m['losses']} | {m['descriptive_cluster_interval']} |")
    lines += ["", "按22个政策组重采样；开发样本、缺失评分与单裁判误差限制解释。与RAG和v7的区间均跨0。不能把质量分差当作准确率百分点。", "",
        "| 难度 | vs RAG | vs Low | vs v7 |", "|---|---:|---:|---:|"]
    for difficulty in ("Easy", "Medium", "Hard"):
        lines.append("| " + difficulty + " | " + " | ".join(f"{comparisons[a]['by_difficulty'][difficulty]['delta_mean']:+.3f}（n={comparisons[a]['by_difficulty'][difficulty]['n']}）" for a in comparisons) + " |")
    lines += ["", "Hard相对v7的+4.476来自31道配对题，其中060和091两题贡献115分差，即该组总分差138.75的大部分。091的对照评分本身有尺度不一致，不能把这个子组均值视为已稳定验证的收益。", "",
        "## 生成成本", "", "以下全部基于四方案均成功生成的同一138题。评分成本不计入生成；历史运行环境可能影响耗时。", "",
        "| 方案 | 平均耗时 | P95耗时 | 每题生成Token | 观测查询Embedding次数/题 |", "|---|---:|---:|---:|---:|"]
    for arm, m in costs.items():
        lines.append(f"| {LABEL[arm]} | {m['mean_latency_seconds']:.2f}s | {m['p95_latency_seconds_nearest_rank']:.2f}s | {m['mean_generation_tokens']:.0f} | {m['mean_observed_query_embedding_calls']:.0f} |")
    lines += ["", "v8.1比Low更快、查询Embedding更少，但生成Token更多；比RAG更慢且生成Token约为两倍。v8.1本批向量缓存实际命中0次，不能把减少Low原生扩展路径的调用归功于缓存。v7历史完整144题中仍有一次未结算调用，预留在JSON中单列；上表138题的调用均有已知用量。", "",
        "v8.1新增144题总生成Token为1,141,273；本批裁判另用1,094,263 Token。一次直接HTTPX连接超时经实际库源码与账本核验确认为未发送；保留原始快照及失败评分，仅恢复未尝试条目，没有重跑取优。", "",
        "## 案例与评分可靠性", "",
        "- 实际改善：060明确补上‘查实重大问题不予表扬’，091完成人员缺口计算；281补对剩余2万元，但裁判给错误的旧答案也打了满分，未测出这次改善。原始分数保持不变。",
        "- 实际退步：049错用87%而非90%，导致患者和基金数值错误；不能为了证明改进而排除该题。",
        "- 持续问题：045仍漏2500元保底；046仍混用旧地方奖励标准；048仍混入照护待遇互斥规则；276仍有100元被说成低于100元的矛盾。", ""]
    for q, note in unresolved_rubric_cases.items():
        lines.append(f"- 裁判复核 {q}：{note}")
    lines += ["", "检测到59组完全相同的评分输入，其中57组具有至少两份有效评分，未观察到总分不一致；但这不证明对语义近似答案的稳定性。上面的已核实差异仍须统一量表锚点。为检视影响，统一排除171/276/091/281四题的事后敏感性分析见JSON；它不替换主结果，不用于选择更有利的成绩。", "",
        "原始数值有效答案中，v8.1有125/138份满分，RAG有124/137份满分，量表仍有明显天花板效应。当前结果能揭示部分缺答和事实错误，尚不能可靠区分所有细微表达质量。", "",
        "建议先复核评分尺度和少数关键回归案例，维持真实退步记录；不基于本批微小分差宣称新版本整体优于RAG。95道测试题留到策略与评分口径冻结后再运行。", "",
        "## 文件", "", "- raw_comparison.csv：144题×4方案的逐项生成状态、评分和成本。",
        "- all_answers.md：问题、参考原文、四组答案和裁判理由。",
        "- priority_score_review.md：评分复核材料；新发现的281漏评另见judge_case_281.md。",
        "- paired_costs_and_quality.json：配对分差、分难度、成本及事后敏感性分析。", ""]
    (OUT / "comparison_report.md").write_text("\n".join(lines))
    dump(OUT / "comparison_integrity.json", {"all_protected_hashes_match": True, "grading_jobs": len(jobs),
        "grading_rows": len(grades), "valid_grade_totals_recomputed": True, "same_question_pairing_recomputed": True,
        "no_pending_grading_usage": True, "historical_generation_unsettled_reserves": {a: x["all144_generation_unsettled_reservation"] for a, x in costs.items()}})
    print(json.dumps({"generation_cost_common_n": len(common), "quality_paired_deltas": {a: m["delta_mean"] for a, m in comparisons.items()},
                      "posthoc_sensitivity": {a: m["delta_mean"] for a, m in sensitivity.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
