"""Offline answer/citation scoring; keep semantic review distinct from quote checks."""

import argparse
import json
from pathlib import Path
import statistics

from experiments.evaluation.context_experiment import digest, jsonlines, write_json, write_jsonl


def score_record(row, assessment, point_count):
    empty = {"assessed": False, "answer_correct": False, "grounded_correct": False, "faithful": False, "all_answer_facts_in_claims": False, "correct_points": 0, "point_count": point_count, "claims": 0, "supported_cited_claims": 0, "quotes": 0, "valid_quotes": 0}
    if row["status"] != "ok":
        return empty
    claims = row["parsed"]["claims"]
    if not assessment or len(assessment.get("points", [])) != point_count or len(assessment.get("citation_support", [])) != len(claims):
        raise ValueError("Every valid answer requires complete rubric and citation review")
    if any(type(x) is not bool for x in assessment["points"] + assessment["citation_support"]) or type(assessment.get("faithful")) is not bool or type(assessment.get("all_answer_facts_in_claims")) is not bool or not isinstance(assessment.get("extra_errors"), list) or not assessment.get("reason"):
        raise ValueError("Incomplete semantic assessment")
    citation_support = []
    for i, claim in enumerate(claims):
        checks = [c["valid_quote"] for c in row["citation_checks"] if c["claim_index"] == i]
        citation_support.append(bool(claim["evidence"]) and bool(checks) and all(checks) and assessment["citation_support"][i])
    correct = all(assessment["points"]) and not assessment["extra_errors"]
    return {"assessed": True, "answer_correct": correct, "grounded_correct": correct and assessment["faithful"] and assessment["all_answer_facts_in_claims"] and bool(claims) and all(citation_support), "faithful": assessment["faithful"], "all_answer_facts_in_claims": assessment["all_answer_facts_in_claims"], "correct_points": sum(assessment["points"]), "point_count": point_count, "claims": len(claims), "supported_cited_claims": sum(citation_support), "quotes": len(row["citation_checks"]), "valid_quotes": sum(c["valid_quote"] for c in row["citation_checks"])}


def report(result, prepared, output, assessments_path=None):
    rows = jsonlines((result / "responses.jsonl").read_text())
    config = json.loads((prepared / "config.json").read_text())
    rubric = {r["question_id"]: r for r in json.loads(Path(config["rubric"]).read_text())["questions"]}
    assessments = {}
    if assessments_path:
        review = json.loads(assessments_path.read_text())
        if review["responses_sha256"] != digest(result / "responses.jsonl") or review["rubric_sha256"] != digest(config["rubric"]):
            raise ValueError("Assessment does not match these answers and rubric")
        assessments = {r["request_id"]: r for r in review["assessments"]}
    if set(assessments) != {r["request_id"] for r in rows if r["status"] == "ok"}:
        raise ValueError("Missing or extra semantic assessments")
    scored = [{"request_id": r["request_id"], "question_id": r["question_id"], "policy": r["policy"], "status": r["status"], **score_record(r, assessments.get(r["request_id"]), len(rubric[r["question_id"]]["required_points"]))} for r in rows]
    summaries = {}
    for policy in config["policies"]:
        selected = [r for r in scored if r["policy"] == policy]
        assessed = sum(r["assessed"] for r in selected)
        total_claims = sum(r["claims"] for r in selected)
        quotes = sum(r["quotes"] for r in selected)
        summaries[policy] = {"planned_questions": len(selected), "assessed_questions": assessed, "error_requests": sum(r["status"] == "error" for r in selected), "not_run_after_error": sum(r["status"] == "not_run_after_error" for r in selected), "answer_correct_count": sum(r["answer_correct"] for r in selected) if assessed else None, "answer_accuracy_all_planned": sum(r["answer_correct"] for r in selected) / len(selected) if assessed else None, "grounded_correct_count": sum(r["grounded_correct"] for r in selected) if assessed else None, "faithful_count": sum(r["faithful"] for r in selected) if assessed else None, "correct_points": sum(r["correct_points"] for r in selected) if assessed else None, "possible_points": sum(r["point_count"] for r in selected), "citation_claim_support_rate": sum(r["supported_cited_claims"] for r in selected) / total_claims if total_claims else None, "literal_quote_validity_rate": sum(r["valid_quotes"] for r in selected) / quotes if quotes else None}
    execution = json.loads((result / "execution_summary.json").read_text())
    latencies = [r["provider_response"]["latency_ms"] for r in rows if r["status"] == "ok" and not r["reused_identical_prompt"]]
    summary = {"status": "Not Run: no valid model answers" if not any(r["assessed"] for r in scored) else "assistant-reviewed diagnostic results; not held-out accuracy", "policies": summaries, "execution": execution, "mean_successful_generation_latency_ms": statistics.mean(latencies) if latencies else None, "assessor": "assistant using pre-frozen rubric; not independent human review" if any(r["assessed"] for r in scored) else None, "model": config["model_name"]}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    write_jsonl(output / "scored.jsonl", scored)
    write_json(output / "manifest.json", {"responses_sha256": digest(result / "responses.jsonl"), "rubric_sha256": digest(config["rubric"]), "assessment_sha256": digest(assessments_path) if assessments_path else None, "reporter_sha256": digest(__file__), "cloud_calls": 0})
    lines = ["# 10题生成与引用诊断", "", f"生成模型：`{config['model_name']}`；结果：**{summary['status']}**。", "", f"| 指标 | {config['policies'][0]} | {config['policies'][1]} |", "| --- | ---: | ---: |"]
    for key, label in [("assessed_questions", "得到有效回答并复核的题数"), ("answer_correct_count", "全部答案要点正确题数"), ("grounded_correct_count", "答案正确且有上下文及引用支持题数"), ("faithful_count", "回答忠实于所给上下文题数"), ("correct_points", "正确答案要点数（共18）"), ("citation_claim_support_rate", "引用支持的claim比例"), ("literal_quote_validity_rate", "引用ID正确且原句匹配的比例")]:
        values = [summaries[p][key] for p in config["policies"]]
        rendered = ["Not Run" if v is None else f"{v:.2%}" if key.endswith("rate") else str(v) for v in values]
        lines.append(f"| {label} | {rendered[0]} | {rendered[1]} |")
    lines += ["", f"本次实际请求 {execution['actual_calls']} 次，保存 {execution['condition_records']} 个条件记录。Embedding/Rerank/测试题调用均为0。失败与停止后的未运行记录完整保留，不将没有回答写成回答准确率0%。", "", "## 逐题记录", ""]
    for qid in config["question_ids"]:
        lines += [f"### {qid}", "", rubric[qid]["question"], ""]
        for policy in config["policies"]:
            row = next(r for r in rows if r["question_id"] == qid and r["policy"] == policy)
            lines.append(f"**{policy}** — {row['status']}")
            if row["status"] == "ok":
                lines += ["", row["parsed"]["answer"], "", "复核：" + assessments[row["request_id"]]["reason"]]
            elif row["status"] == "error":
                error = row["provider_response"]
                lines.append("错误：" + str(error.get("error_code") or error.get("error_type")))
            lines.append("")
    lines += ["## 解释边界", "", "10题是基于开发错误选择的诊断集，不代表全库准确率。引文确实存在与引文足以支持答案分开评价；字段/数字吻合也不能单独证明政策条件匹配。语义评分由助手完成，需保留原输出供用户复核。"]
    if any(r["provider_response"].get("error_code") == "AllocationQuota.FreeTierOnly" for r in rows):
        lines += ["", "本次明确的AllocationQuota.FreeTierOnly表示当前模型仅允许免费额度、可用免费额度已耗尽；在百炼提供可用配置后才能继续。参考[官方说明](https://help.aliyun.com/zh/model-studio/new-free-quota)。"]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--assessments")
    args = parser.parse_args()
    report(Path(args.result), Path(args.prepared), Path(args.output), Path(args.assessments) if args.assessments else None)


if __name__ == "__main__":
    main()
