"""Score frozen held-out answers using a predeclared rubric and explicit reviews."""

import argparse
import json
from pathlib import Path
import statistics

from experiments.evaluation.context_experiment import _paired_delta, digest, jsonlines, write_json, write_jsonl
from experiments.evaluation.generation_report import score_record
from experiments.evaluation.heldout_generation import ROOT, verify
from experiments.evaluation.metrics import percentile


def summarize(rows, policies):
    result = {}
    for policy in policies:
        selected = [r for r in rows if r["policy"] == policy]
        claims, quotes = sum(r["claims"] for r in selected), sum(r["quotes"] for r in selected)
        result[policy] = {"planned_questions": len(selected), "assessed": sum(r["assessed"] for r in selected), "answer_correct": sum(r["answer_correct"] for r in selected), "answer_accuracy_all_planned": sum(r["answer_correct"] for r in selected) / len(selected), "grounded_correct": sum(r["grounded_correct"] for r in selected), "faithful": sum(r["faithful"] for r in selected), "correct_points": sum(r["correct_points"] for r in selected), "possible_points": sum(r["point_count"] for r in selected), "claims": claims, "supported_cited_claims": sum(r["supported_cited_claims"] for r in selected), "citation_claim_support_rate": sum(r["supported_cited_claims"] for r in selected) / claims if claims else None, "quotes": quotes, "valid_quotes": sum(r["valid_quotes"] for r in selected), "literal_quote_validity_rate": sum(r["valid_quotes"] for r in selected) / quotes if quotes else None}
    return result


def report(result, assessments_path, output):
    config, requests = verify()
    frozen = json.loads((result / "response_freeze.json").read_text())
    assert digest(result / "responses.jsonl") == frozen["responses_sha256"]
    assert digest(result / "execution_summary.json") == frozen["execution_sha256"]
    rows = jsonlines((result / "responses.jsonl").read_text())
    assert len(rows) == 198 and {r["request_id"] for r in rows} == {r["request_id"] for r in requests}
    rubric = {q["question_id"]: q for q in json.loads(Path(config["rubric"]).read_text())["questions"]}
    review = json.loads(assessments_path.read_text())
    assert review["responses_sha256"] == digest(result / "responses.jsonl") and review["rubric_sha256"] == digest(config["rubric"])
    assessments = {r["request_id"]: r for r in review["assessments"]}
    assert len(assessments) == len(review["assessments"]) and set(assessments) == {r["request_id"] for r in rows if r["status"] == "ok"}
    scored = [{"request_id": r["request_id"], "question_id": r["question_id"], "policy": r["policy"], "status": r["status"], "presentation_status": r["presentation_status"], **score_record(r, assessments.get(r["request_id"]), len(rubric[r["question_id"]]["required_points"]))} for r in rows]
    policies = config["policies"]
    groups = {q: v["policy_group_id"] for q, v in rubric.items()}
    paired = {}
    for metric in ["answer_correct", "grounded_correct", "faithful"]:
        data = {p: {r["question_id"]: {"original_complete": r[metric]} for r in scored if r["policy"] == p} for p in policies}
        paired[metric] = _paired_delta(data["hybrid_ranked"], data["dense_boundary"], groups, {"bootstrap_seed": 20260922, "bootstrap_samples": 2000})
    execution = json.loads((result / "execution_summary.json").read_text())
    summary = {"scope": "99 held-out questions, frozen single-generation system comparison; assistant semantic assessment, not independent human certification", "policies": summarize(scored, policies), "dense_minus_hybrid": paired, "execution": execution, "latency_ms": {}, "presentation_passes": {p: sum(r["presentation_status"] == "ok" for r in rows if r["policy"] == p) for p in policies}}
    for p in policies:
        latencies = [r["provider_response"]["latency_ms"] for r in rows if r["policy"] == p and r["provider_response"]["status"] == "ok"]
        summary["latency_ms"][p] = {"mean": statistics.mean(latencies) if latencies else None, "p50": percentile(latencies, .5), "p95": percentile(latencies, .95)}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    write_jsonl(output / "scored.jsonl", scored)
    lines = ["# 99题冻结方案的生成与引用评价", "", "两组相同模型、短编号提示与生成参数，各题各组生成一次；只改变已冻结的检索/上下文组合。没有pass@3、额外检索或自动修复。语义评价由助手按生成前固定的227个要点完成，尚未经独立人工盲审。", "", "| 指标 | Dense＋boundary | 原生Hybrid＋ranked |", "| --- | ---: | ---: |"]
    for key, label in [("assessed", "有效回答且已语义复核"), ("answer_correct", "主回答完整正确"), ("grounded_correct", "完整正确且有上下文/引用支持"), ("faithful", "忠实于上下文"), ("correct_points", "正确要点（每组227）"), ("supported_cited_claims", "有有效引用支持的claim"), ("valid_quotes", "原句字面有效的引用")]:
        values = [summary["policies"][p][key] for p in policies]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines += ["", "主准确率分母固定99，失败/截断/无效结构保留；引用存在、引用支持、答案正确分别计分。两组检索与选择同时不同，因此是整套方案对照，不能单独归因于边界补全。", "", "## 逐题答案与评分理由", ""]
    for qid in config["question_ids"]:
        lines += ["### " + qid, "", rubric[qid]["question"], "", "预先固定要点：" + "；".join(rubric[qid]["required_points"]), ""]
        for policy in policies:
            row = next(r for r in rows if r["question_id"] == qid and r["policy"] == policy)
            lines += ["**" + policy + "** — " + row["status"], ""]
            if row["status"] == "ok":
                lines += [row["parsed"]["answer"], "", "复核：" + assessments[row["request_id"]]["reason"], ""]
            else:
                lines += ["原始失败输出保留，未重试。", ""]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(output / "manifest.json", {"response_freeze_sha256": digest(result / "response_freeze.json"), "prepared_freeze_sha256": digest(ROOT / "freeze.json"), "assessments_sha256": digest(assessments_path), "rubric_sha256": digest(config["rubric"]), "reporter_sha256": digest(__file__), "model_calls": 0, "outputs": {p.name: digest(p) for p in sorted(output.iterdir())}})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--assessments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.result, args.assessments, args.output)
