"""Offline paired scoring and stage/pipeline costs for coverage revision."""

import argparse
import json
from pathlib import Path
import statistics

from experiments.evaluation.context_experiment import digest, jsonlines, read_verified, write_json, write_jsonl
from experiments.evaluation.generation_report import score_record
from experiments.evaluation.qa_regression import verify_prepared


def summarize(rows, scored):
    summaries = {}
    drafts = {r["question_id"]: r for r in rows if r["policy"] == "draft"}
    for policy in ["draft", "generic", "coverage"]:
        selected = [r for r in scored if r["policy"] == policy]
        stages = [r for r in rows if r["policy"] == policy]
        pipeline = stages if policy == "draft" else stages + list(drafts.values())
        attempted = [r for r in pipeline if r["attempted"]]
        tokens = [r["provider_response"].get("usage", {}).get("total_tokens") for r in attempted]
        stage_tokens = [r["provider_response"].get("usage", {}).get("total_tokens") for r in stages if r["attempted"]]
        latencies = []
        for row in stages:
            parts = [row] if policy == "draft" else [drafts[row["question_id"]], row]
            if all(r["status"] == "ok" and isinstance(r["provider_response"].get("latency_ms"), (int, float)) for r in parts):
                latencies.append(sum(r["provider_response"]["latency_ms"] for r in parts))
        def total(key):
            return sum(r[key] for r in selected)

        summaries[policy] = {
            "planned_questions": len(selected), "assessed_questions": total("assessed"),
            "answer_correct_count": total("answer_correct"), "grounded_correct_count": total("grounded_correct"),
            "correct_points": total("correct_points"), "possible_points": total("point_count"),
            "supported_cited_claims": total("supported_cited_claims"), "claims": total("claims"),
            "valid_quotes": total("valid_quotes"), "quotes": total("quotes"),
            "stage_reported_total_tokens": sum(t for t in stage_tokens if type(t) is int) if any(type(t) is int for t in stage_tokens) else None,
            "pipeline_reported_total_tokens": sum(t for t in tokens if type(t) is int) if any(type(t) is int for t in tokens) else None,
            "pipeline_usage_complete": bool(attempted) and all(type(t) is int for t in tokens),
            "logical_attempted_calls": len(attempted), "complete_pipeline_count": len(latencies),
            "mean_complete_pipeline_latency_ms": statistics.mean(latencies) if latencies else None,
        }
    return summaries


def report(prepared, result, assessments_path, output):
    verify_prepared(prepared)
    config = json.loads((prepared / "config.json").read_text())
    response_freeze = json.loads((result / "response_freeze.json").read_text())
    for key, path in [("responses_sha256", "responses.jsonl"), ("requests_sha256", "requests.jsonl"), ("execution_sha256", "execution_summary.json")]:
        read_verified(result / path, response_freeze[key])
    review = json.loads(assessments_path.read_text())
    if review["responses_sha256"] != digest(result / "responses.jsonl") or review["rubric_sha256"] != digest(config["rubric"]):
        raise ValueError("Assessment input hash mismatch")
    rows = jsonlines((result / "responses.jsonl").read_text())
    rubric = {r["question_id"]: r for r in json.loads(Path(config["rubric"]).read_text())["questions"]}
    assessments = {r["request_id"]: r for r in review["assessments"]}
    expected = {f"{qid}-{p}" for qid in config["question_ids"] for p in config["policies"]}
    if len(rows) != len(expected) or {r["request_id"] for r in rows} != expected:
        raise ValueError("Planned condition denominator changed")
    if len(assessments) != len(review["assessments"]) or set(assessments) != {r["request_id"] for r in rows if r["status"] == "ok"}:
        raise ValueError("Missing, duplicate or extra assessments")
    scored = [{"request_id": r["request_id"], "question_id": r["question_id"], "policy": r["policy"], "status": r["status"], **score_record(r, assessments.get(r["request_id"]), len(rubric[r["question_id"]]["required_points"]))} for r in rows]
    pairs = {}
    for left, right in [("draft", "generic"), ("draft", "coverage"), ("generic", "coverage")]:
        a = {r["question_id"]: r for r in scored if r["policy"] == left}
        b = {r["question_id"]: r for r in scored if r["policy"] == right}
        pairs[f"{left}_to_{right}"] = {metric: {"gained": [qid for qid in a if not a[qid][metric] and b[qid][metric]], "lost": [qid for qid in a if a[qid][metric] and not b[qid][metric]]} for metric in ["answer_correct", "grounded_correct"]}
    summary = {"policies": summarize(rows, scored), "pairs": pairs, "coverage_trace_format_valid": sum(r.get("coverage_trace_valid", False) for r in rows), "execution": json.loads((result / "execution_summary.json").read_text()), "assessor": "assistant semantic review against frozen rubric and sent context, not independent human review", "status": "reused dev10 diagnostic; not held-out validation"}
    delta = pairs["generic_to_coverage"]["grounded_correct"]
    summary["supports_expanding_development_validation"] = bool(delta["gained"]) and not delta["lost"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    write_jsonl(output / "scored.jsonl", scored)
    write_json(output / "manifest.json", {"reporter_sha256": digest(__file__), "assessments_sha256": digest(assessments_path), "prepared_freeze_sha256": digest(prepared / "freeze.json"), "response_freeze_sha256": digest(result / "response_freeze.json"), "cloud_calls": 0})
    lines = ["# 子问覆盖复查：逐题评分", "", "固定10道开发题，助手语义复核；每种条件都保留10题分母。", ""]
    for qid in config["question_ids"]:
        lines += [f"## {qid}", "", rubric[qid]["question"], "", "评分要点：" + "；".join(rubric[qid]["required_points"]), ""]
        for row in [r for r in rows if r["question_id"] == qid]:
            score = next(s for s in scored if s["request_id"] == row["request_id"])
            lines += [f"### {row['policy']} — {row['presentation_status']}", ""]
            if row["status"] == "ok":
                lines += [row["parsed"]["answer"], "", assessments[row["request_id"]]["reason"], "", f"要点 {score['correct_points']}/{score['point_count']}；完整正确 {score['answer_correct']}；完整有据 {score['grounded_correct']}。", ""]
                for claim in row["parsed"]["claims"]:
                    lines.append("- " + claim["statement"])
                    for e in claim["evidence"]:
                        lines.append(f"  - [{e['chunk_id']}] {e['quote']}")
                if row["policy"] == "coverage":
                    lines += ["", "检查记录（模型自报）：", "```json", json.dumps(row["parsed"].get("coverage"), ensure_ascii=False, indent=2), "```"]
            else:
                lines.append("未得到可评分答案，保留在分母；未重试。")
            lines.append("")
    (output / "review.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["prepared", "result", "assessments", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    report(args.prepared, args.result, args.assessments, args.output)


if __name__ == "__main__":
    main()
