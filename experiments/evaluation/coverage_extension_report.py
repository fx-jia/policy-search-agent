"""Paired dev41 analysis; shared scoring and costs, policy-group uncertainty."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from experiments.evaluation.context_experiment import digest, jsonlines, read_verified, write_json, write_jsonl
from experiments.evaluation.coverage_revision_report import summarize
from experiments.evaluation.generation_report import score_record
from experiments.evaluation.qa_regression import verify_prepared


def paired_analysis(scored, rubric):
    result = {}
    for left, right in [("draft", "generic"), ("draft", "coverage"), ("generic", "coverage")]:
        a = {r["question_id"]: r for r in scored if r["policy"] == left}
        b = {r["question_id"]: r for r in scored if r["policy"] == right}
        if set(a) != set(b) or set(a) != set(rubric):
            raise ValueError("Paired question sets differ")
        result[f"{left}_to_{right}"] = {}
        for metric in ["answer_correct", "grounded_correct"]:
            groups = defaultdict(list)
            for qid in a:
                groups[rubric[qid]["policy_group_id"]].append(int(b[qid][metric]) - int(a[qid][metric]))
            rng = random.Random(20260923)
            units = list(groups.values())
            samples = []
            for _ in range(10000):
                sample = [x for group in rng.choices(units, k=len(units)) for x in group]
                samples.append(sum(sample) / len(sample))
            samples.sort()
            strata = {}
            for subgroup in sorted({q["subgroup"] for q in rubric.values()}):
                ids = [qid for qid in a if rubric[qid]["subgroup"] == subgroup]
                strata[subgroup] = {"questions": len(ids), "left_correct": sum(a[qid][metric] for qid in ids), "right_correct": sum(b[qid][metric] for qid in ids)}
            result[f"{left}_to_{right}"][metric] = {
                "questions": len(a), "gained": [qid for qid in a if not a[qid][metric] and b[qid][metric]], "lost": [qid for qid in a if a[qid][metric] and not b[qid][metric]],
                "delta": sum(int(b[qid][metric]) - int(a[qid][metric]) for qid in a) / len(a),
                "cluster_ci95": [samples[249], samples[9749]], "policy_groups": len(groups), "subgroups": strata,
            }
    return result


def report(prepared, result, assessments_path, output):
    verify_prepared(prepared)
    config = json.loads((prepared / "config.json").read_text())
    frozen = json.loads((result / "response_freeze.json").read_text())
    for key, name in [("responses_sha256", "responses.jsonl"), ("requests_sha256", "requests.jsonl"), ("execution_sha256", "execution_summary.json")]:
        read_verified(result / name, frozen[key])
    rows = jsonlines((result / "responses.jsonl").read_text())
    review = json.loads(assessments_path.read_text())
    if review["responses_sha256"] != digest(result / "responses.jsonl") or review["rubric_sha256"] != digest(config["rubric"]):
        raise ValueError("Review input hash mismatch")
    rubric = {q["question_id"]: q for q in json.loads(Path(config["rubric"]).read_text())["questions"]}
    expected = {f"{qid}-{p}" for qid in config["question_ids"] for p in config["policies"]}
    if len(rows) != 123 or {r["request_id"] for r in rows} != expected:
        raise ValueError("Missing or duplicate planned condition")
    assessments = {a["request_id"]: a for a in review["assessments"]}
    if len(assessments) != len(review["assessments"]) or set(assessments) != {r["request_id"] for r in rows if r["status"] == "ok"}:
        raise ValueError("Missing or duplicate semantic assessment")
    scored = [{"request_id": r["request_id"], "question_id": r["question_id"], "policy": r["policy"], "status": r["status"], **score_record(r, assessments.get(r["request_id"]), len(rubric[r["question_id"]]["required_points"]))} for r in rows]
    by_qid = defaultdict(list)
    for row in rows:
        by_qid[row["question_id"]].append(row)
    common = [qid for qid, records in by_qid.items() if all(r["status"] == "ok" for r in records)]
    common_scored = [r for r in scored if r["question_id"] in common]
    summary = {"status": "assistant-reviewed development extension, not independent held-out test", "policies": summarize(rows, scored), "pairs": paired_analysis(scored, rubric), "common_complete_triplets": len(common), "common_triplet_correct": {p: sum(r["grounded_correct"] for r in common_scored if r["policy"] == p) for p in config["policies"]}, "coverage_trace_format_valid": sum(r.get("coverage_trace_valid", False) for r in rows), "execution": json.loads((result / "execution_summary.json").read_text())}
    latency = {}
    for p in config["policies"]:
        values = []
        for qid in common:
            records = {r["policy"]: r for r in by_qid[qid]}
            parts = [records["draft"]] if p == "draft" else [records["draft"], records[p]]
            values.append(sum(r["provider_response"]["latency_ms"] for r in parts))
        latency[p] = statistics.mean(values) if values else None
    summary["mean_common_triplet_pipeline_latency_ms"] = latency
    summary["supports_continuing_this_version"] = all(summary["pairs"][pair]["grounded_correct"]["delta"] > 0 for pair in ["draft_to_coverage", "generic_to_coverage"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    write_jsonl(output / "scored.jsonl", scored)
    write_json(output / "manifest.json", {"reporter_sha256": digest(__file__), "assessments_sha256": digest(assessments_path), "prepared_freeze_sha256": digest(prepared / "freeze.json"), "response_freeze_sha256": digest(result / "response_freeze.json"), "bootstrap_seed": 20260923, "bootstrap_replicates": 10000, "additional_model_calls": 0})
    lines = ["# 41题扩验：逐题复核", "", "评分为助手检查，不是独立人工验收。所有条件均保留41题分母。", ""]
    for qid in config["question_ids"]:
        lines += [f"## {qid}", "", rubric[qid]["question"], "", "必答要点：" + "；".join(rubric[qid]["required_points"]), ""]
        for row in by_qid[qid]:
            s = next(r for r in scored if r["request_id"] == row["request_id"])
            lines += [f"### {row['policy']} — {row['presentation_status']}", ""]
            if row["status"] == "ok":
                lines += [row["parsed"]["answer"], "", f"要点{s['correct_points']}/{s['point_count']}；完整答案{s['answer_correct']}；完整有据{s['grounded_correct']}。", "", assessments[row["request_id"]]["reason"], ""]
                for claim in row["parsed"]["claims"]:
                    lines += ["- " + claim["statement"]]
                    lines += [f"  - [{e['chunk_id']}] {e['quote']}" for e in claim["evidence"]]
                if row["policy"] == "coverage":
                    lines += ["", "模型检查记录：", "```json", json.dumps(row["parsed"].get("coverage"), ensure_ascii=False, indent=2), "```"]
            else:
                lines.append("调用失败、无法解析或因依赖失败而跳过；计入计划分母，不重试。")
            lines.append("")
    (output / "review.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["prepared", "result", "assessments", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    report(args.prepared, args.result, args.assessments, args.output)
