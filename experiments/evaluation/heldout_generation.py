"""Frozen single-generation comparison over saved held-out contexts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

from experiments.evaluation.context_experiment import digest, jsonlines, read_verified, write_json, write_jsonl
from experiments.evaluation.generation_pilot import build_messages, citation_view, sanitize_content
from experiments.evaluation.heldout_retrieval import verify as verify_retrieval
from experiments.evaluation.qa_regression import execution_summary
from experiments.qa import GenerationClient, process_response, validate_question


ROOT = Path("experiments/datasets/private/heldout_generation_v1")
RETRIEVAL = Path("experiments/results/private/heldout-retrieval-test99-v1")
PROTOCOL = Path("docs/heldout_generation_protocol_v1.md")
CONDITIONS = [("dense", "boundary"), ("hybrid", "ranked")]
SOURCES = ["experiments/evaluation/heldout_generation.py", "experiments/evaluation/heldout_generation_report.py", "experiments/qa.py", "experiments/evaluation/generation_worker.py", "experiments/evaluation/generation_pilot.py", "experiments/evaluation/generation_report.py", "experiments/evaluation/qa_regression.py", "experiments/evaluation/context_completion.py", "experiments/evaluation/context_experiment.py"]
CONTACT = r"联\s*系\s*人|联系.{0,5}(?:电话|地址)|电子邮箱|传真|[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}|(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}[-— ]?\d{7,8})(?!\d)|QQ群|工作群号"


def sanitize(content):
    if re.search(r"(?:患者姓名|姓名|住院病历号|身份证号码|社会保障卡号|家庭住址)\s*[:：]\s*\S|(?<!\d)\d{17}[\dXx](?!\d)", content):
        raise ValueError("Individual record requires separate local handling")
    lines = [line for line in content.splitlines() if not re.search(CONTACT, line) and not re.match(r"^\s*[\u4e00-\u9fff·]{2,8}(?:委员|代表|先生|女士)\s*[:：]\s*$", line)]
    return sanitize_content("\n".join(lines))


def redact(content, spans):
    end, parts = 0, []
    for start, stop in spans:
        if type(start) is not int or type(stop) is not int or not end <= start < stop <= len(content):
            raise ValueError("Invalid reviewed redaction span")
        parts.extend([content[end:start], "[已省略非政策个人信息]"])
        end = stop
    return "".join([*parts, content[end:]])


def assemble(queries, selections, chunks, reviewed):
    relevant = [r for r in selections if (r["method"], r["policy"]) in CONDITIONS]
    selected = {(r["question_id"], r["method"], r["policy"]): r for r in relevant}
    expected = {(q["question_id"], *condition) for q in queries for condition in CONDITIONS}
    if set(selected) != expected or len(selected) != len(relevant) or len(queries) != len({q["question_id"] for q in queries}):
        raise ValueError("Missing, extra or duplicate held-out condition")
    requests = []
    for i, q in enumerate(queries):
        validate_question(q["question"])
        for method, policy in (CONDITIONS if i % 2 == 0 else list(reversed(CONDITIONS))):
            row = selected[(q["question_id"], method, policy)]
            ids = row["selected_ids"]
            if not ids or len(ids) != len(set(ids)) or row["tokens"] != sum(chunks[cid]["tokens"] for cid in ids) or row["tokens"] > 2560:
                raise ValueError("Saved context changed or exceeds budget")
            safe = []
            for cid in ids:
                c, approval = chunks[cid], reviewed.get(cid)
                if c["source_document_id"] == "79870" or not approval or approval["source_document_id"] != c["source_document_id"] or approval["raw_sha256"] != hashlib.sha256(c["content"].encode()).hexdigest():
                    raise ValueError("Unreviewed, blocked or modified outbound content")
                text = sanitize(redact(c["content"], approval.get("redactions", [])))
                if approval["safe_sha256"] != hashlib.sha256(text.encode()).hexdigest():
                    raise ValueError("Sanitized outbound content changed")
                safe.append({"id": cid, "source_document_id": c["source_document_id"], "content": text})
            messages = build_messages(q["question"], safe, short_ids=True)
            name = method + "_" + policy
            requests.append({"request_id": q["question_id"] + "-" + name, "question_id": q["question_id"], "policy": name, "question": q["question"], "mode": "short_ids", "status": "prepared", "selected_ids": ids, "selection_tokens_cl100k": row["tokens"], "chunks": safe, "messages": messages, "prompt_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(), "citation_map": {c["id"]: {"original_chunk_id": c["original_chunk_id"], "source_document_id": c["source_document_id"]} for c in citation_view(safe, True)}})
    return requests


def stop_after(response, consecutive):
    return response.get("error_category") in {"quota", "auth", "configuration", "worker_protocol"} or consecutive >= 2


def prepared_requests():
    config = verify_retrieval()
    frozen = json.loads((RETRIEVAL / "retrieval_freeze.json").read_text())
    for name in ["queries", "selections", "rankings"]:
        read_verified(RETRIEVAL / (name + ".jsonl"), frozen[name + "_sha256"])
    chunks = {c["id"]: c for c in json.loads(Path(config["snapshot"]).read_text())["chunks"]}
    review = json.loads((ROOT / "privacy_review.json").read_text())
    if review.get("status") != "complete" or review["blocked_source_ids"] != ["79870"]:
        raise ValueError("Outbound review incomplete")
    queries = jsonlines((RETRIEVAL / "queries.jsonl").read_text())
    return assemble(queries, jsonlines((RETRIEVAL / "selections.jsonl").read_text()), chunks, review["chunks"])


def prepare():
    requests = prepared_requests()
    rubric = json.loads((ROOT / "answer_rubric.json").read_text())
    benchmark = jsonlines(Path("experiments/datasets/private/benchmark_split_v1/test.jsonl").read_text())
    questions = {q["question_id"]: q for q in benchmark}
    if len(requests) != 198 or len(rubric["questions"]) != 99 or {q["question_id"] for q in rubric["questions"]} != set(questions):
        raise ValueError("Expected full 99-question paired evaluation")
    for row in rubric["questions"]:
        if row["question"] != questions[row["question_id"]]["question"] or row["evidence"] != questions[row["question_id"]]["evidence"] or not row["required_points"] or any(not p.strip() for p in row["required_points"]):
            raise ValueError("Rubric must retain the original question and evidence")
    runtime = json.loads(Path("experiments/configs/policy_qa_v1.json").read_text())
    config = {**runtime["generation"], "max_calls": 198, "worker_batch_calls": 20, "policies": [m + "_" + p for m, p in CONDITIONS], "question_ids": list(questions), "rubric": str(ROOT / "answer_rubric.json"), "privacy_review": str(ROOT / "privacy_review.json"), "answer_mode": "short_ids", "retrieval_calls": 0}
    write_json(ROOT / "config.json", config)
    write_jsonl(ROOT / "requests.jsonl", requests)
    paths = [*SOURCES, str(PROTOCOL), *(str(ROOT / name) for name in ["config.json", "requests.jsonl", "privacy_review.json", "privacy_decisions.json", "answer_rubric.json", "rubric_points.json", "sanitization_impact.json"]), str(RETRIEVAL / "retrieval_freeze.json"), str(RETRIEVAL / "selections.jsonl"), "experiments/configs/policy_qa_v1.json", "experiments/datasets/private/benchmark_split_v1/test.jsonl", "experiments/datasets/private/context_completion_v1/chunks.json"]
    for p in [*SOURCES, str(PROTOCOL)]:
        target = ROOT / "code" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    write_json(ROOT / "freeze.json", {"created_at": datetime.now(timezone.utc).isoformat(), "files": {p: digest(p) for p in paths}, "generation_calls_so_far": 0, "maximum_calls": 198, "no_gold_in_prompts": True, "rubric_authored_before_answers": True, "selection": "previously frozen Dense boundary and native Hybrid ranked", "repeats_per_condition": 1})
    print(json.dumps({"prepared": str(ROOT), "questions": 99, "requests": len(requests), "unique_chunks": len({c["id"] for r in requests for c in r["chunks"]}), "rubric_points": sum(len(q["required_points"]) for q in rubric["questions"]), "model_calls": 0}))


def verify():
    frozen = json.loads((ROOT / "freeze.json").read_text())
    for p, sha in frozen["files"].items():
        read_verified(p, sha)
    requests = jsonlines((ROOT / "requests.jsonl").read_text())
    if requests != prepared_requests():
        raise ValueError("Outbound prompts no longer reconstruct exactly")
    return json.loads((ROOT / "config.json").read_text()), requests


def run(output, allow_model_api):
    if not allow_model_api:
        raise ValueError("Requires --allow-model-api")
    config, requests = verify()
    if (ROOT / "execution_lock.json").exists():
        raise ValueError("This held-out generation run has already started")
    output.mkdir(parents=True, exist_ok=False)
    write_json(ROOT / "execution_lock.json", {"started_at": datetime.now(timezone.utc).isoformat(), "output": str(output), "freeze_sha256": digest(ROOT / "freeze.json")})
    write_json(output / "run_manifest.json", {"freeze_sha256": digest(ROOT / "freeze.json"), "planned_requests": 198, "worker_batch_limit": 20, "no_retries": True, "retrieval_calls": 0, "embedding_calls": 0, "rerank_calls": 0, "pass_at_k": "not run"})
    client, calls, consecutive, stopped, rows = None, 0, 0, False, []
    try:
        with (output / "responses.jsonl").open("x") as stream:
            for request in requests:
                provider = {"status": "not_run_after_error"}
                if not stopped:
                    if client is None or client.calls >= config["worker_batch_calls"]:
                        if client:
                            client.close()
                            client = None
                        try:
                            client = GenerationClient({**config, "max_calls": min(config["worker_batch_calls"], 198 - calls)})
                        except Exception as exc:
                            stopped = True
                            provider = {"status": "not_run_after_error", "error_type": type(exc).__name__, "error_category": "worker_initialization"}
                    if not stopped:
                        calls += 1
                        try:
                            provider = client.generate(request)
                        except Exception as exc:
                            provider = {"status": "error", "error_type": type(exc).__name__, "error_category": "worker_protocol"}
                        consecutive = consecutive + 1 if provider["status"] != "ok" else 0
                        stopped = stop_after(provider, consecutive)
                result = process_response(provider, request)
                row = {k: request[k] for k in ["request_id", "question_id", "policy", "prompt_sha256", "selection_tokens_cl100k"]}
                row.update({**result, "reused_identical_prompt": False, "presentation_status": result["status"]})
                if result["status"] == "invalid_citations":
                    row["status"] = "ok"
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(json.dumps({"request_id": request["request_id"], "status": row["status"], "presentation": row["presentation_status"], "calls": calls}), flush=True)
    finally:
        if client:
            client.close()
    summary = execution_summary(rows, calls, stopped)
    summary.update({"heldout_question_count": 99, "heldout_generation_conditions": 198, "retrieval_calls": 0, "worker_batch_limit": 20})
    write_json(output / "execution_summary.json", summary)
    write_json(output / "response_freeze.json", {"responses_sha256": digest(output / "responses.jsonl"), "execution_sha256": digest(output / "execution_summary.json"), "prepared_freeze_sha256": digest(ROOT / "freeze.json"), "finished_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-model-api", action="store_true")
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run(args.output, args.allow_model_api)
