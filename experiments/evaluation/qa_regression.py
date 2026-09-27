"""Frozen paired dev-only regression for the integrated QA generation step."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from experiments.evaluation.context_experiment import TEST_PATH, TEST_SHA, digest, jsonlines, read_verified, write_json, write_jsonl
from experiments.qa import GenerationClient, load_runtime, prepare_request, process_response


PROTOCOL = Path("docs/policy_qa_protocol_v1.md")
SOURCES = ["experiments/qa.py", "experiments/evaluation/qa_regression.py", "experiments/evaluation/generation_worker.py", "experiments/evaluation/generation_pilot.py", "experiments/evaluation/generation_report.py", "experiments/evaluation/context_completion.py", "experiments/evaluation/context_experiment.py"]


def prepare(output):
    config, snapshot, reviewed, blocked = load_runtime()
    if digest(TEST_PATH) != TEST_SHA:
        raise ValueError("Held-out file identity changed")
    reference_config = json.loads(Path("experiments/configs/citation_format_v1.json").read_text())
    for path, expected in reference_config["input_sha256"].items():
        read_verified(path, expected)
    previous = {r["question_id"]: r for r in jsonlines(Path("experiments/datasets/private/citation_format_v1/prepared/requests.jsonl").read_text()) if r["policy"] == "short_ids"}
    rankings = {r["question_id"]: r for r in jsonlines(Path(config["saved_rankings"]).read_text())}
    chunks = {c["id"]: c for c in snapshot["chunks"]}
    requests = []
    for i, qid in enumerate(reference_config["question_ids"]):
        row = rankings[qid]
        for mode in (["short_ids", "parts"] if i % 2 == 0 else ["parts", "short_ids"]):
            request = prepare_request(row["question"], row["retrieved_chunks"], chunks, reviewed, blocked, mode)
            if request["chunks"] != previous[qid]["chunks"] or (mode == "short_ids" and request["messages"] != previous[qid]["messages"]):
                raise ValueError("Question/context/control prompt changed from previous experiment")
            requests.append({**request, "request_id": f"{qid}-{mode}", "question_id": qid, "policy": mode})
    if len(requests) != 20 or len({r["question_id"] for r in requests}) != 10:
        raise ValueError("Expected exactly 10 paired questions")
    output.mkdir(parents=True, exist_ok=False)
    run_config = {**config["generation"], "max_calls": 20, "question_ids": reference_config["question_ids"], "policies": ["short_ids", "parts"], "rubric": reference_config["rubric"], "runtime": config}
    write_json(output / "config.json", run_config)
    write_jsonl(output / "requests.jsonl", requests)
    outbound = {c["id"]: {"source_document_id": c["source_document_id"], "safe_sha256": reviewed[c["id"]]["safe_sha256"]} for r in requests for c in r["chunks"]}
    write_json(output / "outbound_review.json", {"chunks": outbound, "chunk_count": len(outbound), "source_count": len({c["source_document_id"] for c in outbound.values()}), "blocked_sources": sorted(blocked), "basis": "exact same 45 sanitized, previously reviewed chunks as citation-format-dev10-v1", "no_gold_in_prompts": True})
    files = {str(p): digest(p) for p in [*SOURCES, str(PROTOCOL), str(output / "config.json"), str(output / "requests.jsonl"), str(output / "outbound_review.json"), reference_config["rubric"]]}
    files.update(config["input_sha256"])
    for path in [*SOURCES, str(PROTOCOL)]:
        copied = output / "code" / path
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copied)
    write_json(output / "freeze.json", {"created_at": datetime.now(timezone.utc).isoformat(), "files": files, "test_sha256": TEST_SHA, "max_actual_calls": 20, "generation_only": True})
    print(json.dumps({"prepared": str(output), "requests": 20, "unique_safe_chunks": len(outbound), "embedding_calls": 0, "test_queries": 0}))


def verify_prepared(prepared):
    freeze = json.loads((prepared / "freeze.json").read_text())
    for path, expected in freeze["files"].items():
        read_verified(path, expected)
    if digest(TEST_PATH) != freeze["test_sha256"]:
        raise ValueError("Held-out file changed")


def execution_summary(rows, calls, stopped):
    usages = [r["provider_response"].get("usage") for r in rows if r["provider_response"].get("usage")]
    values = {key: [u[key] for u in usages if type(u.get(key)) is int] for key in ["prompt_tokens", "completion_tokens", "total_tokens"]}
    return {"actual_calls": calls, "condition_records": len(rows), "successful_conditions": sum(r["status"] == "ok" for r in rows), "reused_conditions": 0, "reported_usage": {k: sum(v) if v else None for k, v in values.items()}, "calls_with_usage": len(usages), "calls_without_usage": calls - len(usages), "usage_complete": all(len(v) == calls for v in values.values()), "stopped_due_to_errors": stopped, "embedding_calls": 0, "rerank_calls": 0, "test_queries": 0}


def run(prepared, output, allow_model_api):
    if not allow_model_api:
        raise ValueError("Generation requires --allow-model-api")
    verify_prepared(prepared)
    config = json.loads((prepared / "config.json").read_text())
    requests = jsonlines((prepared / "requests.jsonl").read_text())
    # Reconstruct outbound prompts from the frozen runtime inputs before keys load.
    runtime = config["runtime"]
    snapshot = {c["id"]: c for c in json.loads(Path(runtime["snapshot"]).read_text())["chunks"]}
    review = json.loads(Path(runtime["privacy_review"]).read_text())
    hold = json.loads(Path(runtime["privacy_hold"]).read_text())
    blocked = set(review["blocked_source_ids"]) | {hold["source_document_id"]}
    rankings = {r["question_id"]: r for r in jsonlines(Path(runtime["saved_rankings"]).read_text())}
    for req in requests:
        row = rankings[req["question_id"]]
        rebuilt = prepare_request(row["question"], row["retrieved_chunks"], snapshot, review["chunks"], blocked, req["mode"])
        if any(req[k] != v for k, v in rebuilt.items()):
            raise ValueError("Prepared request differs from the reviewed inputs")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_manifest.json", {"freeze_sha256": digest(prepared / "freeze.json"), "max_calls": 20, "retrieval": "saved Dense64 + boundary2560", "no_retries": True, "test_queries": 0})
    client = GenerationClient(config)
    rows, consecutive, stopped = [], 0, False
    try:
        with (output / "responses.jsonl").open("x") as stream:
            for request in requests:
                if stopped:
                    provider = {"status": "not_run_after_error"}
                else:
                    try:
                        provider = client.generate(request)
                    except Exception as exc:
                        provider = {"status": "error", "error_type": type(exc).__name__, "error_category": "worker_protocol"}
                result = process_response(provider, request)
                record = {k: request[k] for k in ["request_id", "question_id", "policy", "prompt_sha256", "selection_tokens_cl100k"]}
                record.update({**result, "reused_identical_prompt": False, "presentation_status": result["status"]})
                # Bad quotes are scored separately from answer correctness, just
                # as in the existing evaluator. Runtime presentation stays gated.
                if result["status"] == "invalid_citations":
                    record["status"] = "ok"
                rows.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                consecutive = consecutive + 1 if provider["status"] != "ok" else 0
                stopped = stopped or consecutive >= 2 or provider.get("error_category") in {"auth", "quota", "configuration", "worker_protocol"}
                print(json.dumps({"request_id": request["request_id"], "status": record["status"], "calls": client.calls}), flush=True)
    finally:
        client.close()
    write_json(output / "execution_summary.json", execution_summary(rows, client.calls, stopped))
    write_json(output / "response_freeze.json", {"responses_sha256": digest(output / "responses.jsonl"), "execution_sha256": digest(output / "execution_summary.json"), "prepared_freeze_sha256": digest(prepared / "freeze.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--prepared")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-model-api", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(Path(args.output))
    else:
        run(Path(args.prepared), Path(args.output), args.allow_model_api)


if __name__ == "__main__":
    main()
