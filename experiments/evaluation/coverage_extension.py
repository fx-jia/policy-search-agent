"""Extend the frozen revision experiment to the other 41 development questions."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from experiments.evaluation.context_experiment import DEV_SHA, TEST_SHA, digest, jsonlines, read_verified, write_json, write_jsonl
from experiments.evaluation.coverage_revision import COVERAGE_REVIEW, GENERIC_REVIEW, execute
from experiments.evaluation.generation_pilot import build_messages, citation_view
from experiments.evaluation.heldout_generation import sanitize
from experiments.evaluation.qa_regression import execution_summary, verify_prepared
from experiments.qa import GenerationClient


ROOT = Path("experiments/datasets/private/coverage_revision_dev41_v1")
DEV = Path("experiments/datasets/private/benchmark_split_v1/dev.jsonl")
PILOT = Path("experiments/datasets/private/coverage_revision_v1")
SELECTIONS = Path("experiments/results/private/context_completion-dev51-v1/selections.jsonl")
PROTOCOL = Path("docs/coverage_extension_protocol_v1.md")


def remaining_questions(dev, previous):
    if len({q["question_id"] for q in dev}) != len(dev) or any(q["split"] != "dev" or q["review_status"] != "human_approved" for q in dev):
        raise ValueError("Expected unique approved development questions")
    return [q for q in dev if q["question_id"] not in previous]


def prepare():
    verify_prepared(PILOT)
    old_config = json.loads((PILOT / "config.json").read_text())
    queries = remaining_questions(jsonlines(read_verified(DEV, DEV_SHA)), set(old_config["question_ids"]))
    if len(queries) != 41:
        raise ValueError("Expected exactly 41 remaining dev questions")
    runtime = json.loads(Path("experiments/configs/policy_qa_v1.json").read_text())
    for path, sha in runtime["input_sha256"].items():
        read_verified(path, sha)
    chunks = {c["id"]: c for c in json.loads(Path(runtime["snapshot"]).read_text())["chunks"]}
    selections = {r["question_id"]: r for r in jsonlines(SELECTIONS.read_text()) if r["method"] == "dense" and r["policy"] == "boundary"}
    rubric = json.loads((ROOT / "answer_rubric.json").read_text())
    questions = {q["question_id"]: q for q in queries}
    if len(rubric["questions"]) != 41 or {q["question_id"] for q in rubric["questions"]} != set(questions):
        raise ValueError("Rubric scope mismatch")
    for row in rubric["questions"]:
        q = questions[row["question_id"]]
        if row["question"] != q["question"] or row["evidence"] != q["evidence"] or not row["required_points"]:
            raise ValueError("Rubric changed original question/evidence")
    prior_reviews = {}
    review_paths = [Path("experiments/datasets/private/generation_pilot_v1/privacy_review.json"), Path("experiments/datasets/private/heldout_generation_v1/privacy_review.json")]
    for path in review_paths:
        prior_reviews.update(json.loads(path.read_text())["chunks"])
    reviewed, requests = {}, []
    for q in queries:
        selected = selections[q["question_id"]]
        if selected["tokens"] > 2560 or selected["tokens"] != sum(chunks[cid]["tokens"] for cid in selected["selected_ids"]):
            raise ValueError("Fixed context token budget changed")
        safe = []
        for cid in selected["selected_ids"]:
            c = chunks[cid]
            if c["source_document_id"] == "79870":
                raise ValueError("Blocked source")
            raw_hash = hashlib.sha256(c["content"].encode()).hexdigest()
            if cid in prior_reviews and (raw_hash != prior_reviews[cid]["raw_sha256"] or prior_reviews[cid].get("redactions")):
                raise ValueError("Prior review changed or has unhandled redactions")
            content = sanitize(c["content"])
            reviewed[cid] = {"source_document_id": c["source_document_id"], "raw_sha256": raw_hash, "safe_sha256": hashlib.sha256(content.encode()).hexdigest(), "basis": "prior reviewed chunk" if cid in prior_reviews else "assistant read policy text locally before this run", "changed": content != c["content"]}
            safe.append({"id": cid, "source_document_id": c["source_document_id"], "content": content})
        messages = build_messages(q["question"], safe, True)
        requests.append({"request_id": q["question_id"] + "-draft", "question_id": q["question_id"], "question": q["question"], "policy": "draft", "mode": "short_ids", "status": "prepared", "selected_ids": selected["selected_ids"], "selection_tokens_cl100k": selected["tokens"], "chunks": safe, "messages": messages, "prompt_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(), "citation_map": {c["id"]: {"original_chunk_id": c["original_chunk_id"], "source_document_id": c["source_document_id"]} for c in citation_view(safe, True)}})
    write_json(ROOT / "privacy_review.json", {"reviewer": "assistant", "blocked_source_ids": ["79870"], "chunks": reviewed, "scope": "115 selected chunks; not whole corpus", "new_local_reviews": 60})
    write_jsonl(ROOT / "base_requests.jsonl", requests)
    write_json(ROOT / "revision_prompts.json", {"generic": GENERIC_REVIEW, "coverage": COVERAGE_REVIEW})
    config = {**runtime["generation"], "max_calls": 15, "max_actual_calls": 123, "question_ids": list(questions), "policies": ["draft", "generic", "coverage"], "rubric": str(ROOT / "answer_rubric.json"), "question_batch_size": 10}
    write_json(ROOT / "config.json", config)
    original = json.loads((PILOT / "freeze.json").read_text())
    files = dict(original["files"])
    sources = [str(PROTOCOL), __file__, "experiments/evaluation/test_coverage_extension.py", "experiments/evaluation/coverage_revision_report.py", "experiments/evaluation/heldout_generation.py"]
    for path in [*sources, *review_paths, SELECTIONS, DEV, PILOT / "freeze.json", *(ROOT / name for name in ["answer_rubric.json", "privacy_review.json", "base_requests.jsonl", "revision_prompts.json", "config.json"])]:
        files[str(path)] = digest(path)
    for source in sources:
        shutil.copyfile(source, ROOT / Path(source).name)
    write_json(ROOT / "freeze.json", {"created_at": datetime.now(timezone.utc).isoformat(), "files": files, "test_sha256": TEST_SHA, "max_actual_calls": 123, "no_gold_in_prompts": True, "policy_groups": len({q["policy_group_id"] for q in queries}), "unique_safe_chunks": len(reviewed)})
    print(json.dumps({"questions": len(queries), "safe_chunks": len(reviewed), "max_calls": 123, "rubric_points": sum(len(q["required_points"]) for q in rubric["questions"])}))


def run_batches(bases, config, output, client_factory=GenerationClient):
    if not 1 <= len(bases) <= 41 or len({b["question_id"] for b in bases}) != len(bases):
        raise ValueError("Expected at most 41 unique questions")
    rows, requests, calls, stopped = [], [], 0, False
    for offset in range(0, len(bases), 10):
        batch = bases[offset:offset + 10]
        if stopped:
            for base in batch:
                for policy in ["draft", "generic", "coverage"]:
                    rows.append({"request_id": f"{base['question_id']}-{policy}", "question_id": base["question_id"], "policy": policy, "status": "not_run_after_error", "presentation_status": "not_run_after_error", "attempted": False, "provider_response": {"status": "not_run_after_error"}})
            continue
        batch_output = output / f"batch-{offset // 10 + 1:02d}"
        batch_output.mkdir()
        execute(batch, config, batch_output, client_factory)
        batch_summary = json.loads((batch_output / "execution_summary.json").read_text())
        rows.extend(jsonlines((batch_output / "responses.jsonl").read_text()))
        requests.extend(jsonlines((batch_output / "requests.jsonl").read_text()))
        calls += batch_summary["actual_calls"]
        stopped = batch_summary["stopped_due_to_errors"]
    assert calls <= len(bases) * 3 <= 123
    write_jsonl(output / "responses.jsonl", rows)
    write_jsonl(output / "requests.jsonl", requests)
    write_json(output / "execution_summary.json", execution_summary(rows, calls, stopped))
    write_json(output / "response_freeze.json", {"responses_sha256": digest(output / "responses.jsonl"), "requests_sha256": digest(output / "requests.jsonl"), "execution_sha256": digest(output / "execution_summary.json")})


def run(output, allow_model_api):
    if not allow_model_api:
        raise ValueError("Requires --allow-model-api")
    verify_prepared(ROOT)
    config = json.loads((ROOT / "config.json").read_text())
    bases = jsonlines((ROOT / "base_requests.jsonl").read_text())
    if len(bases) != 41 or config["max_actual_calls"] != 123:
        raise ValueError("Frozen scope changed")
    output.mkdir(parents=True, exist_ok=False)
    write_json(ROOT / "execution_lock.json", {"started_at": datetime.now(timezone.utc).isoformat(), "output": str(output)})
    write_json(output / "run_manifest.json", {"freeze_sha256": digest(ROOT / "freeze.json"), "max_calls": 123, "no_retries": True, "question_batch_size": 10, "prior_pilot_questions_repeated": 0, "test_queries": 0})
    run_batches(bases, config, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-model-api", action="store_true")
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run(args.output, args.allow_model_api)
