"""Dev-only paired draft revision with fixed, previously reviewed evidence."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from experiments.evaluation.context_experiment import digest, jsonlines, write_json, write_jsonl
from experiments.evaluation.generation_pilot import build_messages, safe_chunk
from experiments.evaluation.qa_regression import execution_summary, verify_prepared
from experiments.qa import GenerationClient, process_response


SOURCE = Path("experiments/datasets/private/policy_qa_v1")
PROTOCOL = Path("docs/coverage_revision_protocol_v1.md")
POLICIES = ["draft", "generic", "coverage"]
REVIEW_COMMON = """上一条assistant消息是待复查的初稿，可能存在错误，不是新增证据。重新核对最初的问题和原有政策片段，修订初稿，输出最终完整JSON。只依据原有片段，遵守原系统指令中的适用范围和连续引文要求；保留正确内容，纠正发现的问题，不增加用户未问的政策事实。"""
GENERIC_REVIEW = REVIEW_COMMON + """
认真复查答案的正确性、完整性、相关性和引用质量，并作必要修订。保持原answer、claims、insufficient_evidence格式，answer给出统一、简洁的最终回答。"""
COVERAGE_REVIEW = REVIEW_COMMON + """
本次按用户问题的子问覆盖逐项复查：
1. 只从最初的问题识别全部待答事项。分别识别并列对象、年份、事项；各项继承问题共享的地区、时间和适用条件，不从文档扩展题意。
2. 对每项检查初稿answer是否已直接回答，不能把只出现在claims或引文里的答案当作answer已经回答。再检查原有证据是否支持该答案。
3. 已回答且有据则保留；漏答或答错但有据则补正；无充分证据则在最终answer明确该项无法确认，不能猜测补齐。
4. 生成一份统一的最终answer，并同步更新claims与连续原文引用。不要分别生成多份独立答案再拼接。
在原JSON中额外增加简短coverage数组，每项只记录：
{"span":"逐字摘录问题中定位这一待答事项的连续短语，不加预设答案", "status":"covered或repaired或unsupported", "evidence_ids":["支持这一项的E编号"]}
covered表示初稿answer已回答且有依据，repaired表示本轮补正，unsupported表示所给证据不足（其evidence_ids可为空）。coverage仅是检查记录，不能替代最终answer回答。"""


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def revision_request(base, draft, policy):
    if policy not in {"generic", "coverage"} or "parsed" not in draft:
        raise ValueError("Revision requires a parsed common draft and known policy")
    request = copy.deepcopy(base)
    # Only the generated answer is appended: rubric/assessment fields never enter messages.
    answer = {k: draft["parsed"][k] for k in ["answer", "claims", "insufficient_evidence"]}
    request["messages"] += [
        {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        {"role": "user", "content": GENERIC_REVIEW if policy == "generic" else COVERAGE_REVIEW},
    ]
    request.update({"policy": policy, "request_id": f"{base['question_id']}-{policy}", "parent_draft_sha256": _hash_json(answer), "prompt_sha256": _hash_json(request["messages"])})
    return request


def valid_coverage(trace, request):
    if not isinstance(trace, list) or not 1 <= len(trace) <= 12:
        return False
    seen = set()
    for item in trace:
        if not isinstance(item, dict) or set(item) != {"span", "status", "evidence_ids"}:
            return False
        span, status, ids = item["span"], item["status"], item["evidence_ids"]
        if not isinstance(span, str) or not span.strip() or span not in request["question"] or span in seen:
            return False
        if status not in {"covered", "repaired", "unsupported"} or not isinstance(ids, list) or any(not isinstance(cid, str) or cid not in request["citation_map"] for cid in ids):
            return False
        if status != "unsupported" and not ids:
            return False
        seen.add(span)
    return True


def prepare(output):
    verify_prepared(SOURCE)
    old_config = json.loads((SOURCE / "config.json").read_text())
    bases = [r for r in jsonlines((SOURCE / "requests.jsonl").read_text()) if r["policy"] == "short_ids"]
    if len(bases) != 10 or [r["question_id"] for r in bases] != old_config["question_ids"]:
        raise ValueError("Unexpected development question scope")
    runtime = old_config["runtime"]
    snapshot = {c["id"]: c for c in json.loads(Path(runtime["snapshot"]).read_text())["chunks"]}
    review = json.loads(Path(runtime["privacy_review"]).read_text())
    hold = json.loads(Path(runtime["privacy_hold"]).read_text())
    blocked = set(review["blocked_source_ids"]) | {hold["source_document_id"], "79870"}
    for base in bases:
        rebuilt = [safe_chunk(snapshot[cid], review["chunks"], blocked) for cid in base["selected_ids"]]
        if rebuilt != base["chunks"] or build_messages(base["question"], rebuilt, True) != base["messages"]:
            raise ValueError("Outbound evidence or baseline prompt changed")
        base.update({"policy": "draft", "request_id": f"{base['question_id']}-draft"})
    output.mkdir(parents=True, exist_ok=False)
    config = {**runtime["generation"], "max_calls": 15, "max_actual_calls": 30, "question_ids": old_config["question_ids"], "policies": POLICIES, "rubric": old_config["rubric"]}
    write_json(output / "config.json", config)
    write_jsonl(output / "base_requests.jsonl", bases)
    write_json(output / "revision_prompts.json", {"generic": GENERIC_REVIEW, "coverage": COVERAGE_REVIEW})
    old_freeze = json.loads((SOURCE / "freeze.json").read_text())
    files = dict(old_freeze["files"])
    new_files = [Path(__file__), PROTOCOL, Path("experiments/evaluation/test_coverage_revision.py"), SOURCE / "freeze.json", output / "config.json", output / "base_requests.jsonl", output / "revision_prompts.json"]
    for path in new_files:
        files[str(path)] = digest(path)
    for path in [Path(__file__), PROTOCOL]:
        shutil.copyfile(path, output / path.name)
    write_json(output / "freeze.json", {"created_at": datetime.now(timezone.utc).isoformat(), "files": files, "test_sha256": old_freeze["test_sha256"], "max_actual_calls": 30, "unique_safe_chunks": len({c["id"] for b in bases for c in b["chunks"]}), "generation_only": True})
    print(json.dumps({"prepared": str(output), "questions": 10, "max_calls": 30, "embedding_calls": 0}))


def execute(bases, config, output, client_factory=GenerationClient):
    if not 1 <= len(bases) <= 10 or config["max_calls"] != 15:
        raise ValueError("Experiment limited to 10 questions and 15 calls per session")
    if len({b["question_id"] for b in bases}) != len(bases):
        raise ValueError("Duplicate question")
    rows, calls, consecutive, stopped, client = [], 0, 0, False, None
    try:
        with (output / "responses.jsonl").open("x") as responses, (output / "requests.jsonl").open("x") as requests:
            for i, base in enumerate(bases):
                draft = None
                for policy in (["draft", "generic", "coverage"] if i % 2 == 0 else ["draft", "coverage", "generic"]):
                    request = {**base, "policy": policy, "request_id": f"{base['question_id']}-{policy}"}
                    sent = False
                    if stopped:
                        provider = {"status": "not_run_after_error"}
                    elif policy != "draft" and "parsed" not in draft:
                        provider = {"status": "not_run_invalid_draft"}
                    else:
                        if policy != "draft":
                            request = revision_request(base, draft, policy)
                        requests.write(json.dumps(request, ensure_ascii=False) + "\n")
                        requests.flush()
                        try:
                            if client is None or client.calls >= config["max_calls"]:
                                if client is not None:
                                    client.close()
                                client = None
                                client = client_factory(config)
                            if calls >= 30:
                                raise ValueError("Global call budget reached")
                            calls += 1
                            sent = True
                            provider = client.generate(request)
                        except Exception as exc:
                            provider = {"status": "error", "error_type": type(exc).__name__, "error_category": "worker_protocol"}
                        consecutive = consecutive + 1 if provider["status"] != "ok" else 0
                        stopped = consecutive >= 2 or provider.get("error_category") in {"auth", "quota", "configuration", "worker_protocol"}
                    result = process_response(provider, request)
                    record = {k: request[k] for k in ["request_id", "question_id", "policy", "prompt_sha256", "selection_tokens_cl100k"]}
                    record.update({**result, "attempted": sent, "presentation_status": result["status"], "reused_identical_prompt": False, "parent_draft_sha256": request.get("parent_draft_sha256")})
                    if result["status"] == "invalid_citations":
                        record["status"] = "ok"
                    if policy == "coverage":
                        record["coverage_trace_valid"] = valid_coverage(result.get("parsed", {}).get("coverage"), request)
                    if policy == "draft":
                        draft = result
                    rows.append(record)
                    responses.write(json.dumps(record, ensure_ascii=False) + "\n")
                    responses.flush()
                    print(json.dumps({"request_id": record["request_id"], "status": record["presentation_status"], "calls": calls}), flush=True)
    finally:
        if client is not None:
            client.close()
    write_json(output / "execution_summary.json", execution_summary(rows, calls, stopped))
    write_json(output / "response_freeze.json", {"responses_sha256": digest(output / "responses.jsonl"), "requests_sha256": digest(output / "requests.jsonl"), "execution_sha256": digest(output / "execution_summary.json")})


def run(prepared, output, allow_model_api):
    if not allow_model_api:
        raise ValueError("Generation requires --allow-model-api")
    verify_prepared(prepared)
    config = json.loads((prepared / "config.json").read_text())
    bases = jsonlines((prepared / "base_requests.jsonl").read_text())
    if config["max_actual_calls"] != 30 or len(bases) != 10:
        raise ValueError("Frozen experiment scope mismatch")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_manifest.json", {"prepared_freeze_sha256": digest(prepared / "freeze.json"), "max_actual_calls": 30, "no_retries": True, "test_queries": 0, "physical_drafts_shared_between_review_arms": True})
    execute(bases, config, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-model-api", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output)
    else:
        run(args.prepared, args.output, args.allow_model_api)


if __name__ == "__main__":
    main()
