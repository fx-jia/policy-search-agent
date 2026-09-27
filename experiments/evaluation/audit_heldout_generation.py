"""Independently recount saved generation outputs without invoking any model."""

from collections import Counter
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re


PREPARED = Path("experiments/datasets/private/heldout_generation_v1")
RESULT = Path("experiments/results/private/heldout-generation-test99-v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def canonical(text):
    return "".join(re.sub(r"<[^>]+>", "", html.unescape(text)).split())


def audit():
    freeze = json.loads((PREPARED / "freeze.json").read_text())
    for path, expected in freeze["files"].items():
        assert sha(path) == expected, path
    requests = {r["request_id"]: r for r in read_rows(PREPARED / "requests.jsonl")}
    responses = read_rows(RESULT / "responses.jsonl")
    assert len(requests) == len(responses) == 198
    assert {r["request_id"] for r in responses} == set(requests)
    selections = {(r["question_id"], r["method"] + "_" + r["policy"]): r for r in read_rows("experiments/results/private/heldout-retrieval-test99-v1/selections.jsonl")}
    quotes, valid, calls, usage_count = 0, 0, 0, 0
    usage = Counter()
    provider_ids = []
    invalid_schema_quotes = []
    for response in responses:
        request = requests[response["request_id"]]
        assert response["prompt_sha256"] == request["prompt_sha256"]
        assert request["selected_ids"] == selections[(request["question_id"], request["policy"])]["selected_ids"]
        assert len(request["chunks"]) == len(request["selected_ids"])
        assert all(c["source_document_id"] != "79870" for c in request["chunks"])
        provider = response["provider_response"]
        if provider["status"] != "not_run_after_error":
            calls += 1
        if provider.get("requested_at"):
            assert datetime.fromisoformat(freeze["created_at"]) < datetime.fromisoformat(provider["requested_at"])
        if provider.get("provider_request_id"):
            provider_ids.append(provider["provider_request_id"])
        if provider.get("usage"):
            usage_count += 1
            usage.update({k: provider["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")})
        if "parsed" not in response:
            if provider["status"] == "ok":
                raw = json.loads(provider["text"])
                sent = {"E" + str(i): c for i, c in enumerate(request["chunks"], 1)}
                for claim in raw.get("claims", []):
                    for evidence in claim.get("evidence", []):
                        cid = evidence["chunk_id"]
                        quote = canonical(evidence["quote"])
                        good = bool(quote) and cid in sent and quote in canonical(sent[cid]["content"])
                        invalid_schema_quotes.append({"request_id": response["request_id"], "chunk_id": cid, "valid_quote": good})
            continue
        chunks = {"E" + str(i): c for i, c in enumerate(request["chunks"], 1)}
        rebuilt = []
        for index, claim in enumerate(response["parsed"]["claims"]):
            for evidence in claim["evidence"]:
                cid, quote = evidence["chunk_id"], canonical(evidence["quote"])
                found = chunks.get(cid)
                good = bool(quote) and bool(found) and quote in canonical(found["content"])
                rebuilt.append({"claim_index": index, "chunk_id": cid, "original_chunk_id": found["id"] if found else None, "chunk_was_sent": bool(found), "valid_quote": good})
                quotes += 1
                valid += good
        assert rebuilt == response["citation_checks"], response["request_id"]
    assert len(provider_ids) == len(set(provider_ids))
    execution = json.loads((RESULT / "execution_summary.json").read_text())
    assert execution["actual_calls"] == calls and calls <= 198
    assert execution["reported_usage"] == dict(usage)
    assert execution["calls_with_usage"] == usage_count
    review = json.loads((RESULT / "assessments.json").read_text())
    assert review["responses_sha256"] == sha(RESULT / "responses.jsonl")
    assert review["rubric_sha256"] == sha(PREPARED / "answer_rubric.json")
    assessments = {a["request_id"]: a for a in review["assessments"]}
    scored = {s["request_id"]: s for s in read_rows(RESULT / "evaluation/scored.jsonl")}
    for response in responses:
        rid = response["request_id"]
        s = scored[rid]
        if response["status"] != "ok":
            assert not s["assessed"] and not s["answer_correct"] and not s["grounded_correct"]
            continue
        a = assessments[rid]
        assert a["response_row_sha256"] == hashlib.sha256(json.dumps(response, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        supported = [semantic and bool(claim["evidence"]) and all(c["valid_quote"] for c in response["citation_checks"] if c["claim_index"] == i) for i, (semantic, claim) in enumerate(zip(a["citation_support"], response["parsed"]["claims"]))]
        correct = all(a["points"]) and not a["extra_errors"]
        grounded = correct and a["faithful"] and a["all_answer_facts_in_claims"] and bool(supported) and all(supported)
        assert s["answer_correct"] == correct and s["grounded_correct"] == grounded
        assert s["correct_points"] == sum(a["points"]) and s["supported_cited_claims"] == sum(supported)
    report = {"requests_checked": len(requests), "responses_checked": len(responses), "actual_calls": calls, "unique_provider_ids": len(provider_ids), "quotes_checked": quotes, "valid_quotes": valid, "invalid_quotes": quotes - valid, "usage": dict(usage), "calls_with_usage": usage_count, "status_counts": dict(Counter(r["status"] for r in responses)), "presentation_counts": dict(Counter(r["presentation_status"] for r in responses)), "finish_reasons": dict(Counter(r["provider_response"].get("finish_reason", "unavailable") for r in responses)), "semantic_assessments_checked": len(assessments), "planned_denominator_each": 99, "no_model_calls_by_audit": True, "legacy_summary_test_queries_note": "execution_summary.test_queries=0 is inherited retrieval-call telemetry; this run DOES generate answers for 99 held-out questions in 198 conditions", "audit_sha256": sha(__file__)}
    report["invalid_schema_raw_quote_checks"] = invalid_schema_quotes
    (RESULT / "independent_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    audit()
