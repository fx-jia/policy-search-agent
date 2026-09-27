"""Summarize recorded native Agent runs without judging their answers."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re


_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_ANNOTATION_FIELDS = ("content_complete", "grounded", "citation_issue", "evidence_complete")


def _planned_ids(config):
    if not isinstance(config, dict) or not isinstance(config.get("question_ids"), list):
        raise ValueError("config.question_ids must be a list")
    ids = config["question_ids"]
    if not ids or any(not isinstance(qid, str) or not _SAFE_ID.fullmatch(qid) for qid in ids) or len(ids) != len(set(ids)):
        raise ValueError("question_ids must be distinct safe nonempty strings")
    return ids


def _index_by_id(rows, allowed, kind):
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{kind} row must be an object")
        qid = row.get("question_id")
        if qid not in allowed or qid in indexed:
            raise ValueError(f"duplicate_or_unplanned_{kind}_question_id: {qid}")
        indexed[qid] = row
    return indexed


def _usage(call):
    usage = call.get("usage")
    if call.get("transport_status") == "not_sent":
        if call.get("transport_error") != "ConnectError" or usage != {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}:
            raise ValueError("invalid_unsent_usage")
        return usage
    if usage is None:
        return None
    if not isinstance(usage, dict) or type(usage.get("total_tokens")) is not int or usage["total_tokens"] <= 0:
        raise ValueError("invalid_provider_usage")
    for field in ("prompt_tokens", "completion_tokens"):
        if field in usage and (type(usage[field]) is not int or usage[field] < 0):
            raise ValueError("invalid_provider_usage")
    if all(field in usage for field in ("prompt_tokens", "completion_tokens")) and usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]:
        raise ValueError("inconsistent_provider_usage_split")
    return usage


def _calls_for_record(record, seen_calls):
    calls = record.get("calls")
    if not isinstance(calls, list):
        raise ValueError("calls must be a list")
    unique = []
    qid = record["question_id"]
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("call_id"), str) or not call["call_id"] or not isinstance(call.get("request"), dict):
            raise ValueError("call_id/request missing")
        cid = call["call_id"]
        usage = _usage(call)
        canonical = {"call_id": cid, "request": call["request"], "usage": usage}
        if "transport_status" in call:
            canonical.update(transport_status=call["transport_status"], transport_error=call.get("transport_error"))
        previous = seen_calls.get(cid)
        if previous is not None:
            if previous[0] != qid or previous[1] != canonical:
                raise ValueError(f"conflicting_or_cross_question_call_id: {cid}")
            continue
        seen_calls[cid] = (qid, canonical)
        unique.append(canonical)
    return unique


def _annotation(row):
    for field in _ANNOTATION_FIELDS:
        if type(row.get(field)) is not bool:
            raise ValueError(f"annotation.{field} must be boolean")
    return {field: row[field] for field in _ANNOTATION_FIELDS}


def build_report(config, records, annotations):
    """Count planned outcomes and unique physical requests; retain review inputs."""
    question_ids = _planned_ids(config)
    experiment = {field: deepcopy(config.get(field)) for field in ("model", "reasoning", "native_patch", "configuration_changes")}
    if "evidence_integrity_patch" in config:
        experiment["evidence_integrity_patch"] = deepcopy(config["evidence_integrity_patch"])
    allowed = set(question_ids)
    by_record = _index_by_id(records, allowed, "record")
    by_annotation = _index_by_id(annotations, allowed, "annotation")
    seen_calls = {}
    per_question = []
    packets = []
    known_total = known_prompt = known_completion = 0
    pending_ids = []
    known_split_complete = True
    completed_ids = set()
    for qid in question_ids:
        record = by_record.get(qid)
        if record is None:
            per_question.append(
                {
                    "question_id": qid,
                    "technical_status": "not_run",
                    "provider_call_count": 0,
                    "known_provider_tokens": 0,
                    "provider_tokens": 0,
                    "pending_call_ids": [],
                    "final_evidence_candidate_count": 0,
                    **{field: None for field in _ANNOTATION_FIELDS},
                }
            )
            continue
        status = record.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"missing technical status: {qid}")
        answer = record.get("answer", "")
        if not isinstance(answer, str):
            raise ValueError(f"answer must be a string: {qid}")
        evidence = record.get("final_evidence", [])
        if not isinstance(evidence, list):
            raise ValueError(f"final_evidence must be a list: {qid}")
        calls = _calls_for_record(record, seen_calls)
        technical_status = "invalid_empty_answer" if status == "ok" and not answer.strip() else status
        if technical_status == "ok":
            completed_ids.add(qid)
        question_known = 0
        question_pending = []
        for call in calls:
            usage = call["usage"]
            if usage is None:
                pending_ids.append(call["call_id"])
                question_pending.append(call["call_id"])
                continue
            known_total += usage["total_tokens"]
            question_known += usage["total_tokens"]
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                known_prompt += usage["prompt_tokens"]
                known_completion += usage["completion_tokens"]
            else:
                known_split_complete = False
        annotation = by_annotation.get(qid)
        if annotation is not None and technical_status != "ok":
            raise ValueError(f"annotation_for_incomplete_question: {qid}")
        labels = _annotation(annotation) if annotation is not None else {field: None for field in _ANNOTATION_FIELDS}
        per_question.append(
            {
                "question_id": qid,
                "technical_status": technical_status,
                "provider_call_count": sum(c.get("transport_status") != "not_sent" for c in calls),
                "send_attempt_count": len(calls),
                "known_provider_tokens": question_known,
                "provider_tokens": None if question_pending else question_known,
                "pending_call_ids": question_pending,
                "final_evidence_candidate_count": len(evidence),
                **labels,
            }
        )
        packets.append(
            {
                "experiment": deepcopy(experiment),
                "question_id": qid,
                "technical_status": technical_status,
                "answer": answer,
                "final_evidence": deepcopy(evidence),
                "final_request": deepcopy(record.get("final_request")),
                "evidence_mapping_note": "Literal candidates require source/citation review; pre_summary is model-derived, not original policy text.",
                "calls": deepcopy(calls),
                "annotation": deepcopy(labels) if annotation is not None else None,
            }
        )
    if set(by_annotation) - completed_ids:
        raise ValueError("annotations require completed answers")
    status_counts = Counter(row["technical_status"] for row in per_question)
    assessed = [row for row in per_question if row["content_complete"] is not None]
    historical = config.get("historical_native_report")
    if historical is not None and (not isinstance(historical, str) or not historical):
        raise ValueError("historical_native_report must be a path string")
    usage_complete = not pending_ids
    content_complete = sum(row["content_complete"] for row in assessed)
    complete_and_grounded = sum(row["content_complete"] and row["grounded"] for row in assessed)
    return {
        "experiment": experiment,
        "planned_questions": len(question_ids),
        "attempted_questions": len(by_record),
        "completed_questions": len(completed_ids),
        "not_run_questions": [qid for qid in question_ids if qid not in by_record],
        "technical_status_counts": dict(sorted(status_counts.items())),
        "assessed_questions": len(assessed),
        "unassessed_completed_questions": len(completed_ids) - len(assessed),
        "semantic_review_partial": len(assessed) != len(question_ids),
        "content_complete_questions": content_complete,
        "content_complete_fraction_planned": content_complete / len(question_ids),
        "content_complete_fraction_assessed": content_complete / len(assessed) if assessed else None,
        "content_complete_and_grounded_questions": complete_and_grounded,
        "content_complete_and_grounded_fraction_planned": complete_and_grounded / len(question_ids),
        "content_complete_and_grounded_fraction_assessed": complete_and_grounded / len(assessed) if assessed else None,
        "final_evidence_complete_questions": sum(row["evidence_complete"] for row in assessed),
        "citation_issue_questions": sum(row["citation_issue"] for row in assessed),
        "provider_usage": {
            "physical_request_count": sum(c[1].get("transport_status") != "not_sent" for c in seen_calls.values()),
            "send_attempt_count": len(seen_calls),
            "not_sent_attempt_count": sum(c[1].get("transport_status") == "not_sent" for c in seen_calls.values()),
            "known_total_tokens": known_total,
            "total_tokens": known_total if usage_complete else None,
            "known_prompt_tokens": known_prompt,
            "known_completion_tokens": known_completion,
            "usage_complete": usage_complete,
            "usage_split_complete": usage_complete and known_split_complete,
            "pending_call_ids": pending_ids,
        },
        "historical_native": {"source": "historical_report", "path": historical, "paired_contemporaneously": False} if historical else None,
        "semantic_assessment": "explicit_annotations_only; no_automatic_llm_judge",
        "per_question": per_question,
        "review_packets": packets,
    }


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    config = json.loads((root / "config.json").read_text())
    records = _jsonl(root / "records.jsonl")
    annotations_path = root / "annotations.jsonl"
    annotations = _jsonl(annotations_path) if annotations_path.exists() else []
    report = build_report(config, records, annotations)
    packets = report.pop("review_packets")
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    packet_dir = root / "review_packets"
    packet_dir.mkdir(exist_ok=True)
    for packet in packets:
        (packet_dir / f"{packet['question_id']}.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"planned_questions": report["planned_questions"], "completed_questions": report["completed_questions"], "assessed_questions": report["assessed_questions"]}))


if __name__ == "__main__":
    main()
