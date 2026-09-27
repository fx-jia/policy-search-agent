"""Focused, offline contracts for native Agent result accounting."""

import importlib
import json
import subprocess
import sys

import pytest


def report_module():
    return importlib.import_module("experiments.evaluation.native_agent_report")


def test_failed_attempt_keeps_planned_denominator_and_unrun_id():
    report = report_module().build_report(
        {"question_ids": ["q1", "q2"]},
        [{"question_id": "q1", "status": "budget_censored", "calls": [], "answer": "", "final_evidence": []}],
        [],
    )
    assert report["planned_questions"] == 2
    assert report["attempted_questions"] == 1
    assert report["completed_questions"] == 0
    assert report["not_run_questions"] == ["q2"]
    assert report["assessed_questions"] == 0
    assert report["technical_status_counts"] == {"budget_censored": 1, "not_run": 1}
    assert report["per_question"][1]["known_provider_tokens"] == 0
    assert report["per_question"][1]["provider_tokens"] == 0
    assert report["per_question"][1]["pending_call_ids"] == []


def test_physical_call_id_is_counted_once_and_pending_usage_is_not_zero():
    call = {"call_id": "q1:1", "request": {"model": "m"}, "usage": {"prompt_tokens": 70, "completion_tokens": 30, "total_tokens": 100}}
    report = report_module().build_report(
        {"question_ids": ["q1"]},
        [{"question_id": "q1", "status": "ok", "calls": [call, dict(call), {"call_id": "q1:2", "request": {"model": "m"}, "usage": None}], "answer": "答复", "final_evidence": []}],
        [],
    )
    assert report["provider_usage"]["physical_request_count"] == 2
    assert report["provider_usage"]["known_total_tokens"] == 100
    assert report["provider_usage"]["total_tokens"] is None
    assert report["provider_usage"]["pending_call_ids"] == ["q1:2"]
    assert report["provider_usage"]["usage_complete"] is False
    assert report["per_question"][0]["provider_call_count"] == 2


def test_unsent_connect_failure_is_separate_from_sent_and_unknown_usage():
    failed = {"call_id": "q1:1", "request": {"model": "m"}, "transport_status": "not_sent", "transport_error": "ConnectError",
              "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    sent = {"call_id": "q1:2", "request": {"model": "m"}, "usage": {"total_tokens": 10}}
    record = {"question_id": "q1", "status": "ok", "calls": [failed, sent], "answer": "答", "final_evidence": []}
    report = report_module().build_report({"question_ids": ["q1"]}, [record], [])
    assert report["provider_usage"]["send_attempt_count"] == 2
    assert report["provider_usage"]["not_sent_attempt_count"] == 1
    assert report["provider_usage"]["physical_request_count"] == 1
    assert report["provider_usage"]["total_tokens"] == 10
    assert report["provider_usage"]["pending_call_ids"] == []
    assert report["per_question"][0]["provider_call_count"] == 1
    assert report["review_packets"][0]["calls"][0]["transport_status"] == "not_sent"
    failed["transport_error"] = "ReadError"
    with pytest.raises(ValueError, match="usage"):
        report_module().build_report({"question_ids": ["q1"]}, [record], [])


def test_conflicting_or_cross_question_call_identity_is_rejected():
    base = {"call_id": "same", "request": {"model": "m"}, "usage": {"total_tokens": 10}}
    with pytest.raises(ValueError, match="call_id"):
        report_module().build_report(
            {"question_ids": ["q1"]},
            [{"question_id": "q1", "status": "ok", "calls": [base, {**base, "usage": {"total_tokens": 11}}], "answer": "答", "final_evidence": []}],
            [],
        )
    with pytest.raises(ValueError, match="call_id"):
        report_module().build_report(
            {"question_ids": ["q1", "q2"]},
            [
                {"question_id": "q1", "status": "ok", "calls": [base], "answer": "一", "final_evidence": []},
                {"question_id": "q2", "status": "ok", "calls": [base], "answer": "二", "final_evidence": []},
            ],
            [],
        )


def test_annotations_are_explicit_and_final_evidence_is_preserved_for_review():
    evidence = [{"chunk_id": "final-1", "source_document_id": "doc-a", "content": "已审政策句"}]
    record = {"question_id": "q1", "status": "ok", "calls": [], "answer": "答案 [ID:0]", "final_evidence": evidence, "searched_evidence": [{"chunk_id": "unused"}]}
    annotation = {"question_id": "q1", "content_complete": True, "grounded": True, "citation_issue": False, "evidence_complete": True}
    report = report_module().build_report({"question_ids": ["q1", "q2"]}, [record], [annotation])
    assert report["content_complete_questions"] == 1
    assert report["content_complete_and_grounded_questions"] == 1
    assert report["final_evidence_complete_questions"] == 1
    assert report["assessed_questions"] == 1
    assert report["review_packets"][0]["final_evidence"] == evidence
    assert "searched_evidence" not in report["review_packets"][0]
    assert report["review_packets"][0]["answer"] == "答案 [ID:0]"
    assert report["per_question"][1]["content_complete"] is None


def test_partial_semantic_results_use_planned_denominator_without_hiding_missing_questions():
    report = report_module().build_report(
        {"question_ids": ["q1", "q2"]},
        [{"question_id": "q1", "status": "ok", "calls": [], "answer": "答", "final_evidence": []}],
        [{"question_id": "q1", "content_complete": True, "grounded": True, "citation_issue": False, "evidence_complete": True}],
    )
    assert report["content_complete_fraction_planned"] == 0.5
    assert report["content_complete_fraction_assessed"] == 1.0
    assert report["content_complete_and_grounded_fraction_planned"] == 0.5
    assert report["semantic_review_partial"] is True


def test_historical_native_reference_is_never_called_contemporaneous():
    report = report_module().build_report(
        {"question_ids": ["q1"], "historical_native_report": "old/native_report.json"}, [], []
    )
    assert report["historical_native"] == {
        "source": "historical_report",
        "path": "old/native_report.json",
        "paired_contemporaneously": False,
    }


@pytest.mark.parametrize("patched", [False, True], ids=["native", "compose-fix"])
def test_report_and_every_packet_preserve_experiment_identity(patched):
    identity = {
        "model": "qwen3.7-flash-2026-07-15",
        "reasoning": 2,
        "native_patch": {
            "name": "compose_current_evidence_v1",
            "original_function_sha256": "a" * 64,
            "patched_function_sha256": "b" * 64,
        } if patched else None,
        "configuration_changes": ["physical_request_budget"] + (["compose_current_evidence_v1"] if patched else []),
    }
    config = {"question_ids": ["q1", "q2"], **identity}
    if not patched:
        config.pop("native_patch")
    report = report_module().build_report(
        config,
        [
            {"question_id": "q1", "status": "ok", "calls": [], "answer": "答", "final_evidence": []},
            {"question_id": "q2", "status": "guard_stopped", "calls": [], "answer": "", "final_evidence": []},
        ],
        [],
    )
    assert report["experiment"] == identity
    assert [packet["experiment"] for packet in report["review_packets"]] == [identity, identity]


def test_report_preserves_separate_integrity_condition():
    api = report_module()
    integrity = {"name": "evidence_integrity_v1", "module_sha256": "c" * 64}
    report = api.build_report({"question_ids": ["q1"], "evidence_integrity_patch": integrity}, [], [])
    assert report["experiment"]["evidence_integrity_patch"] == integrity
    integrity["name"] = "changed_after_report"
    assert report["experiment"]["evidence_integrity_patch"]["name"] == "evidence_integrity_v1"


def test_cli_writes_report_and_packets_without_annotations(tmp_path):
    identity = {
        "model": "qwen3.7-flash-2026-07-15",
        "reasoning": 2,
        "native_patch": {"name": "compose_current_evidence_v1", "original_function_sha256": "a" * 64, "patched_function_sha256": "b" * 64},
        "configuration_changes": ["compose_current_evidence_v1"],
    }
    (tmp_path / "config.json").write_text(json.dumps({"question_ids": ["q1", "q2"], **identity}))
    (tmp_path / "records.jsonl").write_text(json.dumps({"question_id": "q1", "status": "ok", "calls": [], "answer": "答", "final_evidence": [{"chunk_id": "c1"}]}) + "\n")
    subprocess.run([sys.executable, "-m", "experiments.evaluation.native_agent_report", "--input", str(tmp_path)], check=True, capture_output=True, text=True)
    summary = json.loads((tmp_path / "report.json").read_text())
    packet = json.loads((tmp_path / "review_packets" / "q1.json").read_text())
    assert summary["planned_questions"] == 2
    assert summary["assessed_questions"] == 0
    assert summary["experiment"] == identity
    assert packet["experiment"] == identity
    assert "review_packets" not in summary
    assert packet["final_evidence"] == [{"chunk_id": "c1"}]
    assert not (tmp_path / "review_packets" / "q2.json").exists()
