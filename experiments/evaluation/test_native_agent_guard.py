"""Offline tests of the real durable budget/scope boundary."""

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def api():
    path = Path(__file__).with_name("native_agent_guard.py")
    assert path.exists(), "native_agent_guard implementation missing"
    spec = importlib.util.spec_from_file_location("native_agent_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ledger(api, path, **limits):
    return api.CallLedger(path, **{"run_limit": 100, "question_limit": 100, "call_limit": 2, **limits})


def test_scope_cannot_widen(api):
    assert api.validate_scope(["b"], {"a", "b"}) == ["b"]
    for value in (None, [], ["c"], "a", ["a", ""]):
        with pytest.raises(ValueError):
            api.validate_scope(value, {"a", "b"})


def test_pending_reservations_block_overspend(api, tmp_path):
    log = ledger(api, tmp_path / "calls.jsonl")
    log.start("q")
    log.reserve("q:1", 60, {"messages": []})
    with pytest.raises(ValueError, match="budget"):
        log.reserve("q:2", 60, {"messages": []})
    assert log.summary()["provider_calls"] == 1
    assert log.summary()["pending_calls"] == ["q:1"]


def test_abort_survives_fallback(api, tmp_path):
    log = ledger(api, tmp_path / "calls.jsonl")
    log.start("q")
    log.abort("unreviewed_chunk")
    with pytest.raises(ValueError, match="unreviewed_chunk"):
        log.reserve("q:1", 20, {"messages": []})
    assert log.summary()["provider_calls"] == 0


def test_independent_usage_and_out_of_order_completion(api, tmp_path):
    path = tmp_path / "calls.jsonl"
    log = ledger(api, path)
    log.start("q")
    log.reserve("q:1", 40, {"messages": []})
    log.reserve("q:2", 40, {"messages": []})
    log.finish("q:2", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    log.finish("q:1", {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25})
    assert log.summary()["provider_total_tokens"] == 40
    assert log.summary()["pending_calls"] == []
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["event"] for e in events] == ["start", "reserve", "reserve", "finish", "finish"]
    resumed = ledger(api, path)
    resumed.start("next")
    with pytest.raises(ValueError, match="budget"):
        resumed.reserve("next:1", 61, {"messages": []})


def test_unknown_usage_and_duplicate_attempt_block_resume(api, tmp_path):
    path = tmp_path / "calls.jsonl"
    log = ledger(api, path)
    log.start("q")
    log.reserve("q:1", 40, {"messages": []})
    with pytest.raises(ValueError, match="usage"):
        log.finish("q:1", {})
    with pytest.raises(ValueError):
        ledger(api, path).start("q2")
    assert log.summary()["pending_calls"] == ["q:1"]


def test_duplicate_call_and_question_are_rejected(api, tmp_path):
    log = ledger(api, tmp_path / "calls.jsonl")
    log.start("q")
    log.reserve("q:1", 40, {"messages": []})
    log.finish("q:1", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    with pytest.raises(ValueError, match="duplicate"):
        log.reserve("q:1", 40, {"messages": []})
    with pytest.raises(ValueError):
        log.start("q")


def test_call_limit_and_question_budget(api, tmp_path):
    log = ledger(api, tmp_path / "calls.jsonl", question_limit=30, call_limit=1)
    log.start("q")
    log.reserve("q:1", 20, {"messages": []})
    log.finish("q:1", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    with pytest.raises(ValueError, match="call_limit"):
        log.reserve("q:2", 10, {"messages": []})


def test_usage_exceeding_reservation_locks_further_calls(api, tmp_path):
    log = ledger(api, tmp_path / "calls.jsonl")
    log.start("q")
    log.reserve("q:1", 10, {"messages": []})
    with pytest.raises(ValueError, match="reservation"):
        log.finish("q:1", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    assert log.summary()["provider_total_tokens"] == 15
    with pytest.raises(ValueError):
        log.reserve("q:2", 10, {"messages": []})


def test_not_sent_releases_reservation_but_preserves_attempt_on_replay(api, tmp_path):
    path = tmp_path / "calls.jsonl"
    log = ledger(api, path)
    log.start("q")
    log.reserve("q:1", 100, {"messages": []})
    log.finish_not_sent("q:1")
    resumed = ledger(api, path)
    assert resumed.calls["q:1"]["transport_status"] == "not_sent"
    assert resumed.calls["q:1"]["transport_error"] == "ConnectError"
    assert resumed.calls["q:1"]["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert resumed.summary()["pending_calls"] == []
    assert resumed.summary()["provider_calls"] == 1
    assert resumed.summary()["provider_not_sent_calls"] == 1
    assert resumed.summary()["provider_total_tokens"] == 0
    resumed.reserve("q:2", 100, {"messages": []})
    resumed.finish("q:2", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    assert resumed.summary()["provider_calls"] == 2
    assert resumed.summary()["provider_total_tokens"] == 15
    with pytest.raises(ValueError, match="question_call_limit"):
        resumed.reserve("q:3", 1, {"messages": []})


def test_not_sent_rejects_unknown_or_already_finished_attempt(api, tmp_path):
    log = ledger(api, tmp_path / "calls.jsonl")
    log.start("q")
    with pytest.raises(ValueError, match="unknown_or_duplicate_completion"):
        log.finish_not_sent("missing")
    log.reserve("q:1", 100, {})
    log.finish_not_sent("q:1")
    with pytest.raises(ValueError, match="unknown_or_duplicate_completion"):
        log.finish_not_sent("q:1")


def test_normal_completion_still_rejects_zero_usage(api, tmp_path):
    log = ledger(api, tmp_path / "calls.jsonl")
    log.start("q")
    log.reserve("q:1", 100, {})
    with pytest.raises(ValueError, match="provider_usage_unknown"):
        log.finish("q:1", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    assert log.summary()["pending_calls"] == ["q:1"]
    assert "transport_status" not in log.calls["q:1"]


def _write_preflight_receipt(root, log, identity):
    evidence = root / "reconciliation"
    evidence.mkdir(exist_ok=True)
    (root / "session.json").write_text(json.dumps(identity))
    report = evidence / "review.md"
    report.write_text("Independently reviewed source-attested pre-transport rejection.\n")
    original_sha = hashlib.sha256(log.path.read_bytes()).hexdigest()
    proof = {"schema": "policy-preflight-not-sent-proof/v1", "question_id": log.question_id,
             "identity": identity, "original_ledger_sha256": original_sha, "author": "reviewer",
             "basis": "source_attested_pre_transport_guard_rejection",
             "report_path": "reconciliation/review.md", "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
             "source_sha256": {name: "a" * 64 for name in ("native_chat_model", "native_embedding_model", "policy_search_service", "policy_transport", "deployment_manifest", "container_image")},
             "facts": {"guard_error": "unbounded_generation_request", "model_name": "qwen3.8-max", "requested_max_tokens": 600,
                       "native_cleaner_deleted_limit": True, "family_restored_limit": False,
                       "offline_outer_attempts": 1, "offline_inner_sends": 0, "setup_model_posts": 0,
                       "streaming_branch": False, "audited_client_only": True, "sdk_max_retries": 0}}
    snapshots = {"native-chat-model.py": b"offline native chat snapshot\n", "native-embedding-model.py": b"offline native embedding snapshot\n"}
    for name, content in snapshots.items():
        (evidence / name).write_bytes(content)
    proof["source_sha256"]["native_chat_model"] = hashlib.sha256(snapshots["native-chat-model.py"]).hexdigest()
    proof["source_sha256"]["native_embedding_model"] = hashlib.sha256(snapshots["native-embedding-model.py"]).hexdigest()
    deployment = {"container_id": "c" * 64, "image": "offline-image", "started_at": "2026-10-11T01:17:56Z", "all_file_hashes_verified": True,
                  "scope": {"tenant_id": identity["tenant_id"], "kb_id": identity["kb_id"]},
                  "files": [{"target": target, "sha256": "a" * 64, "container_sha256": "a" * 64, "verified": True, "mode": "ro"}
                            for target in ("/ragflow/api/db/services/policy_search_service.py", "/ragflow/experiments/search_agent/policy_transport.py")]}
    (evidence / "first-question-deployment.json").write_text(json.dumps(deployment))
    proof["source_sha256"]["deployment_manifest"] = hashlib.sha256((evidence / "first-question-deployment.json").read_bytes()).hexdigest()
    (evidence / "first-question-session.json").write_text(json.dumps(identity))
    (evidence / "first-question-original-ledger.jsonl").write_bytes(log.path.read_bytes())
    first_question = {"question_id": log.question_id, "result": {"status": "technical_error", "stop_reason": "resolution_error", "error": "Expecting value: line 1 column 1 (char 0)", "session_id": identity["session_id"], "usage": {"unknown_usage": True}},
                      "ledger": log.summary(), "calls": [], "models": {"generation": "qwen3.8-max"}}
    (evidence / "first-question.json").write_text(json.dumps(first_question))
    runtime = {"schema": "policy-native-runtime-attestation/v1", "container_id": deployment["container_id"], "image": deployment["image"], "started_at": deployment["started_at"],
               "image_id": "sha256:" + proof["source_sha256"]["container_image"], "attested_by": "independent-approver", "report_sha256": proof["report_sha256"],
               "native_source_sha256": {key: proof["source_sha256"][key] for key in ("native_chat_model", "native_embedding_model")}}
    (evidence / "native-runtime.json").write_text(json.dumps(runtime))
    proof["evidence_sha256"] = {name: hashlib.sha256((evidence / name).read_bytes()).hexdigest() for name in (
        "first-question-deployment.json", "first-question-session.json", "first-question-original-ledger.jsonl", "first-question.json",
        "native-chat-model.py", "native-embedding-model.py", "native-runtime.json")}
    proof["limitations"] = {"historical_live_stderr_preserved": False, "historical_packet_trace": False, "provider_usage_receipt": False, "basis_is_source_attested": True}
    proof_path = evidence / "proof.json"
    proof_path.write_text(json.dumps(proof))
    receipt = {"schema": "policy-preflight-not-sent-receipt/v1", "question_id": log.question_id,
               "identity": identity, "original_ledger_sha256": original_sha,
               "basis": "source_attested_pre_transport_guard_rejection", "proof_author": "reviewer", "approved_by": "independent-approver",
               "proof_path": "reconciliation/proof.json", "proof_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest()}
    receipt_path = evidence / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    return "reconciliation/receipt.json"


def _preflight_ledger(api, root):
    identity = {"tenant_id": "t", "kb_id": "k", "session_id": "s", "budget": 100}
    log = ledger(api, root / "q.jsonl", identity=identity, artifact_root=root)
    log.start("q")
    log.abort("provider_usage_unknown")
    return log, identity


def test_preflight_receipt_is_append_only_scoped_and_replayed(api, tmp_path):
    log, identity = _preflight_ledger(api, tmp_path)
    before = log.path.read_bytes()
    assert log.summary() == {"provider_calls": 0, "provider_not_sent_calls": 0, "provider_total_tokens": 0,
                             "pending_calls": [], "fatal_reason": "provider_usage_unknown"}
    receipt = _write_preflight_receipt(tmp_path, log, identity)
    log._reconcile_preflight_not_sent(receipt)
    assert log.path.read_bytes().startswith(before)
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["start", "abort", "preflight_not_sent_reconciled"]
    assert "usage" not in rows[-1]
    assert rows[-1]["prior_ledger_sha256"] == hashlib.sha256(before).hexdigest()
    assert log.calls == {}
    summary = log.summary()
    assert summary["fatal_reason"] is None
    assert summary["preflight_not_sent_reconciliation"]["basis"] == "source_attested_pre_transport_guard_rejection"
    assert summary["preflight_not_sent_reconciliation"]["provider_usage_receipt"] is False
    assert summary["preflight_not_sent_reconciliation"]["original_fatal_reason"] == "provider_usage_unknown"
    assert ledger(api, log.path, identity=identity, artifact_root=tmp_path).summary() == summary
    after = log.path.read_bytes()
    with pytest.raises(ValueError):
        log._reconcile_preflight_not_sent(receipt)
    with pytest.raises(ValueError):
        log.reserve("q:later", 20, {})
    assert log.path.read_bytes() == after


@pytest.mark.parametrize("reserved,finished,reason", [(True, False, "provider_usage_unknown"), (True, True, "provider_usage_unknown"), (False, False, "embedding_usage_unknown"), (False, False, "provider_http_error")])
def test_preflight_receipt_cannot_clear_reserved_or_other_failures(api, tmp_path, reserved, finished, reason):
    identity = {"tenant_id": "t", "kb_id": "k", "session_id": "s", "budget": 100}
    log = ledger(api, tmp_path / "q.jsonl", identity=identity, artifact_root=tmp_path)
    log.start("q")
    if reserved:
        log.reserve("q:1", 20, {})
        if finished:
            log.finish("q:1", {"total_tokens": 10})
    log.abort(reason)
    receipt = _write_preflight_receipt(tmp_path, log, identity)
    before = log.path.read_bytes()
    with pytest.raises(ValueError):
        log._reconcile_preflight_not_sent(receipt)
    assert log.path.read_bytes() == before
    assert log.summary()["fatal_reason"] == reason


@pytest.mark.parametrize("target,field,value", [
    ("receipt", "question_id", "foreign"), ("receipt", "original_ledger_sha256", "b" * 64),
    ("receipt", "approved_by", "reviewer"), ("receipt", "approved_by", "REVIEWER"),
    ("receipt", "proof_author", "foreign"), ("receipt", "proof_path", "/private/tmp/proof.json"),
    ("receipt", "proof_path", "../proof.json"), ("receipt", "proof_sha256", "b" * 64),
    ("proof", "question_id", "foreign"), ("proof", "author", "foreign"),
    ("proof", "original_ledger_sha256", "b" * 64), ("proof", "basis", "unknown_manual_clear"),
    ("proof", "report_sha256", "b" * 64), ("proof", "report_path", "../review.md"),
    ("receipt", "tenant_id", "foreign"), ("receipt", "kb_id", "foreign"),
    ("receipt", "session_id", "foreign"), ("receipt", "budget", 99),
    ("receipt", "budget", 100.0),
    ("proof", "tenant_id", "foreign"), ("proof", "kb_id", "foreign"),
    ("proof", "session_id", "foreign"), ("proof", "budget", 99),
    ("proof", "budget", 100.0),
    ("proof", "offline_inner_sends", 1), ("proof", "sdk_max_retries", True),
])
def test_preflight_receipt_rejects_forged_identity_or_evidence(api, tmp_path, target, field, value):
    log, identity = _preflight_ledger(api, tmp_path)
    receipt_name = _write_preflight_receipt(tmp_path, log, identity)
    path = tmp_path / "reconciliation" / (target + ".json")
    data = json.loads(path.read_text())
    if field in identity:
        data["identity"][field] = value
    elif field in {"offline_inner_sends", "sdk_max_retries"}:
        data["facts"][field] = value
    else:
        data[field] = value
    path.write_text(json.dumps(data))
    if target == "proof":
        receipt_path = tmp_path / receipt_name
        receipt = json.loads(receipt_path.read_text())
        receipt["proof_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt))
    before = log.path.read_bytes()
    with pytest.raises(ValueError):
        log._reconcile_preflight_not_sent(receipt_name)
    assert log.path.read_bytes() == before
    assert log.summary()["fatal_reason"] == "provider_usage_unknown"


@pytest.mark.parametrize("tamper", ["prior_ledger", "receipt", "proof", "report", "scope", "missing_context"])
def test_preflight_replay_rechecks_prior_bytes_and_persistent_artifacts(api, tmp_path, tamper):
    log, identity = _preflight_ledger(api, tmp_path)
    receipt = _write_preflight_receipt(tmp_path, log, identity)
    log._reconcile_preflight_not_sent(receipt)
    if tamper == "prior_ledger":
        log.path.write_bytes(log.path.read_bytes().replace(b'{"event": "start"', b'{"event":  "start"', 1))
    elif tamper in {"receipt", "proof", "report"}:
        path = tmp_path / "reconciliation" / ("review.md" if tamper == "report" else tamper + ".json")
        path.write_bytes(path.read_bytes() + b" ")
    elif tamper == "scope":
        identity = dict(identity, tenant_id="foreign")
    kwargs = {} if tamper == "missing_context" else {"identity": identity, "artifact_root": tmp_path}
    with pytest.raises(ValueError):
        ledger(api, log.path, **kwargs)


@pytest.mark.parametrize("tamper", ["missing_question", "question_bytes", "question_identity", "session_scope", "deployment_scope", "claimed_source", "runtime_image", "native_snapshot", "false_limitations"])
def test_preflight_requires_actual_frozen_evidence_and_consistent_sources(api, tmp_path, tamper):
    log, identity = _preflight_ledger(api, tmp_path)
    receipt = _write_preflight_receipt(tmp_path, log, identity)
    evidence = tmp_path / "reconciliation"
    proof_path = evidence / "proof.json"
    proof = json.loads(proof_path.read_text())
    if tamper == "missing_question":
        (evidence / "first-question.json").unlink()
    elif tamper == "question_bytes":
        path = evidence / "first-question.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif tamper in {"question_identity", "session_scope", "deployment_scope", "runtime_image"}:
        filename = {"question_identity": "first-question.json", "session_scope": "first-question-session.json", "deployment_scope": "first-question-deployment.json", "runtime_image": "native-runtime.json"}[tamper]
        path = evidence / filename
        data = json.loads(path.read_text())
        if tamper == "question_identity":
            data["question_id"] = "foreign"
        elif tamper == "session_scope":
            data["tenant_id"] = "foreign"
        elif tamper == "deployment_scope":
            data["scope"]["kb_id"] = "foreign"
        else:
            data["image_id"] = "sha256:" + "e" * 64
        path.write_text(json.dumps(data))
        proof["evidence_sha256"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        if tamper == "deployment_scope":
            proof["source_sha256"]["deployment_manifest"] = proof["evidence_sha256"][filename]
    elif tamper == "claimed_source":
        proof["source_sha256"]["policy_search_service"] = "e" * 64
    elif tamper == "native_snapshot":
        path = evidence / "native-chat-model.py"
        path.write_bytes(path.read_bytes() + b" ")
        proof["evidence_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        proof["limitations"]["provider_usage_receipt"] = True
    proof_path.write_text(json.dumps(proof))
    receipt_path = tmp_path / receipt
    approved = json.loads(receipt_path.read_text())
    approved["proof_sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(approved))
    before = log.path.read_bytes()
    with pytest.raises(ValueError):
        log._reconcile_preflight_not_sent(receipt)
    assert log.path.read_bytes() == before
