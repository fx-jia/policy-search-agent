"""Offline tests of the real durable budget/scope boundary."""

import importlib.util
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
