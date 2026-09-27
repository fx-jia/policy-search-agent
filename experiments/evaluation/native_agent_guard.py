"""Durable, fail-closed accounting for an isolated native Agent experiment."""

import json
import os
from pathlib import Path
from threading import RLock


def validate_scope(requested, allowed):
    if not isinstance(requested, (list, tuple)) or not requested or any(not isinstance(x, str) or x not in allowed for x in requested):
        raise ValueError("invalid_or_empty_document_scope")
    return list(requested)


class CallLedger:
    def __init__(self, path, *, run_limit, question_limit, call_limit):
        self.path = Path(path)
        self.limits = (run_limit, question_limit, call_limit)
        self.lock = RLock()
        self.question_id = None
        self.started = set()
        self.calls = {}
        self.fatal_reason = None
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                self._apply(json.loads(line))

    def _apply(self, row):
        event = row["event"]
        if event == "start":
            self.question_id = row["question_id"]
            self.started.add(self.question_id)
            self.fatal_reason = None
        elif event == "reserve":
            self.calls[row["call_id"]] = dict(row)
        elif event == "finish":
            self.calls[row["call_id"]]["usage"] = row["usage"]
            if row.get("transport_status") == "not_sent":
                self.calls[row["call_id"]]["transport_status"] = "not_sent"
                self.calls[row["call_id"]]["transport_error"] = row["transport_error"]
        elif event == "abort":
            self.fatal_reason = row["reason"]
        else:
            raise ValueError("unknown_ledger_event")

    def _append(self, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._apply(row)

    def start(self, question_id):
        with self.lock:
            if not question_id or question_id in self.started:
                raise ValueError("duplicate_or_empty_question")
            if any("usage" not in row for row in self.calls.values()):
                raise ValueError("unresolved_provider_usage")
            self._append({"event": "start", "question_id": question_id})

    def abort(self, reason):
        with self.lock:
            if self.fatal_reason is None:
                self._append({"event": "abort", "question_id": self.question_id, "reason": reason})

    def assert_open(self):
        if not self.question_id:
            raise ValueError("no_active_question")
        if self.fatal_reason:
            raise ValueError(self.fatal_reason)

    def reserve(self, call_id, reserve_tokens, request):
        with self.lock:
            self.assert_open()
            if call_id in self.calls or not call_id:
                raise ValueError("duplicate_call_id")
            if not isinstance(reserve_tokens, int) or reserve_tokens <= 0:
                raise ValueError("invalid_reservation")
            current = [r for r in self.calls.values() if r["question_id"] == self.question_id]
            def committed(rows):
                return sum(r["usage"]["total_tokens"] if "usage" in r else r["reserve_tokens"] for r in rows)
            run_limit, question_limit, call_limit = self.limits
            reason = None
            if len(current) >= call_limit:
                reason = "question_call_limit"
            elif committed(self.calls.values()) + reserve_tokens > run_limit:
                reason = "run_token_budget"
            elif committed(current) + reserve_tokens > question_limit:
                reason = "question_token_budget"
            if reason:
                self.abort(reason)
                raise ValueError(reason)
            self._append({"event": "reserve", "question_id": self.question_id, "call_id": call_id, "reserve_tokens": reserve_tokens, "request": request})

    def finish(self, call_id, usage):
        with self.lock:
            row = self.calls.get(call_id)
            if not row or "usage" in row:
                raise ValueError("unknown_or_duplicate_completion")
            if not isinstance(usage, dict) or not isinstance(usage.get("total_tokens"), int) or usage["total_tokens"] <= 0:
                self.abort("provider_usage_unknown")
                raise ValueError("provider_usage_unknown")
            self._append({"event": "finish", "call_id": call_id, "usage": usage})
            if usage["total_tokens"] > row["reserve_tokens"]:
                self.abort("usage_exceeded_reservation")
                raise ValueError("usage_exceeded_reservation")

    def finish_not_sent(self, call_id):
        """Release only an attempt whose physical send raised ConnectError."""
        with self.lock:
            row = self.calls.get(call_id)
            if not row or "usage" in row:
                raise ValueError("unknown_or_duplicate_completion")
            self._append({
                "event": "finish", "call_id": call_id,
                "transport_status": "not_sent", "transport_error": "ConnectError",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

    def summary(self):
        with self.lock:
            return {
                "provider_calls": len(self.calls),
                "provider_not_sent_calls": sum(r.get("transport_status") == "not_sent" for r in self.calls.values()),
                "provider_total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in self.calls.values()),
                "pending_calls": [cid for cid, row in self.calls.items() if "usage" not in row],
                "fatal_reason": self.fatal_reason,
            }
