"""Single-process HTTP owner of one explicitly authorized policy session."""

import asyncio
import copy
import fcntl
import inspect
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID


def _append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _text(value):
    return value if isinstance(value, str) else None


def _url(value):
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value)
        return value if parts.scheme in {"http", "https"} and parts.hostname and not parts.username and not parts.password else None
    except ValueError:
        return None


def _source(row):
    coverage = row.get("coverage")
    if isinstance(coverage, dict):
        coverage = {
            key: value
            for key, value in coverage.items()
            if key in {"complete", "foundational_complete"} and isinstance(value, bool) or key in {"city", "last_discovery_success_at", "last_recheck_success_at"} and isinstance(value, str)
        }
        if isinstance(row.get("coverage", {}).get("sources"), list):
            coverage["sources"] = [url for item in row["coverage"]["sources"] if (url := _url(item))]
    else:
        coverage = None
    return {
        "release_id": _text(row.get("release_id")),
        "as_of": _text(row.get("as_of")),
        "scanned_at": _text(row.get("scanned_at")),
        "applicability": _text(row.get("applicability")),
        "stale": row.get("stale") if isinstance(row.get("stale"), bool) else None,
        "coverage": coverage,
        "pending_count": len(row["pending"]) if isinstance(row.get("pending"), list) else None,
    }


def _turn(row, request_id):
    result = row.get("result") or {}
    chunks = (result.get("reference") or {}).get("chunks") or []
    citations = []
    for index, chunk in enumerate(chunks):
        metadata = chunk.get("document_metadata") or {}
        temporal = metadata.get("temporal_metadata") or metadata.get("temporal") or {}
        citations.append(
            {
                "index": index,
                "chunk_id": _text(chunk.get("chunk_id")),
                "document_id": _text(chunk.get("doc_id")),
                "title": _text(chunk.get("docnm_kwd")) or _text(metadata.get("title")),
                "text": _text(chunk.get("content_with_weight")) or "",
                "source_url": _url(chunk.get("url")) or _url(metadata.get("source_url")),
                "document_number": _text(metadata.get("document_number")),
                "published_at": _text(metadata.get("published_at")),
                "effective_from": _text(metadata.get("effective_from")) or _text(temporal.get("effective_from")),
                "effective_to": _text(metadata.get("effective_to")) or _text(temporal.get("effective_to")),
                "confirmation_status": _text(metadata.get("confirmation_status")) or _text(temporal.get("status")),
                "revision_id": _text(metadata.get("revision_id")),
            }
        )
    trace = []
    for item in result.get("trace") or []:
        event = item.get("event")
        if event not in {"resolution", "initial_search", "decision", "tool", "fallback", "controller_error", "stop"}:
            continue
        value = item.get("value") or {}
        action = item.get("action") or value.get("action") or {}
        if not isinstance(action, dict):
            action = {}
        needs = value.get("needs")
        trace.append(
            {
                "event": event,
                "query": _text(item.get("query")) or _text(action.get("query")),
                "needs": [str(need.get("text", "")) + "：" + str(need.get("status", "")) for need in needs if isinstance(need, dict)] if isinstance(needs, list) else None,
                "action": _text(action.get("type")),
                "chunk_ids": [v for v in item["chunk_ids"] if isinstance(v, str)] if isinstance(item.get("chunk_ids"), list) else None,
                "new_chunk_ids": [v for v in item["new_chunk_ids"] if isinstance(v, str)] if isinstance(item.get("new_chunk_ids"), list) else None,
            }
        )
    status = result.get("status")
    if status not in {"ok", "needs_clarification", "timeout", "technical_error", "generation_error", "cancelled"}:
        status = "technical_error"
    return {
        "request_id": request_id,
        "question_id": _text(row.get("question_id")),
        "question": _text(result.get("question")) or "",
        "effective_question": _text(result.get("effective_question")) or "",
        "status": status,
        "answer": (_text(result.get("answer")) or "") if status in {"ok", "needs_clarification"} else "查询未能完成，请查看服务状态。",
        "citations": citations,
        "trace": trace,
        "stop_reason": _text(result.get("stop_reason")),
        "latency_ms": result.get("latency_ms") if type(result.get("latency_ms")) is int else None,
        "fallback_used": result.get("fallback_used") is True,
        "source_state": _source(row),
    }


class MedicalAssistantService:
    """Keep commands durable and generation alive independently of HTTP callers."""

    def __init__(self, policy, *, owner_user_id, access_check=None):
        self.policy = policy
        self.owner_user_id = owner_user_id
        self._access_check = access_check
        self._lock = asyncio.Lock()
        self._task = None
        self._current = None
        self._cached = None
        self._commands = {}
        self._journal = policy.output / "web_commands.jsonl"
        self._owner_file = (policy.output / "web_owner.lock").open("a+")
        try:
            fcntl.flock(self._owner_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity_path = policy.output / "web_owner.json"
            identity = {"owner_user_id": owner_user_id, "session_id": policy.session_id}
            if identity_path.exists():
                if json.loads(identity_path.read_text()) != identity:
                    raise ValueError("session_owner_mismatch")
            else:
                with identity_path.open("x") as handle:
                    json.dump(identity, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
            if self._journal.exists():
                for line in self._journal.read_text().splitlines():
                    row = json.loads(line)
                    rid = row["request_id"]
                    if row["event"] == "accepted":
                        if rid in self._commands:
                            raise ValueError("command_journal_invalid")
                        self._commands[rid] = row
                    elif row["event"] == "completed" and rid in self._commands:
                        self._commands[rid]["result"] = row["result"]
                    else:
                        raise ValueError("command_journal_invalid")
            self._unconfirmed = next((rid for rid, command in self._commands.items() if "result" not in command), None)
        except BaseException:
            self._owner_file.close()
            raise ValueError("session_owner_unavailable") from None

    async def authorize(self, user_id):
        if user_id != self.owner_user_id:
            raise PermissionError("assistant_owner_required")
        if self._access_check is None:
            from api.db.services.knowledgebase_service import KnowledgebaseService

            allowed = await asyncio.to_thread(KnowledgebaseService.accessible, self.policy.kb_id, user_id)
        else:
            allowed = self._access_check(user_id, self.policy.kb_id)
            if inspect.isawaitable(allowed):
                allowed = await allowed
        if not allowed:
            raise PermissionError("assistant_scope_forbidden")
        await self.policy.authorize()

    @staticmethod
    def _validate_id(request_id):
        if not isinstance(request_id, str):
            raise ValueError("invalid_request_id")
        try:
            if str(UUID(request_id)) != request_id:
                raise ValueError()
        except ValueError:
            raise ValueError("invalid_request_id") from None

    def _existing(self, request_id, kind, question=None):
        command = self._commands.get(request_id)
        if command is None:
            return None
        if command["kind"] != kind or command.get("question") != question:
            raise ValueError("request_id_body_mismatch")
        if "result" in command:
            return copy.deepcopy(command["result"])
        if kind == "clear":
            return self._snapshot()
        return {"state": "running" if self._current == request_id else "unconfirmed", "request_id": request_id}

    async def _refresh(self):
        self._cached = await self.policy.read_state()

    def _snapshot(self, *, running=True):
        cached = self._cached or {}
        turns = []
        for row in cached.get("records", []):
            rid = next((rid for rid, cmd in self._commands.items() if cmd.get("result", {}).get("turn", {}).get("question_id") == row.get("question_id")), None)
            turns.append(_turn(row, rid))
        reason = cached.get("blocked_reason") or None
        state = "blocked" if reason else "ready"
        if cached.get("source_ready") is False and not reason:
            state, reason = "unavailable", "source_unavailable"
        if self._unconfirmed:
            state, reason = "unconfirmed", "command_unconfirmed"
        if self._current and running:
            state, reason = "running", None
        scope = {"city": "上海", "label": "上海医保政策", **_source(cached.get("source_state") or (cached.get("records") or [{}])[-1])}
        return {
            "session_id": self.policy.session_id,
            "can_submit": state == "ready",
            "state": state,
            "blocked_reason": reason,
            "current_request_id": self._current or self._unconfirmed,
            "turns": turns,
            "scope": scope,
        }

    async def snapshot(self):
        if self._current:
            return copy.deepcopy(self._snapshot())
        async with self._lock:
            if not self._current:
                await self._refresh()
            return copy.deepcopy(self._snapshot())

    async def _execute(self, request_id, kind, question):
        try:
            if kind == "turn":
                row = await self.policy.ask(question)
                result = {"state": "completed", "turn": _turn(row, request_id)}
            else:
                await self.policy.clear()
                await self._refresh()
                result = self._snapshot(running=False)
                result["current_request_id"] = self._unconfirmed
            _append(self._journal, {"event": "completed", "request_id": request_id, "result": result})
            self._commands[request_id]["result"] = result
            if kind == "turn":
                await self._refresh()
            return copy.deepcopy(result)
        except BaseException:
            self._unconfirmed = request_id
            raise ValueError("command_unconfirmed") from None
        finally:
            self._current = None

    async def _command(self, request_id, kind, question=None):
        self._validate_id(request_id)
        async with self._lock:
            if self._cached is None and not self._current:
                await self._refresh()
            existing = self._existing(request_id, kind, question)
            if existing is not None:
                return existing
            if self._current:
                raise ValueError("assistant_running")
            await self._refresh()
            if kind == "turn" and not self._snapshot()["can_submit"]:
                raise ValueError(self._snapshot()["blocked_reason"])
            command = {"event": "accepted", "request_id": request_id, "kind": kind, "question": question}
            _append(self._journal, command)
            self._commands[request_id] = command
            self._current = request_id
            task = asyncio.create_task(self._execute(request_id, kind, question))
            self._task = task
            # Retrieve background exceptions even after a caller disconnects.
            self._task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        return await asyncio.shield(task)

    async def submit(self, request_id, question):
        if not isinstance(question, str) or not question.strip() or len(question) > 10000:
            raise ValueError("invalid_question")
        return await self._command(request_id, "turn", question)

    async def clear(self, request_id):
        return await self._command(request_id, "clear")


_host = None
_host_lock = asyncio.Lock()


async def get_medical_assistant_service():
    """Load a deployment binding once; never manufacture a replacement session."""
    global _host
    if _host is not None:
        return _host
    async with _host_lock:
        if _host is not None:
            return _host
        config_path = os.environ.get("MEDICAL_ASSISTANT_CONFIG")
        if not config_path:
            return None
        path = Path(config_path)
        if not path.is_absolute():
            raise ValueError("invalid_assistant_config")
        config = json.loads(path.read_text())
        required = {"owner_user_id", "tenant_id", "kb_id", "session_id", "output_dir", "budget"}
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError("invalid_assistant_config")
        if any(not isinstance(config[key], str) or not config[key].strip() for key in required - {"budget"}):
            raise ValueError("invalid_assistant_config")
        if type(config["budget"]) is not int or not 0 < config["budget"] <= 60000:
            raise ValueError("invalid_assistant_config")
        output = Path(config["output_dir"])
        identity = {key: config[key] for key in ("tenant_id", "kb_id", "session_id", "budget")}
        if not output.is_absolute() or not (output / "session.json").is_file() or json.loads((output / "session.json").read_text()) != identity:
            raise ValueError("authorized_session_required")
        from api.db.services.policy_search_service import PolicySearchService

        policy = PolicySearchService(**identity, output_dir=output)
        _host = MedicalAssistantService(policy, owner_user_id=config["owner_user_id"])
        return _host


def unavailable_session():
    return {
        "session_id": None,
        "can_submit": False,
        "state": "unavailable",
        "blocked_reason": "service_unavailable",
        "current_request_id": None,
        "turns": [],
        "scope": {"city": "上海", "label": "上海医保政策", **_source({})},
    }
