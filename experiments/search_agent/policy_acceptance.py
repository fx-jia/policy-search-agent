"""Observe two accelerated native Shanghai rounds; restore the daily interval."""
import argparse
import copy
import hashlib
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import signal
import time
from urllib.parse import urlsplit, urlunsplit

def _approved_scope():
    scope = []
    for name in ("POLICY_APPROVED_TENANT_ID", "POLICY_APPROVED_KB_ID"):
        value = os.environ.get(name, "")
        if not re.fullmatch(r"[0-9a-f]{32}", value):
            raise ValueError(f"{name} must configure an approved 32-character identifier")
        scope.append(value)
    return tuple(scope)


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items() if key.lower() not in {
            "api_key", "apikey", "secret", "password", "authorization", "access_token", "refresh_token", "credentials"}}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        value = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted-key]", value)
        value = re.sub(r"(?i)(?:bearer\s+|api[_-]?key\s*[:=]\s*)\S+", "[redacted-credential]", value)
        if value.startswith(("http://", "https://")):
            parsed = urlsplit(value)
            if parsed.username or parsed.password:
                value = urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))
        return value
    return value


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_safe(value), handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _manifest(state):
    snapshot = state["snapshot"]
    return {"sources": snapshot["sources"], "doc_ids": sorted(snapshot["doc_ids"]), "chunks": {
        cid: {key: chunk.get(key) for key in ("id", "document_id", "sha256", "order", "ordered")}
        for cid, chunk in snapshot["chunks"].items()}}


def _counter_delta(before, after):
    previous = {row["model_id"]: row for row in before}
    result = []
    for row in after:
        old = previous.get(row["model_id"], {}).get("used_tokens")
        now = row.get("used_tokens")
        result.append({"model_id": row["model_id"], "before": old, "after": now,
                       "delta": now - old if type(old) is int and type(now) is int else None})
    return result


def _document_tokens(state):
    rows = state["documents"]
    return sum(row["token_num"] for row in rows) if all(type(row.get("token_num")) is int for row in rows) else None


def _native_timestamp(value, state):
    point = datetime.fromisoformat(str(value))
    if point.tzinfo is None:
        point = point.replace(tzinfo=timezone(timedelta(seconds=state["native_utc_offset_seconds"])))
    return point.timestamp()


def _creation_timestamp(task, state):
    if type(task.get("create_time")) in (int, float):
        return task["create_time"] / 1000
    return _native_timestamp(task["create_date"], state)


def _validate_original_schedule(task, state):
    queued_date = _native_timestamp(task["update_date"], state)
    created = _creation_timestamp(task, state)
    if type(task.get("update_time")) in (int, float):
        queued_epoch = task["update_time"] / 1000
        if not 0 <= queued_epoch - queued_date < 1:
            raise RuntimeError("original SCHEDULE timestamp fields disagree")
        queued_end = queued_epoch + 0.001
    else:
        queued_end = queued_date + 1
    if queued_end <= created:
        raise RuntimeError("original SCHEDULE was backdated before task creation")


def _finished_health(state, task):
    coverage = state["scope_health"].get("coverage_manifest") or {}
    scan = coverage.get("scan_health") or {}
    try:
        start = _native_timestamp(task["time_started"], state)
        done = _native_timestamp(task["update_date"], state)
        created = _creation_timestamp(task, state)
        # The worker writes DONE before awaiting scan-health persistence, then
        # creates the next SCHEDULE. This row is the actual completion fence.
        newer = [row for row in state["tasks"] if row["id"] != task["id"]
                 and _creation_timestamp(row, state) >= created]
        if any(row["status"] in {"1", "2", "3", "4"} for row in newer):
            return False, False
        schedules = [row for row in newer if row["status"] == "5" and _creation_timestamp(row, state) >= done]
        if not schedules:
            return False, False
        following = min(schedules, key=lambda row: _creation_timestamp(row, state))
        end = _creation_timestamp(following, state)
        end += 0.001 if type(following.get("create_time")) in (int, float) else 1
        # Native epoch fields truncate milliseconds; date-only fallback truncates seconds.
        def current(value):
            return value is not None and start <= _native_timestamp(value, state) < end
        observed = scan.get("observation_complete") is True and current(coverage.get("last_observation_success_at"))
        complete = observed and scan.get("scan_complete") is True and scan.get("discovery_complete") is True and current(coverage.get("last_discovery_success_at")) and current(state["scope_health"].get("last_scan_success_at"))
        return observed, complete
    except (ValueError, TypeError, KeyError):
        return False, False


def _source_targets(state):
    """Freeze sealed current heads, including unresolved bindings (never ready)."""
    revisions = {row["id"]: row for row in state["revision_manifest"]}
    targets = {}
    for key, head in state["snapshot"]["source_heads"].items():
        if head["revision_id"] is None:
            observation = revisions.get(head.get("observation_id"))
            if observation is None or observation["policy_key"] != key or observation.get("revision_hash") is not None:
                raise ValueError("unsealed source head lacks matching observation evidence")
            continue
        revision = revisions.get(head["revision_id"])
        if revision is None:
            raise ValueError("current source head lacks revision evidence")
        if not revision.get("revision_hash"):
            raise ValueError("sealed source head has no sealed revision hash")
        if revision["policy_key"] != key:
            raise ValueError("current source head revision ownership mismatch")
        build = head["parse_build_key"]
        targets[key] = {"revision_id": revision["id"], "revision_hash": revision["revision_hash"],
                        "parse_build_key": build, "assets": copy.deepcopy(revision["assets"]),
                        "bindings": copy.deepcopy(revision["parsing_builds"].get(build, []))}
    return targets


def _targets_published(state, targets, approved_scope):
    """Match frozen admission identities against a later immutable release."""
    snapshot = state["snapshot"]
    documents = {row["id"]: row for row in state["documents"]}
    seen = set()
    for key, target in targets.items():
        assets, bindings = target["assets"], target["bindings"]
        expected = {(asset["role"], asset["sha256"]) for asset in assets}
        if not expected or len(expected) != len(assets) or len(bindings) != len(assets):
            return False
        if {(binding["role"], binding["sha256"]) for binding in bindings} != expected:
            return False
        for binding in bindings:
            did = binding.get("doc_id")
            if not did or did in seen or not binding.get("fingerprint"):
                return False
            seen.add(did)
            build_config = binding.get("parse_build_config")
            if not isinstance(build_config, dict):
                return False
            digest = hashlib.sha256(json.dumps(build_config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            if digest != target["parse_build_key"]:
                return False
            wanted = {"policy_key": key, "revision_id": target["revision_id"], "revision_hash": target["revision_hash"],
                      "parse_build_key": target["parse_build_key"], "asset_role": binding["role"],
                      "asset_sha256": binding["sha256"], "fingerprint": binding["fingerprint"],
                      "embedding_config": build_config.get("embedding_config"),
                      "tenant_id": approved_scope[0], "kb_id": approved_scope[1],
                      "embedding_config_sha256": hashlib.sha256(json.dumps(build_config.get("embedding_config"), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()}
            source = snapshot["sources"].get(did, {})
            if (did not in snapshot["doc_ids"] or any(source.get(field) != value for field, value in wanted.items())
                    or type(source.get("embedding_dimension")) is not int or source["embedding_dimension"] < 1):
                return False
            doc = documents.get(did, {})
            chunks = [(cid, chunk) for cid, chunk in snapshot["chunks"].items() if chunk["document_id"] == did]
            if (str(doc.get("run")) != "3" or doc.get("progress", 0) < 1 or not chunks
                    or doc.get("content_hash") != binding["fingerprint"] or doc.get("chunk_num") != len(chunks)
                    or any(chunk.get("id") != cid or not chunk.get("sha256") for cid, chunk in chunks)):
                return False
    return True


def _completion(state, task, complete):
    return {"task": copy.deepcopy(task), "strict_scan_complete": complete,
            "targets": _source_targets(state), "state": copy.deepcopy(state)}


def _publication_ready(state):
    documents = {row["id"]: row for row in state["documents"]}
    return not any(row.get("revision_hash") for row in state["snapshot"]["pending"]) and all(
        did in documents and str(documents[did]["run"]) == "3" and documents[did]["progress"] >= 1
        and documents[did]["chunk_num"] > 0 for did in state["snapshot"]["doc_ids"])


def _finalization_ready(state, completed, baseline_ids, approved_scope):
    """Use the same success gate at the barrier and actual restoration readback."""
    tasks = state["tasks"]
    for row in tasks:
        if row["status"] == "5":
            _validate_original_schedule(row, state)
        if row["id"] not in baseline_ids and row["status"] in {"2", "4"}:
            raise RuntimeError("automatic native round failed: " + str(row["id"]))
    published_ids = set(state["snapshot"]["doc_ids"])
    if any(row["id"] in published_ids and (str(row["run"]) == "4" or row["progress"] < 0) for row in state["documents"]):
        raise RuntimeError("native document parsing failed")
    if state.get("source_fence_consistent") is not True:
        return False
    current_tasks = {row["id"]: row for row in tasks}
    if any(current_tasks.get(item["task"]["id"]) != item["task"] for item in completed):
        raise RuntimeError("native completion task changed during finalization")
    return (state["connector"]["refresh_freq"] == 1440
            and not any(row["status"] == "1" for row in tasks)
            and any(row["status"] == "5" for row in tasks)
            and _publication_ready(state)
            and all(_targets_published(state, item["targets"], approved_scope) for item in completed))


def run_acceptance(adapter, tenant_id, kb_id, output, *, rounds=2, refresh_minutes=5,
                   timeout_seconds=10800, initial_timeout_seconds=1800, poll_seconds=20, clock=time.monotonic, sleep=time.sleep, emit=print):
    """Only the adapter interval setter mutates native state; no task scheduling."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "scheduler-acceptance-summary.json").exists():
        raise ValueError("acceptance output already has a completed attempt; use another output directory")
    result = {"status": "failed", "mode": "accelerated_acceptance", "partial_coverage": True,
              "tenant_id": tenant_id, "kb_id": kb_id, "requested_rounds": rounds,
              "refresh_minutes": refresh_minutes, "normal_refresh_minutes": 1440,
              "completed_round_ids": [], "rounds": [], "observer_windows": [], "error": "", "restoration": {"verified": False},
              "usage_note": "document.token_num records successful ingestion tokens; excludes probes/retries/failed calls. Legacy model counters may not update. Missing counters are unknown, not zero; these are not provider call ledgers."}
    start, validated, changed = clock(), False, False
    queued, baseline_ids, previous = {}, set(), None
    completions, initial_completion = {}, None
    finalizing = False
    try:
        approved_scope = _approved_scope()
        if (tenant_id, kb_id) != approved_scope or rounds != 2 or refresh_minutes != 5 or not 0 < poll_seconds <= 30:
            raise PermissionError("only the approved dedicated Shanghai two-round five-minute acceptance is allowed")
        identity = adapter.validate()
        if (identity.get("tenant_id") != tenant_id or identity.get("kb_id") != kb_id
                or identity.get("source") != "shanghai_medical" or identity.get("exclusive") is not True):
            raise PermissionError("dedicated connector ownership/source/exclusive scope mismatch")
        validated = True
        result["identity"] = identity
        while True:
            if clock() - start >= timeout_seconds:
                raise TimeoutError("native acceptance timeout")
            if previous is None and clock() - start >= initial_timeout_seconds:
                raise TimeoutError("initial publication timeout")
            state = adapter.read()
            tasks = state["tasks"]
            for row in tasks:
                if row["status"] == "5":
                    _validate_original_schedule(row, state)
                    queued.setdefault(row["id"], copy.deepcopy(row))
            sources = state["snapshot"]["sources"]
            published = len({row["policy_key"] for row in sources.values()})
            current_ids = set(state["snapshot"]["doc_ids"])
            if any(row["id"] in current_ids and (str(row["run"]) == "4" or row["progress"] < 0) for row in state["documents"]):
                raise RuntimeError("native document parsing failed")
            emit(json.dumps({"elapsed_seconds": round(clock() - start), "published_policies": published,
                             "pending": len(state["snapshot"]["pending"]), "rounds_done": len(result["rounds"]),
                             "tasks": [{key: row.get(key) for key in ("id", "status", "error_count", "new_docs_indexed")} for row in tasks[-3:]]}, ensure_ascii=False))
            active = [row for row in tasks if row["id"] not in baseline_ids]
            if previous is None:
                initial_sources = [row for row in tasks if row["status"] != "5"]
                if initial_sources and initial_sources[-1]["status"] in {"2", "4"}:
                    raise RuntimeError("initial native scan failed")
                done = [row for row in tasks if row["status"] == "3"]
                scheduled = [row for row in tasks if row["status"] == "5"]
                observed, complete = _finished_health(state, done[-1]) if done else (False, False)
                if (initial_completion is None and done and observed and state.get("source_fence_consistent") is True):
                    initial_completion = _completion(state, done[-1], complete)
                    _write(output / "scheduler-initial-source-completion.json", initial_completion)
                if initial_completion and state.get("source_fence_consistent") is True:
                    current_initial = next((row for row in tasks if row["id"] == initial_completion["task"]["id"]), None)
                    if current_initial != initial_completion["task"]:
                        raise RuntimeError("native initial completion task changed during publication wait")
                if (initial_completion and scheduled and published >= 20 and state.get("source_fence_consistent") is True
                        and _publication_ready(state) and _targets_published(state, initial_completion["targets"], approved_scope)):
                    if state["connector"]["refresh_freq"] != 1440:
                        raise ValueError("initial connector interval is not the expected daily 1440")
                    baseline_ids = {row["id"] for row in tasks if row["status"] != "5"}
                    previous = copy.deepcopy(state)
                    result["initial_task_id"] = initial_completion["task"]["id"]
                    complete = initial_completion["strict_scan_complete"]
                    result["strict_scan_complete"] = complete
                    result["scan_health"] = "complete" if complete else "partial"
                    _write(output / "scheduler-round-00.json", state)
                    _write(output / "scheduler-before-interval.json", {"identity": identity, "connector": state["connector"], "queued_tasks": queued})
                    changed = True
                    adapter.set_refresh(5)
                    if adapter.read()["connector"]["refresh_freq"] != 5:
                        raise RuntimeError("five-minute interval update did not read back")
                else:
                    sleep(poll_seconds)
                    continue
            else:
                failed = next((row for row in active if row["status"] in {"2", "4"}), None)
                if failed is not None:
                    raise RuntimeError("automatic native round failed: " + str(failed["id"]))
                for row in active:
                    if len(result["rounds"]) == rounds:
                        break
                    if row["status"] != "3" or row["id"] in result["completed_round_ids"]:
                        continue
                    if state.get("source_fence_consistent") is not True:
                        continue
                    if row["id"] not in completions:
                        observed, complete = _finished_health(state, row)
                        if not observed:
                            continue
                        original = queued.get(row["id"])
                        if original is None:
                            raise RuntimeError("automatic round lacks an observed original SCHEDULE record")
                        if original.get("create_date") != row.get("create_date") or original.get("create_time") != row.get("create_time"):
                            raise RuntimeError("native task creation timestamp changed")
                        elapsed = _native_timestamp(row["time_started"], state) - _native_timestamp(original["update_date"], state)
                        if elapsed < refresh_minutes * 60:
                            raise RuntimeError("automatic round started before the original native interval")
                        completion = _completion(state, row, complete)
                        completion.update(queued_task=copy.deepcopy(original), queued_to_started_seconds=elapsed)
                        completions[row["id"]] = completion
                        _write(output / f'scheduler-source-completion-{row["id"]}.json', completion)
                    completion = completions[row["id"]]
                    if row != completion["task"]:
                        raise RuntimeError("native completion task changed during publication wait")
                    if any(identifier not in result["completed_round_ids"] for identifier in list(completions)[:list(completions).index(row["id"])]):
                        continue
                    if not _publication_ready(state) or not _targets_published(state, completion["targets"], approved_scope):
                        continue
                    original, elapsed = completion["queued_task"], completion["queued_to_started_seconds"]
                    complete = completion["strict_scan_complete"]
                    before_tokens, after_tokens = _document_tokens(previous), _document_tokens(state)
                    window = {"before_task_ids": [item["id"] for item in previous["tasks"]],
                              "after_task_ids": [item["id"] for item in state["tasks"]],
                              "before_release_id": previous["snapshot"]["release_id"], "after_release_id": state["snapshot"]["release_id"],
                              "scope": "observer interval; may overlap later scans; not attributed to a source round",
                              "new_parser_task_ids": sorted(set(state["parser_task_ids"]) - set(previous["parser_task_ids"])),
                              "model_counter_delta": _counter_delta(previous["model_counters"], state["model_counters"]),
                              "recorded_document_token_delta": after_tokens - before_tokens if before_tokens is not None and after_tokens is not None else None}
                    result["observer_windows"].append(window)
                    capture = {"round": len(result["rounds"]) + 1, "task": completion["task"], "queued_task": original,
                               "source_completion": completion, "observer_window": len(result["observer_windows"]) - 1,
                               "metrics_scope": window["scope"],
                               "queued_to_started_seconds": elapsed, "state": state,
                               "manifest_unchanged": _manifest(previous) == _manifest(state),
                               "new_parser_task_ids": None, "model_counter_delta": None, "recorded_document_token_delta": None}
                    result["rounds"].append(capture)
                    result["strict_scan_complete"] = result["strict_scan_complete"] and complete
                    result["scan_health"] = "complete" if result["strict_scan_complete"] else "partial"
                    result["completed_round_ids"].append(row["id"])
                    _write(output / f'scheduler-round-{capture["round"]:02}.json', capture)
                    previous = copy.deepcopy(state)
                    if len(result["rounds"]) == rounds:
                        break
                if len(result["rounds"]) == rounds:
                    if not finalizing:
                        adapter.set_refresh(1440)
                        changed = False
                        finalizing = True
                        # Re-read after restoration; the same overall deadline
                        # still bounds a later source's native completion.
                        continue
                    if state["connector"]["refresh_freq"] != 1440:
                        raise RuntimeError("daily interval restoration did not read back")
                    completed = [completions[identifier] for identifier in result["completed_round_ids"]]
                    if _finalization_ready(state, completed, baseline_ids, approved_scope):
                        result["finalization"] = {"verified": True, "state": state,
                                                  "completed_round_ids": list(result["completed_round_ids"])}
                        _write(output / "scheduler-finalization.json", result["finalization"])
                        result["status"] = "passed"
                        break
            sleep(poll_seconds)
    except BaseException as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
    finally:
        if validated:
            try:
                if changed or adapter.read()["connector"]["refresh_freq"] != 1440:
                    adapter.set_refresh(1440)
                restored = adapter.read()
                result["restoration"] = {"verified": restored["connector"]["refresh_freq"] == 1440,
                                         "refresh_minutes": restored["connector"]["refresh_freq"],
                                         "next_schedule_ids": [row["id"] for row in restored["tasks"] if row["status"] == "5"]}
                _write(output / "scheduler-restoration.json", restored)
                if not result["restoration"]["verified"]:
                    raise RuntimeError("daily interval restoration did not read back")
                if result["status"] == "passed":
                    completed = [completions[identifier] for identifier in result["completed_round_ids"]]
                    if not _finalization_ready(restored, completed, baseline_ids, approved_scope):
                        raise RuntimeError("actual restoration readback no longer satisfies finalization readiness")
                    result["restoration"]["publication_verified"] = True
            except BaseException as exc:
                result["status"] = "failed"
                result["restoration"]["error"] = type(exc).__name__ + ": " + str(exc)
        result["elapsed_seconds"] = round(clock() - start)
        _write(output / "scheduler-acceptance-summary.json", result)
    return _safe(result)


class _NativeState:
    def __init__(self, tenant_id, kb_id):
        from common import settings
        settings.init_settings()
        from api.db.db_models import DB, Connector, Connector2Kb, Document, Knowledgebase, SyncLogs, Task, TenantLLM
        from api.db.services.policy_catalog_service import PolicyCatalogService
        from api.db.services.policy_publication_service import PolicyPublicationService
        self.db, self.Connector, self.Link, self.Document, self.KB, self.SyncLogs, self.Task, self.TenantLLM = DB, Connector, Connector2Kb, Document, Knowledgebase, SyncLogs, Task, TenantLLM
        self.tenant_id, self.kb_id = tenant_id, kb_id
        self.catalog = PolicyCatalogService()
        self.publisher = PolicyPublicationService(self.catalog)
        self.scope_id = self.connector_id = None
        self.models = []

    def validate(self):
        from api.db.joint_services.tenant_model_service import resolve_model_config, get_model_config_by_id
        with self.db.connection_context():
            kb = self.KB.get_or_none(self.KB.id == self.kb_id)
            scopes = list(self.catalog.Scope.select().where((self.catalog.Scope.tenant_id == self.tenant_id) & (self.catalog.Scope.kb_id == self.kb_id)))
            if kb is None or kb.tenant_id != self.tenant_id or len(scopes) != 1:
                raise PermissionError("dedicated knowledge base/scope ownership mismatch")
            self.scope_id, self.connector_id = scopes[0].id, scopes[0].connector_id
            connector_scopes = list(self.catalog.Scope.select().where(self.catalog.Scope.connector_id == self.connector_id))
            if len(connector_scopes) != 1 or connector_scopes[0].id != self.scope_id:
                raise PermissionError("policy connector has multiple catalog scopes")
            connector = self.Connector.get_or_none(self.Connector.id == self.connector_id)
            links = list(self.Link.select().where((self.Link.kb_id == self.kb_id) | (self.Link.connector_id == self.connector_id)))
            if (connector is None or connector.tenant_id != self.tenant_id or connector.source != "shanghai_medical"
                    or len(links) != 1 or links[0].kb_id != self.kb_id or links[0].connector_id != self.connector_id or links[0].auto_parse != "1"):
                raise PermissionError("Shanghai connector is foreign, shared, or not linked for parsing")
            chat_ref = (kb.parser_config or {}).get("llm_id")
            if not chat_ref or not kb.tenant_embd_id:
                raise ValueError("dedicated chat and embedding model references are required")
            chat = resolve_model_config(self.tenant_id, "chat", chat_ref)
            embedding = get_model_config_by_id(self.tenant_id, "embedding", kb.tenant_embd_id)
            self.models = [{"model_id": ref, "role": role, "llm_name": config["llm_name"], "llm_factory": config["llm_factory"]}
                           for ref, role, config in ((chat_ref, "chat", chat), (kb.tenant_embd_id, "embedding", embedding))]
            return {"tenant_id": self.tenant_id, "kb_id": self.kb_id, "scope_id": self.scope_id,
                    "connector_id": self.connector_id, "source": connector.source, "exclusive": True}

    def set_refresh(self, minutes):
        from api.db.services.connector_service import ConnectorService
        if minutes not in {5, 1440}:
            raise ValueError("unapproved refresh interval")
        self.validate()
        with self.db.connection_context():
            ConnectorService.update_by_id(self.connector_id, {"refresh_freq": minutes})

    def read(self):
        with self.db.connection_context():
            connector = self.Connector.get_by_id(self.connector_id)
            tasks = list(self.SyncLogs.select().where((self.SyncLogs.connector_id == self.connector_id)
                         & (self.SyncLogs.kb_id == self.kb_id) & (self.SyncLogs.task_type == "sync")).order_by(self.SyncLogs.create_time).dicts())
            scope_before = self.catalog.get_scope(self.scope_id)
            snapshot = self.publisher.acquire_snapshot(self.tenant_id, self.kb_id)
            # Acceptance verifies ingestion of every sealed asset, including
            # future/expired policies filtered from the Search business view.
            if snapshot["release_id"]:
                release = self.catalog.Release.get_by_id(snapshot["release_id"])
                if release.scope_id != self.scope_id:
                    raise ValueError("acceptance release scope mismatch")
                snapshot["sources"] = copy.deepcopy(release.document_manifest)
                snapshot["doc_ids"] = sorted(snapshot["sources"])
                snapshot["chunks"] = self.publisher._manifest(scope_before, release)
            snapshot.pop("as_of", None)
            snapshot["view"] = "complete_published_release"
            doc_fields = [getattr(self.Document, key) for key in ("id", "run", "progress", "chunk_num", "token_num", "content_hash", "create_date", "update_date")]
            docs = list(self.Document.select(*doc_fields).where(self.Document.kb_id == self.kb_id).dicts())
            doc_ids = [row["id"] for row in docs]
            parse_ids = [row.id for row in self.Task.select(self.Task.id).where(self.Task.doc_id.in_(doc_ids))] if doc_ids else []
            counters = []
            for model in self.models:
                rows = list(self.TenantLLM.select(self.TenantLLM.used_tokens).where((self.TenantLLM.tenant_id == self.tenant_id)
                            & (self.TenantLLM.llm_name == model["llm_name"]) & (self.TenantLLM.llm_factory == model["llm_factory"])))
                counters.append({**model, "used_tokens": rows[0].used_tokens if len(rows) == 1 else None,
                                 "counter_source": "legacy tenant_llm.used_tokens; not an actual provider usage ledger"})
            revisions = list(self.catalog.Revision.select().where(self.catalog.Revision.scope_id == self.scope_id))
            revision_manifest = [{key: row.__data__.get(key) for key in ("id", "policy_key", "revision_hash", "state", "observed_at", "raw_object_keys", "assets", "parsing_builds", "temporal_metadata", "relations")} for row in revisions]
            health = self.catalog.status(self.scope_id)
            scope_after = self.catalog.get_scope(self.scope_id)
            tasks_after = list(self.SyncLogs.select().where((self.SyncLogs.connector_id == self.connector_id)
                               & (self.SyncLogs.kb_id == self.kb_id) & (self.SyncLogs.task_type == "sync")).order_by(self.SyncLogs.create_time).dicts())
            fence_fields = ("source_heads", "scan_sequence", "coverage_manifest", "last_scan_success_at", "active_release_id")
            consistent = (tasks == tasks_after and all(scope_before.get(key) == scope_after.get(key) for key in fence_fields)
                          and snapshot.get("source_heads", {}) == scope_before["source_heads"]
                          and snapshot["release_id"] == scope_before["active_release_id"]
                          and all(health.get(key) == scope_before.get(key) for key in fence_fields[1:]))
            snapshot["chunks"] = _manifest({"snapshot": snapshot})["chunks"]
            return {"connector": {"id": connector.id, "refresh_freq": connector.refresh_freq, "status": connector.status},
                    "native_utc_offset_seconds": int(datetime.now().astimezone().utcoffset().total_seconds()),
                    "source_fence_consistent": consistent, "tasks": tasks, "snapshot": snapshot, "documents": docs, "parser_task_ids": sorted(parse_ids),
                    "model_counters": counters, "revision_manifest": revision_manifest,
                    "scope_health": {key: health.get(key) for key in ("last_scan_success_at", "coverage_manifest", "scan_sequence", "active_release_id", "last_error", "pending_count", "covered_policy_count", "published_policy_count", "stale")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, choices=[2], default=2)
    parser.add_argument("--refresh-minutes", type=int, choices=[5], default=5)
    parser.add_argument("--timeout-minutes", type=int, default=180)
    parser.add_argument("--initial-timeout-minutes", type=int, default=30)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        approved_scope = _approved_scope()
    except ValueError as exc:
        parser.error(str(exc))
    if (args.tenant_id, args.kb_id) != approved_scope:
        parser.error("this script only operates the approved dedicated Shanghai knowledge base")
    if args.timeout_minutes <= 0 or args.initial_timeout_minutes <= 0 or not 0 < args.poll_seconds <= 30:
        parser.error("timeout must be positive and polling must be between 1 and 30 seconds")
    def stop(signum, frame):
        raise KeyboardInterrupt("acceptance interrupted")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def deadline(signum, frame):
        raise TimeoutError("native acceptance wall-clock timeout")
    signal.signal(signal.SIGALRM, deadline)
    adapter = _NativeState(args.tenant_id, args.kb_id)
    signal.alarm(args.timeout_minutes * 60)
    try:
        result = run_acceptance(adapter, args.tenant_id, args.kb_id, args.output,
                                rounds=args.rounds, refresh_minutes=args.refresh_minutes, timeout_seconds=args.timeout_minutes * 60,
                                initial_timeout_seconds=args.initial_timeout_minutes * 60,
                                poll_seconds=args.poll_seconds, emit=lambda value: print(value, flush=True))
    finally:
        signal.alarm(0)
    print(json.dumps({key: result[key] for key in ("status", "completed_round_ids", "restoration", "error")}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
