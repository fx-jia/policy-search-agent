"""Runtime worker behavior review; real worker classes extracted to avoid server bootstrap."""

import ast
import asyncio
import importlib.util
import logging
import queue
import sys
import threading
import traceback
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import xxhash

from common.constants import ConnectorTaskType, FileSource, TaskStatus

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("collector_test_util", ROOT / "test/unit_test/common/test_shanghai_medical_connector.py")
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)


@pytest.fixture
def runtime(monkeypatch):
    responses, calls = test.http.__wrapped__(monkeypatch)
    responses[test.LIST] = (200, test.listing(), "text/html")
    responses[test.DETAIL] = (200, test.article(), "text/html")
    captures, binds, finishes, uploaded, logs = [], [], [], [], []
    cancel_calls = []
    catalog = SimpleNamespace(covered_sources=lambda scope: {"covered_urls": [], "covered_document_numbers": []})
    service = SimpleNamespace(
        catalog=catalog,
        begin=lambda task, config: ("scope01", 1),
        capture=lambda scope, seq, cap, build: captures.append(cap) or {"revision_id": "revision01", "scope_id": scope, "parse_build_key": build},
        bind=lambda scope, docs, conf, task: binds.append(docs),
        finish=lambda *args, **kwargs: finishes.append((*args, kwargs)),
    )
    service_module = ModuleType("api.db.services.policy_sync_service")
    service_module.PolicySyncService = lambda: service
    monkeypatch.setitem(sys.modules, "api.db.services.policy_sync_service", service_module)
    monkeypatch.setitem(sys.modules, "common.data_source.shanghai_medical_connector", test.collector)
    monkeypatch.setattr(test.collector.ShanghaiMedicalConnector, "cancel", lambda self: (cancel_calls.append(True), self._cancelled.set(), self._session.close()))
    documents = SimpleNamespace(list_doc_headers_by_kb_and_source_type=lambda *args: [])

    def duplicate(kb, docs, *args, **kwargs):
        assert binds and binds[-1] == docs, "expected IDs must be catalogued before upload"
        uploaded.extend(docs)
        return [], [d["id"] for d in docs]

    sync = SimpleNamespace(
        duplicate_and_parse=duplicate,
        increase_docs=lambda *a: logs.append(("increase", a)),
        done=lambda *a: logs.append(("done", a)),
        start=lambda *a: logs.append(("start", a)),
        update_by_id=lambda *a: logs.append(("update", a)),
        schedule=lambda *a, **k: logs.append(("schedule", a, k)),
    )
    namespace = dict(
        asyncio=asyncio,
        threading=threading,
        traceback=traceback,
        datetime=datetime,
        timezone=timezone,
        Any=Any,
        FileSource=FileSource,
        ConnectorTaskType=ConnectorTaskType,
        TaskStatus=TaskStatus,
        logging=logging,
        Callable=Callable,
        Iterator=Iterator,
        queue=queue,
        logger=logging.getLogger("test"),
        _QUEUE_END=object(),
        _QUEUE_POLL_SECONDS=0.02,
        DocumentService=documents,
        SyncLogsService=sync,
        KnowledgebaseService=SimpleNamespace(get_by_id=lambda *a: (True, object())),
        _CursorPersistingSyncBase=type("CursorPlaceholder", (), {}),
        SYNC_BATCH_PAUSE_SECONDS=0,
        task_limiter=asyncio.Semaphore(1),
        hash128=lambda text: xxhash.xxh128(text.encode()).hexdigest(),
    )
    extract(ROOT / "common/data_source/sitemap_connector.py", {"_WorkerThreadIterator", "iter_in_worker_thread"}, namespace)
    extract(ROOT / "api/db/services/connector_service.py", {"connector_doc_id_candidates", "resolve_connector_doc_id"}, namespace)
    extract(ROOT / "rag/svr/sync_data_source.py", {"_iterate_document_batches", "SyncBase", "ShanghaiMedical"}, namespace)
    worker = namespace["ShanghaiMedical"]({"list_urls": [test.LIST], "seed_urls": [], "parse_build_key": "build01"})
    task = {"id": "task01", "connector_id": "connector01", "kb_id": "kb01", "tenant_id": "tenant01", "poll_range_start": None, "auto_parse": True, "timeout_secs": 10}
    return SimpleNamespace(
        worker=worker,
        task=task,
        responses=responses,
        calls=calls,
        captures=captures,
        binds=binds,
        finishes=finishes,
        uploaded=uploaded,
        logs=logs,
        service=service,
        sync=sync,
        namespace=namespace,
        cancels=cancel_calls,
    )


def test_worker_archives_before_binding_before_upload_then_finishes(runtime):
    asyncio.run(runtime.worker(runtime.task))
    assert len(runtime.captures) == len(runtime.uploaded) == 1
    assert runtime.uploaded[0]["metadata"]["revision_id"] == "revision01"
    assert len(runtime.uploaded[0]["id"]) == 32
    assert runtime.finishes[-1][1] is True
    assert any(log[0] == "done" for log in runtime.logs)
    assert runtime.cancels


def test_archive_sink_failure_fails_and_closes_source(runtime):
    def failed(*a):
        raise RuntimeError("object-store archive unavailable")

    runtime.service.capture = failed
    asyncio.run(runtime.worker(runtime.task))
    assert not runtime.uploaded
    assert runtime.finishes[-1][1] is False
    assert any(log[0] == "update" and log[1][1]["status"] == TaskStatus.FAIL for log in runtime.logs)
    assert not any(log[0] == "done" for log in runtime.logs)
    assert runtime.cancels


def test_duplicate_and_parse_returned_errors_never_become_done(runtime):
    runtime.sync.duplicate_and_parse = lambda kb, docs, *a, **k: (["failed attachment"], [])
    asyncio.run(runtime.worker(runtime.task))
    assert runtime.finishes[-1][1] is False
    assert not any(log[0] == "done" for log in runtime.logs)
    assert any(log[0] == "update" and log[1][1]["status"] == TaskStatus.FAIL for log in runtime.logs)


def test_http_failed_attachment_persists_gap_and_schedules_next_observation(runtime):
    runtime.responses[test.DETAIL] = (200, test.article(attachments=[test.ATTACH]), "text/html")
    runtime.responses[test.ATTACH] = (404, b"not found", "text/plain")
    asyncio.run(runtime.worker(runtime.task))
    assert len(runtime.captures) == 1 and runtime.captures[0]["revision_hash"] is None
    assert not runtime.uploaded
    assert runtime.finishes[-1][1] is False
    assert runtime.finishes[-1][-1]["scan_health"]["observation_complete"] is True
    assert runtime.finishes[-1][-1]["retryable"] is True
    assert any(log[0] == "done" for log in runtime.logs)
    assert any(log[0] == "schedule" for log in runtime.logs)
    assert any(log[0] == "update" and log[1][1].get("error_count", 0) > 0 for log in runtime.logs)
    assert runtime.cancels


def test_policy_cap_keeps_partial_coverage_visible_and_normal_schedule(runtime):
    runtime.worker.conf["max_policies"] = 1
    second = test.OLDER
    runtime.responses[test.LIST] = (200, test.listing() + test.listing(second), "text/html")
    runtime.responses[second] = (200, test.article(), "text/html")
    asyncio.run(runtime.worker(runtime.task))
    assert runtime.finishes[-1][1] is False
    assert runtime.finishes[-1][-1]["scan_health"]["observation_complete"] is True
    assert runtime.finishes[-1][-1]["retryable"] is False
    assert any(log[0] == "done" for log in runtime.logs)
    assert any(log[0] == "schedule" for log in runtime.logs)
    assert any(log[0] == "update" and log[1][1].get("error_count", 0) > 0 for log in runtime.logs)


def test_failed_listing_is_fatal_and_never_schedules(runtime):
    runtime.responses[test.LIST] = (503, b"unavailable", "text/plain")
    asyncio.run(runtime.worker(runtime.task))
    assert runtime.finishes[-1][1] is False
    assert not any(log[0] in {"done", "schedule"} for log in runtime.logs)


def test_timeout_while_source_produces_closes_connector(runtime, monkeypatch):
    started = threading.Event()

    def blocking(self):
        started.set()
        while not self._cancelled.wait(0.01):
            pass
        if False:
            yield []

    monkeypatch.setattr(test.collector.ShanghaiMedicalConnector, "load_from_state", blocking)
    runtime.task["timeout_secs"] = 0.1
    asyncio.run(runtime.worker(runtime.task))
    assert started.is_set()
    assert runtime.cancels
    assert runtime.finishes[-1][1] is False
    assert not any(log[0] == "done" for log in runtime.logs)


def test_worker_passes_admitted_urls_and_document_numbers_to_collector(runtime):
    runtime.service.catalog.covered_sources = lambda scope: {"covered_urls": [test.DETAIL], "covered_document_numbers": ["沪医保规〔2026〕1号"]}
    asyncio.run(runtime.worker(runtime.task))
    assert runtime.worker.connector.config.covered_urls == [test.DETAIL]
    assert runtime.worker.connector.config.covered_document_numbers == ["沪医保规〔2026〕1号"]
    assert runtime.finishes[-1][1] is True
