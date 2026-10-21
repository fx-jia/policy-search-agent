"""Offline checks for dedicated scheduler acceptance and guaranteed restoration."""
import copy
from contextlib import nullcontext
import hashlib
import io
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

TENANT = "1" * 32
KB = "2" * 32


def task(identifier, status, minute, started=None):
    point = datetime(2026, 10, 10, 11) + timedelta(minutes=minute)
    return {"id": identifier, "status": status, "task_type": "sync", "create_date": point.isoformat(),
            "update_date": point.isoformat(), "time_started": started, "error_count": 0,
            "new_docs_indexed": 0, "total_docs_indexed": 20}


EMBEDDING = {"llm_name": "fixture", "llm_factory": "fixture"}
BUILD_CONFIG = {"embedding_config": EMBEDDING}
BUILD = hashlib.sha256(json.dumps(BUILD_CONFIG, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def state(tasks):
    sources = {f"d{i}": {"policy_key": f"p{i}", "revision_id": f"r{i}", "revision_hash": "h",
                           "tenant_id": TENANT, "kb_id": KB, "source_url": "https://ybj.sh.gov.cn/policy", "embedding_config_sha256": hashlib.sha256(json.dumps(EMBEDDING, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(), "embedding_dimension": 2, "asset_role": "body", "asset_sha256": f"asset{i}", "fingerprint": f"fingerprint{i}",
                           "parse_build_key": BUILD, "embedding_config": EMBEDDING} for i in range(20)}
    return {"connector": {"refresh_freq": 1440, "status": "5"}, "tasks": tasks,
            "native_utc_offset_seconds": 0, "source_fence_consistent": True,
            "revision_manifest": [{"id": f"r{i}", "policy_key": f"p{i}", "revision_hash": "h", "assets": [{"role": "body", "sha256": f"asset{i}"}],
                "parsing_builds": {BUILD: [{"doc_id": f"d{i}", "role": "body", "sha256": f"asset{i}", "fingerprint": f"fingerprint{i}",
                    "parse_build_config": BUILD_CONFIG, "embedding_model": EMBEDDING}]}} for i in range(20)],
            "scope_health": {"pending_count": 0, "coverage_manifest": {
                "scan_health": {"observation_complete": True, "discovery_complete": False},
                "last_observation_success_at": next((row["update_date"] + "Z" for row in reversed(tasks) if row["status"] == "3"), None)}},
            "snapshot": {"source_heads": {f"p{i}": {"revision_id": f"r{i}", "parse_build_key": BUILD, "capture_sequence": 1} for i in range(20)}, "release_id": "release", "doc_ids": list(sources), "sources": sources,
            "chunks": {f"c{i}": {"id": f"c{i}", "document_id": f"d{i}", "sha256": "chunk", "order": None, "ordered": False} for i in range(20)},
            "pending": [], "coverage": {"foundational_complete": False}, "scanned_at": "2026-10-10T00:00:00Z"},
            "documents": [{"id": f"d{i}", "run": "3", "progress": 1, "chunk_num": 1, "token_num": 100, "content_hash": f"fingerprint{i}"} for i in range(20)],
            "parser_task_ids": [f"parse{i}" for i in range(20)], "model_counters": [{"model_id": "dedicated-embedding", "used_tokens": None}]}


class Adapter:
    def __init__(self, frames):
        self.frames, self.index, self.changes, self.now = frames, 0, [], 0
        self.frequency = 1440
        self.validations = 0
        self.identity = {"tenant_id": TENANT, "kb_id": KB, "connector_id": "connector", "source": "shanghai_medical", "exclusive": True}

    def validate(self):
        self.validations += 1
        return self.identity

    def read(self):
        value = copy.deepcopy(self.frames[self.index])
        value["connector"]["refresh_freq"] = self.frequency
        return value

    def set_refresh(self, minutes):
        self.changes.append(minutes)
        self.frequency = minutes

    def sleep(self, seconds):
        self.now += seconds
        self.index = min(self.index + 1, len(self.frames) - 1)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        approval = patch.dict(os.environ, {"POLICY_APPROVED_TENANT_ID": TENANT, "POLICY_APPROVED_KB_ID": KB})
        approval.start()
        self.addCleanup(approval.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.attempt = 0

    def run_acceptance(self, adapter, **kwargs):
        from experiments.search_agent.policy_acceptance import run_acceptance
        output = Path(self.directory.name) if self.attempt == 0 else Path(self.directory.name) / str(self.attempt)
        self.attempt += 1
        return run_acceptance(adapter, kwargs.get("tenant_id", TENANT), kwargs.get("kb_id", KB), output,
                              rounds=kwargs.get("rounds", 2), refresh_minutes=kwargs.get("refresh_minutes", 5),
                              timeout_seconds=kwargs.get("timeout_seconds", 200), poll_seconds=kwargs.get("poll_seconds", 10),
                              clock=lambda: adapter.now, sleep=adapter.sleep, emit=lambda value: None)

    def frames(self):
        initial = task("initial", "3", -1, "2026-10-10T10:58:00")
        first = task("first", "5", 0)
        running1 = task("first", "1", 5, "2026-10-10T11:05:30")
        done1 = task("first", "3", 6, "2026-10-10T11:05:30")
        second = task("second", "5", 6)
        running2 = task("second", "1", 11, "2026-10-10T11:11:30")
        done2 = task("second", "3", 12, "2026-10-10T11:11:30")
        frames = [state([initial, first]), state([initial, running1]), state([initial, done1, second]),
                  state([initial, done1, running2]), state([initial, done1, done2, task("third", "5", 12)])]
        created = {}
        for frame in frames:
            for row in frame["tasks"]:
                row["create_date"] = created.setdefault(row["id"], row["create_date"])
        return frames

    def test_two_original_auto_rounds_restore_and_keep_schedule_timestamps(self):
        adapter = Adapter(self.frames())
        result = self.run_acceptance(adapter)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(adapter.changes, [5, 1440])
        self.assertEqual(result["completed_round_ids"], ["first", "second"])
        self.assertTrue(result["restoration"]["verified"])
        self.assertEqual(result["rounds"][0]["queued_task"]["update_date"], "2026-10-10T11:00:00")
        self.assertTrue(result["rounds"][0]["manifest_unchanged"])
        self.assertIsNone(result["rounds"][0]["model_counter_delta"])
        self.assertEqual(result["observer_windows"][0]["recorded_document_token_delta"], 0)
        self.assertTrue((Path(self.directory.name) / "scheduler-round-02.json").exists())

    def test_missing_or_invalid_approval_never_reaches_native_state(self):
        for name in ("POLICY_APPROVED_TENANT_ID", "POLICY_APPROVED_KB_ID"):
            for value in (None, "", "replace-with-private-id"):
                with self.subTest(name=name, value=value), patch.dict(os.environ):
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
                    adapter = Adapter(self.frames())
                    result = self.run_acceptance(adapter)
                    self.assertEqual(result["status"], "failed")
                    self.assertIn(name, result["error"])
                    self.assertEqual(adapter.validations, 0)
                    self.assertEqual(adapter.changes, [])

    def test_requested_scope_cannot_authorize_itself(self):
        for field in ("tenant_id", "kb_id"):
            with self.subTest(field=field):
                adapter = Adapter(self.frames())
                result = self.run_acceptance(adapter, **{field: "3" * 32})
                self.assertEqual(result["status"], "failed")
                self.assertEqual(adapter.validations, 0)
                self.assertEqual(adapter.changes, [])

    def test_approved_scope_still_requires_bounded_rounds_and_interval(self):
        for changed in ({"rounds": 3}, {"refresh_minutes": 1}, {"poll_seconds": 0}, {"poll_seconds": 31}):
            with self.subTest(changed=changed):
                adapter = Adapter(self.frames())
                result = self.run_acceptance(adapter, **changed)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(adapter.validations, 0)
                self.assertEqual(adapter.changes, [])

    def test_cli_rejects_unapproved_scope_before_constructing_native_adapter(self):
        from experiments.search_agent import policy_acceptance
        for missing, tenant in ((True, TENANT), (False, "3" * 32)):
            with self.subTest(missing=missing), patch.dict(os.environ), patch("sys.stderr", io.StringIO()):
                if missing:
                    os.environ.pop("POLICY_APPROVED_TENANT_ID", None)
                with patch.object(policy_acceptance, "_NativeState", side_effect=AssertionError("native access forbidden")):
                    with self.assertRaises(SystemExit) as failure:
                        policy_acceptance.main(["--tenant-id", tenant, "--kb-id", KB, "--output", self.directory.name])
                    self.assertEqual(failure.exception.code, 2)

    def test_native_failure_stops_and_restores(self):
        frames = self.frames()[:2]
        frames[-1]["tasks"][-1].update(status="4", error_count=1, error_msg="attachment failure")
        adapter = Adapter(frames)
        result = self.run_acceptance(adapter)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(adapter.changes, [5, 1440])
        self.assertIn("failed", result["error"])

    def test_timeout_restores_without_creating_a_task(self):
        adapter = Adapter(self.frames()[:1])
        result = self.run_acceptance(adapter, timeout_seconds=20)
        self.assertEqual(result["status"], "failed")
        self.assertIn("timeout", result["error"])
        self.assertEqual(adapter.changes, [5, 1440])

    def test_foreign_or_shared_scope_never_mutates(self):
        for key, value in (("tenant_id", "foreign"), ("source", "other"), ("exclusive", False)):
            adapter = Adapter(self.frames())
            adapter.identity[key] = value
            result = self.run_acceptance(adapter)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(adapter.changes, [])

    def test_early_manual_task_cannot_count_as_scheduler_round(self):
        frames = self.frames()
        frames[1]["tasks"][-1]["time_started"] = "2026-10-10T11:01:00"
        frames[2]["tasks"][1]["time_started"] = "2026-10-10T11:01:00"
        adapter = Adapter(frames)
        result = self.run_acceptance(adapter)
        self.assertEqual(result["status"], "failed")
        self.assertIn("interval", result["error"])
        self.assertEqual(adapter.changes, [5, 1440])

    def test_partial_done_with_review_backlog_and_historical_failures_is_allowed(self):
        frames = self.frames()
        for frame in frames:
            frame["tasks"].insert(0, task("old-failed", "4", -20))
            for row in frame["tasks"]:
                if row["status"] == "3":
                    row.update(error_count=8, error_msg="bounded discovery / needs review")
            frame["snapshot"]["pending"] = [{"revision_hash": None, "pending_reasons": ["required attachment unavailable"]}]
            frame["documents"].append({"id": "superseded", "run": "4", "progress": -1, "token_num": 0})
        result = self.run_acceptance(Adapter(frames))
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["strict_scan_complete"])
        self.assertEqual(result["rounds"][0]["task"]["error_count"], 8)
        self.assertEqual(result["rounds"][0]["state"]["snapshot"]["pending"][0]["revision_hash"], None)

    def test_sealed_pending_waits_for_publication_after_done(self):
        frames = self.frames()
        delayed = copy.deepcopy(frames[2])
        delayed["snapshot"]["pending"] = [{"revision_hash": "sealed", "parse_build_key": "current", "parsing_builds": {"current": [{"doc_id": "new-doc"}]}}]
        frames.insert(2, delayed)
        adapter = Adapter(frames)
        result = self.run_acceptance(adapter)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["rounds"][0]["state"]["snapshot"]["pending"], [])
        self.assertGreaterEqual(result["elapsed_seconds"], 50)

    def test_published_failed_document_still_fails_immediately(self):
        frames = self.frames()
        frames[0]["documents"][0].update(run="4", progress=-1)
        adapter = Adapter(frames)
        result = self.run_acceptance(adapter)
        self.assertEqual(result["status"], "failed")
        self.assertIn("parsing failed", result["error"])
        self.assertEqual(adapter.changes, [])

    def delayed_frames(self, initial=False):
        frames = self.frames()
        index = 0 if initial else 2
        delayed = copy.deepcopy(frames[index])
        delayed["snapshot"]["doc_ids"].remove("d0")
        del delayed["snapshot"]["sources"]["d0"]
        del delayed["snapshot"]["chunks"]["c0"]
        delayed["snapshot"]["pending"] = [{"revision_id": "r0", "revision_hash": "h", "parse_build_key": BUILD, "parsing_builds": delayed["revision_manifest"][0]["parsing_builds"]}]
        delayed["documents"][0].update(run="4", progress=-1, chunk_num=0)
        if initial:
            frames.insert(0, delayed)
        else:
            frames[2] = delayed
        return frames

    def test_initial_unpublished_failure_waits_for_native_recovery(self):
        adapter = Adapter(self.delayed_frames(initial=True))
        result = self.run_acceptance(adapter)
        self.assertEqual(result["status"], "passed", result["error"])
        self.assertEqual(result["completed_round_ids"], ["first", "second"])
        self.assertEqual(adapter.changes, [5, 1440])

    def test_done_fence_survives_parse_failure_until_next_source_running(self):
        adapter = Adapter(self.delayed_frames())
        result = self.run_acceptance(adapter)
        self.assertEqual(result["status"], "passed", result["error"])
        first = result["rounds"][0]
        self.assertEqual(first["source_completion"]["state"]["tasks"][-1]["status"], "5")
        self.assertEqual(first["state"]["tasks"][-1]["status"], "1")
        self.assertEqual(first["source_completion"]["targets"]["p0"]["revision_id"], "r0")
        self.assertIn("not attributed", first["metrics_scope"])
        self.assertIsNone(first["model_counter_delta"])
        self.assertEqual(result["observer_windows"][0]["model_counter_delta"][0]["delta"], None)
        self.assertTrue((Path(self.directory.name) / "scheduler-source-completion-first.json").exists())

    def test_unpublished_failure_exhausted_or_unknown_waits_to_deadline(self):
        for retries in (0, 3, None):
            frames = self.delayed_frames(initial=True)[:1]
            frames[0]["snapshot"]["pending"][0].update(retry_count=retries, errors=["permanent or unknown"])
            adapter = Adapter(frames)
            result = self.run_acceptance(adapter, timeout_seconds=30)
            self.assertIn("timeout", result["error"])
            self.assertEqual(adapter.now, 30)
            self.assertEqual(adapter.changes, [])

    def test_later_revision_build_asset_or_fingerprint_cannot_replace_frozen_target(self):
        for field in ("revision_id", "revision_hash", "parse_build_key", "asset_role", "asset_sha256", "fingerprint", "embedding_config", "embedding_config_sha256", "tenant_id", "kb_id"):
            with self.subTest(field=field):
                frames = self.delayed_frames()
                for frame in frames[3:]:
                    frame["snapshot"]["sources"]["d0"][field] = "replacement"
                result = self.run_acceptance(Adapter(frames), timeout_seconds=70)
                self.assertEqual(result["status"], "failed")
                self.assertNotIn("first", result["completed_round_ids"])
                self.assertIn("timeout", result["error"])

    def test_frozen_target_requires_complete_manifest_and_native_done(self):
        for mutation in ("missing_chunk", "wrong_doc", "wrong_count", "not_done"):
            frames = self.delayed_frames()
            for frame in frames[3:]:
                if mutation == "missing_chunk":
                    del frame["snapshot"]["chunks"]["c0"]
                elif mutation == "wrong_doc":
                    frame["snapshot"]["chunks"]["c0"]["document_id"] = "foreign"
                elif mutation == "wrong_count":
                    frame["documents"][0]["chunk_num"] = 2
                else:
                    frame["documents"][0].update(run="1", progress=0.1)
            result = self.run_acceptance(Adapter(frames), timeout_seconds=70)
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("first", result["completed_round_ids"])

    def test_inconsistent_source_read_cannot_freeze_or_borrow_later_health(self):
        frames = self.delayed_frames()
        frames[2]["source_fence_consistent"] = False
        result = self.run_acceptance(Adapter(frames), timeout_seconds=70)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("first", result["completed_round_ids"])
        self.assertFalse((Path(self.directory.name) / "scheduler-source-completion-first.json").exists())

    def test_final_count_waits_for_a_consistent_publication_read(self):
        frames = self.frames()
        inconsistent = copy.deepcopy(frames[-1])
        inconsistent["source_fence_consistent"] = False
        frames.insert(-1, inconsistent)
        result = self.run_acceptance(Adapter(frames))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["elapsed_seconds"], 50)

    def test_new_pending_target_cannot_be_hidden_by_an_older_completion(self):
        frames = self.delayed_frames()
        for frame in frames[3:]:
            frame["snapshot"]["pending"] = [{"revision_hash": "later-sealed"}]
        result = self.run_acceptance(Adapter(frames), timeout_seconds=70)
        self.assertEqual(result["completed_round_ids"], [])
        self.assertIn("timeout", result["error"])

    def test_source_fail_or_cancel_during_publication_wait_still_fails(self):
        for status in ("2", "4"):
            frames = self.delayed_frames()[:4]
            frames[-1]["tasks"][-1]["status"] = status
            result = self.run_acceptance(Adapter(frames))
            self.assertIn("automatic native round failed", result["error"])
            self.assertLess(result["elapsed_seconds"], 200)

    def test_native_reader_detects_tasks_heads_and_health_changing_mid_read(self):
        from experiments.search_agent.policy_acceptance import _NativeState

        class Field:
            def __init__(self, name):
                self.name = name

            def __eq__(self, other):
                return self

            def __and__(self, other):
                return self

            def in_(self, other):
                return self

        class Query:
            def __init__(self, rows, fields=()):
                self.rows, self.fields = rows, fields

            def where(self, *args):
                return self

            def order_by(self, *args):
                return self

            def dicts(self):
                return [{field.name: row.get(field.name) for field in self.fields} for row in self.rows] if self.fields else self.rows

            def __iter__(self):
                return iter(self.rows)

        def model(rows, fields):
            value = SimpleNamespace(**{key: Field(key) for key in fields})
            value.select = lambda *args: Query(rows, args)
            return value

        frame = self.frames()[2]
        frame["tasks"][1].update(from_beginning="0", docs_removed_from_index=0, full_exception_trace="old trace", poll_range_start="start", poll_range_end="end")
        base_scope = dict(frame["scope_health"], source_heads=frame["snapshot"]["source_heads"], scan_sequence=2, active_release_id="release")
        for changed in (None, "tasks", "source_heads", "scan_sequence", "coverage_manifest", "active_release_id", "expired", "future", "from_beginning", "docs_removed_from_index", "full_exception_trace", "poll_range_start", "poll_range_end"):
            with self.subTest(changed=changed):
                reader = _NativeState.__new__(_NativeState)
                reader.db = SimpleNamespace(connection_context=nullcontext)
                reader.scope_id, reader.connector_id, reader.tenant_id, reader.kb_id = "scope", "connector", TENANT, KB
                reader.models = []
                reader.Connector = SimpleNamespace(get_by_id=lambda _: SimpleNamespace(id="connector", refresh_freq=5, status="5"))
                task_fields = ("id", "status", "task_type", "create_date", "create_time", "update_date", "update_time", "time_started",
                               "error_count", "error_msg", "new_docs_indexed", "total_docs_indexed", "connector_id", "kb_id", "from_beginning", "docs_removed_from_index", "full_exception_trace", "poll_range_start", "poll_range_end")
                reader.SyncLogs = model([], task_fields)
                second_tasks = copy.deepcopy(frame["tasks"])
                if changed == "tasks":
                    second_tasks[-1]["status"] = "1"
                elif changed in {"from_beginning", "docs_removed_from_index", "full_exception_trace", "poll_range_start", "poll_range_end"}:
                    second_tasks[1][changed] = "changed"
                reads = iter([copy.deepcopy(frame["tasks"]), second_tasks])
                reader.SyncLogs.select = lambda *args: Query(next(reads), args)
                reader.Document = model(frame["documents"], ("id", "run", "progress", "chunk_num", "token_num", "content_hash", "create_date", "update_date", "kb_id"))
                reader.Task = model([], ("id", "doc_id"))
                business_snapshot = copy.deepcopy(frame["snapshot"])
                full_sources = copy.deepcopy(frame["snapshot"]["sources"])
                if changed in {"expired", "future"}:
                    full_sources["d0"]["effective_from" if changed == "future" else "effective_to"] = "2099-01-01" if changed == "future" else "2000-01-01"
                    full_sources["d0"]["confirmation_status"] = "confirmed"
                    business_snapshot["doc_ids"].remove("d0")
                    del business_snapshot["sources"]["d0"]
                    del business_snapshot["chunks"]["c0"]
                reader.publisher = SimpleNamespace(acquire_snapshot=lambda *args: copy.deepcopy(business_snapshot),
                                                   _manifest=lambda scope, release: copy.deepcopy(frame["snapshot"]["chunks"]))
                second_scope = copy.deepcopy(base_scope)
                if changed in {"source_heads", "coverage_manifest"}:
                    second_scope[changed]["changed"] = True
                elif changed == "scan_sequence":
                    second_scope[changed] += 1
                elif changed == "active_release_id":
                    second_scope[changed] = "later-release"
                scopes = iter([copy.deepcopy(base_scope), second_scope])
                reader.catalog = SimpleNamespace(
                    get_scope=lambda _: next(scopes), status=lambda _: copy.deepcopy(base_scope),
                    Revision=model([SimpleNamespace(__data__=row) for row in frame["revision_manifest"]], ("scope_id",)),
                    Release=SimpleNamespace(get_by_id=lambda _: SimpleNamespace(scope_id="scope", document_manifest=full_sources)))
                actual = reader.read()
                self.assertEqual(actual["source_fence_consistent"], changed in (None, "expired", "future"))
                if changed in {"expired", "future"}:
                    from experiments.search_agent.policy_acceptance import _source_targets, _targets_published
                    self.assertIn("d0", actual["snapshot"]["sources"])
                    self.assertTrue(_targets_published(actual, _source_targets(actual), (TENANT, KB)))
                    self.assertNotIn("d0", business_snapshot["sources"])
                self.assertEqual(actual["snapshot"]["source_heads"], frame["snapshot"]["source_heads"])
                self.assertEqual(actual["tasks"], frame["tasks"])

    def test_hash_null_observation_heads_are_explicit_partial_backlog(self):
        frames = self.frames()
        for frame in frames:
            frame["snapshot"]["source_heads"]["deferred-policy"] = {
                "revision_id": None, "observation_id": "deferred-observation", "parse_build_key": BUILD}
            frame["revision_manifest"].append({"id": "deferred-observation", "policy_key": "deferred-policy", "revision_hash": None})
        result = self.run_acceptance(Adapter(frames))
        self.assertEqual(result["status"], "passed", result["error"])
        self.assertNotIn("deferred-policy", result["rounds"][0]["source_completion"]["targets"])
        self.assertIn("deferred-policy", result["rounds"][0]["source_completion"]["state"]["snapshot"]["source_heads"])

    def test_null_revision_head_requires_its_matching_unsealed_observation(self):
        for corruption in ("missing_observation", "wrong_policy", "sealed_observation"):
            frames = self.frames()
            frame = frames[0]
            frame["snapshot"]["source_heads"]["deferred-policy"] = {
                "revision_id": None, "observation_id": "deferred-observation", "parse_build_key": BUILD}
            if corruption != "missing_observation":
                frame["revision_manifest"].append({"id": "deferred-observation", "policy_key": "other" if corruption == "wrong_policy" else "deferred-policy",
                                                   "revision_hash": "sealed" if corruption == "sealed_observation" else None})
            adapter = Adapter(frames)
            result = self.run_acceptance(adapter)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(adapter.changes, [])

    def test_simultaneous_ready_completions_count_exactly_two(self):
        frames = self.delayed_frames()
        for frame in frames[3:]:
            frame["snapshot"]["pending"] = [{"revision_hash": "h"}]
        third = copy.deepcopy(frames[-1])
        third["snapshot"]["pending"] = []
        third["tasks"][-1] = task("third", "3", 18, "2026-10-10T11:17:30")
        third["tasks"][-1]["create_date"] = frames[-1]["tasks"][-1]["create_date"]
        third["tasks"].append(task("fourth", "5", 18))
        third["scope_health"]["coverage_manifest"]["last_observation_success_at"] = "2026-10-10T11:18:00Z"
        frames.append(third)
        result = self.run_acceptance(Adapter(frames))
        self.assertEqual(result["status"], "passed", result["error"])
        self.assertEqual(result["completed_round_ids"], ["first", "second"])
        self.assertEqual(len(result["rounds"]), 2)

    def test_done_task_row_cannot_change_while_waiting_for_publication(self):
        for field, value in (("create_date", "2026-10-10T11:01:00"), ("time_started", "2026-10-10T11:06:00"), ("error_count", 12), ("full_exception_trace", "changed trace"), ("from_beginning", "1"), ("docs_removed_from_index", 3), ("poll_range_start", "changed"), ("poll_range_end", "changed")):
            frames = self.delayed_frames()
            for frame in frames[3:]:
                frame["tasks"][1][field] = value
            result = self.run_acceptance(Adapter(frames))
            self.assertEqual(result["status"], "failed")
            self.assertIn("completion task changed", result["error"])

    def test_nonnull_revision_head_cannot_silently_skip_unsealed_revision(self):
        frames = self.frames()
        frames[0]["revision_manifest"][0]["revision_hash"] = None
        adapter = Adapter(frames)
        result = self.run_acceptance(adapter)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(adapter.changes, [])

    def test_initial_frozen_done_row_must_remain_unchanged(self):
        frames = self.delayed_frames(initial=True)
        for frame in frames[1:]:
            frame["tasks"][0]["error_count"] = 9
        result = self.run_acceptance(Adapter(frames))
        self.assertEqual(result["status"], "failed")
        self.assertIn("completion task changed", result["error"])

    def test_only_proven_unsealed_heads_preserve_ready_release_partial_acceptance(self):
        frames = self.frames()
        for frame in frames:
            for key in frame["snapshot"]["source_heads"]:
                observation = "unsealed-" + key
                frame["snapshot"]["source_heads"][key] = {"revision_id": None, "observation_id": observation, "parse_build_key": BUILD}
                frame["revision_manifest"].append({"id": observation, "policy_key": key, "revision_hash": None})
            frame["snapshot"]["pending"] = [{"revision_hash": None, "pending_reasons": ["needs_review"]}]
        result = self.run_acceptance(Adapter(frames))
        self.assertEqual(result["status"], "passed", result["error"])
        self.assertEqual(result["completed_round_ids"], ["first", "second"])
        self.assertEqual(result["rounds"][0]["source_completion"]["targets"], {})
        self.assertEqual(result["scan_health"], "partial")

    def finalization_frames(self):
        frames = self.frames()
        for frame in frames[2:]:
            frame["snapshot"]["doc_ids"].remove("d0")
            del frame["snapshot"]["sources"]["d0"]
            del frame["snapshot"]["chunks"]["c0"]
            frame["snapshot"]["pending"] = [{"revision_id": "r0", "revision_hash": "h", "parse_build_key": BUILD}]
            frame["documents"][0].update(run="1", progress=0.1, chunk_num=0)
        running = state(copy.deepcopy(frames[-1]["tasks"]))
        running["tasks"][-1].update(status="1", time_started="2026-10-10T11:17:30")
        finished = copy.deepcopy(running)
        finished["tasks"][-1].update(status="3", update_date="2026-10-10T11:18:00")
        finished["tasks"].append(task("fourth", "5", 18))
        return [*frames, running, finished]

    def test_two_counted_rounds_restore_then_wait_for_running_source_finalization(self):
        adapter = Adapter(self.finalization_frames())
        changes = []
        setter = adapter.set_refresh
        def tracked_setter(minutes):
            changes.append((adapter.index, minutes))
            setter(minutes)
        adapter.set_refresh = tracked_setter
        result = self.run_acceptance(adapter)
        self.assertEqual(result["status"], "passed", result["error"])
        self.assertEqual(result["completed_round_ids"], ["first", "second"])
        self.assertEqual(changes, [(0, 5), (5, 1440)])
        self.assertEqual(result["elapsed_seconds"], 60)
        self.assertEqual(result["restoration"]["next_schedule_ids"], ["fourth"])
        self.assertTrue(result["finalization"]["verified"])
        self.assertFalse((Path(self.directory.name) / "scheduler-source-completion-third.json").exists())

    def test_finalization_uses_remaining_deadline_without_reset(self):
        adapter = Adapter(self.finalization_frames()[:-1])
        result = self.run_acceptance(adapter, timeout_seconds=70)
        self.assertEqual(result["status"], "failed")
        self.assertIn("timeout", result["error"])
        self.assertEqual(result["elapsed_seconds"], 70)
        self.assertEqual(result["completed_round_ids"], ["first", "second"])
        self.assertEqual(adapter.changes, [5, 1440])

    def test_finalization_interrupt_does_not_wait_or_pass(self):
        adapter = Adapter(self.finalization_frames())
        ordinary_sleep = adapter.sleep
        def interrupting_sleep(seconds):
            if adapter.index == 5:
                raise KeyboardInterrupt("fixture SIGTERM")
            ordinary_sleep(seconds)
        adapter.sleep = interrupting_sleep
        result = self.run_acceptance(adapter)
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["completed_round_ids"], ["first", "second"])
        self.assertEqual(adapter.changes, [5, 1440])
        self.assertEqual(adapter.index, 5)

    def test_finalization_rejects_changed_counted_task_and_later_source_failure(self):
        for corruption in ("task_change", "source_fail", "source_cancel", "published_failure"):
            frames = self.finalization_frames()
            if corruption == "task_change":
                frames[-1]["tasks"][1]["full_exception_trace"] = "changed after counting"
            elif corruption == "published_failure":
                frames[-1]["documents"][0].update(run="4", progress=-1)
            else:
                frames[-1]["tasks"][-2]["status"] = "4" if corruption == "source_fail" else "2"
            result = self.run_acceptance(Adapter(frames))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["completed_round_ids"], ["first", "second"])
            expected = "completion task changed" if corruption == "task_change" else "parsing failed" if corruption == "published_failure" else "automatic native round failed"
            self.assertIn(expected, result["error"])

    def test_finalization_waits_for_consistency_and_all_frozen_target_identities(self):
        for corruption in ("inconsistent", "target_changed", "new_pending"):
            frames = self.finalization_frames()
            if corruption == "inconsistent":
                frames[-1]["source_fence_consistent"] = False
            elif corruption == "target_changed":
                frames[-1]["snapshot"]["sources"]["d0"]["asset_sha256"] = "changed"
            else:
                frames[-1]["snapshot"]["pending"] = [{"revision_hash": "later-sealed"}]
            result = self.run_acceptance(Adapter(frames), timeout_seconds=80)
            self.assertEqual(result["status"], "failed")
            self.assertIn("timeout", result["error"])
            self.assertEqual(result["elapsed_seconds"], 80)

    def test_actual_restoration_readback_rechecks_full_success_gate(self):
        for corruption in ("running", "task", "target", "inconsistent", "published_failure", "source_fail", "source_cancel", "schedule_backdate"):
            with self.subTest(corruption=corruption):
                class ChangedReadback(Adapter):
                    final_reads = 0

                    def read(self):
                        value = super().read()
                        if self.index != len(self.frames) - 1:
                            return value
                        self.final_reads += 1
                        if self.final_reads <= 1:
                            return value
                        if corruption == "running":
                            value["tasks"][-2]["status"] = "1"
                        elif corruption == "task":
                            value["tasks"][1]["full_exception_trace"] = "changed in final readback"
                        elif corruption == "target":
                            value["snapshot"]["sources"]["d0"]["asset_sha256"] = "changed"
                        elif corruption == "inconsistent":
                            value["source_fence_consistent"] = False
                        elif corruption == "published_failure":
                            value["documents"][0].update(run="4", progress=-1)
                        elif corruption in {"source_fail", "source_cancel"}:
                            value["tasks"][-2]["status"] = "4" if corruption == "source_fail" else "2"
                        else:
                            value["tasks"][-1]["update_date"] = "1970-01-01T00:00:00"
                        return value

                adapter = ChangedReadback(self.finalization_frames())
                result = self.run_acceptance(adapter)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["completed_round_ids"], ["first", "second"])
                self.assertEqual(adapter.changes, [5, 1440])
                self.assertIn("error", result["restoration"])
                self.assertEqual(result["elapsed_seconds"], 60)

    def test_stale_observation_health_cannot_count_round(self):
        frames = self.frames()
        for frame in frames[2:]:
            frame["scope_health"]["coverage_manifest"]["last_observation_success_at"] = "2026-10-10T10:59:00Z"
        adapter = Adapter(frames)
        result = self.run_acceptance(adapter, timeout_seconds=70)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["completed_round_ids"], [])
        self.assertEqual(adapter.changes, [5, 1440])

    def test_native_local_task_dates_compare_to_utc_observation_health(self):
        frames = self.frames()
        for frame in frames:
            frame["native_utc_offset_seconds"] = 8 * 3600
            coverage = frame["scope_health"]["coverage_manifest"]
            observation = datetime.fromisoformat(coverage["last_observation_success_at"])
            coverage["last_observation_success_at"] = (observation - timedelta(hours=8)).isoformat()
        result = self.run_acceptance(Adapter(frames))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["rounds"][0]["queued_to_started_seconds"], 330)

    def test_failed_observation_health_never_changes_interval(self):
        frames = self.frames()[:1]
        frames[0]["scope_health"]["coverage_manifest"]["scan_health"]["observation_complete"] = False
        adapter = Adapter(frames)
        result = self.run_acceptance(adapter, timeout_seconds=20)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(adapter.changes, [])

    def test_health_written_after_done_and_before_native_next_schedule_is_valid(self):
        frames = self.frames()
        for frame in frames:
            coverage = frame["scope_health"]["coverage_manifest"]
            latest = next(row for row in reversed(frame["tasks"]) if row["status"] == "3")
            coverage["last_observation_success_at"] = (datetime.fromisoformat(latest["update_date"]) + timedelta(seconds=2)).isoformat() + "Z"
            for row in frame["tasks"]:
                if row["id"] in {"second", "third"}:
                    row["create_date"] = (datetime.fromisoformat(row["create_date"]) + timedelta(seconds=3)).isoformat()
                    if row["status"] == "5":
                        row["update_date"] = row["create_date"]
        result = self.run_acceptance(Adapter(frames))
        self.assertEqual(result["status"], "passed")

    def test_explicit_incomplete_scan_cannot_claim_strict_completeness(self):
        from experiments.search_agent.policy_acceptance import _finished_health
        frame = self.frames()[0]
        coverage = frame["scope_health"]["coverage_manifest"]
        coverage["scan_health"].update(discovery_complete=True, scan_complete=False)
        coverage["last_discovery_success_at"] = coverage["last_observation_success_at"]
        frame["scope_health"]["last_scan_success_at"] = coverage["last_observation_success_at"]
        self.assertEqual(_finished_health(frame, frame["tasks"][0]), (True, False))

    def test_health_from_newer_running_round_cannot_count_previous_done(self):
        from experiments.search_agent.policy_acceptance import _finished_health
        frame = self.frames()[3]
        self.assertEqual(_finished_health(frame, frame["tasks"][1]), (False, False))

    def test_native_millisecond_creation_orders_same_second_failure_and_running(self):
        from experiments.search_agent.policy_acceptance import _finished_health
        frame = self.frames()[0]
        current, scheduled = frame["tasks"]
        created = datetime(2026, 10, 10, 10, 57, tzinfo=timezone.utc)
        base_ms = int(created.timestamp() * 1000)
        current.update(create_date="2026-10-10T10:57:00", create_time=base_ms + 500)
        scheduled["create_time"] = int(datetime(2026, 10, 10, 11, tzinfo=timezone.utc).timestamp() * 1000) + 600
        older = task("older-failed", "4", -3)
        older.update(create_date=current["create_date"], create_time=base_ms + 100)
        frame["tasks"].insert(0, older)
        self.assertEqual(_finished_health(frame, current), (True, False))
        newer = copy.deepcopy(older)
        newer.update(id="newer-running", status="1", create_time=base_ms + 900)
        frame["tasks"].append(newer)
        self.assertEqual(_finished_health(frame, current), (False, False))

    def test_backdated_original_schedule_cannot_count_short_wait_as_five_minutes(self):
        for with_epoch in (False, True):
            with self.subTest(with_epoch=with_epoch):
                frames = self.frames()
                frames[0]["tasks"][-1]["update_date"] = "1970-01-01T00:00:00"
                if with_epoch:
                    frames[0]["tasks"][-1]["update_time"] = 0
                for frame in frames[1:]:
                    for row in frame["tasks"]:
                        if row["id"] == "first":
                            row["time_started"] = "2026-10-10T11:00:30"
                adapter = Adapter(frames)
                result = self.run_acceptance(adapter)
                self.assertEqual(result["status"], "failed")
                self.assertIn("backdated", result["error"])
                self.assertEqual(result["completed_round_ids"], [])
                self.assertEqual(adapter.changes, [])

    def test_natural_schedule_truncated_dates_and_millisecond_epochs_are_allowed(self):
        frames = self.frames()
        for frame in frames:
            for row in frame["tasks"]:
                row["create_time"] = int(datetime.fromisoformat(row["create_date"]).replace(tzinfo=timezone.utc).timestamp() * 1000) + 750
                row["update_time"] = int(datetime.fromisoformat(row["update_date"]).replace(tzinfo=timezone.utc).timestamp() * 1000) + 750
        result = self.run_acceptance(Adapter(frames))
        self.assertEqual(result["status"], "passed")


if __name__ == "__main__":
    unittest.main()
