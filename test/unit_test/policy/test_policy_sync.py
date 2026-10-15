"""Recovery rules exercise adapters without running external services."""

import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
import xxhash
from test.unit_test.api.db.services.test_policy_catalog import CatalogFixture, capture


def load():
    path = Path(__file__).resolve().parents[3] / "api/db/services/policy_sync_service.py"
    spec = importlib.util.spec_from_file_location("policy_sync_unit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecoveryTests(unittest.TestCase):
    def test_retry_delay_is_bounded_and_survives_restart(self):
        mod = load()
        self.assertEqual([mod.retry_delay(i) for i in range(1, 5)], [60, 300, 900, None])

    def test_scan_schedule_distinguishes_retry_from_regular_scan(self):
        mod = load()
        now = "2026-10-10T00:00:00+00:00"
        scope = {"retry_count": 1, "next_retry_at": None, "last_error": "network", "next_scan_at": "2026-10-11T00:00:00+00:00"}
        self.assertEqual(mod.scan_action(scope, now), ("persist_retry", "2026-10-10T00:01:00+00:00"))
        scope["next_retry_at"] = "2026-10-09T23:59:00+00:00"
        self.assertEqual(mod.scan_action(scope, now)[0], "schedule_retry")
        scope.update(retry_count=4, next_retry_at=None)
        self.assertIsNone(mod.scan_action(scope, now))
        scope["next_scan_at"] = now
        self.assertEqual(mod.scan_action(scope, now)[0], "schedule_regular")

    def test_running_document_is_not_requeued_without_stale_task_evidence(self):
        mod = load()
        self.assertFalse(mod.parse_retryable({"run": "1", "progress": 0.3}))
        self.assertTrue(mod.parse_retryable({"run": "4", "progress": -1}))
        self.assertTrue(mod.parse_retryable({"run": "0", "progress": 0}))
        self.assertFalse(mod.parse_retryable({"run": "3", "progress": 1, "chunk_num": 1}))


class SyncBehaviorTests(CatalogFixture):
    def setUp(self):
        super().setUp()
        self.mod = load()
        self.kb = {
            "id": "kb",
            "tenant_id": "tenant",
            "parser_id": "naive",
            "parser_config": {"chunk_token_num": 128},
            "embd_id": "embed",
            "tenant_embd_id": "model-id",
            "embedding_config": {"llm_factory": "Test", "llm_name": "Embed", "api_base": "http://local", "dimension": 16},
        }
        self.config = {"parse_build_key": self.mod.parse_build_key(self.kb)}
        self.connector = SimpleNamespace(status="5", config=self.config, timeout_secs=3600)
        self.logs = []
        self.docs = {}
        self.tasks = {}
        self.runs = []
        self.index = {}
        self.metadata = {}
        self.metadata_writes = []
        test = self

        class Logs:
            @staticmethod
            def query(**kwargs):
                return [r for r in test.logs if all(str(getattr(r, k, None)) == str(v) for k, v in kwargs.items())]

            @staticmethod
            def get_latest_task(*args, **kwargs):
                return test.logs[-1] if test.logs else None

            @staticmethod
            def schedule(connector_id, kb_id, **kwargs):
                existing = [r for r in test.logs if r.status == "5"]
                if existing:
                    for record in existing:
                        record.immediate = kwargs.get("run_immediately")
                    return None
                test.logs.append(
                    SimpleNamespace(
                        id=str(len(test.logs)), status="5", connector_id=connector_id, kb_id=kb_id, task_type="sync", immediate=kwargs.get("run_immediately"), time_started=datetime.now(timezone.utc)
                    )
                )
                return True

            @staticmethod
            def update_by_id(log_id, values):
                for record in test.logs:
                    if record.id == log_id:
                        for key, value in values.items():
                            setattr(record, key, value)

        class Docs:
            @staticmethod
            def get_by_id(doc_id):
                data = test.docs.get(doc_id)
                return bool(data), SimpleNamespace(id=doc_id, to_dict=lambda: dict(data)) if data else None

            @staticmethod
            def begin2parse(doc_id):
                test.docs[doc_id]["run"] = "1"
                test.docs[doc_id]["progress"] = 0

            @staticmethod
            def run(tenant, data, tables):
                test.runs.append(data["id"])

        class Metadata:
            @staticmethod
            def get_document_metadata(doc_id, *, strict=False):
                return dict(test.metadata.get(doc_id, {}))

            @staticmethod
            def update_document_metadata(doc_id, values, **kwargs):
                test.metadata_writes.append((doc_id, dict(values)))
                test.metadata[doc_id] = dict(values)
                return True

        self.native = SimpleNamespace(
            ConnectorService=SimpleNamespace(get_by_id=lambda cid: (True, self.connector)),
            SyncLogsService=Logs,
            DocumentService=Docs,
            DocMetadataService=Metadata,
            TaskService=SimpleNamespace(query=lambda doc_id: self.tasks.get(doc_id, [])),
            KnowledgebaseService=SimpleNamespace(get_by_id=lambda kid: (True, SimpleNamespace(to_dict=lambda: self.kb))),
            linked=lambda cid, kid: True,
            FileService=SimpleNamespace(get_parser=lambda kind, filename, default: default),
            filename_type=lambda filename: "doc",
            timezone="UTC",
        )
        self.publisher = SimpleNamespace(reconcile=lambda sid: {"published": False}, _chunks=lambda scope, doc_id: self.index.get(doc_id, []))
        self.service = self.mod.PolicySyncService(self.catalog, self.publisher)
        self.patch = patch.object(self.service, "_native", return_value=self.native)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def prepared(self, run="0", count=0):
        value = capture()
        value["parse_build_key"] = self.config["parse_build_key"]
        row = self.observe(value)
        self.catalog.bind_documents(
            self.sid,
            row["revision_id"],
            [
                {
                    "doc_id": "doc",
                    "role": "body",
                    "sha256": value["assets"][0]["sha256"],
                    "parse_build_key": self.config["parse_build_key"],
                    "parser_id": "naive",
                    "parser_config": self.kb["parser_config"],
                    "fingerprint": xxhash.xxh128(value["assets"][0]["blob"]).hexdigest(),
                    "embedding_model": self.mod.canonical_embedding_config(self.kb["embedding_config"]),
                    "parse_build_config": self.mod._parse_build_config(self.kb),
                    "tenant_embd_id": self.kb["tenant_embd_id"],
                    "embd_id": self.kb["embd_id"],
                }
            ],
        )
        self.docs["doc"] = {"id": "doc", "kb_id": "kb", "source_type": "shanghai_medical/connector", "run": run, "progress": 1 if run == "3" else 0, "chunk_num": count, "parser_id": "naive", "parser_config": self.kb["parser_config"], "content_hash": xxhash.xxh128(value["assets"][0]["blob"]).hexdigest()}
        return row

    def test_recovery_restores_missing_admission_before_queue_and_preserves_other_metadata(self):
        import sys
        from test.unit_test.policy.test_policy_ingestion import load as load_ingestion

        row = self.prepared("4")
        self.metadata["doc"] = {"category": ["medical"], "scope_id": self.sid}
        queued_metadata = []

        def guarded_queue(tenant, data, tables):
            task = dict(data, doc_id=data["id"], tenant_id=tenant, embd_id="embed", tenant_embd_id="model-id")
            with patch.dict(sys.modules, {"api.db.services.policy_sync_service": self.mod}):
                load_ingestion().validate_policy_parse_task(task, self.kb["embedding_config"], catalog=self.catalog, metadata_reader=self.native.DocMetadataService.get_document_metadata)
            queued_metadata.append(dict(self.metadata[data["id"]]))

        self.native.DocumentService.run = guarded_queue
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(queued_metadata, [{"category": ["medical"], "scope_id": self.sid, "revision_id": row["revision_id"], "parse_build_key": self.config["parse_build_key"]}])
        self.assertEqual(len(self.metadata_writes), 1)

    def test_recovery_metadata_failure_never_queues_and_records_error(self):
        row = self.prepared("4")
        for write_result in (False, True):
            with self.subTest(write_result=write_result):
                self.models[1].update(retry_count=0, next_retry_at=None).where(self.models[1].id == row["revision_id"]).execute()
                self.native.DocMetadataService.update_document_metadata = lambda *args, **kwargs: write_result
                self.service._recover(self.catalog.get_scope(self.sid))
                revision = self.models[1].get_by_id(row["revision_id"])
                self.assertEqual(self.runs, [])
                self.assertEqual(self.docs["doc"]["run"], "4")
                self.assertTrue(any("metadata" in error for error in revision.errors))
                self.assertEqual(revision.retry_count, 1)

    def test_recovery_metadata_failure_does_not_reset_completed_task(self):
        row = self.prepared("3", 1)
        self.tasks["doc"] = [SimpleNamespace(id="completed", progress=1)]
        self.native.TaskService.update_by_id = lambda tid, values: setattr(self.tasks["doc"][0], "progress", values["progress"])
        self.native.DocMetadataService.update_document_metadata = lambda *args, **kwargs: False
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.tasks["doc"][0].progress, 1)
        self.assertEqual(self.docs["doc"]["run"], "3")
        self.assertEqual(self.runs, [])
        self.assertTrue(self.models[1].get_by_id(row["revision_id"]).errors)

    def test_recovery_strict_read_failure_preserves_unknown_existing_metadata_and_never_queues(self):
        row = self.prepared("4")
        self.metadata["doc"] = {"category": "medical", "revision_id": "other"}

        def unavailable(doc_id, *, strict=False):
            if strict:
                raise ConnectionError("metadata read unavailable")
            return {}

        self.native.DocMetadataService.get_document_metadata = unavailable
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.metadata["doc"], {"category": "medical", "revision_id": "other"})
        self.assertEqual(self.metadata_writes, [])
        self.assertEqual(self.runs, [])
        self.assertEqual(self.docs["doc"]["run"], "4")
        self.assertTrue(any("read unavailable" in error for error in self.models[1].get_by_id(row["revision_id"]).errors))

    def test_recovery_keeps_complete_matching_admission_without_rewrite(self):
        row = self.prepared("4")
        self.metadata["doc"] = {"scope_id": self.sid, "revision_id": row["revision_id"], "parse_build_key": self.config["parse_build_key"], "category": "medical"}
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, ["doc"])
        self.assertEqual(self.metadata_writes, [])

    def test_recovery_fingerprint_mismatch_or_missing_binding_blocks_before_metadata_read(self):
        row = self.prepared("4")
        reads = []

        def read(doc_id, **kwargs):
            reads.append(doc_id)
            return {}

        self.native.DocMetadataService.get_document_metadata = read
        original = self.docs["doc"]["content_hash"]
        for fingerprint in ("wrong", None, ""):
            with self.subTest(fingerprint=fingerprint):
                self.models[1].update(retry_count=0, next_retry_at=None).where(self.models[1].id == row["revision_id"]).execute()
                self.docs["doc"]["content_hash"] = fingerprint
                self.service._recover(self.catalog.get_scope(self.sid))
                self.assertEqual(self.runs, [])
                self.assertEqual(reads, [])
                self.assertEqual(self.metadata_writes, [])
        self.docs["doc"]["content_hash"] = original
        revision = self.models[1].get_by_id(row["revision_id"])
        builds = revision.parsing_builds
        builds[self.config["parse_build_key"]][0].pop("fingerprint")
        self.models[1].update(parsing_builds=builds, retry_count=0, next_retry_at=None).where(self.models[1].id == revision.id).execute()
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, [])
        self.assertEqual(reads, [])
        self.assertEqual(self.metadata_writes, [])

    def test_recovery_never_overwrites_conflicting_admission(self):
        row = self.prepared("4")
        for key in ("scope_id", "revision_id", "parse_build_key"):
            with self.subTest(key=key):
                self.models[1].update(retry_count=0, next_retry_at=None).where(self.models[1].id == row["revision_id"]).execute()
                self.metadata["doc"] = {key: "foreign", "category": "medical"}
                self.service._recover(self.catalog.get_scope(self.sid))
                self.assertEqual(self.metadata["doc"], {key: "foreign", "category": "medical"})
                self.assertEqual(self.metadata_writes, [])
                self.assertEqual(self.runs, [])
                self.assertTrue(self.models[1].get_by_id(row["revision_id"]).errors)

    def test_recovery_refuses_foreign_source_and_model_binding_before_metadata_write(self):
        row = self.prepared("4")
        self.docs["doc"]["source_type"] = "shanghai_medical/other"
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, [])
        self.assertEqual(self.metadata_writes, [])
        self.docs["doc"]["source_type"] = "shanghai_medical/connector"
        revision = self.models[1].get_by_id(row["revision_id"])
        builds = revision.parsing_builds
        builds[self.config["parse_build_key"]][0]["embedding_model"]["llm_name"] = "foreign-model"
        self.models[1].update(parsing_builds=builds, retry_count=0, next_retry_at=None).where(self.models[1].id == revision.id).execute()
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, [])
        self.assertEqual(self.metadata_writes, [])

    def test_recovery_refuses_ambiguous_document_binding(self):
        row = self.prepared("4")
        other = self.observe(dict(capture(text="other", key="other"), parse_build_key=self.config["parse_build_key"]))
        binding = dict(row["assets"][0], doc_id="doc", parse_build_key=self.config["parse_build_key"], sha256=other["assets"][0]["sha256"])
        self.catalog.bind_documents(self.sid, other["revision_id"], [binding])
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, [])
        self.assertEqual(self.metadata_writes, [])

    def test_finish_partial_waits_for_daily_scan_and_transport_failure_retries_are_bounded(self):
        health = {"discovery_complete": True, "observation_complete": True, "scan_complete": False, "needs_review_count": 2}
        self.service.finish(self.sid, False, "manual review pending", scan_health=health, retryable=False)
        scope = self.catalog.get_scope(self.sid)
        self.assertEqual(scope["coverage_manifest"]["scan_health"], health)
        self.assertIsNone(self.mod.scan_action(scope, datetime.now(timezone.utc)))
        self.assertEqual(self.mod.scan_action(scope, datetime.fromisoformat(scope["next_scan_at"]))[0], "schedule_regular")
        for attempt, delay in enumerate((60, 300, 900, None), 1):
            self.service.finish(self.sid, False, "HTTP failure", scan_health={"discovery_complete": False, "observation_complete": False}, retryable=True)
            scope = self.catalog.get_scope(self.sid)
            self.assertEqual(scope["retry_count"], attempt)
            now = datetime.now(timezone.utc)
            action = self.mod.scan_action(scope, now)
            if delay is None:
                self.assertIsNone(action)
            else:
                self.assertEqual(action, ("persist_retry", (now + timedelta(seconds=delay)).isoformat()))

    def test_parse_build_changes_with_embedding_identity_and_parser(self):
        original = self.mod.parse_build_key(self.kb)
        self.assertNotEqual(original, self.mod.parse_build_key(dict(self.kb, tenant_embd_id="new-model")))
        self.assertNotEqual(original, self.mod.parse_build_key(dict(self.kb, parser_config={"chunk_token_num": 256})))
        self.assertEqual(original, self.mod.parse_build_key(dict(reversed(list(self.kb.items())))))

    def test_source_failure_is_persisted_and_does_not_abort_capture_loop(self):
        value = capture(errors=["missing attachment"])
        result = self.service.capture(self.sid, self.catalog.begin_scan(self.sid), value, self.config["parse_build_key"])
        self.assertIn("revision_id", result)
        self.assertEqual(len(self.catalog.pending(self.sid)), 1)
        self.assertTrue(value["errors"])

    def test_archive_failure_is_added_to_original_capture_errors(self):
        self.objects.put = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline"))
        value = capture()
        self.service.capture(self.sid, self.catalog.begin_scan(self.sid), value, self.config["parse_build_key"])
        self.assertTrue(any("offline" in e for e in value["errors"]))
        self.assertIsNone(value["revision_hash"])

    def test_begin_refuses_parse_build_drift(self):
        self.kb["embd_id"] = "changed"
        with self.assertRaises(ValueError):
            self.service.begin({"tenant_id": "tenant", "kb_id": "kb", "connector_id": "connector"}, self.config)
        self.assertEqual(self.catalog.get_scope(self.sid)["scan_sequence"], 0)

    def test_embedding_identity_is_canonical_and_contains_no_credentials(self):
        config = dict(self.kb["embedding_config"], api_key="secret", unrelated="value")
        normalized = self.mod.canonical_embedding_config(config)
        self.assertEqual(set(normalized), {"llm_factory", "llm_name", "api_base", "max_tokens", "dimension", "dimensions", "embedding_dimension"})
        self.assertNotIn("secret", str(normalized))
        self.assertEqual(self.mod.embedding_config_sha256(config), self.mod.embedding_config_sha256(dict(reversed(list(config.items())))))
        self.assertNotEqual(self.mod.embedding_config_sha256(config), self.mod.embedding_config_sha256(dict(config, api_base="http://another")))

    def test_native_running_scan_does_not_block_parse_recovery(self):
        self.prepared()
        self.logs.append(SimpleNamespace(id="scan", connector_id="connector", kb_id="kb", task_type="sync", status="1", time_started=datetime.now(timezone.utc)))
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, ["doc"])
        self.assertEqual(len(self.logs), 1)

    def test_done_but_missing_index_requeues_only_unpublished_asset(self):
        self.prepared("3", 1)
        self.tasks["doc"] = [SimpleNamespace(id="done-task", progress=1)]
        self.native.TaskService.update_by_id = lambda tid, values: setattr(self.tasks["doc"][0], "progress", values["progress"])
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, ["doc"])

    def test_missing_document_schedules_immediate_source_recovery(self):
        self.prepared()
        self.docs.clear()
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(len(self.logs), 1)
        self.assertTrue(self.logs[0].immediate)

    def test_running_source_missing_upload_does_not_burn_revision_retries(self):
        row = self.prepared()
        self.docs.clear()
        source = SimpleNamespace(id="uploading", connector_id="connector", kb_id="kb", task_type="sync", status="1", time_started=datetime.now(timezone.utc))
        self.logs.append(source)
        for _ in range(4):
            self.service._recover(self.catalog.get_scope(self.sid))
        revision = self.models[1].get_by_id(row["revision_id"])
        self.assertEqual(revision.retry_count, 0)
        self.assertIsNone(revision.next_retry_at)
        self.assertEqual(self.logs, [source])
        self.assertEqual(self.runs, [])

        # Once the source finishes/fails, missing-document recovery may consume
        # its actual bounded retry budget and request the native source queue.
        source.status = "4"
        for attempt in range(1, 4):
            self.models[1].update(next_retry_at=None).where(self.models[1].id == row["revision_id"]).execute()
            self.service._recover(self.catalog.get_scope(self.sid))
            revision = self.models[1].get_by_id(row["revision_id"])
            self.assertEqual(revision.retry_count, attempt)
            self.assertIsNotNone(revision.next_retry_at)
            self.assertEqual(len(self.logs), 2)
            self.assertTrue(self.logs[-1].immediate)
            self.service._recover(self.catalog.get_scope(self.sid))
            self.assertEqual(self.models[1].get_by_id(row["revision_id"]).retry_count, attempt)
        self.models[1].update(next_retry_at=None).where(self.models[1].id == row["revision_id"]).execute()
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.models[1].get_by_id(row["revision_id"]).retry_count, 3)
        self.assertEqual(len(self.logs), 2)

    def test_running_source_unbound_assets_wait_without_burning_retries(self):
        row = self.prepared()
        self.models[1].update(parsing_builds={}).where(self.models[1].id == row["revision_id"]).execute()
        self.logs.append(SimpleNamespace(id="binding", connector_id="connector", kb_id="kb", task_type="sync", status="1", time_started=datetime.now(timezone.utc)))
        self.service._recover(self.catalog.get_scope(self.sid))
        revision = self.models[1].get_by_id(row["revision_id"])
        self.assertEqual(revision.retry_count, 0)
        self.assertIsNone(revision.next_retry_at)
        self.assertEqual(len(self.logs), 1)

    def test_running_source_does_not_disable_recovery_for_existing_failed_parse(self):
        row = self.prepared("4")
        self.logs.append(SimpleNamespace(id="source", connector_id="connector", kb_id="kb", task_type="sync", status="1", time_started=datetime.now(timezone.utc)))
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, ["doc"])
        self.assertEqual(self.models[1].get_by_id(row["revision_id"]).retry_count, 1)
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, ["doc"])
        self.assertEqual(len(self.logs), 1)

    def test_missing_document_source_recovery_respects_scope_backoff_and_exhaustion(self):
        row = self.prepared()
        self.docs.clear()
        now = datetime.now(timezone.utc)
        for retries, next_retry in ((4, None), (1, (now + timedelta(minutes=5)).isoformat()), (1, None)):
            with self.subTest(retries=retries, next_retry=next_retry):
                self.models[0].update(retry_count=retries, last_error="source failure", next_retry_at=next_retry, next_scan_at=(now + timedelta(days=1)).isoformat()).where(
                    self.models[0].id == self.sid
                ).execute()
                self.service._recover(self.catalog.get_scope(self.sid))
                self.assertEqual(self.logs, [])
                revision = self.models[1].get_by_id(row["revision_id"])
                self.assertEqual(revision.retry_count, 0)
                self.assertIsNone(revision.next_retry_at)
                scope = self.catalog.get_scope(self.sid)
                self.assertEqual(scope["retry_count"], retries)
                if retries == 1:
                    self.assertIsNotNone(scope["next_retry_at"])

    def test_missing_document_source_recovery_runs_when_scope_retry_is_due(self):
        row = self.prepared()
        self.docs.clear()
        now = datetime.now(timezone.utc)
        self.models[0].update(retry_count=1, last_error="source failure", next_retry_at=(now - timedelta(seconds=1)).isoformat(), next_scan_at=(now + timedelta(days=1)).isoformat()).where(
            self.models[0].id == self.sid
        ).execute()
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(len(self.logs), 1)
        self.assertTrue(self.logs[0].immediate)
        self.assertEqual(self.models[1].get_by_id(row["revision_id"]).retry_count, 1)
        self.assertEqual(self.catalog.get_scope(self.sid)["retry_count"], 1)

    def test_scope_source_exhaustion_does_not_block_existing_failed_parse_recovery(self):
        row = self.prepared("4")
        self.models[0].update(retry_count=4, last_error="source failure", next_scan_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat()).where(self.models[0].id == self.sid).execute()
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, ["doc"])
        self.assertEqual(self.logs, [])
        self.assertEqual(self.models[1].get_by_id(row["revision_id"]).retry_count, 1)

    def test_daily_scan_resets_exhausted_parser_retry(self):
        row = self.prepared("4")
        self.models[1].update(retry_count=3).where(self.models[1].id == row["revision_id"]).execute()
        self.models[0].update(next_scan_at=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat()).where(self.models[0].id == self.sid).execute()
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, ["doc"])
        self.assertEqual(self.models[1].get_by_id(row["revision_id"]).retry_count, 1)

    def test_cancelled_or_unlinked_connector_never_requeues(self):
        self.prepared()
        self.connector.status = "2"
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, [])
        self.connector.status = "5"
        self.native.linked = lambda *args: False
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, [])

    def test_done_missing_index_invalidates_completed_task_reuse(self):
        self.prepared("3", 1)
        self.tasks["doc"] = [SimpleNamespace(id="parse-task", progress=1, chunk_ids="gone-chunk")]

        def update_task(task_id, values):
            for value in self.tasks["doc"]:
                if value.id == task_id:
                    value.progress = values["progress"]

        self.native.TaskService.update_by_id = update_task
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.tasks["doc"][0].progress, 0)
        self.assertEqual(self.tasks["doc"][0].chunk_ids, "gone-chunk")
        self.assertEqual(self.runs, ["doc"])

    def test_expired_native_source_scan_records_failure_without_immediate_hot_retry(self):
        self.logs.append(SimpleNamespace(id="expired", connector_id="connector", kb_id="kb", task_type="sync", status="1", time_started=datetime.now(timezone.utc) - timedelta(hours=2)))
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.logs[0].status, "4")
        scope = self.catalog.get_scope(self.sid)
        self.assertEqual(scope["retry_count"], 1)
        self.assertIn("timeout", scope["last_error"])
        self.assertIsNotNone(scope["next_retry_at"])
        self.assertEqual(len(self.logs), 1)

    def test_unknown_running_start_never_guesses_timeout(self):
        self.logs.append(SimpleNamespace(id="unknown", connector_id="connector", kb_id="kb", task_type="sync", status="1", time_started=None))
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.logs[0].status, "1")
        self.assertEqual(self.catalog.get_scope(self.sid)["retry_count"], 0)

    def test_provider_change_changes_build_but_api_key_does_not(self):
        original = self.mod.parse_build_key(self.kb)
        self.kb["embedding_config"]["api_key"] = "never-archive-secret"
        self.assertEqual(self.mod.parse_build_key(self.kb), original)
        self.kb["embedding_config"]["llm_name"] = "NewEmbedding"
        self.assertNotEqual(self.mod.parse_build_key(self.kb), original)
        with self.assertRaises(ValueError):
            self.mod.parse_build_key(dict(self.kb, pipeline_id="pipeline"))

    def test_bind_freezes_actual_upload_config_and_rejects_midscan_drift(self):
        value = capture()
        row = self.catalog.observe(self.sid, self.catalog.begin_scan(self.sid), dict(value, parse_build_key=self.config["parse_build_key"]))
        doc = {
            "id": "doc",
            "semantic_identifier": "policy.txt",
            "extension": ".txt",
            "metadata": {
                "scope_id": self.sid,
                "revision_id": row["revision_id"],
                "parse_build_key": self.config["parse_build_key"],
                "asset_role": "body",
                "asset_sha256": value["assets"][0]["sha256"],
            },
        }
        task = {"tenant_id": "tenant", "kb_id": "kb", "connector_id": "connector"}
        self.service.bind(self.sid, [doc], self.config, task)
        binding = self.models[1].get_by_id(row["revision_id"]).parsing_builds[self.config["parse_build_key"]][0]
        self.assertEqual(binding["parser_config"], {"chunk_token_num": 128})
        self.assertEqual(binding["embedding_model"]["llm_name"], "Embed")
        self.assertEqual(binding["parse_build_config"], self.mod._parse_build_config(self.kb))
        self.assertEqual(binding["parse_build_config"]["embedding_config"], binding["embedding_model"])
        self.kb["parser_config"] = {"chunk_token_num": 256}
        with self.assertRaises(ValueError):
            self.service.bind(self.sid, [doc], self.config, task)

    def test_missing_document_expedites_existing_daily_queue(self):
        self.prepared()
        self.docs.clear()
        self.logs.append(SimpleNamespace(id="daily", connector_id="connector", kb_id="kb", task_type="sync", status="5", immediate=False))
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertTrue(self.logs[0].immediate)
        self.assertEqual(len(self.logs), 1)

    def test_old_build_is_not_requeued_with_new_embedding_config(self):
        row = self.prepared()
        self.kb["embedding_config"]["llm_name"] = "new"
        self.config["parse_build_key"] = self.mod.parse_build_key(self.kb)
        self.service._recover(self.catalog.get_scope(self.sid))
        self.assertEqual(self.runs, [])
        self.assertEqual(self.models[1].get_by_id(row["revision_id"]).state, "needs_review")


class NativeScheduleTests(unittest.TestCase):
    def test_parse_generation_creates_new_identity_for_same_configuration(self):
        mod = load()
        kb = {"parser_id": "naive", "parser_config": {}, "embd_id": "Embed", "embedding_config": {"llm_factory": "Test", "llm_name": "Embed"}}
        self.assertNotEqual(mod.parse_build_key(kb, 0), mod.parse_build_key(kb, 1))
        for generation in (-1, 1000001, True, "1"):
            with self.assertRaises(ValueError):
                mod.parse_build_key(kb, generation)


class MaintenanceLoopTests(unittest.TestCase):
    def test_threading_event_wait_keeps_dispatch_running_and_stops_promptly(self):
        import asyncio
        import threading

        mod = load()
        stop, heartbeat, finished = threading.Event(), threading.Event(), threading.Event()
        second_tick = threading.Event()
        ticks, errors = [], []

        def tick():
            ticks.append(True)
            if len(ticks) >= 2:
                second_tick.set()

        async def exercise():
            maintenance = asyncio.create_task(mod.maintain_policies(stop, interval=0.01))
            await asyncio.sleep(0.02)
            heartbeat.set()  # Another native worker coroutine must remain runnable.
            await asyncio.to_thread(second_tick.wait, 0.2)
            stop.set()
            await asyncio.wait_for(maintenance, timeout=0.2)

        def run():
            try:
                asyncio.run(exercise())
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        # A separate loop thread makes the regression itself bounded even when
        # the broken implementation synchronously blocks inside Event.wait().
        with patch.object(mod, "PolicySyncService", return_value=SimpleNamespace(tick=tick)):
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            try:
                dispatched = heartbeat.wait(timeout=0.5)
                stopped = finished.wait(timeout=0.5) if dispatched else False
            finally:
                stop.set()
                worker.join(timeout=1)
        self.assertTrue(dispatched, "synchronous threading.Event.wait blocked event-loop dispatch")
        self.assertTrue(stopped, "maintenance did not stop after the event was set")
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(ticks), 2)
