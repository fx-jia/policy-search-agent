"""Policy bookkeeping and bounded recovery through native source and parse queues."""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo


def canonical_embedding_config(config):
    """Credential-free identity shared by ingestion, publication and queries."""
    if not isinstance(config, dict) or not config.get("llm_name") or not config.get("llm_factory"):
        raise ValueError("policy embedding configuration could not be resolved")
    return {key: config.get(key) for key in ("llm_factory", "llm_name", "api_base", "max_tokens", "dimension", "dimensions", "embedding_dimension")}


def embedding_config_sha256(config):
    return hashlib.sha256(json.dumps(canonical_embedding_config(config), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _embedding_config(kb):
    if "embedding_config" in kb:
        config = kb["embedding_config"]
    else:
        from api.db.joint_services.tenant_model_service import get_model_config_by_id, resolve_model_config
        from common.constants import LLMType

        config = (
            get_model_config_by_id(kb["tenant_id"], LLMType.EMBEDDING, kb["tenant_embd_id"]) if kb.get("tenant_embd_id") else resolve_model_config(kb["tenant_id"], LLMType.EMBEDDING, kb["embd_id"])
        )
    return canonical_embedding_config(config)


def _parse_build_config(kb: dict, parse_generation=0) -> dict:
    """Freeze actual parser/model configuration without provider credentials."""
    if type(parse_generation) is not int or not 0 <= parse_generation <= 1000000:
        raise ValueError("parse_generation must be an integer from 0 through 1000000")
    if kb.get("pipeline_id"):
        raise ValueError("policy MVP requires the native parser; pipeline_id is unsupported")
    config = {key: kb.get(key) for key in ("parser_id", "tenant_embd_id", "embd_id")}
    config.update(parser_config=kb.get("parser_config") or {}, embedding_config=_embedding_config(kb), implementation="shanghai-policy-parser-v1", parse_generation=parse_generation)
    return config


def parse_build_key(kb: dict, parse_generation=0) -> str:
    config = _parse_build_config(kb, parse_generation)
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def retry_delay(attempt):
    return {1: 60, 2: 300, 3: 900}.get(attempt)


def scan_action(scope, now):
    now = datetime.fromisoformat(now) if isinstance(now, str) else now

    def due(value):
        return value and datetime.fromisoformat(value) <= now

    if due(scope.get("next_scan_at")):
        return "schedule_regular", (now + timedelta(hours=24)).isoformat()
    if scope.get("last_error"):
        delay = retry_delay(scope.get("retry_count", 0))
        if delay is None:
            return None
        if not scope.get("next_retry_at"):
            return "persist_retry", (now + timedelta(seconds=delay)).isoformat()
        if due(scope["next_retry_at"]):
            return "schedule_retry", None
    return None


def parse_retryable(doc):
    return str(doc.get("run", "0")) in {"0", "4"} or float(doc.get("progress") or 0) < 0


def validate_policy_config(source, config):
    """Validate stored settings without source requests and close the temporary session."""
    if source != "shanghai_medical":
        return config
    from common.data_source.shanghai_medical_connector import ShanghaiMedicalConnector

    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    connector = ShanghaiMedicalConnector.build_connector(config)
    try:
        if config.get("sync_deleted_files"):
            raise ValueError("policy history cannot be pruned by missing source links")
    finally:
        connector.cancel()
    return config


class PolicySyncService:
    def __init__(self, catalog=None, publisher=None):
        if catalog is None:
            from api.db.services.policy_catalog_service import PolicyCatalogService

            catalog = PolicyCatalogService()
        if publisher is None:
            from api.db.services.policy_publication_service import PolicyPublicationService

            publisher = PolicyPublicationService(catalog)
        self.catalog, self.publisher = catalog, publisher

    @staticmethod
    def _native():
        from api.db.db_models import Connector2Kb
        from api.db.services.connector_service import ConnectorService, SyncLogsService
        from api.db.services.document_service import DocumentService
        from api.db.services.doc_metadata_service import DocMetadataService
        from api.db.services.file_service import FileService
        from api.db.services.knowledgebase_service import KnowledgebaseService
        from api.db.services.task_service import TaskService
        from api.utils.file_utils import filename_type
        from common import settings

        return SimpleNamespace(
            ConnectorService=ConnectorService,
            SyncLogsService=SyncLogsService,
            DocumentService=DocumentService,
            DocMetadataService=DocMetadataService,
            TaskService=TaskService,
            KnowledgebaseService=KnowledgebaseService,
            FileService=FileService,
            filename_type=filename_type,
            timezone=settings.TIMEZONE,
            linked=lambda cid, kid: bool(Connector2Kb.get_or_none((Connector2Kb.connector_id == cid) & (Connector2Kb.kb_id == kid))),
        )

    @staticmethod
    def _knowledgebase(native, task, config):
        ok, model = native.KnowledgebaseService.get_by_id(task["kb_id"])
        if not ok:
            raise ValueError("policy knowledge base is missing")
        kb = dict(model.to_dict())
        kb["embedding_config"] = _embedding_config(kb)
        if kb.get("tenant_id") != task["tenant_id"]:
            raise ValueError("policy knowledge base tenant mismatch")
        if config.get("parse_build_key") != parse_build_key(kb, config.get("parse_generation", 0)):
            raise ValueError("policy parser/embedding configuration changed; initialize a new parse build")
        return kb

    def begin(self, task, config):
        self._knowledgebase(self._native(), task, config)
        scope = self.catalog.ensure_scope(task["tenant_id"], task["kb_id"], task["connector_id"], coverage={"city": "Shanghai", "sources": config.get("list_urls", []), "foundational_complete": False})
        return scope["id"], self.catalog.begin_scan(scope["id"])

    def capture(self, scope_id, sequence, capture, build):
        remaining = capture.get("_asset_budget_remaining")
        if remaining is not None:
            if type(remaining) is not int or remaining < 0:
                raise ValueError("asset budget remaining must be a nonnegative integer")
            if len(capture.get("assets", [])) > remaining and not self.catalog.is_published_capture(scope_id, capture, build):
                if "asset_limit: deferred" not in capture.setdefault("errors", []):
                    capture["errors"].append("asset_limit: deferred")
                capture["revision_hash"] = None
        result = self.catalog.observe(scope_id, sequence, dict(capture, parse_build_key=build))
        if not result.get("revision_hash"):
            # The collector will skip this group and continue persisting other
            # discoveries. Raising here would discard its partially filled batch.
            capture.setdefault("errors", []).extend(e for e in result.get("errors", []) if e not in capture.get("errors", []))
            capture["revision_hash"] = None
        metadata = {key: str(result[key]) for key in ("revision_id", "scope_id", "parse_build_key")}
        metadata["policy_ingestion_required"] = "0" if self.catalog.is_published_revision(scope_id, result["revision_id"], build) else "1"
        return metadata

    def bind(self, scope_id, docs, config, task):
        native = self._native()
        kb = self._knowledgebase(native, task, config)
        scope = self.catalog.get_scope(scope_id)
        if any(scope[key] != task[key] for key in ("tenant_id", "kb_id", "connector_id")):
            raise ValueError("policy scope does not match the native task")
        groups = {}
        for doc in docs:
            meta = doc.get("metadata") or {}
            revision = meta.get("revision_id")
            if not revision or meta.get("scope_id") != scope_id or meta.get("parse_build_key") != config["parse_build_key"]:
                raise ValueError("policy document has no trusted capture/build binding")
            filename = doc["semantic_identifier"]
            if doc["extension"] not in filename:
                filename += doc["extension"]
            groups.setdefault(revision, []).append(
                {
                    "doc_id": doc["id"],
                    "role": meta["asset_role"],
                    "sha256": meta["asset_sha256"],
                    "parse_build_key": meta["parse_build_key"],
                    "fingerprint": doc.get("fingerprint"),
                    "parser_id": native.FileService.get_parser(native.filename_type(filename), filename, kb["parser_id"]),
                    "parser_config": kb.get("parser_config") or {},
                    "embedding_model": _embedding_config(kb),
                    "parse_build_config": _parse_build_config(kb, config.get("parse_generation", 0)),
                    "tenant_embd_id": kb.get("tenant_embd_id"),
                    "embd_id": kb.get("embd_id"),
                }
            )
        for revision, assets in groups.items():
            self.catalog.bind_documents(scope_id, revision, assets)

    def finish(self, scope_id, complete, error=None, *, scan_health=None, retryable=True):
        self.catalog.record_scan_result(scope_id, complete, error, scan_health=scan_health, retryable=retryable)

    @staticmethod
    def _expired_scan(record, connector, now, tz):
        started = getattr(record, "time_started", None)
        timeout = getattr(connector, "timeout_secs", None)
        if not started or not timeout or timeout <= 0:
            return False
        if isinstance(started, str):
            started = datetime.fromisoformat(started)
        if started.tzinfo is None:
            started = started.replace(tzinfo=ZoneInfo(tz))
        return (now - started).total_seconds() > timeout

    def _restore_admission(self, native, scope, revision, asset, data, kb, config):
        """Repair absent metadata only from a unique, sealed current binding."""
        build = config["parse_build_key"]
        if revision["scope_id"] != scope["id"] or not revision["revision_hash"] or revision["parse_build_key"] != build:
            raise ValueError("policy recovery requires a sealed current revision")
        if data.get("id") != asset["doc_id"] or data.get("kb_id") != scope["kb_id"] or data.get("source_type") != "shanghai_medical/" + scope["connector_id"]:
            raise ValueError("policy recovery document scope/source mismatch")
        if data.get("tenant_id", scope["tenant_id"]) != scope["tenant_id"]:
            raise ValueError("policy recovery document tenant mismatch")
        bindings = [
            (row.id, key)
            for row in self.catalog.Revision.select().where(self.catalog.Revision.scope_id == scope["id"])
            for key, assets in row.parsing_builds.items()
            for item in assets if item.get("doc_id") == data["id"]
        ]
        if bindings != [(revision["id"], build)]:
            raise ValueError("policy recovery document has no unique admitted binding")
        if not asset.get("fingerprint") or data.get("content_hash") != asset["fingerprint"]:
            raise ValueError("policy recovery document fingerprint differs from its admitted asset")
        if not asset.get("parser_id") or any(data.get(key) != asset.get(key) for key in ("parser_id", "parser_config")):
            raise ValueError("policy recovery document parser mismatch")
        if asset.get("parse_build_config") != _parse_build_config(kb, config.get("parse_generation", 0)):
            raise ValueError("policy recovery parse build configuration mismatch")
        if canonical_embedding_config(asset.get("embedding_model")) != _embedding_config(kb) or any(asset.get(key) != kb.get(key) for key in ("embd_id", "tenant_embd_id")):
            raise ValueError("policy recovery embedding binding mismatch")
        expected = {"scope_id": scope["id"], "revision_id": revision["id"], "parse_build_key": build}
        metadata = native.DocMetadataService.get_document_metadata(data["id"], strict=True)
        if not isinstance(metadata, dict):
            raise ValueError("policy recovery metadata is not an object")
        if any(key in metadata and metadata[key] != value for key, value in expected.items()):
            raise ValueError("policy recovery metadata admission conflict")
        if any(key not in metadata for key in expected):
            if native.DocMetadataService.update_document_metadata(data["id"], dict(metadata, **expected)) is not True:
                raise ValueError("policy recovery metadata persistence failed")
        readback = native.DocMetadataService.get_document_metadata(data["id"], strict=True) or {}
        if any(readback.get(key) != value for key, value in expected.items()):
            raise ValueError("policy recovery metadata admission readback mismatch")

    def _recover(self, scope):
        native = self._native()
        Scope, Revision = self.catalog.Scope, self.catalog.Revision
        ok, connector = native.ConnectorService.get_by_id(scope["connector_id"])
        if not ok or str(connector.status) == "2" or not native.linked(scope["connector_id"], scope["kb_id"]):
            return
        now = datetime.now(timezone.utc)
        self._knowledgebase(native, scope, connector.config)
        latest = native.SyncLogsService.get_latest_task(scope["connector_id"], scope["kb_id"], task_type="sync")
        if latest and str(latest.status) == "2":
            return
        running = list(native.SyncLogsService.query(connector_id=scope["connector_id"], kb_id=scope["kb_id"], status="1", task_type="sync"))
        for record in running:
            if self._expired_scan(record, connector, now, native.timezone):
                reason = f"policy recovery: native source timeout exceeded ({connector.timeout_secs}s)"
                native.SyncLogsService.update_by_id(record.id, {"status": "4", "error_msg": reason})
                self.finish(scope["id"], False, reason)
        running = [record for record in running if not self._expired_scan(record, connector, now, native.timezone)]
        scope = self.catalog.get_scope(scope["id"])
        action = scan_action(scope, now) if not running else None
        did_schedule = False
        if action:
            kind, due = action
            if kind == "persist_retry":
                Scope.update(next_retry_at=due).where(Scope.id == scope["id"]).execute()
            else:
                native.SyncLogsService.schedule(scope["connector_id"], scope["kb_id"], reindex=True, run_immediately=True)
                did_schedule = True
                Scope.update(
                    next_scan_at=due if kind == "schedule_regular" else scope["next_scan_at"], next_retry_at=None, retry_count=0 if kind == "schedule_regular" else scope["retry_count"]
                ).where(Scope.id == scope["id"]).execute()
                if kind == "schedule_regular":
                    Revision.update(retry_count=0, next_retry_at=None).where(Revision.scope_id == scope["id"]).execute()
        # Missing uploads are source work. A failed source must first pass its
        # normal backoff/budget gate; only a clean source gets initial recovery.
        allow_missing_recovery = not running and (did_schedule or (not scope["last_error"] and not scope["next_retry_at"] and scope["retry_count"] == 0))
        published_ids = {doc for release in self.catalog.Release.select().where(self.catalog.Release.scope_id == scope["id"]) for doc in release.document_manifest}
        missing_documents = False
        for revision in self.catalog.pending(scope["id"]):
            if not revision["revision_hash"]:
                continue
            if revision["parse_build_key"] != connector.config["parse_build_key"]:
                Revision.update(state="needs_review", errors=["old parse build cannot use changed parser/embedding configuration"]).where(Revision.id == revision["id"]).execute()
                continue
            delay = retry_delay(revision["retry_count"] + 1)
            if delay is None or (revision["next_retry_at"] and datetime.fromisoformat(revision["next_retry_at"]) > now):
                continue
            assets = revision["parsing_builds"].get(revision["parse_build_key"], [])
            revision_missing = not assets
            if revision_missing:
                missing_documents = True
            eligible = []
            for asset in assets:
                if asset["doc_id"] in published_ids:
                    continue
                ok, doc = native.DocumentService.get_by_id(asset["doc_id"])
                if not ok:
                    missing_documents = revision_missing = True
                    continue
                data = doc.to_dict()
                if str(data.get("run")) == "2":
                    continue
                if any(field in asset and data.get(field) != asset[field] for field in ("parser_id", "parser_config")):
                    Revision.update(state="needs_review", errors=["document parser configuration differs from parse build"]).where(Revision.id == revision["id"]).execute()
                    continue
                tasks = native.TaskService.query(doc_id=doc.id)
                active_tasks = [t for t in tasks if 0 <= float(t.progress or 0) < 1]
                if active_tasks and not parse_retryable(data):
                    continue
                done = str(data.get("run")) == "3" and float(data.get("progress") or 0) >= 1 and int(data.get("chunk_num") or 0) > 0
                if done:
                    indexed = self.publisher._chunks(scope, doc.id)
                    if len(indexed) == int(data["chunk_num"]) and indexed:
                        continue
                elif not parse_retryable(data) and tasks and not all(float(t.progress or 0) < 0 for t in tasks):
                    continue
                eligible.append((data, asset, tasks if done else []))
            if eligible or (revision_missing and allow_missing_recovery):
                # Missing documents can be in the source's bind-before-upload
                # window; spend their recovery budget only when source recovery
                # is allowed, after both the running and scope retry gates.
                # Persist the attempt before queue I/O, so exceptions/restarts do
                # not create an unlimited hot retry loop.
                Revision.update(retry_count=Revision.retry_count + 1, next_retry_at=(now + timedelta(seconds=delay)).isoformat()).where(Revision.id == revision["id"]).execute()
            for data, asset, completed_tasks in eligible:
                try:
                    kb = self._knowledgebase(native, scope, connector.config)
                    self._restore_admission(native, scope, revision, asset, data, kb, connector.config)
                except Exception as exc:
                    error = f"policy recovery document {data['id']}: {exc}"
                    row = Revision.get_by_id(revision["id"])
                    Revision.update(errors=list(dict.fromkeys([*row.errors, error]))).where(Revision.id == row.id).execute()
                    continue
                for task in completed_tasks:
                    # Native queue_tasks trusts completed task IDs. Invalidate
                    # reuse only after admission passes, retaining cleanup IDs.
                    native.TaskService.update_by_id(task.id, {"progress": 0})
                native.DocumentService.begin2parse(data["id"])
                ok, refreshed = native.DocumentService.get_by_id(data["id"])
                if ok and str(refreshed.to_dict().get("run")) == "1":
                    native.DocumentService.run(scope["tenant_id"], refreshed.to_dict(), {})
        if missing_documents and allow_missing_recovery and not did_schedule:
            native.SyncLogsService.schedule(scope["connector_id"], scope["kb_id"], reindex=True, run_immediately=True)

    def tick(self):
        """Scope leases coordinate recovery and publication across worker processes."""
        Scope = self.catalog.Scope
        results = []
        for row in Scope.select():
            owner, now = uuid.uuid4().hex, time.time()
            claimed = Scope.update(lease_owner=owner, lease_until=now + 300).where((Scope.id == row.id) & (Scope.lease_until <= now)).execute()
            if not claimed:
                continue
            recovery_error = None
            try:
                self._recover(self.catalog.get_scope(row.id))
            except Exception as exc:
                recovery_error = str(exc)
                Scope.update(last_error=str(exc)).where(Scope.id == row.id).execute()
                logging.exception("Policy recovery failed for scope %s", row.id)
            finally:
                Scope.update(lease_owner=None, lease_until=0).where((Scope.id == row.id) & (Scope.lease_owner == owner)).execute()
            if recovery_error:
                results.append({"scope_id": row.id, "published": False, "error": recovery_error})
                continue
            try:
                results.append(self.publisher.reconcile(row.id))
            except Exception as exc:
                results.append({"scope_id": row.id, "published": False, "error": str(exc)})
                logging.exception("Policy publication failed for scope %s", row.id)
        return results


async def maintain_policies(stop_event, interval=60):
    """Run independently of long downloads and wake promptly on worker shutdown."""
    while not stop_event.is_set():
        try:
            await asyncio.to_thread(PolicySyncService().tick)
        except Exception:
            logging.exception("Policy maintenance failed")
        await asyncio.to_thread(stop_event.wait, interval)
