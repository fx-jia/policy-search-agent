"""SQLite exercises the production policy model factory, without runtime services."""

import hashlib
import json
from datetime import datetime
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch
import unittest

from peewee import DateTimeField, Model, SqliteDatabase

from api.db.policy_models import _make_policy_models

# The existing package __init__ eagerly imports UserService and initializes DB.
# Isolate only that package entrypoint; execute the unmodified production modules.
_service_root = Path(__file__).resolve().parents[5] / "api/db/services"
_namespace = types.ModuleType("api.db.services")
_namespace.__path__ = [str(_service_root)]
with patch.dict(sys.modules, {"api.db.services": _namespace}):
    for _name in ("policy_catalog_service", "policy_sync_service", "policy_publication_service"):
        _full_name = "api.db.services." + _name
        _spec = importlib.util.spec_from_file_location(_full_name, _service_root / (_name + ".py"))
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_full_name] = _module
        _spec.loader.exec_module(_module)
        if _name == "policy_catalog_service":
            PolicyCatalogService = _module.PolicyCatalogService
        elif _name == "policy_sync_service":
            PolicySyncModule = _module
        else:
            PolicyPublicationService = _module.PolicyPublicationService


class MemoryObjects:
    def __init__(self):
        self.objects = {}

    def put(self, bucket, key, blob, tenant_id=None):
        self.objects[(tenant_id, bucket, key)] = bytes(blob)

    def get(self, bucket, key, tenant_id=None):
        return self.objects[(tenant_id, bucket, key)]


def capture(text="body", key="policy", errors=None, revision_hash=None):
    blob = text.encode()
    return {
        "policy_key": key,
        "source_url": "https://ybj.sh.gov.cn/gfxwj/policy.html",
        "source_kind": "gfxwj",
        "title": "Policy",
        "document_number": "沪医保1号",
        "published_at": "2026-01-01",
        "signed_at": None,
        "observed_at": "2026-10-10T00:00:00+00:00",
        "raw_html": b"<p>body</p>",
        "text": text,
        "parse_build_key": "build-1",
        "assets": [{"role": "body", "url": "https://ybj.sh.gov.cn/gfxwj/policy.html", "extension": ".txt", "blob": blob, "sha256": hashlib.sha256(blob).hexdigest()}],
        "errors": errors or [],
        "revision_hash": revision_hash or (None if errors else hashlib.sha256(blob).hexdigest()),
    }


class CatalogFixture(unittest.TestCase):
    def setUp(self):
        self.db = SqliteDatabase(":memory:")
        db = self.db

        class Base(Model):
            create_date = DateTimeField(default=datetime.now)

            class Meta:
                database = db

        self.models = _make_policy_models(Base)
        self.db.create_tables(self.models)
        self.catalog = PolicyCatalogService(self.models)
        self.objects = MemoryObjects()
        self.catalog.object_store = self.objects
        self.scope = self.catalog.ensure_scope("tenant", "kb", "connector", {"themes": ["medical"]})
        self.sid = self.scope["id"]

    def tearDown(self):
        self.db.close()

    def observe(self, value=None):
        return self.catalog.observe(self.sid, self.catalog.begin_scan(self.sid), value or capture())


class CatalogCase(CatalogFixture):
    def test_failed_attachment_persists_candidate_without_hash_and_marks_pending(self):
        row = self.observe(capture(errors=["attachment unavailable"]))
        self.assertIsNone(row["revision_hash"])
        self.assertIn("attachment unavailable", row["errors"])
        self.assertEqual(len(self.catalog.pending(self.sid)), 1)
        restarted = PolicyCatalogService(self.models)
        self.assertEqual(restarted.pending(self.sid)[0]["revision_id"], row["revision_id"])
        self.assertEqual(self.catalog.get_scope(self.sid)["source_heads"]["policy"]["observation_id"], row["id"])

    def test_candidate_exists_before_archive_failure(self):
        class Broken(MemoryObjects):
            def put(inner, bucket, key, blob, tenant_id=None):
                self.assertEqual(self.models[1].select().count(), 1)
                self.assertIsNone(self.models[1].select().get().revision_hash)
                raise OSError("storage offline")

        self.catalog.object_store = Broken()
        row = self.observe()
        self.assertEqual(row["state"], "failed")
        self.assertEqual(row["assets"][0]["role"], "body")
        self.assertTrue(any("storage offline" in e for e in row["errors"]))

    def test_sealed_duplicate_reuses_revision_and_documents(self):
        first = self.observe()
        self.catalog.bind_documents(self.sid, first["revision_id"], [{"doc_id": "doc", "role": "body", "sha256": capture()["assets"][0]["sha256"], "parse_build_key": "build-1"}])
        second = self.observe()
        self.assertEqual(first["revision_id"], second["revision_id"])
        self.assertEqual(self.models[1].select().where(self.models[1].revision_hash.is_null(False)).count(), 1)
        self.assertEqual(second["parsing_builds"]["build-1"][0]["doc_id"], "doc")

    def test_old_observation_cannot_replace_new_source_head(self):
        older, newer = self.catalog.begin_scan(self.sid), self.catalog.begin_scan(self.sid)
        new = self.catalog.observe(self.sid, newer, capture("new"))
        self.catalog.observe(self.sid, older, capture("old"))
        head = self.catalog.get_scope(self.sid)["source_heads"]["policy"]
        self.assertEqual(head["revision_id"], new["revision_id"])
        self.assertEqual(head["capture_sequence"], newer)

    def test_bind_refuses_foreign_scope_and_hash(self):
        row = self.observe()
        foreign = self.catalog.ensure_scope("other", "other-kb", "other-connector")
        with self.assertRaises(ValueError):
            self.catalog.bind_documents(foreign["id"], row["revision_id"], [{"doc_id": "x", "role": "body", "sha256": "wrong"}])
        with self.assertRaises(ValueError):
            self.catalog.bind_documents(self.sid, row["revision_id"], [{"doc_id": "x", "role": "body", "sha256": "wrong"}])

    def test_scan_failure_does_not_advance_last_success(self):
        self.catalog.record_scan_result(self.sid, True)
        success = self.catalog.get_scope(self.sid)["last_scan_success_at"]
        self.catalog.record_scan_result(self.sid, False, "missing page")
        scope = self.catalog.get_scope(self.sid)
        self.assertEqual(scope["last_scan_success_at"], success)
        self.assertEqual(scope["last_error"], "missing page")
        self.assertEqual(self.catalog.status(self.sid)["covered_policy_count"], 0)

    def test_review_requires_source_evidence_and_advances_generation(self):
        row = self.observe(capture("本通知自2026年1月1日起实施"))
        before = self.catalog.get_scope(self.sid)["generation"]
        with self.assertRaises(ValueError):
            self.catalog.review_revision(row["revision_id"], {"status": "confirmed", "effective_from": "2026-01-01"})
        reviewed = self.catalog.review_revision(row["revision_id"], {"status": "confirmed", "effective_from": "2026-01-01", "evidence": "本通知自2026年1月1日起实施", "reviewed_by": "maintainer"})
        self.assertEqual(reviewed["temporal_metadata"]["status"], "confirmed")
        self.assertGreater(self.catalog.get_scope(self.sid)["generation"], before)
        with self.assertRaises(ValueError):
            self.catalog.review_revision(row["revision_id"], {"status": "confirmed", "effective_from": "2027-01-01", "effective_to": "2026-01-01", "evidence": "dates", "reviewed_by": "maintainer"})

    def test_inherited_model_timestamps_remain_json_compatible(self):
        self.observe(capture(errors=["pending attachment"]))
        json.dumps(self.catalog.status(self.sid))

    def test_review_rejects_quote_absent_from_archived_source(self):
        row = self.observe()
        with self.assertRaises(ValueError):
            self.catalog.review_revision(row["revision_id"], {"status": "confirmed", "effective_from": "2026-01-01", "evidence": "invented quote", "reviewed_by": "maintainer"})

    def test_deferred_discovery_is_not_reported_as_covered_policy(self):
        self.observe(capture())
        self.observe(capture(key="deferred", errors=["policy_limit: deferred"]))
        status = self.catalog.status(self.sid)
        self.assertEqual(status["covered_policy_count"], 1)
        self.assertEqual(status["discovered_policy_count"], 2)

    def test_explicit_related_urls_are_associations_and_survive_restart(self):
        url = "https://ybj.sh.gov.cn/zcjd/interpretation.html"
        row = self.observe(dict(capture(), related_urls=[url, url]))
        self.assertEqual(self.models[1].get_by_id(row["revision_id"]).relations, [{"type": "association", "url": url}])

    def test_covered_sources_ignore_201_deferred_and_keep_prior_admission(self):
        admitted = self.observe()
        sequence = self.catalog.begin_scan(self.sid)
        for i in range(201):
            value = capture(key=f"deferred-{i}", errors=["policy_limit: deferred"])
            value.update(assets=[], source_url=f"https://ybj.sh.gov.cn/gfxwj/deferred-{i}.html")
            self.catalog.observe(self.sid, sequence, value)
        self.observe(dict(capture(errors=["detail failed"]), assets=[]))
        self.assertEqual(self.catalog.covered_sources(self.sid), {"covered_urls": [admitted["source_url"]], "covered_document_numbers": ["沪医保1号"]})

    def test_covered_sources_fail_loudly_above_admitted_limit(self):
        sequence = self.catalog.begin_scan(self.sid)
        for i in range(201):
            value = capture(text=f"body-{i}", key=f"policy-{i}")
            value["source_url"] = f"https://ybj.sh.gov.cn/gfxwj/{i}.html"
            self.catalog.observe(self.sid, sequence, value)
        with self.assertRaisesRegex(ValueError, "200"):
            self.catalog.covered_sources(self.sid)

    def test_partial_scan_preserves_strict_freshness_and_tracks_discovery_observation(self):
        self.catalog.record_scan_result(self.sid, True, scan_health={"discovery_complete": True, "observation_complete": True, "scan_complete": True})
        initial = self.catalog.get_scope(self.sid)
        self.models[0].update(next_retry_at="2020-01-01T00:00:00+00:00", retry_count=2).where(self.models[0].id == self.sid).execute()
        health = {"discovery_complete": True, "observation_complete": True, "scan_complete": False, "deferred_count": 3}
        self.catalog.record_scan_result(self.sid, False, "asset_limit: deferred", scan_health=health, retryable=False)
        current = self.catalog.get_scope(self.sid)
        self.assertEqual(current["last_scan_success_at"], initial["last_scan_success_at"])
        self.assertEqual(current["last_error"], "asset_limit: deferred")
        self.assertEqual(current["retry_count"], 0)
        self.assertIsNone(current["next_retry_at"])
        self.assertEqual(current["coverage_manifest"]["themes"], ["medical"])
        self.assertEqual(current["coverage_manifest"]["scan_health"], health)
        self.assertGreaterEqual(current["coverage_manifest"]["last_discovery_success_at"], initial["coverage_manifest"]["last_discovery_success_at"])
        self.assertGreaterEqual(current["coverage_manifest"]["last_observation_success_at"], initial["coverage_manifest"]["last_observation_success_at"])
        from datetime import timedelta

        scheduled = datetime.fromisoformat(current["next_scan_at"])
        self.assertLess(abs((scheduled - datetime.fromisoformat(current["coverage_manifest"]["last_observation_success_at"])) - timedelta(hours=24)), timedelta(seconds=1))

    def test_failed_scan_health_retains_prior_success_timestamps_and_complete_refreshes(self):
        self.catalog.record_scan_result(self.sid, True, scan_health={"discovery_complete": True, "observation_complete": True})
        previous = self.catalog.get_scope(self.sid)
        self.catalog.record_scan_result(self.sid, False, "HTTP failure", scan_health={"discovery_complete": False, "observation_complete": False})
        failed = self.catalog.get_scope(self.sid)
        self.assertEqual(failed["retry_count"], 1)
        for key in ("last_discovery_success_at", "last_observation_success_at"):
            self.assertEqual(failed["coverage_manifest"][key], previous["coverage_manifest"][key])
        self.assertEqual(failed["last_scan_success_at"], previous["last_scan_success_at"])
        self.catalog.record_scan_result(self.sid, True, scan_health={"discovery_complete": True, "observation_complete": True, "scan_complete": True})
        recovered = self.catalog.get_scope(self.sid)
        self.assertGreater(recovered["last_scan_success_at"], failed["last_scan_success_at"])
        self.assertEqual(recovered["retry_count"], 0)
        self.assertIsNone(recovered["last_error"])
