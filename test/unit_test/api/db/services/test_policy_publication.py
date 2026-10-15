"""Release and isolation contracts on SQLite and explicit external-service adapters."""

import hashlib
import json

import xxhash
from peewee import DateTimeField

from test.unit_test.api.db.services.test_policy_catalog import CatalogFixture, PolicyPublicationService, PolicySyncModule, capture


FIXTURE_KB = {
    "parser_id": "naive",
    "parser_config": {},
    "embd_id": "embed",
    "tenant_embd_id": "model-id",
    "embedding_config": {"llm_factory": "Fixture", "llm_name": "embed", "api_base": "https://provider.test/v1"},
}
BUILD_CONFIG = PolicySyncModule._parse_build_config(FIXTURE_KB)
BUILD = PolicySyncModule.parse_build_key(FIXTURE_KB)
EMBEDDING = PolicySyncModule.canonical_embedding_config(FIXTURE_KB["embedding_config"])


class PublicationCase(CatalogFixture):
    def setUp(self):
        super().setUp()
        self.documents = {}
        self.chunks = {}
        self.publisher = PolicyPublicationService(self.catalog, self.documents.get, lambda doc: self.chunks.get(doc, []), self.objects)

    def _track_revision_writes(self):
        # Native BaseModel refreshes update_date on every row update. Preserve
        # that observable contract without importing the application database.
        self.models[1]._meta.add_field("update_date", DateTimeField(null=True))
        self.db.execute_sql('ALTER TABLE "policy_revision" ADD COLUMN "update_date" DATETIME')
        self.db.execute_sql('CREATE TABLE "revision_writes" ("revision_id" TEXT)')
        self.db.execute_sql("""
            CREATE TRIGGER revision_state_write AFTER UPDATE OF state, errors ON policy_revision
            BEGIN
                INSERT INTO revision_writes VALUES (NEW.id);
                UPDATE policy_revision SET update_date = datetime(
                    '2026-10-11', '+' || (SELECT COUNT(*) FROM revision_writes) || ' seconds'
                ) WHERE id = NEW.id;
            END
        """)

    def _revision_write_counts(self):
        return dict(self.db.execute_sql("SELECT revision_id, COUNT(*) FROM revision_writes GROUP BY revision_id").fetchall())

    def ready(self, text="body", key="policy", doc="doc", metadata=None):
        value = dict(capture(text, key), parse_build_key=BUILD)
        if metadata:
            value.update(metadata)
        row = self.observe(value)
        self.catalog.bind_documents(
            self.sid,
            row["revision_id"],
            [
                {
                    "doc_id": doc,
                    "role": "body",
                    "sha256": value["assets"][0]["sha256"],
                    "parse_build_key": BUILD,
                    "parse_build_config": BUILD_CONFIG,
                    "embedding_model": EMBEDDING,
                    "embd_id": "embed",
                    "tenant_embd_id": "model-id",
                }
            ],
        )
        self.documents[doc] = {
            "id": doc,
            "kb_id": "kb",
            "tenant_id": "tenant",
            "run": "3",
            "progress": 1,
            "chunk_num": 1,
            "content_hash": xxhash.xxh128(text.encode()).hexdigest(),
            "embedding_config": EMBEDDING,
        }
        self.chunks[doc] = [{"id": "chunk-" + doc, "doc_id": doc, "content_with_weight": text, "docnm_kwd": "Policy", "position_int": [0], "q_2_vec": [0.1, 0.2]}]
        return row

    def test_done_but_empty_index_never_publishes(self):
        self.ready()
        self.chunks["doc"] = []
        result = self.publisher.reconcile(self.sid)
        self.assertFalse(result["published"])
        self.assertEqual(self.publisher.acquire_snapshot("tenant", "kb")["doc_ids"], [])
        self.assertTrue(self.catalog.pending(self.sid))

    def test_repeated_validation_failure_preserves_revision_and_release(self):
        self._track_revision_writes()
        self.ready()
        self.publisher.reconcile(self.sid)
        failed = self.ready("pending", "pending", "pending")
        self.documents["pending"].update(run="1", progress=0, chunk_num=0)
        self.assertTrue(self.publisher.reconcile(self.sid)["published"])
        revision = self.catalog.Revision.get_by_id(failed["revision_id"]).__data__.copy()
        scope = self.catalog.get_scope(self.sid)
        pending = self.catalog.pending(self.sid)
        writes = self._revision_write_counts()

        result = self.publisher.reconcile(self.sid)

        self.assertFalse(result["published"])
        self.assertEqual(self.catalog.Revision.get_by_id(failed["revision_id"]).__data__, revision)
        self.assertEqual(self.catalog.pending(self.sid), pending)
        self.assertEqual(self._revision_write_counts(), writes)
        self.assertEqual(self.catalog.get_scope(self.sid)["generation"], scope["generation"])
        self.assertEqual(result["release_id"], scope["active_release_id"])

    def test_changed_validation_error_updates_only_failed_revision(self):
        self._track_revision_writes()
        published = self.ready()
        self.publisher.reconcile(self.sid)
        failed = self.ready("pending", "pending", "pending")
        self.documents["pending"].update(run="1", progress=0)
        self.publisher.reconcile(self.sid)
        before = self._revision_write_counts()
        old_date = self.catalog.Revision.get_by_id(failed["revision_id"]).update_date
        self.documents["pending"].update(run="3", progress=1)
        self.chunks["pending"] = []

        self.assertTrue(self.publisher.reconcile(self.sid)["published"])

        after = self._revision_write_counts()
        row = self.catalog.Revision.get_by_id(failed["revision_id"])
        self.assertEqual(row.state, "failed")
        self.assertEqual(row.errors, ["document index is empty or incomplete"])
        self.assertNotEqual(row.update_date, old_date)
        self.assertEqual(after[failed["revision_id"]], before[failed["revision_id"]] + 1)
        self.assertEqual(after[published["revision_id"]], before[published["revision_id"]])

    def test_successful_revalidation_clears_published_error_without_new_release(self):
        self._track_revision_writes()
        row = self.ready()
        release = self.publisher.reconcile(self.sid)["release_id"]
        self.documents["doc"]["progress"] = 0.5
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])
        failed = self.catalog.Revision.get_by_id(row["revision_id"])
        self.assertEqual(failed.state, "published")
        self.assertEqual(failed.errors, ["document parse is incomplete or failed"])
        before = self._revision_write_counts()
        old_date = failed.update_date
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])
        self.assertEqual(self._revision_write_counts(), before)
        self.assertEqual(self.catalog.Revision.get_by_id(row["revision_id"]).update_date, old_date)
        generation = self.catalog.get_scope(self.sid)["generation"]
        self.documents["doc"]["progress"] = 1

        result = self.publisher.reconcile(self.sid)

        recovered = self.catalog.Revision.get_by_id(row["revision_id"])
        self.assertFalse(result["published"])
        self.assertEqual(result["release_id"], release)
        self.assertEqual(recovered.state, "published")
        self.assertEqual(recovered.errors, [])
        self.assertNotEqual(recovered.update_date, old_date)
        self.assertEqual(self._revision_write_counts()[row["revision_id"]], before[row["revision_id"]] + 1)
        self.assertEqual(self.catalog.get_scope(self.sid)["generation"], generation)
        recovered_data, writes = recovered.__data__.copy(), self._revision_write_counts()
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])
        self.assertEqual(self.catalog.Revision.get_by_id(row["revision_id"]).__data__, recovered_data)
        self.assertEqual(self._revision_write_counts(), writes)

    def test_first_publication_writes_state_and_generation_conflict_keeps_candidate_unpublished(self):
        self._track_revision_writes()
        row = self.ready()
        before = self._revision_write_counts()
        self.assertTrue(self.publisher.reconcile(self.sid)["published"])
        self.assertEqual(self.catalog.Revision.get_by_id(row["revision_id"]).state, "published")
        self.assertEqual(self._revision_write_counts()[row["revision_id"]], before[row["revision_id"]] + 1)
        candidate = self.ready("new", "new", "new")

        def changed(doc):
            if doc == "new":
                self.catalog.Scope.update(generation=self.catalog.Scope.generation + 1).where(self.catalog.Scope.id == self.sid).execute()
            return self.chunks[doc]

        self.publisher.chunk_reader = changed
        result = self.publisher.reconcile(self.sid)
        self.assertEqual(result["reason"], "generation_conflict")
        self.assertEqual(self.catalog.Revision.get_by_id(candidate["revision_id"]).state, "queued")

    def test_snapshot_is_immutable_and_tenant_isolated(self):
        row = self.ready()
        self.assertTrue(self.publisher.reconcile(self.sid)["published"])
        before = self.publisher.acquire_snapshot("tenant", "kb")
        self.models[1].update(title="tampered", temporal_metadata={"status": "confirmed"}).where(self.models[1].id == row["revision_id"]).execute()
        after = self.publisher.acquire_snapshot("tenant", "kb")
        self.assertEqual(after["sources"]["doc"]["title"], "Policy")
        self.assertEqual(after["sources"], before["sources"])
        self.assertEqual(after["chunks"]["chunk-doc"]["sha256"], hashlib.sha256(b"body").hexdigest())
        self.assertEqual(self.publisher.acquire_snapshot("intruder", "kb")["doc_ids"], [])

    def test_failed_group_retains_old_release_other_group_advances(self):
        self.ready()
        self.publisher.reconcile(self.sid)
        self.observe(capture("new body", errors=["missing attachment"]))
        self.ready("second", "policy-2", "doc-2")
        self.publisher.reconcile(self.sid)
        snapshot = self.publisher.acquire_snapshot("tenant", "kb")
        self.assertEqual(set(snapshot["doc_ids"]), {"doc", "doc-2"})
        self.assertTrue(snapshot["pending"])

    def test_late_old_ready_does_not_replace_newer_desired(self):
        self.ready("old", doc="old")
        self.ready("new", doc="new")
        self.publisher.reconcile(self.sid)
        self.assertEqual(self.publisher.acquire_snapshot("tenant", "kb")["doc_ids"], ["new"])

    def test_generation_cas_rolls_back_release_if_observation_changes_during_read(self):
        self.ready()

        def changed(doc):
            self.observe(capture("later", errors=["attachment pending"]))
            return self.chunks[doc]

        publisher = PolicyPublicationService(self.catalog, self.documents.get, changed, self.objects)
        result = publisher.reconcile(self.sid)
        self.assertFalse(result["published"])
        self.assertEqual(self.models[2].select().count(), 0)

    def test_incomplete_attachment_bindings_do_not_publish(self):
        value = capture()
        value["assets"].append({"role": "attachment:0", "url": "https://ybj.sh.gov.cn/cmsres/a.pdf", "blob": b"pdf", "sha256": hashlib.sha256(b"pdf").hexdigest(), "extension": ".pdf"})
        row = self.observe(value)
        self.catalog.bind_documents(self.sid, row["revision_id"], [{"doc_id": "doc", "role": "body", "sha256": value["assets"][0]["sha256"]}])
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])

    def test_hash_mismatch_and_foreign_document_do_not_publish(self):
        self.ready()
        self.documents["doc"]["content_hash"] = "wrong"
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])
        self.documents["doc"]["content_hash"] = xxhash.xxh128(b"body").hexdigest()
        self.documents["doc"]["kb_id"] = "another-kb"
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])

    def test_ambiguous_positions_disable_neighbors(self):
        self.ready()
        self.documents["doc"]["chunk_num"] = 2
        self.chunks["doc"].append({"id": "chunk-2", "doc_id": "doc", "content_with_weight": "more", "position_int": [0], "q_2_vec": [0.1, 0.2]})
        self.publisher.reconcile(self.sid)
        snap = self.publisher.acquire_snapshot("tenant", "kb")
        self.assertTrue(snap["chunks"])
        self.assertTrue(all(not c["ordered"] and c["order"] is None for c in snap["chunks"].values()))

    def test_rollback_revalidates_index_and_preserves_pointer_on_failure(self):
        self.ready("old", doc="old")
        old = self.publisher.reconcile(self.sid)["release_id"]
        self.ready("new", doc="new")
        new = self.publisher.reconcile(self.sid)["release_id"]
        self.chunks["old"] = []
        with self.assertRaises(ValueError):
            self.publisher.rollback(self.sid, old)
        self.assertEqual(self.catalog.get_scope(self.sid)["active_release_id"], new)

    def test_known_future_and_expired_are_filtered_unknown_retained(self):
        self.ready("future", "future", "future", {"temporal_metadata": {"status": "confirmed", "effective_from": "2027-01-01"}})
        self.ready("expired", "expired", "expired", {"temporal_metadata": {"status": "confirmed", "effective_to": "2025-12-31"}})
        self.ready("unknown", "unknown", "unknown")
        self.publisher.reconcile(self.sid)
        self.assertEqual(self.publisher.acquire_snapshot("tenant", "kb", "2026-10-10")["doc_ids"], ["unknown"])

    def test_published_document_reparse_cannot_mutate_release_chunk_hash(self):
        self.ready()
        first = self.publisher.reconcile(self.sid)["release_id"]
        self.chunks["doc"][0]["content_with_weight"] = "changed after publication"
        result = self.publisher.reconcile(self.sid)
        self.assertFalse(result["published"])
        self.assertEqual(self.catalog.get_scope(self.sid)["active_release_id"], first)

    def test_review_creates_new_release_without_rewriting_old_metadata(self):
        row = self.ready("本通知自2026年1月1日起实施")
        old = self.publisher.reconcile(self.sid)["release_id"]
        self.catalog.review_revision(row["revision_id"], {"status": "confirmed", "effective_from": "2026-01-01", "evidence": "本通知自2026年1月1日起实施", "reviewed_by": "maintainer"})
        new = self.publisher.reconcile(self.sid)["release_id"]
        self.assertNotEqual(old, new)
        self.assertEqual(self.models[2].get_by_id(old).document_manifest["doc"]["confirmation_status"], "unknown")
        self.assertEqual(self.publisher.acquire_snapshot("tenant", "kb")["sources"]["doc"]["confirmation_status"], "confirmed")

    def test_publication_lease_prevents_competing_publisher_then_expires(self):
        self.ready()
        self.models[0].update(lease_owner="other", lease_until=99999999999).where(self.models[0].id == self.sid).execute()
        self.assertEqual(self.publisher.reconcile(self.sid)["reason"], "lease_busy")
        self.models[0].update(lease_until=0).where(self.models[0].id == self.sid).execute()
        self.assertTrue(self.publisher.reconcile(self.sid)["published"])

    def test_missing_raw_capture_archive_blocks_publication(self):
        row = self.ready()
        key = row["raw_object_keys"]["html"]["key"]
        del self.objects.objects[("tenant", "kb", key)]
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])

    def test_partial_done_and_index_count_mismatch_never_publish(self):
        self.ready()
        self.documents["doc"]["progress"] = 0.5
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])
        self.documents["doc"]["progress"] = 1
        self.documents["doc"]["chunk_num"] = 2
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])

    def test_parser_configuration_drift_requires_new_build(self):
        row = self.ready()
        stored = self.models[1].get_by_id(row["revision_id"]).parsing_builds
        stored[BUILD][0].update(parser_id="naive", parser_config={"chunk_token_num": 128})
        self.models[1].update(parsing_builds=stored).where(self.models[1].id == row["revision_id"]).execute()
        self.documents["doc"].update(parser_id="naive", parser_config={"chunk_token_num": 256})
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])

    def test_missing_or_invalid_index_vector_blocks_publication(self):
        self.ready()
        for vector in (None, [], [float("nan"), 0.2], [0.1], [True, 0.2]):
            self.chunks["doc"][0]["q_2_vec"] = vector
            self.assertFalse(self.publisher.reconcile(self.sid)["published"])

    def test_inconsistent_index_dimensions_block_publication(self):
        self.ready()
        self.documents["doc"]["chunk_num"] = 2
        self.chunks["doc"].append({"id": "chunk-2", "doc_id": "doc", "content_with_weight": "more", "q_3_vec": [0.1, 0.2, 0.3]})
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])

    def test_release_freezes_embedding_config_dimension_and_association(self):
        relation = "https://ybj.sh.gov.cn/zcjd/interpretation.html"
        self.ready(metadata={"related_urls": [relation]})
        self.assertTrue(self.publisher.reconcile(self.sid)["published"])
        source = self.publisher.acquire_snapshot("tenant", "kb")["sources"]["doc"]
        self.assertEqual(source["embedding_dimension"], 2)
        self.assertEqual(source["embedding_config"]["llm_name"], "embed")
        self.assertEqual(source["embedding_config_sha256"], hashlib.sha256(json.dumps(source["embedding_config"], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest())
        self.assertEqual(source["relations"], [{"type": "association", "url": relation}])

    def test_embedding_binding_or_current_config_drift_blocks_publication(self):
        row = self.ready()
        self.documents["doc"]["embedding_config"] = {"llm_factory": "Fixture", "llm_name": "different"}
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])
        self.documents["doc"].pop("embedding_config")
        stored = self.models[1].get_by_id(row["revision_id"]).parsing_builds
        binding = next(iter(stored.values()))[0]
        binding["embedding_model"] = {"llm_factory": "Fixture", "llm_name": "different"}
        self.models[1].update(parsing_builds=stored).where(self.models[1].id == row["revision_id"]).execute()
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])

    def test_missing_embedding_build_proof_never_publishes(self):
        row = self.ready()
        stored = self.models[1].get_by_id(row["revision_id"]).parsing_builds
        stored[BUILD][0].pop("parse_build_config")
        self.models[1].update(parsing_builds=stored).where(self.models[1].id == row["revision_id"]).execute()
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])

    def test_frozen_embedding_identity_is_immutable_and_rollback_checks_it(self):
        self.ready()
        release = self.publisher.reconcile(self.sid)["release_id"]
        before = self.publisher.acquire_snapshot("tenant", "kb")["sources"]["doc"]
        self.documents["doc"]["embedding_config"] = dict(EMBEDDING, llm_name="changed-behind-same-id")
        self.assertFalse(self.publisher.reconcile(self.sid)["published"])
        self.assertEqual(self.publisher.acquire_snapshot("tenant", "kb")["sources"]["doc"], before)
        with self.assertRaisesRegex(ValueError, "embedding"):
            self.publisher.rollback(self.sid, release)

    def test_native_vector_discovery_reads_real_fields_including_es9(self):
        # Execute native Retriever.chunk_list and ES get_fields, with just the
        # database HTTP boundary replaced. No assumed vector dimension is supplied.
        import ast
        from collections import UserDict
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import common
        from common.doc_store.doc_store_base import OrderByExpr

        root = Path(__file__).resolve().parents[5]

        def native_method(path, class_name, name, namespace):
            tree = ast.parse((root / path).read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), path, "exec"), namespace)
            return namespace[name]

        def index_name(tenant):
            return "ragflow_" + tenant

        chunk_list = native_method("rag/nlp/search.py", "Dealer", "chunk_list", {"OrderByExpr": OrderByExpr, "index_name": index_name})
        get_fields = native_method("rag/utils/es_conn.py", "ESConnection", "get_fields", {})

        class SDKResponse:
            def __init__(self, body):
                self.body = UserDict(body)

            def get(self, key, default=None):
                return self.body.get(key, default)

        class GetResponse:
            def __init__(self, body):
                self._body = body

            def get(self, key, default=None):
                return self._body.get(key, default)

        self.ready()
        for es9, response_type in ((False, dict), (True, dict), (True, UserDict), (True, SDKResponse), (True, GetResponse)):
            searches = []

            def search(fields, highlights, condition, matches, order, offset, limit, idx, kbs):
                searches.append(fields)
                raw = dict(self.chunks["doc"][0], kb_id=["kb"])
                vec = raw.pop("q_2_vec")
                return response_type({"hits": {"hits": [{"_id": raw["id"], "_source": raw if es9 else dict(raw, q_2_vec=vec), "fields": {"q_2_vec": [vec]} if es9 else {}}]}})

            store = SimpleNamespace(search=search, get=lambda *a: {} if es9 else self.chunks["doc"][0], get_fields=lambda result, fields: get_fields(None, result, fields))
            retriever = SimpleNamespace(dataStore=store)
            retriever.chunk_list = lambda *a, **kw: chunk_list(retriever, *a, **kw)
            settings = SimpleNamespace(retriever=retriever, docStoreConn=store)
            with patch.object(common, "settings", settings, create=True), patch.dict(sys.modules, {"rag.nlp.search": SimpleNamespace(index_name=index_name)}):
                publisher = PolicyPublicationService(self.catalog, self.documents.get, object_store=self.objects)
                rows = publisher._chunks(self.catalog.get_scope(self.sid), "doc")
                self.assertEqual(publisher._embedding_dimension(rows), 2)
                self.assertIn("q_2_vec", searches[-1])
                if es9:
                    self.assertIn(["q_*_vec"], searches)

    def test_document_reference_and_related_link_are_published_without_legal_inference(self):
        url = "https://ybj.sh.gov.cn/zcjd/interpretation.html"
        reference = {"kind": "association_reference", "document_number": "沪医保规〔2024〕1号", "source_span": "涉及沪医保规〔2024〕1号文件，具体效力待核实。", "status": "needs_review"}
        link = {"type": "association", "url": url}
        row = self.ready(text=reference["source_span"], metadata={"relations": [reference, dict(reversed(list(reference.items()))), link], "related_urls": [url, url]})
        self.assertEqual(self.models[1].get_by_id(row["revision_id"]).relations, [reference, link])
        self.assertTrue(self.publisher.reconcile(self.sid)["published"])
        source = self.publisher.acquire_snapshot("tenant", "kb")["sources"]["doc"]
        self.assertEqual(source["relations"], [reference, link])
        self.assertEqual(source["confirmation_status"], "unknown")

    def test_release_native_manifest_omits_vectors_but_retains_positions_and_dimension(self):
        self.ready()
        self.assertTrue(self.publisher.reconcile(self.sid)["published"])
        snapshot = self.publisher.acquire_snapshot("tenant", "kb")
        native = snapshot["chunks"]["chunk-doc"]["native"]
        self.assertNotIn("q_2_vec", native)
        self.assertEqual(native["content_with_weight"], "body")
        self.assertEqual(native["position_int"], [0])
        self.assertEqual(snapshot["sources"]["doc"]["embedding_dimension"], 2)
        self.assertEqual(self.chunks["doc"][0]["q_2_vec"], [0.1, 0.2])

    def test_capture_skips_only_complete_exact_published_revision_and_build(self):
        row = self.ready()
        self.publisher.reconcile(self.sid)
        service = PolicySyncModule.PolicySyncService(self.catalog, self.publisher)

        def observed(value, build=BUILD):
            return service.capture(self.sid, self.catalog.begin_scan(self.sid), value, build)

        self.assertEqual(observed(capture())["policy_ingestion_required"], "0")
        self.assertEqual(observed(capture(), "new-build")["policy_ingestion_required"], "1")
        self.assertEqual(observed(capture("updated"))["policy_ingestion_required"], "1")
        # A published pointer with an incomplete manifest cannot suppress work.
        release_id = self.catalog.get_scope(self.sid)["active_release_id"]
        self.catalog.Release.update(document_manifest={}).where(self.catalog.Release.id == release_id).execute()
        self.assertEqual(observed(capture())["policy_ingestion_required"], "1")
        self.assertEqual(self.models[1].get_by_id(row["revision_id"]).state, "published")

    def test_capture_without_publication_does_not_skip_bound_assets(self):
        self.ready()
        service = PolicySyncModule.PolicySyncService(self.catalog, self.publisher)
        result = service.capture(self.sid, self.catalog.begin_scan(self.sid), capture(), BUILD)
        self.assertEqual(result["policy_ingestion_required"], "1")

    def test_capture_budget_does_not_charge_published_assets_and_persists_deferred_once(self):
        self.ready()
        self.publisher.reconcile(self.sid)
        service = PolicySyncModule.PolicySyncService(self.catalog, self.publisher)
        unchanged = dict(capture(), _asset_budget_remaining=0)
        reused = service.capture(self.sid, self.catalog.begin_scan(self.sid), unchanged, BUILD)
        self.assertEqual(reused["policy_ingestion_required"], "0")
        self.assertFalse(unchanged["errors"])
        changed = dict(capture("new body"), _asset_budget_remaining=0)
        puts = []
        original_put = self.objects.put

        def put(*args, **kwargs):
            puts.append(args[1])
            return original_put(*args, **kwargs)

        self.objects.put = put
        deferred = service.capture(self.sid, self.catalog.begin_scan(self.sid), changed, BUILD)
        self.assertEqual(deferred["policy_ingestion_required"], "1")
        self.assertIsNone(changed["revision_hash"])
        self.assertIn("asset_limit: deferred", changed["errors"])
        row = self.models[1].get_by_id(deferred["revision_id"])
        self.assertIsNone(row.revision_hash)
        self.assertEqual(row.state, "failed")
        self.assertIn("asset_limit: deferred", row.errors)
        self.assertEqual(len(puts), 3)  # html, cleaned text, one body asset, exactly once
        self.assertEqual(self.catalog.pending(self.sid)[0]["revision_id"], row.id)

    def test_capture_budget_new_build_and_changed_asset_require_work(self):
        self.ready()
        self.publisher.reconcile(self.sid)
        service = PolicySyncModule.PolicySyncService(self.catalog, self.publisher)
        value = dict(capture(), _asset_budget_remaining=0)
        result = service.capture(self.sid, self.catalog.begin_scan(self.sid), value, "new-build")
        self.assertEqual(result["policy_ingestion_required"], "1")
        self.assertIn("asset_limit: deferred", value["errors"])
        changed = dict(capture(), _asset_budget_remaining=0)
        changed["assets"][0]["sha256"] = "different"
        self.assertFalse(self.catalog.is_published_capture(self.sid, changed, BUILD))
