"""Controlled OFFLINE update chain; PDF parsing, index and model outputs are fixtures.

Real paths: collector HTTP boundary, production policy model factory on SQLite,
archive/readback, native connector ID resolver, binding, publication, conversation,
PolicyBackend, HTTP audit transports and usage ledger. No live provider or website.
"""

import ast
import asyncio
import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from api.utils.common import hash128
from experiments.search_agent.policy_transport import AsyncAuditTransport, SyncAuditTransport
from experiments.search_agent.resolution import resolution_messages, validate_resolution
from experiments.search_agent.test_policy_backend import render
from test.unit_test.api.db.services.test_policy_catalog import CatalogFixture, PolicyPublicationService
from test.unit_test.api.db.services.test_policy_publication import BUILD, BUILD_CONFIG, EMBEDDING
from test.unit_test.common.test_shanghai_medical_connector import ATTACH, DETAIL, LIST, article, collector, listing
from test.unit_test.policy.test_policy_search_service import ProviderFixture, load_service
from test.unit_test.policy.test_policy_sync import load as load_sync

pytest_plugins = ["test.unit_test.common.test_shanghai_medical_connector"]

ROOT = Path(__file__).resolve().parents[3]
QUOTE = "本通知自2026年1月1日起实施"
BODY = f"测试数据。第一条 医保费用按照附件执行。{QUOTE}。"
OLD_PDF = b"%PDF-1.7\nOFFLINE_PARSE_FIXTURE_OLD"
NEW_PDF = b"%PDF-1.7\nOFFLINE_PARSE_FIXTURE_NEW"
PARSED_PDFS = {OLD_PDF: "测试数据：旧附件规定费用为10元。", NEW_PDF: "测试数据：新附件规定费用为20元。"}


def native_doc_resolver():
    """Load the two real pure native ID functions without importing runtime DB."""
    path = ROOT / "api/db/services/connector_service.py"
    tree = ast.parse(path.read_text())
    names = {"connector_doc_id_candidates", "resolve_connector_doc_id"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"hash128": hash128}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), "exec"), namespace)
    return namespace["resolve_connector_doc_id"]


class UpdateChain:
    def __init__(self, rows, monkeypatch, tmp_path):
        self.fixture = CatalogFixture()
        self.fixture.setUp()
        # File SQLite shares committed releases with the service's I/O thread.
        self.fixture.db.close()
        self.fixture.db.init(str(tmp_path / "catalog.sqlite"))
        self.fixture.db.create_tables(self.fixture.models)
        self.fixture.catalog.ensure_scope("tenant", "kb", "connector", {"themes": ["medical"]})
        self.catalog, self.sid = self.fixture.catalog, self.fixture.sid
        self.documents, self.index, self.roles = {}, {}, {}
        self.publisher = PolicyPublicationService(self.catalog, self.documents.get, lambda doc: self.index.get(doc, []), self.fixture.objects)
        self.sync = load_sync().PolicySyncService(self.catalog, self.publisher)
        self.resolve_doc_id = native_doc_resolver()
        self.connector = collector.ShanghaiMedicalConnector.build_connector({"list_urls": [LIST], "seed_urls": [], "parse_build_key": BUILD})
        # HTTP itself is provided by the real collector's reusable boundary fixture.
        # Avoid its production pacing delay while keeping real monotonic time for asyncio.
        monkeypatch.setattr(self.connector, "_wait", lambda seconds: None)
        self.rows = rows
        rows[LIST] = (200, listing(), "text/html")
        rows[DETAIL] = (200, article(BODY, attachments=[ATTACH]), "text/html")
        rows[ATTACH] = (200, OLD_PDF, "application/pdf")
        self.latest_revision = None

    def close(self):
        self.connector.cancel()
        self.fixture.tearDown()

    def scan_and_fixture_parse(self):
        sequence = self.catalog.begin_scan(self.sid)

        def capture(value):
            result = self.sync.capture(self.sid, sequence, value, BUILD)
            self.latest_revision = result["revision_id"]
            return result

        self.connector.capture_sink = capture
        documents = []
        try:
            for batch in self.connector.load_from_state():
                documents.extend(batch)
        except BaseException as exc:
            self.sync.finish(self.sid, False, str(exc), scan_health=dict(self.connector.scan_health), retryable=True)
            raise
        self.sync.finish(
            self.sid,
            self.connector.scan_complete,
            "; ".join(self.connector.errors)[:4000] if not self.connector.scan_complete else None,
            scan_health=dict(self.connector.scan_health),
            retryable=bool(self.connector.retryable_errors),
        )
        for document in documents:
            metadata = document.metadata
            doc_id = self.resolve_doc_id("kb", "connector", document.id, set(self.documents))
            self.catalog.bind_documents(
                self.sid,
                metadata["revision_id"],
                [
                    {
                        "doc_id": doc_id,
                        "role": metadata["asset_role"],
                        "sha256": metadata["asset_sha256"],
                        "parse_build_key": metadata["parse_build_key"],
                        "fingerprint": document.fingerprint,
                        "parse_build_config": BUILD_CONFIG,
                        "embedding_model": EMBEDDING,
                        "embd_id": "embed",
                        "tenant_embd_id": "model-id",
                    }
                ],
            )
            # This is explicitly a parse/index fixture, not a PDF parser execution.
            text = PARSED_PDFS[document.blob] if document.extension == ".pdf" else document.blob.decode()
            chunk_id = hash128(f"offline-chunk:{doc_id}:{text}")
            self.documents[doc_id] = {
                "id": doc_id,
                "kb_id": "kb",
                "tenant_id": "tenant",
                "run": "3",
                "progress": 1,
                "chunk_num": 1,
                "content_hash": document.fingerprint,
                "embedding_config": EMBEDDING,
            }
            self.roles[doc_id] = metadata["asset_role"]
            self.index[doc_id] = [{"id": chunk_id, "doc_id": doc_id, "kb_id": "kb", "docnm_kwd": document.semantic_identifier, "content_with_weight": text, "q_2_vec": [0.1, 0.2]}]
        return self.publisher.reconcile(self.sid)


class AuditedIndexFixture(ProviderFixture):
    """Real HTTP auditing with deterministic provider/index boundary responses."""

    def __init__(self, chain):
        super().__init__()
        self.chain = chain
        self.kb = {"id": "kb", "tenant_id": "tenant", "embd_id": "offline-fixture", "tenant_embd_id": "fixture-model"}
        self.retrieval_scopes = []
        self.selected_text = ""
        self.on_retrieve = None

    def provider(self, request):
        response = super().provider(request)
        body = json.loads(request.content)
        if request.url.path.endswith("chat/completions") and "response_format" not in body:
            payload = response.json()
            payload["choices"][0]["message"]["content"] = self.selected_text + "##0$$"
            return httpx.Response(200, json=payload)
        return response

    async def factory(self, kb, audit):
        self.factories += 1
        generation = httpx.AsyncClient(transport=AsyncAuditTransport(httpx.MockTransport(self.provider), audit, "generation", "https://provider.test/v1", "chat"))
        embedding = httpx.Client(transport=SyncAuditTransport(httpx.MockTransport(self.provider), audit, "embedding", "https://provider.test/v1", "embed"))

        async def chat(system, messages, config):
            response = await generation.post("https://provider.test/v1/chat/completions", json={"model": "chat", "messages": [{"role": "system", "content": system}, *messages], **config})
            return response.json()["choices"][0]["message"]["content"]

        async def resolve(context):
            system, messages = resolution_messages(context)
            return validate_resolution(json.loads(await chat(system, messages, {"max_tokens": 600, "response_format": {"type": "json_object"}})), context)

        async def retrieve(query, doc_ids):
            self.retrieval_scopes.append(list(doc_ids))
            if self.on_retrieve:
                callback, self.on_retrieve = self.on_retrieve, None
                callback()  # Advance real publication after this backend bound its release.
            embedding.post("https://provider.test/v1/embeddings", json={"model": "embed", "input": query})
            attachment_ids = [doc for doc in doc_ids if self.chain.roles[doc] != "body"]
            assert attachment_ids, "test must retrieve a genuinely published attachment ID"
            selected = copy.deepcopy(self.chain.index[attachment_ids[0]][0])
            self.selected_text = selected["content_with_weight"]
            selected["chunk_id"] = selected["id"]
            return {"chunks": [selected]}

        async def close():
            self.closed += 1
            await generation.aclose()
            embedding.close()

        return {"chat": chat, "resolve": resolve, "retrieve": retrieve, "count_tokens": len, "kb_prompt": render, "close": close}


@pytest.mark.parametrize("change_body", [False, True], ids=["attachment-only-update", "same-url-body-and-attachment-update"])
def test_offline_capture_failure_recovery_and_next_turn_release(http, monkeypatch, tmp_path, change_body):
    rows, source_calls = http
    chain = UpdateChain(rows, monkeypatch, tmp_path)
    try:
        old_result = chain.scan_and_fixture_parse()
        assert old_result["published"]
        old = chain.publisher.acquire_snapshot("tenant", "kb", "2026-10-10")
        assert len(old["doc_ids"]) == 2
        assert all(source["confirmation_status"] == "unknown" for source in old["sources"].values())
        provider = AuditedIndexFixture(chain)
        service = load_service()("tenant", "kb", "offline-update-chain", tmp_path, publication=chain.publisher, kb_reader=provider.kb_reader, request_factory=provider.factory)

        async def conversation():
            first = await service.ask("测试数据中的医保附件费用是多少？", as_of="2026-10-10")
            assert first["result"]["status"] == "ok", first["result"]
            assert first["release_id"] == old["release_id"]
            assert first["applicability"] == "unknown"
            assert "时效尚未确认" in first["result"]["answer"]
            assert "10元" in first["result"]["answer"]
            assert first["result"]["stop_reason"] != "sufficient"

            if change_body:
                rows[DETAIL] = (200, article(BODY + "第二条 测试正文也已更新。", attachments=[ATTACH]), "text/html")
            rows[ATTACH] = (404, b"offline attachment unavailable", "text/plain")
            old_documents = copy.deepcopy(chain.documents)
            old_index = copy.deepcopy(chain.index)
            chain.scan_and_fixture_parse()
            # Discovery and durable observations finished, but the unavailable
            # attachment prevents strict freshness and this policy's replacement.
            health = chain.catalog.get_scope(chain.sid)["coverage_manifest"]["scan_health"]
            assert health["discovery_complete"] is True and health["observation_complete"] is True
            assert health["scan_complete"] is False and health["failed_count"] == 1
            assert health["ingestion_asset_count"] == 0
            assert not chain.connector.scan_complete
            assert chain.catalog.get_scope(chain.sid)["last_error"]
            failed = chain.publisher.acquire_snapshot("tenant", "kb", "2026-10-10")
            assert failed["pending"]
            assert failed["scanned_at"] == old["scanned_at"]
            assert failed["sources"] == old["sources"] and failed["chunks"] == old["chunks"]
            assert chain.documents == old_documents and chain.index == old_index
            # A coverage-only release is allowed; the original content manifest
            # itself remains immutable even if a new release pointer is created.
            original_release = chain.catalog.Release.get_by_id(old["release_id"])
            assert original_release.document_manifest == old["sources"]
            assert chain.publisher._manifest(chain.catalog.get_scope(chain.sid), original_release) == old["chunks"]

            def publish_during_old_turn():
                rows[ATTACH] = (200, NEW_PDF, "application/pdf")
                assert chain.scan_and_fixture_parse()["published"]
                assert chain.connector.scan_complete
                recovered_health = chain.catalog.get_scope(chain.sid)["coverage_manifest"]["scan_health"]
                assert recovered_health["scan_complete"] and recovered_health["ingestion_asset_count"] == 2

            provider.on_retrieve = publish_during_old_turn
            during = await service.ask("继续说明测试附件费用。")
            assert during["result"]["status"] == "ok", during["result"]
            assert during["release_id"] == failed["release_id"]
            assert during["pending"]
            assert "10元" in during["result"]["answer"] and "20元" not in during["result"]["answer"]
            new = chain.publisher.acquire_snapshot("tenant", "kb", "2026-10-10")
            assert new["release_id"] != old["release_id"]
            assert set(old["doc_ids"]).isdisjoint(new["doc_ids"])
            assert new["pending"] == []
            old_body = next(source for source in old["sources"].values() if source["asset_role"] == "body")
            new_body = next(source for source in new["sources"].values() if source["asset_role"] == "body")
            assert (old_body["asset_sha256"] != new_body["asset_sha256"]) is change_body
            old_attachment = next(source for source in old["sources"].values() if source["asset_role"] != "body")
            new_attachment = next(source for source in new["sources"].values() if source["asset_role"] != "body")
            assert old_attachment["asset_sha256"] != new_attachment["asset_sha256"]

            next_turn = await service.ask("更新后的测试附件费用是多少？")
            assert next_turn["result"]["status"] == "ok", next_turn["result"]
            assert next_turn["release_id"] == new["release_id"]
            assert "20元" in next_turn["result"]["answer"]
            cited = next_turn["result"]["reference"]["chunks"][0]
            assert cited["doc_id"] in new["doc_ids"] and cited["doc_id"] not in old["doc_ids"]
            assert hashlib.sha256(cited["content_with_weight"].encode()).hexdigest() == new["chunks"][cited["chunk_id"]]["sha256"]
            assert cited["url"] == DETAIL
            assert cited["document_metadata"]["asset_url"] == ATTACH
            assert next_turn["applicability"] == "unknown"

            chain.catalog.review_revision(chain.latest_revision, {"status": "confirmed", "effective_from": "2026-01-01", "evidence": QUOTE, "reviewed_by": "offline-fixture-maintainer"})
            reviewed = chain.publisher.reconcile(chain.sid)
            assert reviewed["published"] and reviewed["release_id"] != new["release_id"]
            reviewed_turn = await service.ask("已核对的测试附件费用是多少？")
            assert reviewed_turn["result"]["status"] == "ok", reviewed_turn["result"]
            assert reviewed_turn["applicability"] == "confirmed_as_of"
            assert reviewed_turn["release_id"] == reviewed["release_id"]
            assert chain.catalog.Release.get_by_id(new["release_id"]).document_manifest[cited["doc_id"]]["confirmation_status"] == "unknown"
            assert len(service.conversation.turns) == 4
            for result in (first, during, next_turn, reviewed_turn):
                assert result["ledger"]["pending_calls"] == []
                assert result["ledger"]["provider_total_tokens"] == result["ledger"]["provider_calls"] * 20
                assert result["ledger"]["provider_calls"] > 0
            records = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
            assert [record["release_id"] for record in records] == [old["release_id"], failed["release_id"], new["release_id"], reviewed["release_id"]]

        asyncio.run(conversation())
        assert len([call for call in source_calls if call[0] == DETAIL]) == 3
        assert provider.closed == provider.factories == 4
    finally:
        chain.close()
