"""Service behavior with real core, ledger, transport and offline provider responses."""
import copy
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import sys
import time
import hashlib
import threading
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import unittest

import httpx

from experiments.search_agent.policy_transport import AsyncAuditTransport, SyncAuditTransport
from experiments.search_agent.resolution import resolution_messages, validate_resolution
from experiments.evaluation.native_agent_guard import CallLedger
from experiments.search_agent.test_policy_backend import snapshot, render


def load_service():
    path = Path(__file__).resolve().parents[3] / "api/db/services/policy_search_service.py"
    spec = importlib.util.spec_from_file_location("policy_search_service_offline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PolicySearchService


class ProviderFixture:
    def __init__(self):
        self.release = snapshot()
        # This fixture represents a freshly scanned catalog; temporal staleness
        # cases explicitly override scanned_at in the policy backend tests.
        self.release["scanned_at"] = datetime.now(timezone.utc).isoformat()
        self.factories = 0
        self.closed = 0
        self.loaded = 0
        self.requests = []
        self.unknown = False
        self.contexts = []
        self.pause = None
        self.kb = {"id": "k", "tenant_id": "t", "embd_id": "embed@OpenAI", "tenant_embd_id": "model"}

    def kb_reader(self, kb_id):
        return copy.deepcopy(self.kb)

    def acquire_snapshot(self, tenant_id, kb_id, as_of):
        self.loaded += 1
        return copy.deepcopy(self.release)

    def provider(self, request):
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        usage = {} if self.unknown else {"usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}
        if request.url.path == "/v1/embeddings":
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}], **usage})
        if body["messages"][0]["content"].startswith("你是医保问答的会话解析器"):
            context = json.loads(body["messages"][1]["content"])
            self.contexts.append(context)
            user_context = [{"message_id": "pending-question", "quote": context["pending"]["question"]["content"]}] if context.get("pending") else []
            answer = json.dumps({"kind": "question", "clarification": "", "user_context": user_context, "reset": False})
        elif "response_format" in body:
            answer = json.dumps({"action": {"type": "finish"}, "stop_kind": "sufficient", "constraints": [],
                                 "needs": [{"text": "费用", "status": "supported", "evidence_ids": ["s0"]}], "reason": "依据"})
        else:
            answer = "不收费。##0$$"
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}}], **usage})

    async def factory(self, kb, audit):
        self.factories += 1
        async def async_provider(request):
            if self.pause:
                self.pause.set()
                await asyncio.Event().wait()
            return self.provider(request)
        gen = httpx.AsyncClient(transport=AsyncAuditTransport(httpx.MockTransport(async_provider), audit, "generation", "https://provider.test/v1", "chat"))
        emb = httpx.Client(transport=SyncAuditTransport(httpx.MockTransport(self.provider), audit, "embedding", "https://provider.test/v1", "embed"))

        async def chat(system, messages, config):
            response = await gen.post("https://provider.test/v1/chat/completions", json={"model": "chat", "messages": [{"role": "system", "content": system}, *messages], **config})
            return response.json()["choices"][0]["message"]["content"]

        async def resolve(context):
            system, messages = resolution_messages(context)
            return validate_resolution(json.loads(await chat(system, messages, {"max_tokens": 600, "response_format": {"type": "json_object"}})), context)

        async def retrieve(query, doc_ids):
            emb.post("https://provider.test/v1/embeddings", json={"model": "embed", "input": query})
            chunk = self.release["chunks"]["a"]
            return {"chunks": [{"chunk_id": "a", "doc_id": "d", "content_with_weight": chunk["text"]}]}

        async def close():
            self.closed += 1
            await gen.aclose()
            emb.close()
        return {"chat": chat, "resolve": resolve, "retrieve": retrieve, "count_tokens": len, "kb_prompt": render, "close": close}


def model_modules(fixture, *, unsupported=False):
    """Replace database/tokenizer/bundle dependencies; keep real OpenAI clients."""
    from openai import AsyncOpenAI, OpenAI
    original_clients, bundles, selections, retrievals = [], [], [], []

    class ChatModel:
        def __init__(self):
            self.model_name = "chat"
            self.async_client = AsyncOpenAI(api_key="fixture", base_url="https://provider.test/v1")
            original_clients.append(self.async_client)
            self.max_retries = 5

        def _clean_conf(self, config):
            config.pop("max_tokens", None)
            return config

        async def _async_chat(self, history, config, **kwargs):
            response = await self.async_client.chat.completions.create(
                model=self.model_name, messages=history, **config, **kwargs)
            return response.choices[0].message.content, response.usage.total_tokens

        async def async_chat(self, system, messages, config):
            config = dict(config)
            if system and messages and messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": system})
            config = self._clean_conf(config)
            try:
                return await self._async_chat(messages, config)
            except Exception as exc:
                return "**ERROR**: " + str(exc), 0

    class EmbeddingModel:
        def __init__(self):
            self.model_name = "embed"
            self.client = None if unsupported else OpenAI(api_key="fixture", base_url="https://provider.test/v1")
            if self.client:
                original_clients.append(self.client)

        def _call(self, query):
            return self.client.embeddings.create(model=self.model_name, input=query)

    class Bundle:
        def __init__(self, tenant_id, config, **kwargs):
            self.mdl = ChatModel() if config["model_type"] == "chat" else EmbeddingModel()
            self.is_tools = True
            self.closed = False
            bundles.append(self)

        async def async_chat(self, system, messages, config):
            text, _ = await self.mdl.async_chat(system, messages, config)
            return text

        def encode_queries(self, query):
            response = self.mdl._call(query)
            return response.data[0].embedding, response.usage.total_tokens

        def close(self):
            self.closed = True

    def chat_config(tenant, kind):
        selections.append(("chat", tenant, kind))
        return {"llm_name": "chat", "model_type": "chat"}

    def embed_config(tenant, kind, ref):
        selections.append(("embedding", tenant, kind, ref))
        return {"llm_name": "embed", "llm_factory": "OpenAI", "api_base": "https://provider.test/v1", "model_type": "embedding"}

    def resolve_config(tenant, kind, ref):
        if kind == "chat":
            selections.append(("chat_override", tenant, kind, ref))
            if ref == "foreign-chat":
                raise LookupError("model outside tenant scope")
            return {"llm_name": "chat", "llm_factory": "OpenAI", "model_type": "chat"}
        return embed_config(tenant, kind, ref)

    async def retrieval(query, embedding, tenants, kbs, page, page_size, **kwargs):
        retrievals.append((tenants, kbs, page, page_size, kwargs))
        embedding.encode_queries(query)
        chunk = fixture.release["chunks"]["a"]
        return {"chunks": [{"chunk_id": "a", "doc_id": "d", "content_with_weight": chunk["text"]}]}

    modules = {}
    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        modules[name] = value
        return value
    settings = module("common.settings", retriever=SimpleNamespace(retrieval=retrieval))
    module("common", settings=settings)
    module("common.constants", LLMType=SimpleNamespace(CHAT="chat", EMBEDDING="embedding"))
    module("common.token_utils", num_tokens_from_string=len)
    module("api.db.joint_services.tenant_model_service", get_tenant_default_model_by_type=chat_config,
           get_model_config_by_id=embed_config, resolve_model_config=resolve_config)
    module("api.db.services.llm_service", LLMBundle=Bundle)
    module("api.db.db_models", DB=SimpleNamespace(connection_context=nullcontext))
    sync_path = Path(__file__).resolve().parents[3] / "api/db/services/policy_sync_service.py"
    spec = importlib.util.spec_from_file_location("api.db.services.policy_sync_service", sync_path)
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    modules["api.db.services.policy_sync_service"] = sync
    embedding_identity = {"llm_factory": "OpenAI", "llm_name": "embed", "api_base": "https://provider.test/v1",
                          "max_tokens": None, "dimension": None, "dimensions": None, "embedding_dimension": None}
    fixture.release["sources"]["d"].update(embedding_config=embedding_identity,
                                         embedding_config_sha256=hashlib.sha256(json.dumps(embedding_identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                                         embedding_dimension=2)
    module("rag.llm.chat_model", Base=ChatModel)
    module("rag.llm.embedding_model", OpenAIEmbed=EmbeddingModel)
    module("rag.prompts.generator", kb_prompt=render)
    return modules, original_clients, bundles, selections, retrievals


class PolicySearchServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 1

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = ProviderFixture()

    def service(self, **kwargs):
        PolicySearchService = load_service()
        return PolicySearchService("t", "k", "s", self.directory.name, publication=self.fixture,
                                   kb_reader=self.fixture.kb_reader, request_factory=self.fixture.factory, **kwargs)

    async def test_exact_tenant_authorization_precedes_models_and_snapshot(self):
        self.fixture.kb["tenant_id"] = "other"
        with self.assertRaisesRegex(PermissionError, "tenant"):
            await self.service().ask("费用？")
        self.assertEqual(self.fixture.factories, 0)
        self.assertEqual(self.fixture.loaded, 0)
        self.assertEqual(self.fixture.requests, [])

    async def test_each_turn_fresh_models_and_snapshot_actual_usage_records(self):
        service = self.service()
        first = await service.ask("费用？", as_of="2026-10-10")
        self.assertEqual(first["result"]["status"], "ok")
        self.assertEqual(first["release_id"], "r1")
        self.assertEqual(first["ledger"]["provider_total_tokens"], 80)
        self.assertEqual(first["ledger"]["provider_calls"], 4)
        self.assertEqual(first["ledger"]["pending_calls"], [])
        self.fixture.release = snapshot("r2")
        second = await service.ask("还有什么费用？")
        self.assertEqual(second["release_id"], "r2")
        self.assertEqual(second["session_total_tokens"], 160)
        self.assertEqual(self.fixture.factories, 2)
        self.assertEqual(self.fixture.closed, 2)
        records = [json.loads(line) for line in (Path(self.directory.name) / "records.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual(len(service.conversation.turns), 2)
        self.assertEqual(len(list((Path(self.directory.name) / "ledgers").glob("*.jsonl"))), 2)

    async def test_as_of_conflict_clarifies_before_model_then_pending_retains_user(self):
        service = self.service()
        result = await service.ask("2021年的上海医保费用？", as_of="2026-10-10")
        self.assertEqual(result["result"]["status"], "needs_clarification")
        self.assertEqual(self.fixture.requests, [])
        self.assertEqual(self.fixture.loaded, 0)
        self.assertEqual(len(service.conversation.turns), 0)
        result = await service.ask("按历史时点", as_of="2021-06-01")
        self.assertEqual(result["result"]["status"], "ok")
        self.assertEqual(self.fixture.contexts[0]["pending"]["question"]["content"], "2021年的上海医保费用？")
        self.assertIn("2021年的上海医保费用？", result["result"]["effective_question"])
        self.assertIsNone(service.conversation.pending)

    async def test_historical_year_without_explicit_date_requires_clarification(self):
        result = await self.service().ask("2021年上海医保规则？")
        self.assertEqual(result["result"]["status"], "needs_clarification")
        self.assertEqual(result["ledger"]["provider_calls"], 0)
        self.assertEqual(self.fixture.loaded, 0)

    async def test_unknown_usage_blocks_session_and_restarted_service(self):
        service = self.service()
        self.fixture.unknown = True
        result = await service.ask("费用？")
        self.assertEqual(result["result"]["status"], "technical_error")
        self.assertTrue(result["ledger"]["pending_calls"])
        self.assertEqual(len(service.conversation.turns), 0)
        before = len(self.fixture.requests)
        for target in (service, self.service()):
            with self.assertRaisesRegex(ValueError, "unresolved|unknown"):
                await target.ask("继续")
        self.assertEqual(len(self.fixture.requests), before)

    async def test_approved_preflight_receipt_unblocks_same_session_preserving_usage_and_records(self):
        from experiments.evaluation.test_native_agent_guard import _write_preflight_receipt
        service = self.service()
        root = Path(self.directory.name)
        identity = {"tenant_id": "t", "kb_id": "k", "session_id": "s", "budget": 60000}
        log = CallLedger(root / "ledgers" / "preflight.jsonl", run_limit=60000, question_limit=60000, call_limit=7,
                         identity=identity, artifact_root=root)
        log.start("preflight")
        log.abort("provider_usage_unknown")
        known = CallLedger(root / "ledgers" / "known.jsonl", run_limit=60000, question_limit=60000, call_limit=7)
        known.start("known")
        known.reserve("known:1", 30, {})
        known.finish("known:1", {"total_tokens": 20})
        original_record = json.dumps({"question_id": "preflight", "result": {"status": "technical_error", "usage": {"unknown_usage": True}},
                                      "conversation": {"turns": [], "pending": None, "history_truncated": False, "as_of": None}}) + "\n"
        (root / "records.jsonl").write_text(original_record)
        with self.assertRaisesRegex(ValueError, "unresolved_provider_usage"):
            await service.ask("继续")
        self.assertEqual(self.fixture.requests, [])
        original_ledger = log.path.read_bytes()
        receipt = _write_preflight_receipt(root, log, identity)
        with self.assertRaisesRegex(ValueError, "unresolved_provider_usage"):
            await service.ask("尚未追加receipt")
        log._reconcile_preflight_not_sent(receipt)
        result = await service.ask("费用？")
        self.assertEqual(result["result"]["status"], "ok", result["result"]["error"])
        self.assertEqual(result["session_total_tokens"], 100)
        self.assertTrue(log.path.read_bytes().startswith(original_ledger))
        self.assertTrue((root / "records.jsonl").read_text().startswith(original_record))
        restarted = self.service()
        self.assertEqual(restarted._used, 100)
        self.assertEqual(restarted._blocked, "")
        self.assertEqual(restarted.session_id, "s")
        self.assertEqual(restarted.budget, 60000)

    async def test_invalid_preflight_receipt_blocks_existing_and_restarted_session(self):
        from experiments.evaluation.test_native_agent_guard import _write_preflight_receipt
        service = self.service()
        root = Path(self.directory.name)
        identity = {"tenant_id": "t", "kb_id": "k", "session_id": "s", "budget": 60000}
        log = CallLedger(root / "ledgers" / "preflight.jsonl", run_limit=60000, question_limit=60000, call_limit=7,
                         identity=identity, artifact_root=root)
        log.start("preflight")
        log.abort("provider_usage_unknown")
        log._reconcile_preflight_not_sent(_write_preflight_receipt(root, log, identity))
        proof = root / "reconciliation" / "proof.json"
        proof.write_bytes(proof.read_bytes() + b" ")
        for target in (service, self.service()):
            with self.assertRaisesRegex(ValueError, "usage|reconciliation"):
                await target.ask("费用？")
        self.assertEqual(self.fixture.factories, 0)
        self.assertEqual(self.fixture.requests, [])

    async def test_missing_unknown_ledger_keeps_existing_and_restarted_session_blocked(self):
        service = self.service()
        self.fixture.unknown = True
        result = await service.ask("费用？")
        self.assertEqual(result["result"]["status"], "technical_error")
        self.assertTrue(result["ledger"]["pending_calls"])
        (Path(self.directory.name) / "ledgers" / (result["question_id"] + ".jsonl")).unlink()
        before = len(self.fixture.requests)
        for target in (service, self.service()):
            with self.assertRaisesRegex(ValueError, "usage"):
                await target.ask("继续")
        self.assertEqual(len(self.fixture.requests), before)

    async def test_known_raw_usage_remains_authoritative_over_stage_unknown_on_restart(self):
        self.service()
        root = Path(self.directory.name)
        log = CallLedger(root / "ledgers" / "known-error.jsonl", run_limit=60000, question_limit=60000, call_limit=7)
        log.start("known-error")
        log.reserve("known-error:1", 30, {})
        log.finish("known-error:1", {"total_tokens": 20})
        log.abort("provider_http_error")
        record = {"question_id": "known-error", "result": {"status": "technical_error", "usage": {"unknown_usage": True}},
                  "ledger": log.summary(), "conversation": {"turns": [], "pending": None, "history_truncated": False, "as_of": None}}
        (root / "records.jsonl").write_text(json.dumps(record) + "\n")
        result = await self.service().ask("费用？")
        self.assertEqual(result["result"]["status"], "ok", result["result"]["error"])
        self.assertEqual(result["session_total_tokens"], 100)

    async def test_deleted_reconciled_ledger_blocks_the_same_live_service_instance(self):
        from experiments.evaluation.test_native_agent_guard import _write_preflight_receipt
        service = self.service()
        root = Path(self.directory.name)
        identity = {"tenant_id": "t", "kb_id": "k", "session_id": "s", "budget": 60000}
        log = CallLedger(root / "ledgers" / "preflight.jsonl", run_limit=60000, question_limit=60000, call_limit=7, identity=identity, artifact_root=root)
        log.start("preflight")
        log.abort("provider_usage_unknown")
        service._refresh_usage()
        log._reconcile_preflight_not_sent(_write_preflight_receipt(root, log, identity))
        service._refresh_usage()
        self.assertEqual(service._blocked, "")
        log.path.unlink()
        with self.assertRaisesRegex(ValueError, "usage"):
            await service.ask("继续")
        self.assertEqual(self.fixture.requests, [])

    async def test_tiny_session_budget_rejects_send_and_records_actual_zero(self):
        result = await self.service(budget=100).ask("费用？")
        self.assertEqual(self.fixture.requests, [])
        self.assertEqual(result["ledger"]["provider_total_tokens"], 0)
        self.assertEqual(result["result"]["status"], "technical_error")

    async def test_output_scope_identity_cannot_be_reused_by_other_session(self):
        await self.service().ask("2021年政策？")
        PolicySearchService = load_service()
        with self.assertRaisesRegex(ValueError, "session"):
            PolicySearchService("t", "k", "other", self.directory.name, publication=self.fixture,
                                kb_reader=self.fixture.kb_reader, request_factory=self.fixture.factory)

    async def test_cancelled_reserved_call_blocks_same_service_and_records_unknown(self):
        service = self.service()
        self.fixture.pause = asyncio.Event()
        task = asyncio.create_task(service.ask("费用？"))
        await self.fixture.pause.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(service.conversation.turns, [])
        self.fixture.pause = None
        with self.assertRaisesRegex(ValueError, "usage"):
            await service.ask("继续")
        record = json.loads((Path(self.directory.name) / "records.jsonl").read_text().splitlines()[-1])
        self.assertEqual(record["result"]["status"], "cancelled")
        self.assertTrue(record["ledger"]["pending_calls"])

    async def test_restarted_session_restores_successful_history_and_exact_actual_total(self):
        await self.service().ask("费用？", as_of="2026-10-10")
        restarted = self.service()
        self.assertEqual(len(restarted.conversation.turns), 1)
        result = await restarted.ask("继续费用？")
        self.assertEqual(result["session_total_tokens"], 160)
        self.assertEqual(result["as_of"], "2026-10-10")
        self.assertEqual(len(self.fixture.contexts[-1]["history"]), 2)

    async def test_default_factory_wires_real_sdk_clients_scope_and_disables_retries(self):
        modules, originals, bundles, selections, retrievals = model_modules(self.fixture)
        service = self.service()
        service._request_factory = None
        with patch.dict(sys.modules, modules), patch("httpx.AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)), patch("httpx.HTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)):
            result = await service.ask("费用？", as_of="2026-10-10")
        self.assertEqual(result["result"]["status"], "ok", result["result"]["error"])
        self.assertEqual(result["ledger"]["provider_total_tokens"], 80)
        self.assertEqual(selections, [("chat", "t", "chat"), ("embedding", "t", "embedding", "model")])
        tenants, kbs, page, count, kwargs = retrievals[0]
        self.assertEqual((tenants, kbs, page, count), (["t"], ["k"], 1, 8))
        self.assertEqual(kwargs["doc_ids"], ["d"])
        self.assertEqual(kwargs["knn_top_k"], 1024)
        self.assertFalse(kwargs["allow_dense_fallback"])
        self.assertIsNone(kwargs["rerank_mdl"])
        self.assertTrue(all(client.is_closed() for client in originals))
        self.assertTrue(all(bundle.closed for bundle in bundles))
        self.assertEqual(bundles[0].mdl.max_retries, 0)
        self.assertEqual(bundles[0].mdl.async_client.max_retries, 0)
        self.assertEqual(bundles[1].mdl.client.max_retries, 0)

    async def test_default_factory_preserves_bound_despite_native_config_cleaner(self):
        modules, _, _, _, _ = model_modules(self.fixture)
        service = self.service()
        service._request_factory = None
        with patch.dict(sys.modules, modules), patch("httpx.AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)), patch("httpx.HTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)):
            result = await service.ask("费用？")
        self.assertEqual(result["result"]["status"], "ok", result["result"]["error"])
        requests = [body for path, body in self.fixture.requests if path == "/v1/chat/completions"]
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0]["max_tokens"], 600)
        self.assertTrue(all(type(body["max_tokens"]) is int and 0 < body["max_tokens"] <= 1200 for body in requests))
        self.assertTrue(all(body["messages"][0]["role"] == "system" for body in requests))
        self.assertEqual(result["ledger"]["provider_calls"], 4)
        self.assertEqual(result["ledger"]["provider_total_tokens"], 80)
        self.assertEqual(result["ledger"]["pending_calls"], [])

    async def test_bounded_chat_copies_history_and_preserves_an_existing_system(self):
        from experiments.search_agent.policy_transport import AuditedCalls
        modules, _, _, _, _ = model_modules(self.fixture)
        service = self.service()
        ledger = CallLedger(Path(self.directory.name) / "history.jsonl", run_limit=60000, question_limit=60000, call_limit=7)
        ledger.start("history")
        config = {"max_tokens": 600}
        with patch.dict(sys.modules, modules), patch("httpx.AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)), patch("httpx.HTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)):
            request = await service._build_request(self.fixture.kb, AuditedCalls(ledger))
            try:
                for messages, expected in (
                    ([{"role": "user", "content": "问题"}], [{"role": "system", "content": "新系统"}, {"role": "user", "content": "问题"}]),
                    ([{"role": "system", "content": "原系统"}, {"role": "user", "content": "问题"}], [{"role": "system", "content": "原系统"}, {"role": "user", "content": "问题"}]),
                ):
                    original = copy.deepcopy(messages)
                    text = await request["chat"]("新系统", messages, config)
                    self.assertEqual(text, "不收费。##0$$")
                    self.assertEqual(self.fixture.requests[-1][1]["messages"], expected)
                    self.assertEqual(messages, original)
                    self.assertEqual(config, {"max_tokens": 600})
            finally:
                await request["close"]()
        self.assertEqual(ledger.summary()["provider_total_tokens"], 40)
        self.assertEqual(ledger.summary()["pending_calls"], [])

    async def test_default_factory_proxy_keeps_all_sdk_posts_inside_audit(self):
        for proxy in ("http://host.docker.internal:7897", None):
            with self.subTest(proxy=proxy):
                self.fixture = ProviderFixture()
                modules, originals, bundles, _, _ = model_modules(self.fixture)
                service = self.service()
                service._request_factory = None
                transports = {}

                def transport(**kwargs):
                    inner = httpx.MockTransport(self.fixture.provider)
                    transports[inner] = kwargs
                    return inner

                env = {"HTTPS_PROXY": "http://untrusted.test:9999", "ALL_PROXY": "http://untrusted.test:9999"}
                with patch.dict(os.environ, env), patch.dict(sys.modules, modules), patch("httpx.AsyncHTTPTransport", transport), patch("httpx.HTTPTransport", transport):
                    if proxy is None:
                        os.environ.pop("POLICY_MODEL_PROXY", None)
                    else:
                        os.environ["POLICY_MODEL_PROXY"] = proxy
                    result = await service.ask("费用？", as_of="2026-10-10")
                self.assertEqual(result["result"]["status"], "ok", result["result"]["error"])
                self.assertEqual([path for path, _ in self.fixture.requests], ["/v1/chat/completions", "/v1/embeddings", "/v1/chat/completions", "/v1/chat/completions"])
                self.assertEqual(result["ledger"]["provider_calls"], 4)
                self.assertEqual(result["ledger"]["provider_total_tokens"], 80)
                self.assertEqual(result["ledger"]["pending_calls"], [])
                for row, (_, body) in zip(result["calls"], self.fixture.requests, strict=True):
                    self.assertEqual(row["request"], dict(body, call_kind="embedding" if body["model"] == "embed" else "generation"))
                    self.assertEqual(row["usage"]["total_tokens"], 20)
                for client, audit_type, endpoint in (
                    (bundles[0].mdl.async_client._client, AsyncAuditTransport, "chat/completions"),
                    (bundles[1].mdl.client._client, SyncAuditTransport, "embeddings"),
                ):
                    self.assertIsInstance(client._transport, audit_type)
                    self.assertIs(client._transport_for_url(httpx.URL("https://provider.test/v1/" + endpoint)), client._transport)
                    self.assertFalse(client.trust_env)
                    self.assertEqual(transports[client._transport.inner], {"proxy": proxy, "retries": 0, "trust_env": False, "verify": True})
                self.assertTrue(all(client.is_closed() for client in originals))
                self.assertTrue(all(bundle.closed for bundle in bundles))
                self.assertEqual(bundles[0].mdl.async_client.max_retries, 0)
                self.assertEqual(bundles[1].mdl.client.max_retries, 0)

    async def test_invalid_model_proxy_is_rejected_before_model_setup_or_send(self):
        values = ("", " http://host.docker.internal:7897", "http://host.docker.internal:7897/",
                  "https://host.docker.internal:7897", "socks5://host.docker.internal:7897",
                  "http://host.docker.internal:7898", "http://other.test:7897",
                  "http://user:secret@host.docker.internal:7897", "http://host.docker.internal:7897?x=1",
                  "http://host.docker.internal:7897#fragment", "http://HOST.docker.internal:7897")
        for value in values:
            with self.subTest(proxy=value):
                self.fixture = ProviderFixture()
                modules, originals, bundles, selections, _ = model_modules(self.fixture)
                service = self.service()
                service._request_factory = None
                with patch.dict(os.environ, {"POLICY_MODEL_PROXY": value}), patch.dict(sys.modules, modules), patch("httpx.AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)), patch("httpx.HTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)):
                    result = await service.ask("费用？")
                self.assertEqual(result["result"]["status"], "technical_error")
                self.assertIn("invalid_policy_model_proxy", result["result"]["error"])
                if value:
                    self.assertNotIn(value, result["result"]["error"])
                self.assertEqual((originals, bundles, selections, self.fixture.requests), ([], [], [], []))
                self.assertEqual(result["ledger"]["provider_calls"], 0)

    async def test_model_proxy_failure_keeps_unknown_usage_and_blocks_later_sdk_posts(self):
        proxy = "http://host.docker.internal:7897"
        failures = (("generation", httpx.ProxyError), ("embedding", httpx.ProxyError),
                    ("generation", httpx.ConnectError), ("embedding", httpx.ConnectError))
        for failed_kind, error in failures:
            with self.subTest(kind=failed_kind, error=error.__name__), tempfile.TemporaryDirectory() as output:
                self.fixture = ProviderFixture()
                modules, originals, bundles, _, _ = model_modules(self.fixture)
                PolicySearchService = load_service()
                service = PolicySearchService("t", "k", "s", output, publication=self.fixture, kb_reader=self.fixture.kb_reader)
                attempts = []

                def transport(**kwargs):
                    def provider(request):
                        attempts.append((request.method, request.url.path))
                        kind = "embedding" if request.url.path.endswith("/embeddings") else "generation"
                        if kwargs.get("proxy") == proxy and kind == failed_kind:
                            raise error("offline proxy failure", request=request)
                        return self.fixture.provider(request)
                    return httpx.MockTransport(provider)

                with patch.dict(os.environ, {"POLICY_MODEL_PROXY": proxy}), patch.dict(sys.modules, modules), patch("httpx.AsyncHTTPTransport", transport), patch("httpx.HTTPTransport", transport):
                    result = await service.ask("费用？")
                    self.assertEqual(result["result"]["status"], "technical_error")
                    self.assertEqual(result["ledger"]["provider_calls"], 1 if failed_kind == "generation" else 2)
                    self.assertEqual(result["ledger"]["provider_total_tokens"], 0 if failed_kind == "generation" else 20)
                    self.assertEqual(result["ledger"]["provider_not_sent_calls"], 0)
                    self.assertEqual(len(result["ledger"]["pending_calls"]), 1)
                    self.assertEqual(result["ledger"]["fatal_reason"], "provider_usage_unknown")
                    self.assertNotIn("usage", result["calls"][-1])
                    before = list(attempts)
                    restarted = PolicySearchService("t", "k", "s", output, publication=self.fixture, kb_reader=self.fixture.kb_reader)
                    for target in (service, restarted):
                        with self.assertRaisesRegex(ValueError, "unresolved_provider_usage"):
                            await target.ask("继续")
                    self.assertEqual(attempts, before)
                    self.assertTrue(all(method == "POST" for method, _ in attempts))
                    self.assertEqual(len(attempts), result["ledger"]["provider_calls"])
                self.assertTrue(all(client.is_closed() for client in originals))
                self.assertTrue(all(bundle.closed for bundle in bundles))

    async def test_default_factory_rejects_unauditable_provider_before_send(self):
        modules, originals, bundles, _, _ = model_modules(self.fixture, unsupported=True)
        service = self.service()
        service._request_factory = None
        with patch.dict(sys.modules, modules):
            result = await service.ask("费用？")
        self.assertEqual(result["result"]["status"], "technical_error")
        self.assertIn("provider_without_auditable_raw_usage", result["result"]["error"])
        self.assertEqual(self.fixture.requests, [])
        self.assertEqual(result["ledger"]["provider_calls"], 0)
        self.assertTrue(all(client.is_closed() for client in originals))
        self.assertTrue(all(bundle.closed for bundle in bundles))

    async def test_kb_chat_override_resolves_under_request_tenant(self):
        self.fixture.kb["parser_config"] = {"llm_id": "kb-chat"}
        modules, _, _, selections, _ = model_modules(self.fixture)
        service = self.service()
        service._request_factory = None
        with patch.dict(sys.modules, modules), patch("httpx.AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)), patch("httpx.HTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)):
            result = await service.ask("费用？")
        self.assertEqual(result["result"]["status"], "ok", result["result"]["error"])
        self.assertEqual(selections[0], ("chat_override", "t", "chat", "kb-chat"))
        self.assertNotIn(("chat", "t", "chat"), selections)

    async def test_foreign_kb_chat_override_does_not_fall_back_or_send(self):
        self.fixture.kb["parser_config"] = {"llm_id": "foreign-chat"}
        modules, _, bundles, selections, _ = model_modules(self.fixture)
        service = self.service()
        service._request_factory = None
        with patch.dict(sys.modules, modules), patch("httpx.AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)), patch("httpx.HTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)):
            result = await service.ask("费用？")
        self.assertEqual(result["result"]["status"], "technical_error")
        self.assertIn("tenant scope", result["result"]["error"])
        self.assertEqual(selections, [("chat_override", "t", "chat", "foreign-chat")])
        self.assertEqual(bundles, [])
        self.assertEqual(self.fixture.requests, [])
        self.assertEqual(result["ledger"]["provider_calls"], 0)

    async def test_clear_removes_persistent_user_context_and_preserves_usage(self):
        service = self.service()
        await service.ask("费用？", as_of="2026-10-10")
        await service.clear()
        self.assertEqual(service.conversation.turns, [])
        restarted = self.service()
        self.assertEqual(restarted.conversation.turns, [])
        result = await restarted.ask("费用？")
        self.assertEqual(result["session_total_tokens"], 160)
        self.fixture.unknown = True
        await restarted.ask("继续")
        await restarted.clear()
        with self.assertRaisesRegex(ValueError, "usage"):
            await restarted.ask("仍然继续")

    async def test_existing_service_rechecks_unfinished_durable_ledger_before_send(self):
        service = self.service()
        ledger = CallLedger(Path(self.directory.name) / "ledgers" / "interrupted.jsonl",
                            run_limit=60000, question_limit=60000, call_limit=7)
        ledger.start("interrupted")
        ledger.reserve("unfinished", 5000, {"model": "chat"})
        with self.assertRaisesRegex(ValueError, "usage"):
            await service.ask("继续")
        self.assertEqual(self.fixture.requests, [])

    async def test_same_year_chinese_case_date_conflict_clarifies_before_snapshot(self):
        result = await self.service().ask("2026年9月22日的医保待遇？", as_of="2026-10-10")
        self.assertEqual(result["result"]["status"], "needs_clarification")
        self.assertEqual(self.fixture.loaded, 0)
        self.assertEqual(self.fixture.requests, [])

    async def test_invalid_user_case_date_clarifies_without_provider_calls(self):
        result = await self.service().ask("2026年2月30日的医保待遇？")
        self.assertEqual(result["result"]["status"], "needs_clarification")
        self.assertEqual(self.fixture.requests, [])
        with self.assertRaises(ValueError):
            await self.service().ask("医保待遇？", as_of="2026-02-30")

    async def test_resolved_inherited_user_date_conflict_clarifies_before_snapshot(self):
        service = self.service()
        service.conversation._turns = [{"user": "2026年9月22日的上海医保", "assistant": "话题", "effective_question": "旧问题"}]
        original_factory = self.fixture.factory
        async def factory(kb, audit):
            result = await original_factory(kb, audit)
            original = result["resolve"]
            async def resolve(context):
                await original(context)
                return validate_resolution({"kind": "question", "clarification": "", "reset": False,
                                            "user_context": [{"message_id": "t0-user", "quote": "2026年9月22日"}]}, context)
            result["resolve"] = resolve
            return result
        service._request_factory = factory
        result = await service.ask("待遇呢？", as_of="2026-10-10")
        self.assertEqual(result["result"]["status"], "needs_clarification")
        self.assertEqual(self.fixture.loaded, 0)
        self.assertEqual(len(self.fixture.requests), 1)

    async def test_blocking_snapshot_and_authorization_yield_to_event_loop(self):
        service = self.service()
        original = self.fixture.acquire_snapshot
        def slow_snapshot(tenant, kb, as_of):
            time.sleep(.06)
            return original(tenant, kb, as_of)
        self.fixture.acquire_snapshot = slow_snapshot
        service.conversation.limits = replace(service.conversation.limits, resolver_seconds=.02)
        started = time.monotonic()
        result = await service.ask("费用？")
        self.assertEqual(result["result"]["status"], "timeout")
        self.assertLess(time.monotonic() - started, .05)

    async def test_same_dimension_changed_model_identity_rejected_before_embedding(self):
        modules, _, _, _, retrievals = model_modules(self.fixture)
        config = self.fixture.release["sources"]["d"]["embedding_config"]
        config["llm_name"] = "different-model-with-same-dimension"
        self.fixture.release["sources"]["d"]["embedding_config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        service = self.service()
        service._request_factory = None
        with patch.dict(sys.modules, modules), patch("httpx.AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)), patch("httpx.HTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)):
            result = await service.ask("费用？")
        self.assertEqual(result["result"]["status"], "technical_error")
        self.assertIn("embedding", result["result"]["error"])
        self.assertEqual(retrievals, [])
        self.assertEqual([path for path, _ in self.fixture.requests], ["/v1/chat/completions"])

    async def test_actual_query_vector_dimension_rejected_with_known_usage(self):
        modules, _, _, _, _ = model_modules(self.fixture)
        self.fixture.release["sources"]["d"]["embedding_dimension"] = 3
        service = self.service()
        service._request_factory = None
        with patch.dict(sys.modules, modules), patch("httpx.AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)), patch("httpx.HTTPTransport", lambda **kwargs: httpx.MockTransport(self.fixture.provider)):
            result = await service.ask("费用？")
        self.assertEqual(result["result"]["status"], "technical_error")
        self.assertIn("dimension", result["result"]["error"])
        self.assertEqual(result["ledger"]["provider_total_tokens"], 40)
        self.assertEqual(result["ledger"]["pending_calls"], [])

    async def test_cancelled_model_setup_closes_late_created_clients(self):
        modules, originals, bundles, _, _ = model_modules(self.fixture)
        service = self.service()
        created, release = threading.Event(), threading.Event()
        original = service._model_inputs
        def slow_inputs(kb):
            value = original(kb)
            created.set()
            release.wait(1)
            return value
        service._model_inputs = slow_inputs
        ledger = CallLedger(Path(self.directory.name) / "setup.jsonl", run_limit=60000, question_limit=60000, call_limit=7)
        ledger.start("setup")
        from experiments.search_agent.policy_transport import AuditedCalls
        with patch.dict(sys.modules, modules):
            task = asyncio.create_task(service._build_request(self.fixture.kb, AuditedCalls(ledger)))
            await asyncio.to_thread(created.wait, 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            release.set()
            for _ in range(20):
                if all(client.is_closed() for client in originals):
                    break
                await asyncio.sleep(.01)
            self.assertTrue(all(client.is_closed() for client in originals))
            self.assertTrue(all(bundle.closed for bundle in bundles))
        self.assertEqual(ledger.summary()["provider_calls"], 0)
