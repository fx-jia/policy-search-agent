"""Dynamic policy scopes with offline async transport fixtures."""
import copy
import ast
import logging
from pathlib import Path
import re
from dataclasses import replace
import hashlib
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from experiments.search_agent.conversation import Conversation
from experiments.search_agent.core import AgentLimits

from experiments.search_agent import adapter
from experiments.search_agent.evidence import Evidence, pack_evidence, validate_views
from experiments.search_agent.resolution import Resolution


def snapshot(release="r1", text="符合条件者不收费。", status="confirmed"):
    return {"release_id": release, "doc_ids": ["d"],
            "sources": {"d": {"source_url": "https://ybj.sh.gov.cn/policy", "source_kind": "gfxwj",
                                "published_at": "2026-01-01", "effective_from": "2026-02-01",
                                "confirmation_status": status, "temporal_metadata": {"status": status}, "tenant_id": "t", "kb_id": "k"}},
            "chunks": {"a": {"id": "a", "document_id": "d", "title": "办法", "text": text,
                               "order": 0, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                               "native": {}, "ordered": True}},
            "pending": [], "coverage": {"complete": False}, "scanned_at": "2026-10-10T00:00:00Z"}


def render(info, max_tokens):
    blocks = []
    for i, chunk in enumerate(info["chunks"]):
        block = f'ID: {i}\n├── Title: {chunk["docnm_kwd"]}'
        if chunk.get("url"):
            block += f'\n├── URL: {chunk["url"]}'
        for key, value in chunk.get("document_metadata", {}).items():
            if value is not None and str(value):
                block += f'\n├── {key}: {value}'
        blocks.append(block + '\n└── Content:\n' + chunk["content_with_weight"])
    return blocks


class Fixture:
    def __init__(self, data=None):
        self.data = data or snapshot()
        self.loads = self.retrievals = self.chats = 0
        self.resolution = Resolution("question", "费用？")
        self.raw = None
        self.answer = "不收费。##0$$"
        self.usage = {"total_tokens": 20, "physical_calls": 1}
        self.requests = []

    def backend(self, **kwargs):
        from experiments.search_agent.policy_backend import PolicyBackend
        return PolicyBackend(self.load, self.retrieve, self.resolve, self.chat, len,
                             audit=self.audit, kb_prompt=render, tenant_id="t", kb_id="k", **kwargs)

    def load(self, as_of):
        self.loads += 1
        return self.data

    async def resolve(self, context):
        return self.resolution

    async def retrieve(self, query, doc_ids):
        self.retrievals += 1
        if self.raw is not None:
            return self.raw
        chunk = self.data["chunks"]["a"]
        return {"chunks": [{"chunk_id": "a", "doc_id": "d", "content_with_weight": chunk["text"],
                            "url": "https://evil.test", "document_metadata": {"confirmation_status": "forged"}}]}

    async def chat(self, system, messages, config):
        self.chats += 1
        self.requests.append((system, messages, config))
        if "response_format" in config:
            return json.dumps({"action": {"type": "finish"}, "stop_kind": "sufficient", "constraints": [],
                               "needs": [{"text": "费用", "status": "supported", "evidence_ids": ["s0"]}], "reason": "依据"})
        return self.answer

    def audit(self, stage, result):
        return self.usage


class SharedSourceTests(unittest.TestCase):
    def test_judge_and_native_keep_trusted_metadata(self):
        item = Evidence("a", "d", "办法", "不收费。", 0, {"url": "evil"}, source=snapshot()["sources"]["d"])
        _, messages = adapter.judge_messages("费用", [item], None)
        self.assertIn("https://ybj.sh.gov.cn/policy", messages[0]["content"])
        native = adapter.native_kbinfos([item])
        self.assertEqual(native["chunks"][0]["url"], "https://ybj.sh.gov.cn/policy")
        self.assertEqual(native["chunks"][0]["document_metadata"]["confirmation_status"], "confirmed")

    def test_publication_metadata_projection_preserves_business_and_full_reference(self):
        source = snapshot()["sources"]["d"] | {
            "title": "政策业务标题", "document_number": "沪医保规〔2025〕6号",
            "signed_at": "2025-08-29", "observed_at": "2026-10-10T15:22:49Z",
            "effective_to": None, "policy_key": "p" * 64, "revision_hash": "r" * 64,
            "revision_id": "i" * 32, "asset_role": "attachment:internal-id",
            "asset_sha256": "a" * 64, "asset_url": "https://ybj.sh.gov.cn/file.pdf",
            "parse_build_key": "b" * 64, "archive_key": "policy/private/" + "a" * 64,
            "fingerprint": "f" * 32, "embedding_config": {"api_base": "internal", "max_tokens": 1000000},
            "embedding_config_sha256": "e" * 64, "embedding_dimension": 1024,
            "relations": [{"type": "association_reference", "document_number": "沪医保规〔2024〕1号",
                           "source_span": "依据原文规定", "status": "needs_review"}],
        }
        self.assertEqual(len(source), 26)
        source["temporal_metadata"] = {"status": "confirmed", "repealed_at": "2027-01-01",
                                       "evidence": "本办法于2027年1月1日废止。", "reviewed_by": "审核人",
                                       "evidence_archive_key": "private/archive", "evidence_archive_sha256": "h" * 64}
        items = [Evidence(str(i), str(i % 3), "internal-generated-filename-hash.txt", "原文证据。", i,
                          source=copy.deepcopy(source)) for i in range(8)]
        # Deterministic offline character counter; actual native tokenizer is checked separately.
        views = adapter.prepare_judge_views("费用", items, None, lambda s: (len(s) + 1) // 2, 4800,
                                            neighbors={}, gap_contract=True)
        self.assertEqual(len(views), 8)
        _, messages = adapter.judge_messages("费用", views, None)
        projected = json.loads(messages[0]["content"])["evidence"][0]["source"]
        self.assertNotIn("embedding_config", projected)
        self.assertNotIn("archive_key", projected)
        self.assertNotIn("revision_hash", projected)
        self.assertNotIn("tenant_id", projected)
        self.assertEqual(projected["asset_role"], "attachment")
        self.assertEqual(projected["relations"], source["relations"])
        self.assertEqual(projected["temporal_metadata"]["evidence"], source["temporal_metadata"]["evidence"])
        self.assertNotIn("evidence_archive_key", projected["temporal_metadata"])
        prompt = adapter.native_kbinfos(views, for_prompt=True)
        self.assertEqual(prompt["chunks"][0]["document_metadata"], projected)
        self.assertEqual(prompt["chunks"][0]["docnm_kwd"], source["title"])
        rendered = "\n".join(render(prompt, 5600))
        adapter.check_final_request({"messages": [{"role": "system", "content": rendered}]}, views)
        for before, after in (("审核人", "伪造审核人"), ("依据原文规定", "伪造关联")):
            with self.assertRaisesRegex(ValueError, "source_metadata"):
                adapter.check_final_request({"messages": [{"role": "system", "content": rendered.replace(before, after)}]}, views)
        full = adapter.native_kbinfos(views)
        self.assertEqual(full["chunks"][0]["document_metadata"], source)
        self.assertEqual(views[0].source, source)
        for key in ("tenant_id", "revision_hash", "embedding_config_sha256"):
            with self.assertRaisesRegex(ValueError, "source_metadata_changed"):
                validate_views([replace(views[0], source=source | {key: "forged"})], items)

    def test_excerpt_cannot_change_trusted_source(self):
        item = Evidence("a", "d", "办法", "不收费。", 0, source={"source_url": "official"})
        with self.assertRaisesRegex(ValueError, "source"):
            validate_views([replace(item, source={"source_url": "evil"})], [item])

    def test_final_blocks_accept_and_require_trusted_source_fields(self):
        item = Evidence("a", "d", "办法", "不收费。", 0, source=snapshot()["sources"]["d"])
        text = '\n'.join(render(adapter.native_kbinfos([item], for_prompt=True), 5600))
        adapter.check_final_request({"messages": [{"role": "system", "content": text}]}, [item])
        with self.assertRaisesRegex(ValueError, "source"):
            adapter.check_final_request({"messages": [{"role": "system", "content": text.replace("confirmed", "forged")}]}, [item])

    def test_source_metadata_is_in_complete_judge_budget(self):
        item = Evidence("a", "d", "办法", "不收费。", 0, source={"source_url": "x" * 8000})
        with self.assertRaises(ValueError):
            adapter.prepare_judge_views("费用", [item], None, len, 4800, neighbors={}, gap_contract=True)

    def test_final_window_reserves_complete_source_metadata_budget(self):
        item = Evidence("a", "d", "办法", "费用规定。" * 2000, 0, source=snapshot()["sources"]["d"])
        views = pack_evidence([item], None, "费用", len, 5600)
        self.assertTrue(views)
        self.assertLessEqual(len('\n'.join(render(adapter.native_kbinfos(views, for_prompt=True), 5600))), 5600)

    def test_final_selected_source_rejects_extra_text_or_metadata(self):
        item = Evidence("a", "d", "办法", "不收费。", 0, source=snapshot()["sources"]["d"])
        text = '\n'.join(render(adapter.native_kbinfos([item], for_prompt=True), 5600))
        for forged in (text + "未选中的额外条款。", text.replace("└── Content:", "├── forged: yes\n└── Content:")):
            with self.assertRaises(ValueError):
                adapter.check_final_request({"messages": [{"role": "system", "content": forged}]}, [item])


class PolicyBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clock = self.enterContext(patch("experiments.search_agent.policy_backend.datetime", wraps=datetime))
        clock.now.return_value = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)

    async def bind(self, backend):
        await backend.resolve({"current": {"content": "费用？"}})

    async def test_clarification_does_not_load_knowledge(self):
        fixture = Fixture()
        fixture.resolution = Resolution("clarification", clarification="哪一年？")
        backend = fixture.backend()
        self.assertEqual((await backend.resolve({})).kind, "clarification")
        self.assertEqual(fixture.loads, 0)
        with self.assertRaisesRegex(ValueError, "snapshot"):
            await backend.search("费用")

    async def test_release_bound_once_and_next_backend_sees_new_release(self):
        fixture = Fixture()
        first = fixture.backend()
        await self.bind(first)
        fixture.data = snapshot("r2", "新规定收费。")
        fixture.raw = {"chunks": [{"chunk_id": "a", "doc_id": "d", "content_with_weight": "符合条件者不收费。"}]}
        self.assertEqual((await first.search("费用"))[0].text, "符合条件者不收费。")
        self.assertEqual(first.snapshot["release_id"], "r1")
        second = fixture.backend()
        await self.bind(second)
        self.assertEqual(second.snapshot["release_id"], "r2")
        self.assertEqual(fixture.loads, 2)

    async def test_empty_release_never_calls_unscoped_retrieval(self):
        fixture = Fixture({"release_id": None, "doc_ids": [], "sources": {}, "chunks": {}, "pending": [], "coverage": {}, "scanned_at": None})
        backend = fixture.backend()
        await self.bind(backend)
        self.assertEqual(await backend.search("费用"), [])
        self.assertEqual(fixture.retrievals, 0)

    async def test_scope_unseen_chunk_hash_and_tenant_rejected(self):
        for raw, error in [({"chunks": [{"chunk_id": "bad", "doc_id": "d", "content_with_weight": "符合条件者不收费。"}]}, "chunk"),
                           ({"chunks": [{"chunk_id": "a", "doc_id": "d", "content_with_weight": "已篡改。"}]}, "hash"),
                           ({"chunks": [{"chunk_id": "a", "doc_id": "other", "content_with_weight": "符合条件者不收费。"}]}, "scope")]:
            fixture = Fixture()
            backend = fixture.backend()
            await self.bind(backend)
            fixture.raw = raw
            with self.assertRaisesRegex(ValueError, error):
                await backend.search("费用")
        fixture = Fixture()
        fixture.data["sources"]["d"]["tenant_id"] = "other"
        with self.assertRaisesRegex(ValueError, "tenant"):
            await self.bind(fixture.backend())

    async def test_narrow_scope_requires_seen_document_and_empty_rejected(self):
        backend = Fixture().backend()
        await self.bind(backend)
        for docs in (["d"], [], ["other"]):
            with self.assertRaises(ValueError):
                await backend.search("费用", docs)
        await backend.search("费用")
        self.assertTrue(await backend.search("费用", ["d"]))

    async def test_native_singleton_kb_scope_accepted_and_multi_foreign_rejected(self):
        for kb, accepted in (("k", True), (["k"], True), (["other"], False), (["k", "other"], False), ([], False)):
            fixture = Fixture()
            fixture.raw = {"chunks": [{"chunk_id": "a", "doc_id": "d", "content_with_weight": "符合条件者不收费。", "kb_id": kb}]}
            backend = fixture.backend()
            await self.bind(backend)
            if accepted:
                self.assertEqual([item.id for item in await backend.search("费用")], ["a"])
            else:
                with self.assertRaisesRegex(ValueError, "scope"):
                    await backend.search("费用")

    async def test_unordered_neighbors_disabled(self):
        fixture = Fixture()
        fixture.data["chunks"]["a"]["ordered"] = False
        backend = fixture.backend()
        await self.bind(backend)
        seed = (await backend.search("费用"))[0]
        self.assertEqual(backend.neighbor_directions(seed), [])
        self.assertEqual(await backend.neighbors(seed, "both"), [])

    async def test_unordered_publisher_null_order_can_retrieve_without_neighbors(self):
        fixture = Fixture()
        fixture.data["chunks"]["a"].update(ordered=False, order=None)
        backend = fixture.backend()
        await self.bind(backend)
        seed = (await backend.search("费用"))[0]
        self.assertIsNone(backend.snapshot["chunks"]["a"]["order"])
        self.assertEqual(seed.order, 0)
        self.assertEqual(backend.neighbor_directions(seed), [])
        self.assertEqual(await backend.neighbors(seed, "both"), [])
        result = await backend.generate("费用", [seed])
        self.assertEqual(result["answer"], "不收费。##0$$")

    async def test_source_cannot_be_overwritten_and_unknown_only_not_sufficient(self):
        fixture = Fixture(snapshot(status="unknown"))
        backend = fixture.backend(as_of="2026-10-10")
        await self.bind(backend)
        items = await backend.search("费用")
        self.assertEqual(items[0].source["confirmation_status"], "unknown")
        decision = await backend.decide("费用", items, None)
        self.assertEqual(decision["needs"][0]["status"], "uncertain")
        self.assertNotEqual(decision["stop_kind"], "sufficient")
        self.assertIn("2026-10-10", fixture.requests[0][0] + fixture.requests[0][1][0]["content"])

    async def test_real_conversation_unknown_time_remains_uncertain(self):
        fixture = Fixture(snapshot(status="unknown"))
        backend = fixture.backend()
        result = await Conversation("session").ask("费用？", backend, AgentLimits(max_actions=0))
        self.assertEqual(result.status, "ok")
        self.assertNotEqual(result.stop_reason, "sufficient")
        self.assertIn("政策时效尚未确认", result.answer)
        self.assertEqual(backend.applicability, "unknown")

    async def test_effective_dates_are_checked_against_historical_as_of(self):
        for as_of, supported in (("2026-01-01", False), ("2026-03-01", True), ("2027-01-01", True), ("2027-01-02", False)):
            fixture = Fixture()
            fixture.data["sources"]["d"]["effective_to"] = "2027-01-01"
            backend = fixture.backend(as_of=as_of)
            await self.bind(backend)
            value = await backend.decide("费用", await backend.search("费用"), None)
            self.assertEqual(value["needs"][0]["status"] == "supported", supported)

    async def test_repeal_date_excludes_policy_at_that_date(self):
        fixture = Fixture()
        fixture.data["sources"]["d"]["repealed_at"] = "2026-10-10"
        backend = fixture.backend(as_of="2026-10-10")
        await self.bind(backend)
        value = await backend.decide("费用", await backend.search("费用"), None)
        self.assertEqual(value["needs"][0]["status"], "uncertain")

    async def test_large_pending_catalog_remains_available_with_bounded_final_notice(self):
        fixture = Fixture()
        fixture.data["pending"] = [{"policy_key": str(i), "title": "待处理政策", "errors": ["待解析"],
                                     "parsing_builds": {"large": "x" * 8000}} for i in range(40)]
        backend = fixture.backend()
        await self.bind(backend)
        value = await backend.generate("费用", await backend.search("费用"))
        self.assertEqual(len(value["pending"]), 40)
        self.assertIn('"total":40', fixture.requests[0][0])
        self.assertIn('"omitted":37', fixture.requests[0][0])
        self.assertNotIn("parsing_builds", fixture.requests[0][0])

    async def test_native_prompt_margin_retains_full_budgeted_selected_blocks(self):
        path = Path(__file__).resolve().parents[2] / "rag/prompts/generator.py"
        tree = ast.parse(path.read_text())
        unit = ast.Module(body=[node for node in tree.body if getattr(node, "name", None) in {"get_value", "_kb_block", "kb_prompt"}], type_ignores=[])
        namespace = {"re": re, "logging": logging, "num_tokens_from_string": len}
        exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
        fixture = Fixture(snapshot(text="费用不收费。" * 2000))
        backend = fixture.backend()
        backend._kb_prompt = namespace["kb_prompt"]
        await self.bind(backend)
        selected = pack_evidence(await backend.search("费用"), None, "费用", len, 5600)
        value = await backend.generate("费用", selected)
        self.assertTrue(value["answer"])
        self.assertIn(selected[0].text, fixture.requests[0][0])
        self.assertLessEqual(len(fixture.requests[0][0] + "费用"), 7000)

    async def test_selected_newer_policy_pending_downgrades_and_requires_lag_notice(self):
        fixture = Fixture()
        fixture.data["sources"]["d"].update(policy_key="same", revision_id="old")
        fixture.data["pending"] = [{"policy_key": "same", "revision_id": "new", "observed_at": "2026-10-10T00:00:00Z", "pending_reasons": ["attachment missing"]}]
        backend = fixture.backend(as_of="2026-10-10")
        result = await Conversation("session").ask("最新现行医保费用？", backend, AgentLimits(max_actions=0))
        self.assertEqual(result.status, "ok")
        self.assertNotEqual(result.stop_reason, "sufficient")
        self.assertEqual(backend.applicability, "unknown")
        self.assertIn("更新滞后", result.answer)
        self.assertIn("attachment missing", fixture.requests[0][0] + fixture.requests[0][1][0]["content"])

    async def test_stale_current_downgrades_but_historical_and_unrelated_pending_do_not(self):
        with patch("experiments.search_agent.policy_backend.datetime", wraps=datetime) as clock:
            clock.now.return_value = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
            for as_of, stale, same, expected in (("2026-10-10", True, False, False),
                                               ("2026-10-10", False, False, True),
                                               ("2026-03-01", True, True, True)):
                fixture = Fixture()
                fixture.data["sources"]["d"].update(policy_key="same", revision_id="old")
                fixture.data["scanned_at"] = "2026-09-01T00:00:00Z" if stale else "2026-10-10T00:00:00Z"
                fixture.data["pending"] = [{"policy_key": "same" if same else "other", "revision_id": "new", "observed_at": "2026-10-10T00:00:00Z"}]
                backend = fixture.backend(as_of=as_of)
                await self.bind(backend)
                value = await backend.decide("医保费用", await backend.search("费用"), None)
                self.assertEqual(value["needs"][0]["status"] == "supported", expected)

    async def test_scan_freshness_notice_distinguishes_absent_stale_fresh_and_invalid(self):
        cases = ((None, ["尚无完整成功扫描记录"]),
                 ("2026-10-07T00:00:00Z", ["超过48小时没有完整成功扫描"]),
                 ("2026-10-10T00:00:00Z", []),
                 ("invalid-date", ["未能确认完整扫描时间"]),
                 ("2026-10-10", ["未能确认完整扫描时间"]),
                 (7, ["未能确认完整扫描时间"]))
        with patch("experiments.search_agent.policy_backend.datetime", wraps=datetime) as clock:
            clock.now.return_value = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
            for scanned, expected in cases:
                with self.subTest(scanned=scanned):
                    fixture = Fixture()
                    fixture.data["scanned_at"] = scanned
                    backend = fixture.backend(as_of="2026-10-10")
                    await self.bind(backend)
                    self.assertEqual(backend._lag_reasons(fixture.data["sources"]["d"]), expected)
                    self.assertEqual(fixture.chats, 0)
                    self.assertEqual(fixture.retrievals, 0)

    async def test_unreviewed_association_to_selected_document_requires_lag_notice(self):
        fixture = Fixture()
        fixture.data["sources"]["d"].update(policy_key="formal", document_number="沪医保〔2026〕1号")
        fixture.data["pending"] = [{"policy_key": "notice", "observed_at": "2026-10-10T00:00:00Z", "relations": [
            {"kind": "association_reference", "document_number": "沪医保〔2026〕1号", "status": "needs_review"}]}]
        backend = fixture.backend(as_of="2026-10-10")
        await self.bind(backend)
        selected = await backend.search("费用")
        value = await backend.decide("当前费用", selected, None)
        self.assertEqual(value["needs"][0]["status"], "uncertain")
        answer = await backend.generate("当前费用", selected)
        self.assertIn("关联通知尚未核对", answer["answer"])

    async def test_same_document_neighbors_use_manifest_and_require_seen_seed(self):
        fixture = Fixture()
        text = "期限十天。"
        fixture.data["chunks"]["b"] = {"id": "b", "document_id": "d", "title": "办法", "text": text,
                                        "order": 1, "sha256": hashlib.sha256(text.encode()).hexdigest(), "native": {}, "ordered": True}
        fixture.data["doc_ids"].append("e")
        fixture.data["sources"]["e"] = copy.deepcopy(fixture.data["sources"]["d"])
        fixture.data["chunks"]["c"] = {"id": "c", "document_id": "e", "title": "其他", "text": text,
                                        "order": 0, "sha256": hashlib.sha256(text.encode()).hexdigest(), "native": {}, "ordered": True}
        backend = fixture.backend()
        await self.bind(backend)
        seed = (await backend.search("费用"))[0]
        self.assertEqual(backend.neighbor_directions(seed), ["next"])
        with self.assertRaises(ValueError):
            await backend.neighbors(replace(seed, order=1), "both")
        self.assertEqual([e.id for e in await backend.neighbors(seed, "both")], ["b"])
        with self.assertRaisesRegex(ValueError, "neighbor"):
            await backend.neighbors(Evidence("c", "e", "其他", text, 0), "both")

    async def test_actual_remaining_calls_reserves_final_and_physical_retry_limit(self):
        fixture = Fixture()
        backend = fixture.backend(remaining_calls=lambda: 2)
        await self.bind(backend)
        self.assertEqual(backend.remaining_generation_calls(), 2)
        fixture.usage = {"total_tokens": 20, "physical_calls": 5}
        with self.assertRaisesRegex(ValueError, "physical_call"):
            await backend.generate("费用", await backend.search("费用"))

    async def test_provider_exception_audits_and_closes_request(self):
        fixture = Fixture()
        audited = []
        async def broken(query, docs):
            raise RuntimeError("provider failure")
        def audit(stage, result):
            audited.append((stage, isinstance(result, RuntimeError)))
            return {"total_tokens": 0, "physical_calls": 1}
        backend = fixture.backend()
        await self.bind(backend)
        backend._retrieve = broken
        backend._audit_callback = audit
        with self.assertRaisesRegex(RuntimeError, "provider failure"):
            await backend.search("费用")
        self.assertEqual(audited, [("search", True)])
        with self.assertRaises(ValueError):
            await backend.generate("费用", [])

    async def test_generation_one_selected_prompt_and_citation_bounds(self):
        fixture = Fixture()
        backend = fixture.backend()
        await self.bind(backend)
        selected = await backend.search("费用")
        result = await backend.generate("费用", selected)
        self.assertEqual(result["reference"]["chunks"][0]["url"], "https://ybj.sh.gov.cn/policy")
        self.assertEqual(result["release_id"], "r1")
        self.assertEqual(fixture.chats, 1)
        self.assertEqual(fixture.retrievals, 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            await backend.generate("费用", selected)
        for bad in ("错误##1$$", "错误##bad$$", "错误[ID:1]", "错误##$$", "错误##0", "错误[ID: 1]"):
            fixture = Fixture()
            fixture.answer = bad
            backend = fixture.backend()
            await self.bind(backend)
            with self.assertRaisesRegex(ValueError, "citation"):
                await backend.generate("费用", await backend.search("费用"))

    async def test_unknown_usage_closes_request_before_final(self):
        fixture = Fixture()
        backend = fixture.backend()
        await self.bind(backend)
        fixture.usage = None
        with self.assertRaisesRegex(ValueError, "usage"):
            await backend.search("费用")
        with self.assertRaisesRegex(ValueError, "usage"):
            await backend.generate("费用", [])
        self.assertEqual(fixture.chats, 0)

    async def test_missing_physical_usage_is_unknown_and_known_limit_is_distinct(self):
        fixture = Fixture()
        fixture.usage = {"total_tokens": 20}
        backend = fixture.backend()
        with self.assertRaisesRegex(ValueError, "usage"):
            await self.bind(backend)
        self.assertTrue(backend.metrics["unknown_usage"])
        fixture = Fixture()
        fixture.usage = {"total_tokens": 70000, "physical_calls": 1}
        backend = fixture.backend()
        with self.assertRaisesRegex(ValueError, "token_limit"):
            await self.bind(backend)
        self.assertFalse(backend.metrics["unknown_usage"])

    async def test_snapshot_requires_release_for_published_documents(self):
        fixture = Fixture()
        fixture.data["release_id"] = None
        with self.assertRaisesRegex(ValueError, "release"):
            await self.bind(fixture.backend())

    async def test_final_prompt_distinguishes_review_state_excerpt_and_missing_scope(self):
        visible = "可以使用家庭共济资金支付。"
        fixture = Fixture(snapshot(text=visible + "本通知自2026年9月15日起施行。", status="unknown"))
        backend = fixture.backend()
        await self.bind(backend)
        original = (await backend.search("家庭成员范围与生效日期"))[0]
        selected = [replace(original, text=visible, spans=((0, len(visible)),), truncated=True)]
        result = await backend.generate("哪些家庭成员、哪些项目可以支付，何时生效？", selected)
        system, messages, config = fixture.requests[0]
        self.assertIn("时效状态unknown表示人工核验尚未完成，不表示原文没有记载生效日期", system)
        self.assertIn("所选片段未展示某项事实（包括截断片段）时，不得断言整个文件未记载", system)
        self.assertIn("题目要求的具体家庭成员或项目范围缺少原文依据时，明确说明未能确认具体范围", system)
        self.assertIn("不能用‘符合条件的家庭成员’等笼统表述代替所问范围", system)
        self.assertNotIn("2026年9月15日", system)
        self.assertEqual(result["applicability"], "unknown")
        self.assertEqual(result["answer"], "政策时效尚未确认；以下仅说明已收录原文与证据缺口。\n" + fixture.answer)
        self.assertEqual(config["max_tokens"], 1200)
        self.assertEqual(messages[0]["content"], "哪些家庭成员、哪些项目可以支付，何时生效？")

    async def test_final_prompt_projection_keeps_full_returned_citations(self):
        fixture = Fixture()
        fixture.data["sources"]["d"].update(archive_key="internal/" * 1000,
                                             embedding_config={"api_base": "private-model-endpoint"})
        backend = fixture.backend()
        await self.bind(backend)
        selected = await backend.search("费用")
        result = await backend.generate("费用", selected)
        self.assertEqual(result["reference"]["chunks"][0]["document_metadata"], fixture.data["sources"]["d"])
        self.assertNotIn("archive_key", fixture.requests[0][0])
        self.assertNotIn("private-model-endpoint", fixture.requests[0][0])
        self.assertIn("confirmation_status", fixture.requests[0][0])

    async def test_final_metadata_budget_does_not_silently_drop_sources(self):
        fixture = Fixture()
        fixture.data["sources"]["d"]["relations"] = [{"type": "association_reference", "source_span": "x" * 6000}]
        backend = fixture.backend()
        await self.bind(backend)
        with self.assertRaisesRegex(ValueError, "budget"):
            await backend.generate("费用", await backend.search("费用"))
        self.assertEqual(fixture.chats, 0)


if __name__ == "__main__":
    unittest.main()
