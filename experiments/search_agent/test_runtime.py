import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from experiments.search_agent.runtime import build_snapshot, VectorCache, RuntimeBackend


class RuntimeTests(unittest.TestCase):
    def test_snapshot_requires_raw_and_safe_hashes(self):
        raw, safe = "原文", "审核原文"
        def sha(s):
            return hashlib.sha256(s.encode()).hexdigest()
        rows = [{"id": "a", "document_id": "d", "order": 0, "content_sha256": sha(raw)}]
        config = {"review": {"entries": {"a": {"raw_sha256": sha(raw), "sanitized_content": safe, "safe_sha256": sha(safe), "safe_title": "标题", "source_document_id": "1"}}},
                  "document_map": {"documents": {"d": "1"}}, "scope_document_ids": ["d"], "chat_config": {"dataset_ids": ["kb"]}}
        self.assertEqual(build_snapshot(config, rows).by_id["a"].text, safe)
        rows[0]["content_sha256"] = "changed"
        with self.assertRaises(ValueError):
            build_snapshot(config, rows)

    def test_vector_cache_does_not_count_hit_as_physical_embedding(self):
        cache = VectorCache(3)
        calls = []
        def compute():
            calls.append(1)
            return ([1, 2], 5)
        self.assertEqual(cache.get(("model", "q"), compute), ([1, 2], 5))
        self.assertEqual(cache.get(("model", "q"), compute), ([1, 2], 0))
        self.assertEqual(cache.calls, 1)
        self.assertEqual(cache.hits, 1)
        cache.get(("other", "q"), compute)
        self.assertEqual(cache.calls, 2)

    def test_vector_budget_blocks_before_physical_call(self):
        cache = VectorCache(1)
        cache.get("a", lambda: ([1], 2))
        with self.assertRaises(ValueError):
            cache.get("b", lambda: self.fail("must not send"))


class JudgeRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolution_uses_monitored_json_messages_and_user_provenance(self):
        import json
        from experiments.search_agent.resolution import prepare_resolution_context
        answer = {"kind": "question", "clarification": "",
                  "user_context": [{"message_id": "t0-user", "quote": "上海"}], "reset": False}
        chat = SimpleNamespace(async_chat=AsyncMock(return_value=json.dumps(answer, ensure_ascii=False)))
        ledger = SimpleNamespace(assert_open=lambda: None, calls={}, question_id="q", limits=(10000, 10000, 4))
        backend = RuntimeBackend({}, {"count_tokens": len, "chat": chat, "ledger": ledger, "event": lambda e: None}, None, {})
        context = prepare_resolution_context("那退休人员呢？", [{"user": "上海医保条件？", "assistant": "旧答复"}], None, len)
        result = await backend.resolve(context)
        self.assertIn("上海医保条件？", result.question)
        self.assertIn("那退休人员呢？", result.question)
        self.assertEqual(backend.current["stage"], "resolve")
        system, messages, options = chat.async_chat.call_args.args
        self.assertEqual(backend.current["approved_resolution_messages"], [{"role": "system", "content": system}] + messages)
        self.assertEqual(options["response_format"], {"type": "json_object"})

    async def test_resolution_blocks_before_send_if_no_final_call_can_be_reserved(self):
        chat = SimpleNamespace(async_chat=AsyncMock())
        ledger = SimpleNamespace(assert_open=lambda: None, calls={str(i): {"question_id": "q"} for i in range(3)}, question_id="q", limits=(10000, 10000, 4))
        backend = RuntimeBackend({}, {"count_tokens": len, "chat": chat, "ledger": ledger, "event": lambda e: None}, None, {})
        with self.assertRaisesRegex(ValueError, "final_call_reserved"):
            await backend.resolve({})
        self.assertEqual(chat.async_chat.await_count, 0)

    async def test_invalid_resolution_does_not_retry_or_accept_assistant_money(self):
        import json
        from experiments.search_agent.resolution import prepare_resolution_context
        answer = {"kind": "clarification", "clarification": "上海报销380元吗？", "user_context": [], "reset": False}
        chat = SimpleNamespace(async_chat=AsyncMock(return_value=json.dumps(answer)))
        ledger = SimpleNamespace(assert_open=lambda: None, calls={}, question_id="q", limits=(10000, 10000, 4))
        backend = RuntimeBackend({}, {"count_tokens": len, "chat": chat, "ledger": ledger, "event": lambda e: None}, None, {})
        context = prepare_resolution_context("退休人员呢？", [{"user": "上海医保条件？", "assistant": "380元"}], None, len)
        with self.assertRaisesRegex(ValueError, "number"):
            await backend.resolve(context)
        self.assertEqual(chat.async_chat.await_count, 1)

    async def test_json_mode_requested_and_invalid_output_preserved_without_retry(self):
        malformed = '{"needs": []} trailing text'
        chat = SimpleNamespace(async_chat=AsyncMock(return_value=malformed))
        events = []
        backend = RuntimeBackend({}, {"count_tokens": len, "chat": chat,
                                      "ledger": SimpleNamespace(assert_open=lambda: None), "event": events.append}, None, {})
        with self.assertRaises(ValueError):
            await backend.decide("问题", [], None)
        self.assertEqual(chat.async_chat.await_count, 1)
        self.assertEqual(chat.async_chat.call_args.args[2]["response_format"], {"type": "json_object"})
        self.assertEqual(events[0], {"event": "judge_raw_response", "text": malformed})


if __name__ == "__main__":
    unittest.main()
