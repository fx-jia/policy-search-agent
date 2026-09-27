import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from experiments.search_agent.runtime import build_snapshot, VectorCache, RuntimeBackend


class RuntimeTests(unittest.TestCase):
    def test_snapshot_requires_raw_and_safe_hashes(self):
        raw, safe = "原文", "审核原文"
        sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
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
