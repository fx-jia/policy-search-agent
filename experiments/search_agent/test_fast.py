import asyncio
import unittest

from experiments.search_agent.core import AgentLimits
from experiments.search_agent.evidence import Evidence, validate_views
from experiments.search_agent.fast import context_text, run, select_context


def chunk(cid, text, title="政策"):
    return Evidence(cid, "doc-" + cid, title, text, 0)


class ContextTests(unittest.TestCase):
    def test_ranked_unique_sources_and_versions_remain_distinct(self):
        first = chunk("a", "适用于职工，2024年标准为100元。")
        other = chunk("b", "适用于居民，2025年标准为200元。")
        chosen = select_context([first, first, other, chunk("c", "第三条。"), chunk("d", "第四条。")], len, 1000)
        self.assertEqual([e.id for e in chosen], ["a", "b", "c"])
        self.assertEqual(chosen[1].text, other.text)
        validate_views(chosen, [first, other, chunk("c", "第三条。")])

    def test_budget_includes_source_headers_and_keeps_whole_paragraph_prefix(self):
        source = chunk("a", "仅适用于居民。\n每年最多100元。\n下一段内容比较长，不得只保留数字。")
        budget = len(context_text([chunk("a", "仅适用于居民。\n每年最多100元。")]))
        selected = select_context([source], len, budget)
        self.assertEqual(selected[0].text, "仅适用于居民。\n每年最多100元。")
        self.assertTrue(selected[0].truncated)
        self.assertLessEqual(len(context_text(selected)), budget)
        validate_views(selected, [source])

    def test_oversized_first_paragraph_is_not_cut_or_skipped_inside_chunk(self):
        source = chunk("a", "限定条件" * 60 + "。\n可报销100元。")
        self.assertEqual(select_context([source], len, 100), [])

    def test_conflicting_duplicate_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting_chunk"):
            select_context([chunk("a", "居民100元"), chunk("a", "职工200元")], len, 1000)

    def test_soft_linebreak_cannot_separate_numeric_condition(self):
        prefix = "居民医保政策范围内住院医疗费用基金支付比例稳定在"
        source = chunk("a", prefix + "\n70%\n左右。\n后续条款很长。" * 5)
        budget = len(context_text([chunk("a", prefix)]))
        self.assertEqual(select_context([source], len, budget), [])


class Backend:
    count_tokens = staticmethod(len)

    def __init__(self, items):
        self.items = items
        self.calls = []
        self.metrics = {}

    async def search(self, question):
        self.calls.append(("search", question))
        return self.items

    async def generate(self, question, selected, native=False):
        self.calls.append(("generate", question, selected, native))
        return {"answer": "政策回答[ID:0]", "reference": {"chunks": []}}


class RunTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_search_one_answer_no_controller_and_original_question(self):
        backend = Backend([chunk("a", "居民100元。")])
        result = await run("2024年江苏居民能否报销？", backend)
        self.assertEqual(result.status, "ok")
        self.assertEqual([c[0] for c in backend.calls], ["search", "generate"])
        self.assertEqual(backend.calls[0][1], backend.calls[1][1])
        self.assertFalse(backend.calls[1][3])
        self.assertEqual(result.selected[0]["chunk_id"], "a")
        self.assertIn("retrieval_ms", result.usage)
        self.assertIn("generation_ms", result.usage)

    async def test_no_fitting_evidence_does_not_generate_from_empty_context(self):
        backend = Backend([chunk("a", "一个超长段落。" * 500)])
        result = await run("问题", backend)
        self.assertEqual(result.status, "context_unavailable")
        self.assertEqual(len(backend.calls), 1)

    async def test_timeout_cancels_inflight_search_without_generating(self):
        backend = Backend([])
        cancelled = asyncio.Event()

        async def slow_search(_):
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.set()

        backend.search = slow_search
        result = await run("问题", backend, AgentLimits(total_seconds=.01, search_seconds=.005))
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.stop_reason, "retrieval_timeout")
        self.assertTrue(cancelled.is_set())
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
