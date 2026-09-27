"""Offline behavioral contracts; no service or model dependencies."""
import asyncio
import unittest

from experiments.search_agent.core import AgentLimits, run
from experiments.search_agent.evidence import Evidence, evidence_segments, make_views, pack_evidence, validate_decision
from experiments.search_agent.cache import RequestCache


def evidence(cid="a", text="参保人不收费。", doc="doc"):
    return Evidence(cid, doc, "政策原文", text, 0, {})


def decision(ids=("s0",), action=None):
    return {"constraints": [], "needs": [{"text": "是否收费", "status": "supported",
            "evidence_ids": list(ids)}],
            "action": action or {"type": "finish"}, "reason": "证据已经明确"}


class Backend:
    def __init__(self, decisions=None, results=None):
        self.decisions = list(decisions or [decision()])
        self.results = list(results or [[evidence()]])
        self.searches = []
        self.generated = []
        self.count_tokens = len
        self.metrics = {}

    def record_control_event(self, event):
        pass

    def neighbor_directions(self, seed):
        return []

    def remaining_generation_calls(self):
        return 4

    async def search(self, query, document_ids=None):
        self.searches.append((query, document_ids))
        return self.results.pop(0)

    async def neighbors(self, seed, direction):
        return [evidence("b", "补充规定。")]

    async def decide(self, question, views, previous):
        return self.decisions.pop(0)

    async def generate(self, question, selected, *, native=False):
        self.generated.append((question, selected))
        return {"answer": "不收费。[ID:0]", "reference": {"chunks": [{"id": e.id} for e in selected]}}


class CoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_sufficient_finishes_with_one_decision_and_one_generation(self):
        b = Backend()
        r = await run("是否收费？", b, AgentLimits())
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.stop_reason, "sufficient")
        self.assertEqual(len(b.searches), 1)
        self.assertEqual(len(b.generated), 1)

    async def test_missing_need_triggers_one_targeted_search(self):
        d = decision(action={"type": "search_policy", "query": "办理期限", "need_index": 0})
        d["needs"][0].update(status="missing", evidence_ids=[])
        b = Backend([d, decision(["s1"])], [[evidence()], [evidence("b")]])
        r = await run("是否收费及办理期限？", b, AgentLimits())
        self.assertEqual(len(b.searches), 2)
        self.assertIn("是否收费及办理期限？", b.searches[1][0])
        self.assertEqual(r.stop_reason, "sufficient")

    async def test_no_new_evidence_stops(self):
        d = decision(action={"type": "search_policy", "query": "期限", "need_index": 0})
        d["needs"][0].update(status="missing", evidence_ids=[])
        b = Backend([d], [[evidence()], [evidence()]])
        r = await run("期限？", b, AgentLimits())
        self.assertEqual(r.stop_reason, "no_progress")
        self.assertEqual(len(b.generated), 1)

    async def test_invalid_id_can_be_configured_to_fail_without_generation(self):
        b = Backend([decision(["invented"])])
        r = await run("是否收费？", b, AgentLimits(fallback_on_invalid_decision=False))
        self.assertEqual(r.status, "invalid_decision")
        self.assertFalse(b.generated)

    async def test_document_scope_must_have_been_observed(self):
        d = decision(action={"type": "search_policy", "query": "期限", "need_index": 0, "document_ids": ["secret"]})
        d["needs"][0].update(status="missing", evidence_ids=[])
        b = Backend([d])
        r = await run("期限？", b, AgentLimits())
        self.assertEqual(r.controller_error, "unseen_document_scope")
        self.assertEqual(len(b.searches), 1)

    async def test_max_two_actions_and_three_decisions(self):
        ds = []
        for query in ["期限", "程序", "材料"]:
            d = decision(action={"type": "search_policy", "query": query, "need_index": 0})
            d["needs"][0].update(status="missing", evidence_ids=[])
            ds.append(d)
        b = Backend(ds, [[evidence()], [evidence("b")], [evidence("c")]])
        r = await run("是否收费？", b, AgentLimits())
        self.assertEqual(len(b.searches), 3)
        self.assertEqual(r.stop_reason, "action_limit")
        self.assertEqual(len(b.generated), 1)

    async def test_empty_retrieval_is_a_missing_evidence_result(self):
        b = Backend(results=[[]])
        r = await run("未知政策？", b, AgentLimits())
        self.assertEqual(r.stop_reason, "no_evidence")
        self.assertEqual(len(b.generated), 1)

    async def test_disabled_enhancement_preserves_full_chunks_and_order(self):
        original = [evidence("a", "无关。"*500), evidence("b")]
        b = Backend(results=[original])
        r = await run("是否收费？", b, AgentLimits(enhance=False))
        self.assertEqual(b.generated[0][1], original)
        self.assertEqual(r.stop_reason, "native_replay")

    async def test_bad_model_output_without_fallback_remains_technical_failure(self):
        b = Backend([{"needs": []}])
        r = await run("是否收费？", b, AgentLimits(fallback_on_invalid_decision=False))
        self.assertEqual(r.status, "invalid_decision")
        self.assertFalse(b.generated)

    async def test_request_timeout(self):
        class Slow(Backend):
            async def search(self, *args):
                await asyncio.sleep(1)
        r = await run("问题", Slow(), AgentLimits(search_seconds=0.01, total_seconds=0.02))
        self.assertEqual(r.status, "timeout")


class EvidenceTests(unittest.TestCase):
    def test_repeated_quote_uses_occurrence_that_judge_actually_saw(self):
        raw = "乙类" + "普" * 100 + "不收费。" + "甲类" + "新" * 80 + "不收费。"
        items = [evidence(text=raw)]
        views = make_views(items, "甲类", len, 100)
        self.assertIn("甲类", views[0].text)
        d = validate_decision(decision(), views, "甲类是否收费？")
        selected = pack_evidence(items, d, "甲类", len, 110)
        self.assertIn("甲类", selected[0].text)
        self.assertNotIn("乙类", selected[0].text)

    def test_action_rejects_unknown_executable_fields(self):
        d = decision(action={"type": "search_policy", "query": "期限", "need_index": 0, "ignored_key": 1})
        d["needs"][0].update(status="missing", evidence_ids=[])
        with self.assertRaises(ValueError):
            validate_decision(d, [evidence()], "收费期限？")

    def test_coverage_selection_does_not_let_long_first_chunk_evict_answer(self):
        e = [evidence("long", "无关内容。"*500), evidence("answer")]
        views = make_views(e, "是否收费？", len, 120)
        sid = next(sid for sid, s in evidence_segments(views).items() if s["chunk_id"] == "answer")
        d = validate_decision(decision([sid]), views, "是否收费？")
        selected = pack_evidence(e, d, "是否收费？", len, 120)
        self.assertIn("answer", [c.id for c in selected])
        self.assertLessEqual(sum(len(c.text) for c in selected), 120)

    def test_different_document_versions_not_text_deduplicated(self):
        e = [evidence("old", doc="old"), evidence("new", doc="new")]
        selected = pack_evidence(e, None, "收费", len, 100)
        self.assertEqual(len(selected), 2)

    def test_fabricated_action_numbers_rejected(self):
        d = decision(action={"type": "search_policy", "query": "100万元", "need_index": 0})
        d["needs"][0].update(status="missing", evidence_ids=[])
        with self.assertRaises(ValueError):
            validate_decision(d, [evidence()], "是否收费？")


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_identical_calls_coalesce_and_models_are_separate(self):
        cache = RequestCache()
        calls = []
        async def calculate():
            calls.append(1)
            await asyncio.sleep(0)
            return [1, 2]
        a, b = await asyncio.gather(cache.get(("model1", "query"), calculate), cache.get(("model1", "query"), calculate))
        self.assertEqual((a, b), ([1, 2], [1, 2]))
        self.assertEqual(len(calls), 1)
        await cache.get(("model2", "query"), calculate)
        self.assertEqual(len(calls), 2)

    async def test_failed_call_is_not_cached_as_success(self):
        c = RequestCache()
        async def fail():
            raise ValueError("failure")
        with self.assertRaises(ValueError):
            await c.get("q", fail)
        self.assertEqual(await c.get("q", lambda: asyncio.sleep(0, result=7)), 7)


if __name__ == "__main__":
    unittest.main()
