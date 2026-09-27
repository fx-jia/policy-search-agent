"""Regression cases for v6 failures, using source-preserving IDs and fallback."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.search_agent.adapter import judge_messages
from experiments.search_agent.core import AgentLimits, run
from experiments.search_agent.evidence import Evidence, SEPARATOR, pack_evidence, validate_decision
from experiments.search_agent.runtime import RuntimeBackend
from experiments.evaluation.native_agent_guard import CallLedger


def choose(ids=("s0",), *, action=None, status="supported"):
    return {"action": action or {"type": "finish"}, "constraints": [],
            "needs": [{"text": "回答用户问题", "status": status, "evidence_ids": list(ids)}], "reason": "按原文回答"}


class SegmentContractTests(unittest.TestCase):
    def test_more_than_eight_selected_sources_is_rejected_not_silently_dropped(self):
        views = [Evidence(str(i), str(i), "政策", "原文。", 0) for i in range(9)]
        value = choose([f"s{i}" for i in range(6)])
        value["needs"].append({"text": "另一需求", "status": "supported", "evidence_ids": ["s6", "s7", "s8"]})
        with self.assertRaisesRegex(ValueError, "evidence_source_limit"):
            validate_decision(value, views, "问题")

    def test_context_expansion_cannot_evict_another_selected_clause(self):
        raw = "甲" * 20 + "金额10。" + "乙" * 20 + "期限20。"
        item = Evidence("a", "d", "政策", raw, 0)
        view = Evidence("a", "d", "政策", "金额10。" + SEPARATOR + "期限20。", 0,
                        spans=((20, 25), (45, 50)), truncated=True)
        bound = validate_decision(choose(["s0", "s1"]), [view], "金额和期限？")
        chosen = pack_evidence([item], bound, "金额和期限？", len, budget=40)
        self.assertIn("金额10。", chosen[0].text)
        self.assertIn("期限20。", chosen[0].text)
        self.assertLessEqual(len(chosen[0].text), 40)

    def test_copy_free_id_recovers_original_linebreaks_and_source_offsets(self):
        text = "其他内容。可放宽至\n3\n个月。"
        item = Evidence("a", "doc", "政策", text, 0)
        _, messages = judge_messages("最长多久？", [item], None)
        visible = json.loads(messages[0]["content"])["evidence"][0]
        self.assertEqual(visible.get("segments"), [{"id": "s0", "text": "其他内容。"}, {"id": "s1", "text": "可放宽至\n3\n个月。"}])
        raw = choose(["s1"])
        bound = validate_decision(raw, [item], "最长多久？")
        self.assertEqual(bound["needs"][0]["evidence"], [{"chunk_id": "a", "quote": "可放宽至\n3\n个月。", "_source_span": [5, 15]}])
        self.assertNotIn("evidence", raw["needs"][0])

    def test_second_paragraph_does_not_invent_nonadjacent_article_heading(self):
        text = "第十三条 申请人不服可以复议。\n复查期间申请复议的，复查程序终止。"
        item = Evidence("a", "d", "办法", text, 0)
        bound = validate_decision(choose(["s1"]), [item], "如何处理？")
        self.assertEqual(bound["needs"][0]["evidence"][0]["quote"], "\n复查期间申请复议的，复查程序终止。")

    def test_repeated_text_binds_visible_occurrence_and_document_version(self):
        items = [Evidence("old", "d1", "旧版", "不收费。", 0),
                 Evidence("new", "d2", "新版", "不收费。", 0, spans=((100, 104),), truncated=True)]
        bound = validate_decision(choose(["s1"]), items, "是否收费？")
        self.assertEqual(bound["needs"][0]["evidence"][0], {"chunk_id": "new", "quote": "不收费。", "_source_span": [100, 104]})

    def test_cannot_select_hidden_text_or_supply_a_quote_to_override_id(self):
        view = Evidence("a", "d", "政策", "首段。" + SEPARATOR + "末段。", 0, spans=((0, 3), (90, 93)), truncated=True)
        with self.assertRaisesRegex(ValueError, "unknown_evidence_id"):
            validate_decision(choose(["s2"]), [view], "问题")
        raw = choose(["s0"])
        raw["needs"][0]["evidence"] = [{"chunk_id": "a", "quote": "补贴100万元"}]
        with self.assertRaisesRegex(ValueError, "unexpected_need_field"):
            validate_decision(raw, [view], "问题")

    def test_needs_cannot_carry_the_global_action(self):
        raw = choose()
        raw["needs"][0]["action"] = raw.pop("action")
        with self.assertRaises(ValueError):
            validate_decision(raw, [Evidence("a", "d", "标题", "原文。", 0)], "问题")


class Backend:
    def __init__(self, decisions, results=None):
        self.decisions = list(decisions)
        self.results = list(results or [[Evidence("a", "d", "标题", "原文。", 0)]])
        self.generated = []
        self.control_events = []
        self.metrics = {}
        self.count_tokens = len

    async def search(self, query, document_ids=None):
        return self.results.pop(0)

    def neighbor_directions(self, seed):
        return []

    def remaining_generation_calls(self):
        return 4

    async def decide(self, question, views, previous):
        value = self.decisions.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    async def generate(self, question, selected, *, native=False):
        self.generated.append((copy.deepcopy(selected), native))
        return {"answer": "依据原文回答。[ID:0]", "reference": {}}

    def record_control_event(self, event):
        self.control_events.append(event)


class FallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_decision_generates_once_from_original_order_and_logs_degradation(self):
        original = [Evidence("a", "d", "标题", "不裁切。" * 500, 0), Evidence("b", "d", "标题", "原文。", 1)]
        backend = Backend([{"needs": []}], [original])
        result = await run("问题", backend)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.stop_reason, "controller_fallback")
        self.assertTrue(result.fallback_used)
        self.assertTrue(result.controller_error)
        self.assertEqual(backend.generated, [(original, True)])
        self.assertEqual([e["event"] for e in backend.control_events], ["controller_error", "fallback"])

    async def test_unknown_id_is_not_accepted_but_can_use_initial_rag(self):
        backend = Backend([choose(["invented"])])
        result = await run("问题", backend)
        self.assertEqual(result.controller_error, "unknown_evidence_id")
        self.assertEqual(backend.generated[0][1], True)

    async def test_error_after_supplement_falls_back_to_initial_evidence_only(self):
        first = Evidence("a", "d", "标题", "已知条件。", 0)
        added = Evidence("b", "d", "标题", "补充信息。", 1)
        missing = choose([], status="missing", action={"type": "search_policy", "query": "条件", "need_index": 0})
        backend = Backend([missing, {"needs": []}], [[first], [added]])
        result = await run("问题", backend)
        self.assertEqual(result.stop_reason, "controller_fallback")
        self.assertEqual(backend.generated, [([first], True)])

    async def test_source_or_provider_error_does_not_take_fallback(self):
        backend = Backend([ValueError("runtime_source_changed")])
        result = await run("问题", backend)
        self.assertNotEqual(result.status, "ok")
        self.assertFalse(backend.generated)

    async def test_no_fallback_when_insufficient_time_remains(self):
        backend = Backend([{"needs": []}])
        result = await run("问题", backend, AgentLimits(total_seconds=0.1, search_seconds=0.05))
        self.assertEqual(result.status, "invalid_decision")
        self.assertFalse(backend.generated)

    async def test_unsettled_provider_usage_blocks_native_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = CallLedger(Path(temp) / "ledger.jsonl", run_limit=10000, question_limit=10000, call_limit=4)
            ledger.start("q")
            ledger.reserve("q:1", 100, {})
            backend = RuntimeBackend({}, {"count_tokens": len, "ledger": ledger}, None, {})
            with self.assertRaisesRegex(ValueError, "unresolved_provider_usage"):
                await backend.generate("问题", [], native=True)


if __name__ == "__main__":
    unittest.main()
