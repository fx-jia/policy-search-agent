"""Real controller flow with deterministic model/service substitutes."""
import unittest

from experiments.search_agent.adapter import Snapshot, judge_messages
from experiments.search_agent.core import AgentLimits, run
from experiments.search_agent.evidence import Evidence, evidence_segments, validate_decision


def missing(action):
    action = dict(action)
    stop = action.pop("stop_kind", None)
    value = {"action": action, "constraints": [], "needs": [{"text": "办理期限", "status": "missing", "evidence_ids": []}], "reason": "期限尚需核验"}
    if stop:
        value["stop_kind"] = stop
    return value


class RereadBackend:
    count_tokens = staticmethod(len)

    def __init__(self):
        self.item = Evidence("a", "d", "办法", "受理材料和流程。" * 8 + "办理期限为十天。", 0)
        self.metrics, self.events, self.visible = {}, [], []
        self.calls = 0
        self.generated = []

    async def search(self, question, document_ids=None):
        self.calls += 1
        return [self.item]

    def record_control_event(self, event):
        self.events.append(event)

    def neighbor_directions(self, seed):
        return []

    def remaining_generation_calls(self):
        return 4

    async def decide(self, question, views, previous):
        self.visible.append(views)
        if previous is None:
            return missing({"type": "read_chunk", "chunk_id": "a", "need_index": 0})
        ids = [sid for sid, s in evidence_segments(views).items() if "办理期限为十天" in s["quote"]]
        return {"constraints": [], "needs": [{"text": "办理期限", "status": "supported", "evidence_ids": ids}],
                "action": {"type": "finish"}, "reason": "找到期限"}

    async def generate(self, question, selected, *, native=False):
        self.generated.append(selected)
        return {"answer": "十天[ID:0]", "reference": {}}


class RereadTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_chunk_new_span_counts_as_progress_without_embedding(self):
        backend = RereadBackend()
        result = await run("受理材料和流程？", backend, AgentLimits(controller_evidence_tokens=35, final_evidence_tokens=60))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.stop_reason, "sufficient")
        self.assertEqual(backend.calls, 1)
        self.assertEqual(len(backend.visible), 2)
        self.assertNotIn("办理期限为十天", backend.visible[0][0].text)
        self.assertIn("办理期限为十天", backend.visible[1][0].text)
        self.assertIn("办理期限为十天", backend.generated[0][0].text)
        tool = next(t for t in result.trace if t["event"] == "tool")
        self.assertEqual(tool["new_chunk_ids"], [])
        self.assertTrue(tool["new_spans"])

    def test_hidden_tool_rejects_unknown_or_untruncated_sources(self):
        view = Evidence("a", "d", "办法", "完整原文。", 0)
        for cid in ("a", "invented"):
            with self.assertRaises(ValueError):
                validate_decision(missing({"type": "read_chunk", "chunk_id": cid, "need_index": 0}), [view], "期限？")

    def test_neighbor_availability_is_exposed_without_hidden_text(self):
        seed = Evidence("a", "d", "办法", "申请规定。", 0)
        nxt = Evidence("b", "d", "办法", "隐藏的期限。", 1)
        from experiments.search_agent.runtime import RuntimeBackend
        backend = RuntimeBackend({}, {"count_tokens": len}, Snapshot([seed, nxt]), {})
        self.assertEqual(backend.neighbor_directions(seed), ["next"])
        _, history = judge_messages("期限？", [seed], None, neighbors={"a": ["next"]})
        self.assertIn('"next"', history[0]["content"])
        self.assertNotIn("隐藏的期限", history[0]["content"])


class GapTests(unittest.IsolatedAsyncioTestCase):
    def test_observed_root_stop_reason_has_one_validated_location(self):
        item = Evidence("a", "d", "办法", "办理期限为十天。", 0)
        value = {"action":{"type":"finish"},"constraints":[],"needs":[{"text":"期限", "status":"supported", "evidence_ids":["s0"]}],
                 "reason":"已有期限依据", "stop_kind":"sufficient"}
        checked = validate_decision(value, [item], "期限？")
        self.assertEqual(checked['stop_kind'], 'sufficient')
        value['stop_kind'] = 'invented'
        with self.assertRaises(ValueError):
            validate_decision(value, [item], "期限？")

    async def test_reconnect_consumption_leaves_a_call_for_final_answer(self):
        class Backend(RereadBackend):
            def __init__(self):
                super().__init__()
                self.item = Evidence("a", "d", "办法", "申请材料。", 0)
                self.slots = 4
                self.decisions = 0
            def remaining_generation_calls(self):
                return self.slots
            async def decide(self, *args):
                self.decisions += 1
                self.slots -= 2 if self.decisions == 1 else 1
                return missing({"type":"search_policy", "query":"办理期限"+"要求"*self.decisions, "need_index":0})
            async def search(self, *args, **kwargs):
                self.calls += 1
                return [Evidence(str(self.calls),str(self.calls),'规定','仍无期限。',0)]
            async def generate(self, question, selected, *, native=False):
                if self.slots <= 0:
                    raise ValueError('question_call_limit')
                self.slots -= 1
                return await super().generate(question, selected, native=native)
        backend = Backend()
        result = await run("办理期限？", backend)
        self.assertEqual(result.status, 'ok')
        self.assertEqual(result.stop_reason, 'final_call_reserved')
        self.assertEqual(backend.decisions, 2)
        self.assertEqual(backend.slots, 0)

    async def test_already_seen_neighbor_cannot_suppress_targeted_search(self):
        class Backend(RereadBackend):
            def __init__(self):
                super().__init__()
                self.items = [Evidence("a", "d", "办法", "办理期限见其他规定。", 0), Evidence("b", "d", "办法", "申请材料。", 1)]
            async def search(self, question, document_ids=None):
                self.calls += 1
                return self.items if self.calls == 1 else [Evidence("c", "other", "新版", "办理期限为十天。", 0)]
            def neighbor_directions(self, seed):
                return []  # Runtime must filter observed neighbors.
            async def decide(self, question, views, previous):
                if previous is None:
                    return missing({"type": "search_policy", "query": "新版办理期限", "need_index": 0})
                return await super().decide(question, views, previous)
        from experiments.search_agent.runtime import RuntimeBackend
        from experiments.search_agent.adapter import Snapshot
        backend = Backend()
        real = RuntimeBackend({}, {"count_tokens": len}, Snapshot(backend.items), {})
        real.observed = {e.id: e for e in backend.items}
        self.assertEqual(real.neighbor_directions(backend.items[0]), [])
        result = await run("办理期限？", backend)
        self.assertEqual(result.stop_reason, "sufficient")
        self.assertEqual(backend.calls, 2)

    async def test_same_need_can_read_distinct_spans_twice(self):
        class Backend(RereadBackend):
            def __init__(self):
                super().__init__()
                self.item = Evidence("a", "d", "办法", ("已知条款" + "甲" * 995 + "。") * 5 + ("办理期限" + "乙" * 345 + "。") * 20, 0)
            async def decide(self, question, views, previous):
                return missing({"type": "read_chunk", "chunk_id": "a", "need_index": 0})
        result = await run("已知条款？", Backend())
        tools = [t for t in result.trace if t['event'] == 'tool']
        self.assertEqual(len(tools), 2)
        self.assertNotEqual(tools[0]['new_spans'], tools[1]['new_spans'])
        self.assertEqual(result.status, "ok")

    async def test_current_decision_anchors_are_reserved_before_reread(self):
        class Backend(RereadBackend):
            def __init__(self):
                super().__init__()
                self.item = Evidence("a", "d", "办法", ("已知" + "甲" * 897 + "。") * 5 + "办理期限" + "乙" * 695 + "。", 0)
            async def decide(self, question, views, previous):
                ids = [sid for sid, s in evidence_segments(views).items() if s['quote'].startswith('已知')]
                return {"action": {"type":"read_chunk", "chunk_id":"a", "need_index":1},"constraints":[],
                        "needs":[{"text":"已知条件", "status":"supported", "evidence_ids":ids},
                                 {"text":"办理期限", "status":"missing", "evidence_ids":[]}], "reason":"补读期限"}
        backend = Backend()
        result = await run("已知条件？", backend)
        self.assertEqual(result.status, "ok")
        self.assertTrue(backend.generated)
        self.assertNotEqual(result.stop_reason, "technical_error")
        self.assertFalse(result.fallback_used)

    async def test_known_gap_finish_reads_available_neighbor_before_search(self):
        class Backend(RereadBackend):
            def __init__(self):
                super().__init__()
                self.item = Evidence("a", "d", "办法", "办理评估需提交材料，时限见后续规定。", 0)
                self.reads = []

            def neighbor_directions(self, seed):
                return ["next"] if seed.id == "a" else []

            async def neighbors(self, seed, direction):
                self.reads.append((seed.id, direction))
                return [Evidence("b", "d", "办法", "办理期限为十天。", 1)]

            async def decide(self, question, views, previous):
                if previous is None:
                    return missing({"type": "finish", "stop_kind": "no_more_evidence"})
                return await super().decide(question, views, previous)

        backend = Backend()
        result = await run("办理期限？", backend)
        self.assertEqual(result.stop_reason, "sufficient")
        self.assertEqual(backend.calls, 1)
        self.assertEqual(backend.reads, [("a", "next")])
        self.assertIn("办理期限为十天", "".join(e.text for e in backend.generated[0]))
        self.assertTrue(any(t['event'] == 'action_override' for t in result.trace))

    async def test_missing_user_information_does_not_force_policy_search(self):
        class Backend(RereadBackend):
            async def decide(self, *args):
                return missing({"type": "finish", "stop_kind": "missing_user_information"})
        backend = Backend()
        result = await run("我可以报销多少？", backend)
        self.assertEqual(result.stop_reason, "missing_user_information")
        self.assertEqual(backend.calls, 1)

    async def test_gap_without_readable_source_searches_once_then_stops_duplicate(self):
        class Backend(RereadBackend):
            def __init__(self):
                super().__init__()
                self.item = Evidence("a", "d", "办法", "背景。", 0)
            async def decide(self, *args):
                return missing({"type": "finish", "stop_kind": "no_more_evidence"})
        backend = Backend()
        result = await run("办理期限？", backend)
        self.assertEqual(backend.calls, 2)
        self.assertEqual(result.stop_reason, "no_progress")


if __name__ == "__main__":
    unittest.main()
