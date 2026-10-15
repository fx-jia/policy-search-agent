"""Exercise conversation state around the real core with external call substitutes."""
import asyncio
import copy
import unittest

from experiments.search_agent.conversation import Conversation, ConversationLimits
from experiments.search_agent.core import AgentLimits, AgentResult
from experiments.search_agent.resolution import Resolution
from experiments.search_agent.evidence import Evidence


class Backend:
    def __init__(self, resolution=None, *, label="fresh", decisions=None, answer="已核实。[ID:0]"):
        self.resolution = resolution
        self.label = label
        self.decisions = list(decisions or [])
        self.answer = answer
        self.contexts = []
        self.searches = []
        self.generated = []
        self.metrics = {"generation_calls": 0}
        self.count_tokens = lambda value: (len(value) + 3) // 4

    async def resolve(self, context):
        self.contexts.append(copy.deepcopy(context))
        self.metrics["generation_calls"] += 1
        if isinstance(self.resolution, Exception):
            raise self.resolution
        value = self.resolution(context) if callable(self.resolution) else self.resolution
        value = value or {"kind": "question", "question": context["current"]["content"]}
        return Resolution(**value)

    def remaining_generation_calls(self):
        return 4 - self.metrics["generation_calls"]

    def record_control_event(self, event):
        pass

    def neighbor_directions(self, seed):
        return []

    async def search(self, question, document_ids=None):
        self.searches.append(question)
        return [Evidence(f"{self.label}-{len(self.searches)}", self.label, "核验政策", "参保人不收费。", 0)]

    async def decide(self, question, views, previous):
        self.metrics["generation_calls"] += 1
        if self.decisions:
            return self.decisions.pop(0)
        return {"constraints": [], "needs": [{"text": "是否收费", "status": "supported", "evidence_ids": ["s0"]}],
                "action": {"type": "finish"}, "reason": "依据明确"}

    async def generate(self, question, selected, *, native=False):
        self.metrics["generation_calls"] += 1
        self.generated.append(question)
        return {"answer": self.answer, "reference": {"chunks": [{"id": item.id} for item in selected]}}


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    def test_conversation_limits_cannot_exceed_shared_history_bound(self):
        for limits in (ConversationLimits(max_rounds=7), ConversationLimits(history_tokens=3001)):
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, "invalid_conversation_limits"):
                Conversation("invalid", limits=limits)

    async def test_invalid_agent_limits_stop_before_any_model_call(self):
        conversation = Conversation("invalid")
        for limits in (AgentLimits(max_actions=3), AgentLimits(max_actions=-1),
                       AgentLimits(max_decisions=4), AgentLimits(max_decisions=0),
                       AgentLimits(total_seconds=61), AgentLimits(search_seconds=0),
                       AgentLimits(total_seconds=20, search_seconds=21)):
            with self.subTest(limits=limits):
                backend = Backend()
                with self.assertRaisesRegex(ValueError, "invalid_agent_limits"):
                    await conversation.ask("医保？", backend, limits)
                self.assertEqual(backend.metrics["generation_calls"], 0)
                self.assertEqual(backend.contexts, [])
                self.assertEqual(backend.searches, [])
        self.assertEqual(conversation.turns, [])

    async def test_follow_up_uses_resolved_question_and_fresh_evidence(self):
        conversation = Conversation("session")
        first = await conversation.ask("上海2026年居民医保是否收费？", Backend(label="first"))
        backend = Backend({"kind": "question", "question": "上海2026年居民医保异地办理是否收费？"}, label="second")
        result = await conversation.ask("异地办理呢？", backend)
        self.assertIsInstance(result, AgentResult)
        self.assertEqual((result.session_id, result.question, result.status), ("session", "异地办理呢？", "ok"))
        self.assertEqual(result.effective_question, "上海2026年居民医保异地办理是否收费？\n用户本轮原话：异地办理呢？")
        self.assertEqual(next(event for event in result.trace if event["event"] == "initial_search")["query"], result.effective_question)
        self.assertEqual(backend.generated, [result.effective_question])
        self.assertEqual(result.reference["chunks"], [{"id": "second-1"}])
        self.assertEqual(first.reference["chunks"], [{"id": "first-1"}])
        self.assertEqual(backend.contexts[0]["history"][0]["content"], "上海2026年居民医保是否收费？")
        self.assertEqual(conversation.turns[-1]["effective_question"], result.effective_question)

    async def test_current_correction_and_negation_are_preserved_verbatim(self):
        conversation = Conversation("correction")
        await conversation.ask("上海2026年居民医保是否收费？", Backend())
        raw = "改为北京2025年，不是上海，也不是职工医保。"
        result = await conversation.ask(raw, Backend({"kind": "question", "question": "北京2025年居民医保是否收费？"}))
        self.assertEqual(result.effective_question, "北京2025年居民医保是否收费？\n用户本轮原话：" + raw)
        self.assertEqual(conversation.turns[-1]["user"], raw)

    async def test_clarification_has_no_core_calls_and_can_be_continued(self):
        conversation = Conversation("pending")
        backend = Backend({"kind": "clarification", "clarification": "请补充参保类型。"})
        result = await conversation.ask("上海2026年异地办理是否收费？", backend)
        self.assertEqual((result.status, result.stop_reason, result.answer),
                         ("needs_clarification", "missing_user_information", "请补充参保类型。"))
        self.assertEqual(result.usage["generation_calls"], 1)
        self.assertEqual(backend.searches, [])
        self.assertEqual(backend.generated, [])
        self.assertEqual(conversation.turns, [])
        self.assertEqual(conversation.pending, {"question": "上海2026年异地办理是否收费？", "clarification": "请补充参保类型。", "messages": []})
        backend = Backend({"kind": "question", "question": "上海2026年居民医保异地办理是否收费？"})
        result = await conversation.ask("居民医保。", backend)
        self.assertEqual(result.status, "ok")
        self.assertEqual(backend.contexts[0]["pending"]["question"]["content"], "上海2026年异地办理是否收费？")
        self.assertIsNone(conversation.pending)
        self.assertEqual(conversation.turns[0]["user"], "上海2026年异地办理是否收费？\n居民医保。")

    async def test_repeated_clarifications_keep_all_raw_supplements(self):
        conversation = Conversation("pending")
        await conversation.ask("上海2026年待遇？", Backend({"kind": "clarification", "clarification": "参保类型？"}))
        await conversation.ask("居民医保。", Backend({"kind": "clarification", "clarification": "年龄？"}))
        self.assertEqual(conversation.pending["messages"], ["居民医保。"])
        backend = Backend({"kind": "question", "question": "上海2026年成年居民医保待遇？"})
        result = await conversation.ask("成年人。", backend)
        self.assertEqual(result.status, "ok")
        self.assertEqual(conversation.turns[0]["user"], "上海2026年待遇？\n居民医保。\n成年人。")
        self.assertIsNone(conversation.pending)

    async def test_core_missing_user_information_creates_pending(self):
        value = {"constraints": [], "needs": [{"text": "是否收费", "status": "missing", "evidence_ids": []}],
                 "action": {"type": "finish"}, "stop_kind": "missing_user_information", "reason": "需参保类型"}
        conversation = Conversation("core-pending")
        result = await conversation.ask("上海是否收费？", Backend(decisions=[value], answer="请补充参保类型。"))
        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual(result.stop_reason, "missing_user_information")
        self.assertEqual(conversation.turns, [])
        self.assertEqual(conversation.pending["question"], "上海是否收费？")
        self.assertEqual(conversation.pending["clarification"], "请补充参保类型。")

    async def test_reset_discards_old_history_only_after_success(self):
        conversation = Conversation("topic")
        await conversation.ask("上海2026年医保？", Backend())
        await conversation.ask("它是否收费？", Backend({"kind": "clarification", "clarification": "具体业务？"}))
        raw = "换个问题，北京停车怎么办？"
        result = await conversation.ask(raw, Backend({"kind": "question", "question": "北京停车怎么办？", "reset": True}))
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(conversation.turns), 1)
        self.assertEqual(conversation.turns[0]["user"], raw)
        self.assertIsNone(conversation.pending)
        next_backend = Backend()
        await conversation.ask("线上办理呢？", next_backend)
        contents = [message["content"] for message in next_backend.contexts[0]["history"]]
        self.assertNotIn("上海2026年医保？", contents)
        self.assertFalse(next_backend.contexts[0]["history_truncated"])

    async def test_failed_reset_retains_history_and_pending(self):
        conversation = Conversation("topic")
        await conversation.ask("上海医保？", Backend())
        await conversation.ask("它收费吗？", Backend({"kind": "clarification", "clarification": "哪项业务？"}))
        old_turns, old_pending = conversation.turns, conversation.pending
        result = await conversation.ask("换成停车。", Backend({"kind": "question", "question": "停车？", "reset": True}, answer=""))
        self.assertEqual(result.status, "generation_error")
        self.assertEqual(conversation.turns, old_turns)
        self.assertEqual(conversation.pending, old_pending)

    async def test_resolver_failure_rolls_back_pending(self):
        conversation = Conversation("rollback")
        await conversation.ask("上海医保？", Backend({"kind": "clarification", "clarification": "年份？"}))
        old_pending = conversation.pending
        result = await conversation.ask("2026年。", Backend(ValueError("invalid_resolution")))
        self.assertEqual((result.status, result.stop_reason, result.error), ("technical_error", "resolution_error", "invalid_resolution"))
        self.assertEqual(conversation.pending, old_pending)
        self.assertEqual(conversation.turns, [])

    async def test_cancellation_does_not_commit_or_remove_pending(self):
        conversation = Conversation("cancel")
        await conversation.ask("医保？", Backend({"kind": "clarification", "clarification": "地区？"}))
        old_pending = conversation.pending
        entered = asyncio.Event()

        class Blocking(Backend):
            async def resolve(self, context):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(conversation.ask("上海。", Blocking()))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(conversation.pending, old_pending)
        self.assertEqual(conversation.turns, [])

    async def test_returned_state_cannot_mutate_the_conversation(self):
        conversation = Conversation("copies")
        await conversation.ask("上海医保？", Backend())
        turns = conversation.turns
        turns[0]["user"] = "污染"
        turns.clear()
        await conversation.ask("它收费吗？", Backend({"kind": "clarification", "clarification": "业务？"}))
        pending = conversation.pending
        pending["question"] = "污染"
        pending["messages"].append("污染")
        backend = Backend()
        await conversation.ask("登记。", backend)
        self.assertEqual(backend.contexts[0]["history"][0]["content"], "上海医保？")
        self.assertEqual(backend.contexts[0]["pending"]["question"]["content"], "它收费吗？")
        self.assertEqual(backend.contexts[0]["pending"]["messages"], [])

    async def test_history_keeps_complete_latest_rounds_and_reports_truncation(self):
        conversation = Conversation("bounded", limits=ConversationLimits(max_rounds=2))
        for number in range(4):
            await conversation.ask(f"问题{number}？", Backend())
        self.assertEqual([turn["user"] for turn in conversation.turns], ["问题2？", "问题3？"])
        backend = Backend()
        result = await conversation.ask("继续。", backend)
        history = backend.contexts[0]["history"]
        self.assertEqual([message["role"] for message in history], ["user", "assistant", "user", "assistant"])
        self.assertEqual([message["content"] for message in history if message["role"] == "user"], ["问题2？", "问题3？"])
        self.assertTrue(result.history_truncated)
        self.assertTrue(backend.contexts[0]["history_truncated"])

    async def test_oversized_input_stops_before_resolver_and_retains_pending(self):
        conversation = Conversation("budget")
        await conversation.ask("医保？", Backend({"kind": "clarification", "clarification": "条件？"}))
        old_pending = conversation.pending
        backend = Backend()
        result = await conversation.ask("条件" * 12000, backend)
        self.assertEqual(result.status, "technical_error")
        self.assertIn("budget", result.error)
        self.assertEqual(backend.contexts, [])
        self.assertEqual(backend.searches, [])
        self.assertEqual(conversation.pending, old_pending)

    async def test_same_conversation_serializes_before_reading_history(self):
        conversation = Conversation("serial")
        entered, release = asyncio.Event(), asyncio.Event()

        class Blocking(Backend):
            async def resolve(self, context):
                entered.set()
                await release.wait()
                return await super().resolve(context)

        first = asyncio.create_task(conversation.ask("上海医保？", Blocking()))
        await entered.wait()
        backend = Backend()
        second = asyncio.create_task(conversation.ask("异地呢？", backend))
        await asyncio.sleep(0)
        self.assertEqual(backend.contexts, [])
        release.set()
        results = await asyncio.gather(first, second)
        self.assertEqual([result.status for result in results], ["ok", "ok"])
        self.assertEqual(backend.contexts[0]["history"][0]["content"], "上海医保？")

    async def test_independent_conversations_do_not_share_locks_or_history(self):
        blocked = Conversation("blocked")
        free = Conversation("free")
        entered, release = asyncio.Event(), asyncio.Event()

        class Blocking(Backend):
            async def resolve(self, context):
                entered.set()
                await release.wait()
                return await super().resolve(context)

        task = asyncio.create_task(blocked.ask("上海医保？", Blocking()))
        await entered.wait()
        try:
            backend = Backend()
            result = await asyncio.wait_for(free.ask("北京停车？", backend), timeout=1)
            self.assertEqual(result.status, "ok")
            self.assertEqual(backend.contexts[0]["history"], [])
            self.assertFalse(task.done())
        finally:
            release.set()
            await task
        self.assertEqual(blocked.turns[0]["user"], "上海医保？")
        self.assertEqual(free.turns[0]["user"], "北京停车？")

    async def test_clear_waits_for_active_turn_then_removes_all_state(self):
        conversation = Conversation("clear")
        entered, release = asyncio.Event(), asyncio.Event()

        class Blocking(Backend):
            async def resolve(self, context):
                entered.set()
                await release.wait()
                return await super().resolve(context)

        task = asyncio.create_task(conversation.ask("医保？", Blocking()))
        await entered.wait()
        clearing = asyncio.create_task(conversation.clear())
        await asyncio.sleep(0)
        self.assertFalse(clearing.done())
        release.set()
        await asyncio.gather(task, clearing)
        self.assertEqual(conversation.turns, [])
        self.assertIsNone(conversation.pending)

    async def test_resolver_timeout_uses_shared_turn_budget_and_keeps_state(self):
        conversation = Conversation("timeout", limits=ConversationLimits(resolver_seconds=0.01))

        class Slow(Backend):
            async def resolve(self, context):
                await asyncio.sleep(1)
                return await super().resolve(context)

        backend = Slow()
        result = await conversation.ask("医保？", backend, AgentLimits(total_seconds=0.08, search_seconds=0.05))
        self.assertEqual((result.status, result.stop_reason), ("timeout", "resolution_timeout"))
        self.assertEqual(backend.searches, [])
        self.assertEqual(conversation.turns, [])

    async def test_resolver_elapsed_time_is_deducted_from_search_deadline(self):
        conversation = Conversation("elapsed")

        class Slow(Backend):
            async def resolve(self, context):
                await asyncio.sleep(0.04)
                return await super().resolve(context)

            async def search(self, question, document_ids=None):
                await asyncio.sleep(0.04)
                return await super().search(question, document_ids)

        result = await conversation.ask("医保？", Slow(), AgentLimits(total_seconds=0.1, search_seconds=0.06))
        self.assertEqual((result.status, result.stop_reason), ("timeout", "search_timeout"))
        self.assertEqual(conversation.turns, [])

    async def test_resolver_call_counts_toward_final_generation_reservation(self):
        def missing(query):
            return {"constraints": [], "needs": [{"text": "是否收费", "status": "missing", "evidence_ids": []}],
                    "action": {"type": "search_policy", "query": query, "need_index": 0}, "reason": "尚缺依据"}
        backend = Backend(decisions=[missing("收费条件"), missing("登记费用")])
        result = await Conversation("reserved").ask("是否收费？", backend)
        self.assertEqual((result.status, result.stop_reason), ("ok", "final_call_reserved"))
        self.assertEqual(result.usage["generation_calls"], 4)
        self.assertEqual(len(backend.generated), 1)


if __name__ == "__main__":
    unittest.main()
