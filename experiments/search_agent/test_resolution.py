"""Conversation provenance and complete-request budgets without model services."""
import json
import unittest

from experiments.search_agent.resolution import prepare_resolution_context, resolution_messages, validate_resolution
from experiments.search_agent.evidence import DecisionError


def context(question="那退休人员呢？"):
    return {"current": {"id": "current", "role": "user", "content": question},
            "history": [{"id": "u0", "role": "user", "content": "上海2023年居民医保待遇是什么？"},
                        {"id": "a0", "role": "assistant", "content": "报销380元。[ID:0]"}],
            "pending": None, "history_truncated": False}


def proposal(**changes):
    return {"kind": "question", "clarification": "",
            "user_context": [{"message_id": "u0", "quote": "上海2023年"}], "reset": False, **changes}


class ResolutionTests(unittest.TestCase):
    def test_free_text_expansion_cannot_import_assistant_conditions(self):
        value = proposal(question="北京职工医保退休人员免缴费的申请条件？", user_context=[])
        with self.assertRaisesRegex(DecisionError, "fields"):
            validate_resolution(value, context("那条件呢？"))

    def test_selected_user_source_preserves_full_negative_scope(self):
        data = context("报销呢？")
        data["history"][0]["content"] = "上海2026年居民医保，不包括职工医保。"
        value = {"kind": "question", "clarification": "", "user_context": [{"message_id": "u0", "quote": "职工医保"}], "reset": False}
        result = validate_resolution(value, data)
        self.assertIn("上海2026年居民医保，不包括职工医保。", result.question)
        self.assertIn("报销呢？", result.question)

    def test_user_corrections_keep_chronology_when_model_reverses_bindings(self):
        data = context("报销比例呢？")
        data["history"] = [{"id": "t0-user", "role": "user", "content": "上海2026年居民医保报销条件？"},
                           {"id": "t1-user", "role": "user", "content": "改为北京2025年。"}]
        value = proposal(user_context=[{"message_id": "t1-user", "quote": "北京2025年"},
                                       {"message_id": "t0-user", "quote": "上海2026年"}])
        result = validate_resolution(value, data)
        self.assertLess(result.question.index("上海2026年"), result.question.index("改为北京2025年"))
        self.assertTrue(result.question.endswith("用户本轮原话：报销比例呢？"))

    def test_colloquial_and_formal_chinese_amounts_cannot_enter_clarification(self):
        for amount in ("三百八十块钱", "叁佰捌拾圆", "十万人民币"):
            with self.subTest(amount=amount), self.assertRaisesRegex(DecisionError, "number"):
                validate_resolution(proposal(kind="clarification", clarification="是补贴" + amount + "吗？", user_context=[]), context())

    def test_user_condition_binding_and_numbers_are_checked(self):
        result = validate_resolution(proposal(), context())
        self.assertIn("上海2023年居民医保待遇是什么？", result.question)
        self.assertTrue(result.question.endswith("用户本轮原话：那退休人员呢？"))
        self.assertEqual(result.user_context[0]["quote"], "上海2023年")
        for quote in ({"message_id": "a0", "quote": "380元"},
                      {"message_id": "u0", "quote": "北京2024年"},
                      {"message_id": "unknown", "quote": "上海"}):
            with self.subTest(quote=quote), self.assertRaises(DecisionError):
                validate_resolution(proposal(user_context=[quote]), context())

    def test_assistant_answer_amount_cannot_become_question_fact(self):
        with self.assertRaisesRegex(DecisionError, "fields"):
            validate_resolution(proposal(question="上海2023年退休人员报销380元的条件？"), context())

    def test_unselected_old_user_numbers_cannot_leak_into_question(self):
        result = validate_resolution(proposal(user_context=[]), context())
        self.assertNotIn("2023", result.question)
        self.assertEqual(result.question, "那退休人员呢？")

    def test_current_explicit_amount_is_allowed_and_original_question_is_retained(self):
        result = validate_resolution(proposal(user_context=[]), context("我想核实380元，不包括财政补贴。"))
        self.assertIn("380元", result.question)
        self.assertIn("不包括财政补贴", result.question)

    def test_reset_cannot_inherit_earlier_user_conditions(self):
        with self.assertRaisesRegex(DecisionError, "reset"):
            validate_resolution(proposal(reset=True), context("换个问题，杭州的条件是什么？"))
        result = validate_resolution(proposal(reset=True, user_context=[]), context("换个问题，杭州的条件是什么？"))
        self.assertTrue(result.reset)

    def test_ambiguous_followup_produces_clarification(self):
        result = validate_resolution(proposal(kind="clarification", clarification="请说明你指的是哪一项待遇。", user_context=[]), context("第二项呢？"))
        self.assertEqual(result.kind, "clarification")
        self.assertTrue(result.clarification)
        with self.assertRaises(DecisionError):
            validate_resolution(proposal(kind="clarification", clarification="", user_context=[]), context())

    def test_extra_executable_fields_and_non_boolean_reset_are_rejected(self):
        for value in (proposal(tool="search"), proposal(reset="false"), proposal(user_context=[{"message_id": "u0", "quote": "上海", "role": "user"}])):
            with self.subTest(value=value), self.assertRaises(DecisionError):
                validate_resolution(value, context())

    def test_pending_raw_user_sources_can_complete_the_question(self):
        data = {**context("成年人"), "history": [], "pending": {
            "question": {"id": "pending-question", "role": "user", "content": "上海2026年居民医保条件？"},
            "clarification": "请说明年龄。", "messages": []}}
        result = validate_resolution(proposal(user_context=[{"message_id": "pending-question", "quote": "上海2026年"}]), data)
        self.assertIn("2026年", result.question)


class ContextTests(unittest.TestCase):
    def test_history_round_limit_removes_complete_pairs_and_keeps_current(self):
        turns = [{"user": f"旧问题{i}", "assistant": f"旧答复{i}", "effective_question": f"旧问题{i}"} for i in range(8)]
        data = prepare_resolution_context("本轮不包括补贴", turns, None, len, max_rounds=2, budget=10000)
        self.assertEqual([m["content"] for m in data["history"]], ["旧问题6", "旧答复6", "旧问题7", "旧答复7"])
        self.assertEqual(data["current"]["content"], "本轮不包括补贴")
        self.assertTrue(data["history_truncated"])

    def test_exact_system_plus_json_budget_is_enforced(self):
        minimal = prepare_resolution_context("本轮问题", [], None, len, budget=10000)
        system, messages = resolution_messages(minimal)
        budget = len(system + messages[0]["content"]) + 10
        turns = [{"user": "旧问题" * 100, "assistant": "旧回答" * 100, "effective_question": "旧问题"}]
        data = prepare_resolution_context("本轮问题", turns, None, len, budget=budget)
        system, messages = resolution_messages(data)
        self.assertLessEqual(len(system + messages[0]["content"]), budget)
        self.assertEqual(data["history"], [])
        self.assertEqual(json.loads(messages[0]["content"])["current"]["content"], "本轮问题")

    def test_oversized_current_or_pending_is_rejected_without_truncation(self):
        for question, pending in (("本轮" * 4000, None), ("上海", {"question": "原问题" * 4000, "messages": [], "clarification": "哪里？"})):
            with self.subTest(pending=bool(pending)), self.assertRaisesRegex(DecisionError, "budget"):
                prepare_resolution_context(question, [], pending, len, budget=3000)

    def test_pending_is_user_provenance_and_returned_context_is_independent(self):
        pending = {"question": "上海医保条件？", "messages": ["2026年"], "clarification": "哪类医保？"}
        data = prepare_resolution_context("职工医保", [], pending, len, budget=10000)
        self.assertEqual(data["pending"]["question"]["id"], "pending-question")
        self.assertEqual(data["pending"]["messages"][0]["content"], "2026年")
        data["pending"]["messages"][0]["content"] = "污染"
        self.assertEqual(pending["messages"], ["2026年"])


if __name__ == "__main__":
    unittest.main()
