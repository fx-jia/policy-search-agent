"""Shared budgets and literal rereads, without external services."""
import unittest

from experiments.search_agent import evidence as ev


class BudgetTests(unittest.TestCase):
    def test_short_blocks_return_budget_to_long_policy(self):
        long = ev.Evidence("long", "d", "办法", "背景。" * 80 + "领取期限六个月，携带身份证。", 0)
        short = ev.Evidence("short", "other", "其他", "无关。", 0)
        views = ev.make_views([long, short], "按某地办法，领取期限和证件？", len, 400)
        self.assertEqual(views[0].text, long.text)
        self.assertFalse(views[0].truncated)

    def test_all_selected_anchors_share_budget_without_per_block_cap(self):
        a = ev.Evidence("a", "d", "规定", "适用对象甲，待遇和限制条件必须一起保留。", 0)
        b = ev.Evidence("b", "e", "其他", "乙。", 0)
        decision = {"needs": [{"text": "甲待遇", "evidence": [{"chunk_id": "a", "quote": a.text, "_source_span": [0, len(a.text)]}]}]}
        chosen = ev.pack_evidence([a, b], decision, "甲待遇？", len, len(a.text) + 3)
        self.assertEqual(chosen[0].text, a.text)
        self.assertLessEqual(sum(len(e.text) for e in chosen), len(a.text) + 3)

    def test_reread_finds_unseen_literal_clause_and_no_fake_new_chunk(self):
        raw = "申报条件。领取期限六个月。受理程序。"
        item = ev.Evidence("a", "d", "办法", raw, 0)
        view = ev.Evidence("a", "d", "办法", raw[:5], 0, spans=((0, 5),), truncated=True)
        reread = ev.read_chunk(item, "领取期限", [view], len, 25)
        self.assertEqual(reread.id, "a")
        self.assertIn("领取期限六个月。", reread.text)
        ev.validate_views([reread], [item])
        self.assertIsNone(ev.read_chunk(item, "领取期限", [item], len, 25))

    def test_requested_need_is_not_drowned_by_policy_title(self):
        title = "关于完善城乡居民基本医疗保险管理服务办法"
        raw = (title + "相关工作。") * 15 + "领取期限六个月，持身份证领取。"
        item = ev.Evidence("a", "d", title, raw, 0)
        views = ev.make_views([item], "按《" + title + "》，领取期限和证件是什么？", len, 70)
        self.assertIn("领取期限六个月，持身份证领取。", views[0].text)

    def test_budget_never_exceeded_with_multiple_needs(self):
        items = [ev.Evidence("a", "d", "办法", "背景。" * 50 + "费用不收取。", 0),
                 ev.Evidence("b", "e", "办法", "无关。" * 50 + "期限十天。", 0)]
        views = ev.make_views(items, "费用及期限？", len, 80, needs=["费用", "期限"])
        self.assertIn("费用不收取。", views[0].text)
        self.assertIn("期限十天。", views[1].text)
        self.assertLessEqual(sum(len(e.text) for e in views), 80)

    def test_whole_judge_request_budget_includes_segment_metadata(self):
        from experiments.search_agent.adapter import prepare_judge_views, judge_messages
        item = ev.Evidence("a", "d", "政策", "申请。" * 1500, 0)
        views = prepare_judge_views("申请？", [item], None, len, 4800, neighbors={"a": []}, gap_contract=True)
        system, messages = judge_messages("申请？", views, None, neighbors={"a": []}, gap_contract=True)
        self.assertLessEqual(len(system + messages[0]["content"]), 7000)
        self.assertTrue(views)
        ev.validate_views(views, [item])

    def test_reread_can_reach_tail_of_single_oversized_sentence(self):
        raw = "背景" + "甲" * 8000 + "办理期限十天。"
        item = ev.Evidence("a", "d", "政策", raw, 0)
        prefix = ev.Evidence("a", "d", "政策", raw[:4800], 0, spans=((0, 4800),), truncated=True)
        result = ev.read_chunk(item, "办理期限", [prefix], len, 700)
        self.assertIsNotNone(result)
        self.assertIn("办理期限十天。", result.text)
        self.assertLessEqual(len(result.text), 700)
        ev.validate_views([result], [item])


if __name__ == "__main__":
    unittest.main()
