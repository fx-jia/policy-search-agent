from dataclasses import replace
import unittest

from experiments.search_agent.evidence import Evidence, validate_views
from experiments.search_agent.fast import context_text
from experiments.search_agent.fast_context import select_context


def chunk(cid, text, title="医保政策"):
    return Evidence(cid, "doc-" + cid, title, text, 0)


class SelectionTests(unittest.TestCase):
    def select(self, question, items, budget=2000):
        chosen = select_context(question, items, len, context_text, budget)
        validate_views(chosen, items)
        self.assertLessEqual(len(context_text(chosen)), budget)
        self.assertLessEqual(len(chosen), 3)
        return chosen

    def test_relevant_clause_after_first_three_is_retained(self):
        items = [chunk(str(i), "有关部门应当做好医保监督管理工作。") for i in range(3)]
        items += [chunk("answer", "异地就医人员可以通过网上渠道办理备案。")]
        chosen = self.select("异地就医人员如何办理备案？", items)
        self.assertIn("answer", [e.id for e in chosen])

    def test_shortening_retains_full_article_eligibility_and_soft_numeric_wrap(self):
        irrelevant = "第一条\n" + "有关部门履行监督管理职责。" * 30 + "\n"
        relevant = "第二条\n仅限参保居民住院。支付比例为\n70%\n，年度限额100元。\n"
        other = "第三条\n" + "做好宣传管理工作。" * 30
        e = chunk("a", irrelevant + relevant + other)
        chosen = self.select("居民住院支付比例多少？", [e], 160)
        self.assertEqual(len(chosen), 1)
        self.assertIn(relevant, chosen[0].text)
        self.assertTrue(chosen[0].truncated)

    def test_unstructured_window_keeps_adjacent_condition(self):
        e = chunk("a", "本通知适用于参保居民。" + "办理流程另行公布。" * 20 + "仅限首次申请。补助标准为100元。逾期不予办理。" + "经办管理按照规定执行。" * 20)
        chosen = self.select("补助标准是多少？", [e], 160)
        self.assertIn("仅限首次申请。补助标准为100元。逾期不予办理。", chosen[0].text)
        self.assertIn("本通知适用于参保居民。", chosen[0].text)

    def test_article_keeps_its_own_chapter_not_previous_population(self):
        e = chunk("a", "第一章 职工待遇\n第一条\n" + "经办管理按照规定执行。" * 30 +
                  "\n第二章 居民待遇\n第二条\n参保人员住院支付比例为70%。")
        chosen = self.select("居民住院支付比例多少？", [e], 150)
        self.assertIn("第二章 居民待遇", chosen[0].text)
        self.assertIn("参保人员住院支付比例为70%", chosen[0].text)
        self.assertNotIn("第一章 职工待遇", chosen[0].text)

    def test_multi_demand_does_not_fill_all_slots_with_one_topic(self):
        items = [chunk(str(i), "异地就医备案可以网上办理。") for i in range(3)]
        items += [chunk("other", "居民参保缴费应于每年十二月底完成。")]
        chosen = self.select("异地就医备案如何办理？居民参保缴费什么时候完成？", items)
        self.assertIn("other", [e.id for e in chosen])

    def test_never_cuts_an_overbudget_qualifying_article(self):
        e = chunk("a", "第一条\n" + "限定条件" * 100 + "，补助100元。")
        self.assertEqual(self.select("补助多少？", [e], 100), [])

    def test_unfit_high_rank_sources_do_not_hide_a_fitting_fourth(self):
        large = "第一条\n住院报销比例" + "限定条件" * 100 + "。"
        items = [chunk(str(i), large) for i in range(3)] + [chunk("fits", "住院报销70%。")]
        chosen = self.select("住院报销比例多少？", items, 200)
        self.assertIn("fits", [e.id for e in chosen])

    def test_versions_and_original_identity_are_preserved(self):
        a = chunk("a", "2024年居民补助100元。", "2024年办法")
        b = chunk("b", "2025年居民补助200元。", "2025年办法")
        chosen = self.select("2024年与2025年居民补助标准？", [a, a, b])
        self.assertEqual({e.id for e in chosen}, {"a", "b"})
        with self.assertRaisesRegex(ValueError, "conflicting_chunk"):
            self.select("居民补助标准？", [a, replace(a, text=b.text)])

    def test_unmatched_query_is_deterministic_and_bounded(self):
        items = [chunk(str(i), "政策通知。") for i in range(8)]
        self.assertEqual(self.select("???", items), self.select("???", items))
        self.assertEqual(self.select("问题", []), [])


if __name__ == "__main__":
    unittest.main()
