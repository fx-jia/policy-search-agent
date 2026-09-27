"""Adapter tests without importing the installed RAGFlow server."""
import copy
import unittest

from experiments.search_agent.adapter import Snapshot, native_kbinfos, judge_messages, check_final_request
from experiments.search_agent.evidence import Evidence, make_views


class AdapterTests(unittest.TestCase):
    def test_neighbor_order_and_same_document_only(self):
        cs = [Evidence(str(n), "d", "政策", f"条款{n}", n, {"chunk_id": str(n)}) for n in range(3)]
        s = Snapshot(cs)
        self.assertEqual([e.id for e in s.neighbors(cs[1], "both")], ["0", "2"])
        with self.assertRaises(ValueError):
            Snapshot([cs[0], Evidence("bad", "d", "政策", "x", 2)])

    def test_native_fields_and_aggregates_match_selected_sources(self):
        e = Evidence("a", "d", "标题", "已审核内容", 0, {"chunk_id": "a", "doc_id": "d", "content_with_weight": "原文", "vector": [1]})
        k = native_kbinfos([e])
        self.assertEqual(k["chunks"][0]["content_with_weight"], "已审核内容")
        self.assertEqual(k["doc_aggs"], [{"doc_id": "d", "doc_name": "标题", "count": 1}])
        self.assertNotIn("vector", k["chunks"][0])
        self.assertEqual(e.native["content_with_weight"], "原文")

    def test_judge_has_visible_text_not_hidden_full_native_body(self):
        e = Evidence("a", "d", "标题", "收费规定。"*100, 0, {"secret": "DO_NOT_SEND"})
        views = make_views([e], "收费", len, 30)
        s, history = judge_messages("问题", views, None)
        self.assertNotIn("DO_NOT_SEND", str((s, history)))
        self.assertNotIn("reference_answer", str((s, history)))
        self.assertIn("truncated", history[0]["content"])

    def test_final_request_must_retain_chosen_literal_evidence(self):
        e = Evidence("a", "d", "标题", "规定不收费。", 0)
        with self.assertRaises(ValueError):
            check_final_request({"messages": [{"role": "system", "content": "没有条款"}]}, [e])
        check_final_request({"messages": [{"role": "system", "content": "ID: 0\n├── Title: 标题\n└── Content:\n规定不收费。"}]}, [e])

    def test_identical_text_cannot_prove_another_source_was_present(self):
        es = [Evidence("a", "d1", "旧政策", "同一条款。", 0), Evidence("b", "d2", "新政策", "同一条款。", 0)]
        body = {"messages": [{"role": "system", "content": "ID: 0\n├── Title: 旧政策\n└── Content:\n同一条款。"}]}
        with self.assertRaises(ValueError):
            check_final_request(body, es)


if __name__ == "__main__":
    unittest.main()
