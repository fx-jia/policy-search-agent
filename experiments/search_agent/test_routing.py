from dataclasses import replace
import math
import unittest

from experiments.search_agent.evidence import Evidence
from experiments.search_agent.routing import evidence_features, evidence_rule


def chunk(cid, text):
    return Evidence(cid, "doc", "医保管理办法", text, 0)


class EvidenceRoutingTests(unittest.TestCase):
    def test_answer_clause_outside_selected_context_changes_risk(self):
        initial = [chunk(str(i), "有关部门负责医保基金监督管理工作。") for i in range(3)]
        initial.append(chunk("3", "调查举报可以委托社会中介机构。"))
        question = "调查举报能否委托社会中介机构？"
        missing = evidence_features(question, initial, initial[:3])
        present = evidence_features(question, initial, [initial[3], initial[0], initial[1]])
        self.assertGreater(missing["coverage_gap"], 0)
        self.assertLess(missing["selected_retention"], present["selected_retention"])
        self.assertEqual(evidence_rule(missing), "ab")
        self.assertEqual(evidence_rule(present), "fast")

    def test_source_title_alone_cannot_prove_clause_coverage(self):
        e = replace(chunk("a", "发布通知，遵照执行。"), title="关于调查举报委托社会中介机构的办法")
        value = evidence_features("按《调查举报委托社会中介机构办法》，公示期限多久？", [e], [e])
        self.assertEqual(value["chosen_coverage"], 0)
        self.assertEqual(evidence_rule(value), "ab")

    def test_soft_wraps_in_identifiers_are_normalized_without_new_api(self):
        first = chunk("a", "监督管理规定。")
        last = chunk("b", "第八条：调查举报可以委\n托社会中介机构。2018\n年施行。")
        value = evidence_features("按2018年第八条，调查举报可否委托社会中介机构？", [first, last], [first])
        self.assertEqual(value["anchor_loss_fraction"], 1)
        self.assertGreater(value["coverage_gap"], 0)

    def test_hidden_part_of_selected_chunk_counts_as_missing(self):
        e = chunk("a", "第一条监督管理。\n第八条可以委托社会中介机构调查举报。")
        shown = replace(e, text="第一条监督管理。", spans=((0, 8),), truncated=True)
        value = evidence_features("第八条能否委托社会中介机构调查举报？", [e], [shown])
        self.assertGreater(value["coverage_gap"], 0)
        self.assertEqual(value["anchor_loss_fraction"], 1)

    def test_unseen_or_fabricated_excerpt_is_rejected(self):
        e = chunk("a", "居民参保条件。")
        for chosen in ([chunk("bad", "居民参保条件。")], [replace(e, text="职工参保条件。")]):
            with self.assertRaises(ValueError):
                evidence_features("谁可参保？", [e], chosen)

    def test_empty_evidence_and_empty_terms_never_claim_safe(self):
        for initial, chosen in (([], []), ([chunk("a", "规定。")], [chunk("a", "规定。")])):
            value = evidence_features("？？？", initial, chosen)
            self.assertTrue(all(math.isfinite(v) for v in value.values()))
            self.assertEqual(evidence_rule(value), "ab")

    def test_native_metadata_and_chunk_identity_do_not_enter_features(self):
        e = chunk("a", "调查举报可以委托社会中介机构。")
        a = evidence_features("调查举报能否委托社会中介机构？", [e], [e])
        renamed = replace(e, id="another", document_id="another-doc", native={"answer": "YES", "difficulty": "Easy"})
        self.assertEqual(a, evidence_features("调查举报能否委托社会中介机构？", [renamed], [renamed]))


if __name__ == "__main__":
    unittest.main()
