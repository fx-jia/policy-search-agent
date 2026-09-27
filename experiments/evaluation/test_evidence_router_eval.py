import copy
import unittest

from experiments.evaluation.evidence_router_eval import grouped_predictions, runtime_inputs
from experiments.search_agent.routing import evidence_features


class ProtocolTests(unittest.TestCase):
    def test_held_group_labels_cannot_change_its_own_predictions(self):
        x = {"a": [0.], "b": [1.], "c": [2.], "d": [3.]}
        groups = {q: q for q in x}
        folds = {"a": 0, "b": 0, "c": 1, "d": 1}
        first, audit = grouped_predictions(x, {"a": 0, "b": 1, "c": 0, "d": 1}, groups, folds)
        changed, _ = grouped_predictions(x, {"a": 1, "b": 0, "c": 0, "d": 1}, groups, folds)
        self.assertEqual(first["a"], changed["a"])
        self.assertEqual(first["b"], changed["b"])
        self.assertEqual(audit[0]["model"]["mean"], [2.5])

    def test_policy_group_crossing_folds_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "policy_group_leakage"):
            grouped_predictions({"a": [0], "b": [1]}, {"a": 0, "b": 1},
                                {"a": "same", "b": "same"}, {"a": 0, "b": 1})

    def test_only_initial_literal_evidence_is_extracted_from_logs(self):
        row = {"question": "居民能否参加医保？", "answer": "不可作为输入", "reference": {"gold": "不可作为输入"},
               "trace": [{"event": "fast_context", "initial_ids": ["c"]}],
               "seen_chunks": {"c": {"doc_id": "d", "docnm_kwd": "政策", "content_with_weight": "居民可以参加医保。"},
                               "later": {"doc_id": "x", "docnm_kwd": "未初检来源", "content_with_weight": "额外答案"}},
               "selected": [{"chunk_id": "c", "document_id": "d", "text": "居民可以参加医保。", "spans": [], "truncated": False}]}
        question, initial, chosen = runtime_inputs(row)
        self.assertEqual([e.id for e in initial], ["c"])
        before = evidence_features(question, initial, chosen)
        mutated = copy.deepcopy(row)
        mutated.update(answer="另一个答案", difficulty="Hard", grade=100)
        self.assertEqual(before, evidence_features(*runtime_inputs(mutated)))
        mutated["selected"][0]["chunk_id"] = "later"
        with self.assertRaises(ValueError):
            runtime_inputs(mutated)


if __name__ == "__main__":
    unittest.main()
