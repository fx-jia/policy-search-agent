import unittest

from experiments.evaluation.context_completion import Candidate
from experiments.evaluation.context_experiment import evaluate_context, project_candidates


class ContextExperimentTests(unittest.TestCase):
    def test_alternative_does_not_erase_required_policy_context(self):
        question = {"question_id": "q", "evidence": [{"source_document_id": "a", "text": "98天"}, {"source_document_id": "a", "text": "2026年起"}]}
        alt = {"source_document_id": "b", "text": "产假98天"}
        alternatives = [{"evidence": [alt], "required_original_indices": [1]}]
        b = Candidate("b0", "doc-b", "b", 1, 0, "产假98天", 10)
        a = Candidate("a0", "doc-a", "a", 2, 0, "2026年起", 10)
        result = evaluate_context(question, [b], alternatives, {"a", "b"}, {})
        self.assertFalse(result["complete_with_alternatives"])
        result = evaluate_context(question, [b, a], alternatives, {"a", "b"}, {})
        self.assertTrue(result["complete_with_alternatives"])
        self.assertFalse(result["original_complete"])

    def test_multi_source_question_needs_both_evidence_spans(self):
        question = {"question_id": "q", "evidence": [{"source_document_id": "a", "text": "340元"}, {"source_document_id": "b", "text": "420元"}]}
        a = Candidate("a0", "doc-a", "a", 1, 0, "340元", 10)
        result = evaluate_context(question, [a], [], {"a", "b"}, {})
        self.assertFalse(result["original_complete"])
        self.assertEqual(result["original_span_fraction"], 0.5)

    def test_unjudged_is_not_counted_as_unsupported(self):
        question = {"question_id": "q", "evidence": [{"source_document_id": "a", "text": "规则"}]}
        chunks = [Candidate(s, s, s, i + 1, 0, "规则", t) for i, (s, t) in enumerate([("a", 10), ("b", 20), ("c", 30), ("d", 40)])]
        result = evaluate_context(question, chunks, [], {"a"}, {("q", "b"): "不支持", ("q", "c"): "部分支持"})
        self.assertEqual(result["tokens_by_judgment"], {"known_positive_document": 10, "unsupported_document": 20, "partial_document": 30, "unjudged_document": 40})

    def test_projection_ignores_ranking_labels_and_checks_source_identity(self):
        stored = {"id": "c", "document_id": "d", "source_document_id": "s", "order": 0, "content": "规则", "tokens": 10}
        retrieved = [{"id": "c", "document_id": "d", "source_document_id": "s", "rank": 1, "gold_answer": "不允许参与选择"}]
        projected = project_candidates(retrieved, {"c": stored})
        self.assertEqual(projected, [Candidate("c", "d", "s", 1, 0, "规则", 10)])
        retrieved[0]["source_document_id"] = "wrong"
        with self.assertRaises(ValueError):
            project_candidates(retrieved, {"c": stored})


if __name__ == "__main__":
    unittest.main()
