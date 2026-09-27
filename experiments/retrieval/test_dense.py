"""No live services: verify that the strict Dense query cannot use gold/text filters."""

import unittest
from unittest.mock import Mock

from experiments.retrieval.dense import DenseClient
from experiments.retrieval.dense_worker import knn_body
from experiments.retrieval.ragflow import RetrievalError


class DenseQueryTest(unittest.TestCase):
    def test_knn_has_no_lexical_or_oracle_clause(self):
        config = {"candidate_count": 64, "knn_top_k": 1024, "knn_num_candidates": 2048, "similarity_threshold": 0}
        query = knn_body([0.1, 0.2], ["kb"], config)
        self.assertNotIn("query", query)
        self.assertNotIn("doc_id", str(query["knn"]["filter"]))
        self.assertNotIn("match", str(query))
        self.assertEqual(query["knn"]["field"], "q_2_vec")
        self.assertEqual(query["knn"]["k"], 1024)
        self.assertEqual(query["size"], 64)
        self.assertEqual(query["knn"]["filter"]["bool"]["filter"], [{"terms": {"kb_id": ["kb"]}}])

    def test_invalid_vector_rejected(self):
        for vector in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                knn_body(vector, "kb", {})

    def test_fixed_corpus_scope_is_used_for_all_vectors(self):
        config = {"candidate_count": 64, "knn_top_k": 1024, "knn_num_candidates": 2048, "similarity_threshold": 0}
        one = knn_body([0.1, 0.2], ["a", "b"], config, ["d1", "d2"])
        two = knn_body([0.9, 0.3], ["a", "b"], config, ["d1", "d2"])
        self.assertEqual(one["knn"]["filter"], two["knn"]["filter"])
        self.assertEqual(one["knn"]["filter"]["bool"]["filter"][1], {"terms": {"doc_id": ["d1", "d2"]}})

    def test_timeout_cannot_shift_responses_to_the_next_question(self):
        client = DenseClient.__new__(DenseClient)
        client.config = {"retrieval_config": {"candidate_count": 64}}
        client.failed = False
        client._request = Mock(side_effect=RetrievalError("dense_worker_timeout"))
        client.close = Mock()
        with self.assertRaisesRegex(RetrievalError, "timeout"):
            client.retrieve("first", 64)
        with self.assertRaisesRegex(RetrievalError, "unavailable"):
            client.retrieve("second", 64)
        client._request.assert_called_once()
        client.close.assert_called_once()

    def test_mismatched_response_is_not_scored(self):
        client = DenseClient.__new__(DenseClient)
        client.config = {"retrieval_config": {"candidate_count": 64}}
        client.failed = False
        client._request = Mock(return_value={"request_id": "different-question"})
        client.close = Mock()
        with self.assertRaisesRegex(RetrievalError, "mismatch"):
            client.retrieve("first", 64)
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
