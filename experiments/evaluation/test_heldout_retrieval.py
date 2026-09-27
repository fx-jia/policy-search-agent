import unittest

from experiments.evaluation import heldout_retrieval


class HeldoutTests(unittest.TestCase):
    def manifest(self):
        return {"split_version": "v", "assignments": {"q1": "g1", "q2": "g2", "d": "g3"}, "groups": {"g1": {"split": "test", "question_ids": ["q1"]}, "g2": {"split": "test", "question_ids": ["q2"]}, "g3": {"split": "dev", "question_ids": ["d"]}}}

    def rows(self):
        return [{"question_id": f"q{i}", "question": f"问题{i}", "split": "test", "split_version": "v", "policy_group_id": f"g{i}", "review_status": "human_approved", "answer": "不能进入检索", "relevant_document_ids": ["gold"]} for i in [1, 2]]

    def test_query_projection_excludes_answers_and_relevance_labels(self):
        self.assertTrue(hasattr(heldout_retrieval, "query_view"), "test-only query projection missing")
        self.assertEqual(heldout_retrieval.query_view(self.rows(), self.manifest()), [{"question_id": "q1", "question": "问题1"}, {"question_id": "q2", "question": "问题2"}])

    def test_split_and_membership_changes_are_rejected(self):
        self.assertTrue(hasattr(heldout_retrieval, "query_view"), "frozen split guard missing")
        original = self.rows()
        for changed in [original[:1], original + original[:1], [{**r, "split": "dev"} for r in original], [{**r, "policy_group_id": "g3"} for r in original], [{**r, "review_status": "pending"} for r in original]]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                heldout_retrieval.query_view(changed, self.manifest())

    def test_error_stops_on_quota_or_three_consecutive_failures(self):
        self.assertTrue(hasattr(heldout_retrieval, "should_stop"), "call stop rule missing")
        self.assertTrue(heldout_retrieval.should_stop("quota_error", 1))
        self.assertTrue(heldout_retrieval.should_stop("transport_error", 3))
        self.assertTrue(heldout_retrieval.should_stop("dense_worker_unavailable", 1))
        self.assertFalse(heldout_retrieval.should_stop("transport_error", 2))
        self.assertFalse(heldout_retrieval.should_stop(None, 0))


if __name__ == "__main__":
    unittest.main()
