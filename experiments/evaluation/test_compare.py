"""Prevent uncontrolled comparisons and question-wise uncertainty estimates."""

import copy
import unittest

from experiments.evaluation.compare import paired_cluster_interval, validate_manifests


class CompareTest(unittest.TestCase):
    def manifests(self):
        retrieval = {"candidate_count": 64, "context_top_k": 5, "knn_top_k": 1024, "knn_num_candidates": 2048, "similarity_threshold": 0, "vector_similarity_weight": 0.3}
        return [{"method": method, "split": "dev", "knowledge_base_version": "frozen", "retrieval_config": copy.deepcopy(retrieval)} for method in ("dense", "hybrid", "hybrid_rerank")]

    def test_corpus_change_invalidates_comparison(self):
        manifests = self.manifests()
        validate_manifests(manifests)
        manifests[-1]["knowledge_base_version"] = "changed"
        with self.assertRaisesRegex(ValueError, "knowledge_base_version"):
            validate_manifests(manifests)

    def test_candidate_budget_change_rejected(self):
        manifests = self.manifests()
        manifests[-1]["retrieval_config"]["candidate_count"] = 100
        with self.assertRaisesRegex(ValueError, "candidate_count"):
            validate_manifests(manifests)

    def test_repeated_same_group_is_one_independent_group(self):
        reference = [{"question_id": str(i), "metrics": {"score": 0}} for i in range(4)]
        candidate = [{"question_id": str(i), "metrics": {"score": 0.5}} for i in range(4)]
        groups = {str(i): "one-policy" for i in range(4)}
        result = paired_cluster_interval(reference, candidate, groups, "score", repeats=20)
        self.assertEqual(result["policy_groups"], 1)
        self.assertEqual(result["mean_delta"], 0.5)
        self.assertEqual(result["percentile_95_interval"], [0.5, 0.5])

    def test_misaligned_questions_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            paired_cluster_interval([{"question_id": "a"}], [{"question_id": "b"}], {}, "score")


if __name__ == "__main__":
    unittest.main()
