import unittest

from experiments.evaluation.fast_rag_router import assign_folds, features, fit, probability, route_metrics


class RouterTests(unittest.TestCase):
    def test_policy_group_never_spans_folds(self):
        groups = {"a1": "a", "a2": "a", "b1": "b", "c1": "c", "d1": "d"}
        folds = assign_folds(groups, 3)
        self.assertEqual(folds["a1"], folds["a2"])
        self.assertEqual(set(folds.values()), {0, 1, 2})

    def test_features_only_accept_question_text_and_capture_multiple_demands(self):
        self.assertEqual(len(features("江苏职工能否单独参加医保？")), len(features("问题")))
        with self.assertRaises(TypeError):
            features({"question": "问题", "reference_answer": "答案"})
        self.assertNotEqual(features("谁可以办理？"), features("职工与居民有什么区别，分别如何办理？"))

    def test_local_fit_learns_a_separable_case(self):
        model = fit([[-2], [-1], [1], [2]], [0, 0, 1, 1])
        self.assertLess(probability(model, [-1.5]), .5)
        self.assertGreater(probability(model, [1.5]), .5)

    def test_mixed_route_keeps_failed_quality_predictions_visible(self):
        rows = {"fast": {"a": {"latency_ms": 2, "calls": []}, "b": {"latency_ms": 2, "calls": []}},
                "ab": {"a": {"latency_ms": 8, "calls": []}, "b": {"latency_ms": 8, "calls": []}},
                "rag": {"a": {"latency_ms": 5, "calls": []}, "b": {"latency_ms": 5, "calls": []}}}
        quality = {"fast": {"a": 100, "b": 50}, "ab": {"a": 100, "b": 100}, "rag": {"a": 100, "b": 100}}
        result = route_metrics({"a": "fast", "b": "fast"}, quality, rows)
        self.assertEqual(result["quality_delta_vs_ab"], -25)
        self.assertEqual(result["fast_harmful_routes"], 1)
        self.assertEqual(result["latency_mean_ms"], 2)


if __name__ == "__main__":
    unittest.main()
