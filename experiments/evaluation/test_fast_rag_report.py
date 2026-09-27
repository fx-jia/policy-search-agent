import unittest

from experiments.evaluation.fast_rag_report import cost, oracle, scores


class ReportingTests(unittest.TestCase):
    def test_invalid_grade_is_missing_not_zero(self):
        values = scores([{"arm": "fast", "question_id": "q1", "status": "ok", "grade": {"total": 0}},
                         {"arm": "fast", "question_id": "q2", "status": "invalid_grade", "grade": {"total": 50}},
                         {"arm": "fast", "question_id": "q3", "status": "ok", "grade": {"total": None}}])
        self.assertEqual(values, {"fast": {"q1": 0}})

    def test_unknown_call_is_reserved_not_zero_usage(self):
        self.assertEqual(cost({"calls": [{"usage": {"total_tokens": 100}}, {"reserve_tokens": 1200}]}),
                         {"known_tokens": 100, "unknown_reserve": 1200, "unknown_calls": 1})

    def test_hindsight_gate_accounts_for_both_selected_paths(self):
        quality = {"fast": {"a": 100, "b": 75}, "ab": {"a": 100, "b": 100}}
        rows = {"fast": {q: {"latency_ms": 2000, "calls": [{"usage": {"total_tokens": 500}}]} for q in ("a", "b")},
                "ab": {q: {"latency_ms": 8000, "calls": [{"usage": {"total_tokens": 2000}}]} for q in ("a", "b")}}
        result = oracle(["a", "b"], quality, rows)
        self.assertEqual(result["fast_questions"], 1)
        self.assertEqual(result["quality_mean"], 100)
        self.assertEqual(result["latency_mean_ms"], 5000)
        self.assertEqual(result["generation_mean_tokens"], 1250)


if __name__ == "__main__":
    unittest.main()
