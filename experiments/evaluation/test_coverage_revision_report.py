import importlib
import importlib.util
import unittest


class CoverageReportTests(unittest.TestCase):
    def test_logical_cost_includes_shared_draft_but_failure_stays_in_denominator(self):
        name = "experiments.evaluation.coverage_revision_report"
        self.assertIsNotNone(importlib.util.find_spec(name), "coverage report is missing")
        module = importlib.import_module(name)
        rows, scored = [], []
        for qid in ["ok", "failed"]:
            for policy, tokens in [("draft", 10), ("generic", 20), ("coverage", 30)]:
                good = qid == "ok"
                rows.append({"question_id": qid, "policy": policy, "status": "ok" if good else "error", "attempted": good or policy == "draft", "provider_response": {"usage": {"total_tokens": tokens}, "latency_ms": tokens} if good else {}})
                scored.append({"question_id": qid, "policy": policy, "assessed": good, "answer_correct": good, "grounded_correct": good, "correct_points": int(good), "point_count": 1, "claims": int(good), "supported_cited_claims": int(good), "quotes": int(good), "valid_quotes": int(good)})
        summaries = module.summarize(rows, scored)
        self.assertEqual(summaries["coverage"]["planned_questions"], 2)
        self.assertEqual(summaries["coverage"]["grounded_correct_count"], 1)
        self.assertEqual(summaries["coverage"]["stage_reported_total_tokens"], 30)
        self.assertEqual(summaries["coverage"]["pipeline_reported_total_tokens"], 40)
        self.assertEqual(summaries["coverage"]["mean_complete_pipeline_latency_ms"], 40)
        self.assertEqual(summaries["coverage"]["complete_pipeline_count"], 1)
        self.assertFalse(summaries["coverage"]["pipeline_usage_complete"])
        self.assertEqual(summaries["generic"]["pipeline_reported_total_tokens"], 30)
        self.assertEqual(summaries["draft"]["pipeline_reported_total_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
