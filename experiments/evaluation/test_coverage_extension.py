import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments.evaluation.test_coverage_revision import BASE, FakeClient


class CoverageExtensionTests(unittest.TestCase):
    def module(self):
        name = "experiments.evaluation.coverage_extension"
        self.assertIsNotNone(importlib.util.find_spec(name), "development extension missing")
        return importlib.import_module(name)

    def test_completed_pilot_and_non_dev_questions_cannot_enter_extension(self):
        m = self.module()
        rows = [{"question_id": "old", "split": "dev", "review_status": "human_approved"}, {"question_id": "new", "split": "dev", "review_status": "human_approved"}]
        self.assertEqual([r["question_id"] for r in m.remaining_questions(rows, {"old"})], ["new"])
        with self.assertRaises(ValueError):
            m.remaining_questions(rows + [{"question_id": "test", "split": "test", "review_status": "human_approved"}], {"old"})
        with self.assertRaises(ValueError):
            m.remaining_questions(rows + [rows[1]], {"old"})

    def test_multiple_batches_keep_every_record_and_physical_cost_once(self):
        m = self.module()
        bases = [{**BASE, "question_id": f"dev-{i}"} for i in range(11)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            m.run_batches(bases, {"max_calls": 15}, out, FakeClient)
            rows = [json.loads(x) for x in (out / "responses.jsonl").read_text().splitlines()]
            summary = json.loads((out / "execution_summary.json").read_text())
            self.assertEqual(len(rows), 33)
            self.assertEqual(len({r["request_id"] for r in rows}), 33)
            self.assertEqual(summary["actual_calls"], 33)
            self.assertEqual(summary["reported_usage"]["total_tokens"], 330)

    def test_quota_stop_carries_across_batches_and_preserves_denominator(self):
        m = self.module()

        class QuotaClient(FakeClient):
            def generate(self, request):
                self.calls += 1
                return {"status": "error", "error_category": "quota"}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            m.run_batches([{**BASE, "question_id": f"dev-{i}"} for i in range(11)], {"max_calls": 15}, out, QuotaClient)
            rows = [json.loads(x) for x in (out / "responses.jsonl").read_text().splitlines()]
            summary = json.loads((out / "execution_summary.json").read_text())
            self.assertEqual(summary["actual_calls"], 1)
            self.assertEqual(len(rows), 33)
            self.assertTrue(summary["stopped_due_to_errors"])
            self.assertEqual(rows[-1]["status"], "not_run_after_error")
            self.assertFalse(rows[-1]["attempted"])


if __name__ == "__main__":
    unittest.main()
