import tempfile
from pathlib import Path
import unittest

from experiments.evaluation import qa_regression
from experiments.evaluation.context_experiment import write_json


class QARegressionTests(unittest.TestCase):
    def test_frozen_request_changes_are_rejected_before_opening_model_client(self):
        self.assertTrue(hasattr(qa_regression, "verify_prepared"), "frozen-request verification missing")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "requests.jsonl"
            path.write_text("changed")
            write_json(root / "freeze.json", {"files": {str(path): "wrong"}})
            with self.assertRaises(ValueError):
                qa_regression.verify_prepared(root)

    def test_unknown_usage_remains_unknown_instead_of_becoming_zero(self):
        self.assertTrue(hasattr(qa_regression, "execution_summary"), "usage accounting missing")
        rows = [{"status": "ok", "provider_response": {"status": "ok", "usage": {"total_tokens": 12}}}, {"status": "error", "provider_response": {"status": "error"}}]
        summary = qa_regression.execution_summary(rows, 2, False)
        self.assertEqual(summary["reported_usage"]["total_tokens"], 12)
        self.assertFalse(summary["usage_complete"])
        self.assertEqual(summary["calls_without_usage"], 1)
        self.assertIsNone(summary["reported_usage"]["prompt_tokens"])


if __name__ == "__main__":
    unittest.main()
