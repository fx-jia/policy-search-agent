"""Guard unknown accounting and the freshness of generated review evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.evaluation import answer_quality_report as report


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.qid = "q1"
        question = {"question_id": self.qid, "difficulty": "Easy", "policy_group_id": "g1"}
        (self.out / "dev.jsonl").write_text(json.dumps(question) + "\n")
        self.job = {"id": "j1", "question_id": self.qid, "arm": "ab", "stage": "extension",
                    "packet": {"question": "题干", "reference_answer": "答案", "scope": "题意范围", "answer": "已有答案",
                               "reference_evidence": [], "citations": []}}
        for name, value in [("jobs", [self.job]), ("config", {"model": "judge"}),
                            ("manifest", {"question_ids": [self.qid], "rubric": "test", "technical_exclusions": []})]:
            (self.out / (name + ".json")).write_text(json.dumps(value))

    def generate(self):
        with patch.object(report, "BASE", self.out):
            report.report(self.out)
        return json.loads((self.out / "metrics.json").read_text())

    def test_host_timeout_is_unknown_not_zero_accounting(self):
        (self.out / "records.jsonl").write_text(json.dumps({"id": "j1", "status": "host_timeout", "usage_unknown": True}) + "\n")
        metrics = self.generate()
        self.assertEqual(metrics["jobs_without_call_telemetry"], 1)
        self.assertEqual(metrics["unknown_usage_jobs"], 1)
        self.assertIsNone(metrics["table"]["ab"]["judge_latency_mean"])

    def test_updated_records_get_new_review_snapshot_without_overwriting_notes(self):
        self.generate()
        first = json.loads((self.out / "review_artifact.json").read_text())["file"]
        path = self.out / first
        path.write_text(path.read_text() + "\n人工意见：保留。\n")
        row = {"id": "j1", "status": "invalid_grade", "validation_error": "quote", "output": "新的裁判输出"}
        (self.out / "records.jsonl").write_text(json.dumps(row) + "\n")
        self.generate()
        second = json.loads((self.out / "review_artifact.json").read_text())["file"]
        self.assertNotEqual(first, second)
        self.assertIn("新的裁判输出", (self.out / second).read_text())
        self.assertIn("人工意见：保留。", path.read_text())


if __name__ == "__main__":
    unittest.main()
