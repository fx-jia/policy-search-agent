"""Evidence and review preservation checks, using only temporary local data."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from experiments.analysis.profile_dataset import fingerprint
from experiments.datasets.build_benchmark import build, fingerprint_text


class BuildBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.jsonl"
        self.body = "第一段背景\n金额\n100元\n末行"
        record = {
            "系统ID": "a",
            "政策正文": self.body,
            "文件名称": "试验政策",
            "数据来源路径": {},
            "适用区域": "某地",
            "发文日期": "2020-01-01",
            "生效日期": None,
            "废止日期": None,
            "效力状态": None,
            "版本号": 1,
            "政策文件大类": "待遇保障",
        }
        self.source.write_text(json.dumps(record, ensure_ascii=False))
        self.approved = {
            "question_id": "med-001",
            "question": "原审核问题？",
            "reference_answer": "第一段背景",
            "relevant_document_ids": ["a"],
            "answerable": True,
            "dataset_version": fingerprint(self.source),
            "review_status": "human_approved",
            "reviewer": "user",
            "split": "smoke",
            "question_type": "conditions",
            "evidence": [{"source_document_id": "a", "start": 0, "end": 5, "text": "第一段背景", "body_sha256": fingerprint_text(self.body)}],
        }
        self.base = self.root / "approved.jsonl"
        self.base.write_text(json.dumps(self.approved, ensure_ascii=False) + "\n")
        self.specs = self.root / "specs.json"
        self.payload = {
            "source_sha256": fingerprint(self.source),
            "base_benchmark_sha256": fingerprint(self.base),
            "expected_new_questions": 1,
            "questions": [{"question_id": "med-002", "question": "新问题金额是多少？", "question_type": "numeric", "evidence_line_ranges": [["a", 1, 2]], "context_line_ranges": [["a", 0, 0]]}],
        }
        self.output = self.root / "out" / "benchmark.jsonl"

    def run_build(self):
        self.specs.write_text(json.dumps(self.payload, ensure_ascii=False))
        with contextlib.redirect_stdout(io.StringIO()):
            return build(self.source, self.specs, self.output, self.base)

    def test_multiline_evidence_and_approved_base_are_preserved(self):
        original_hash = fingerprint(self.base)
        self.run_build()
        old, new = map(json.loads, self.output.read_text().splitlines())
        self.assertEqual(old, self.approved)
        self.assertEqual(fingerprint(self.base), original_hash)
        evidence = new["evidence"][0]
        self.assertEqual(evidence["text"], "金额\n100元")
        self.assertEqual(self.body[evidence["start"] : evidence["end"]], evidence["text"])
        self.assertEqual(new["reference_answer"], "金额\n100元")
        self.assertEqual(new["review_status"], "pending_human_review")
        self.assertEqual(new["split"], "candidate")

    def test_invalid_range_does_not_create_output(self):
        self.payload["questions"][0]["evidence_line_ranges"] = [["a", 1, 100]]
        with self.assertRaisesRegex(ValueError, "line range"):
            self.run_build()
        self.assertFalse(self.output.exists())

    def test_duplicate_id_does_not_replace_approved_question(self):
        self.payload["questions"][0]["question_id"] = "med-001"
        with self.assertRaisesRegex(ValueError, "Duplicate question"):
            self.run_build()
        self.assertFalse(self.output.exists())

    def test_stale_base_hash_is_rejected(self):
        self.base.write_text(self.base.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Base benchmark fingerprint"):
            self.run_build()

    def test_incomplete_specs_are_rejected(self):
        self.payload["expected_new_questions"] = 2
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.run_build()
        self.assertFalse(self.output.exists())

    def test_user_review_is_never_overwritten_even_without_dataset_output(self):
        self.output.parent.mkdir()
        review = self.output.with_name("review_med-002_med-002.md")
        review.write_text("用户审核：部分支持")
        with self.assertRaises(FileExistsError):
            self.run_build()
        self.assertEqual(review.read_text(), "用户审核：部分支持")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
