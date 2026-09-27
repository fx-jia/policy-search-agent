"""Keep held-out questions and modified split artifacts out of development runs."""

import json
from pathlib import Path
import tempfile
import unittest

from experiments.analysis.profile_dataset import fingerprint
from experiments.evaluation.run import verify_development_split


class SplitGuardTest(unittest.TestCase):
    def test_test_questions_rejected_without_manifest(self):
        with self.assertRaisesRegex(ValueError, "Held-out"):
            verify_development_split([{"split": "test"}], "unused", None, "source")

    def test_versioned_questions_require_manifest(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            verify_development_split([{"split": "dev", "split_version": "v"}], "unused", None, "source")

    def test_frozen_dev_accepted_but_modified_content_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [{"question_id": "q1", "split": "dev", "split_version": "v", "policy_group_id": "g", "question": "原问题"}]
            benchmark = root / "dev.jsonl"
            benchmark.write_text(json.dumps(samples[0]))
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "source_sha256": "source",
                        "output_sha256": {"dev": fingerprint(benchmark)},
                        "split_version": "v",
                        "assignments": {"q1": "g"},
                        "groups": {"g": {"split": "dev", "question_ids": ["q1"]}},
                    }
                )
            )
            self.assertEqual(verify_development_split(samples, benchmark, manifest, "source")["split"], "dev")
            benchmark.write_text(benchmark.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_development_split(samples, benchmark, manifest, "source")


if __name__ == "__main__":
    unittest.main()
