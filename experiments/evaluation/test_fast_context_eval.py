import unittest
import json
from pathlib import Path
import tempfile

from experiments.evaluation.fast_context_eval import patch_runner


class FrozenRunnerTests(unittest.TestCase):
    def test_only_selection_changes_and_original_runner_stays_untouched(self):
        from experiments.evaluation.fast_rag_eval import OUT
        import json
        config = json.loads((OUT / "generation/config.json").read_text())
        old = config["module_sources"]["experiments.search_agent.fast"]
        new = patch_runner(old)
        self.assertIn("from .fast_context import select_context", new)
        self.assertNotIn("def select_context(", new)
        self.assertIn("select_context(question, initial, backend.count_tokens, context_text)", new)
        before = old[old.index("async def run("):]
        after = new[new.index("async def run("):]
        self.assertEqual(after, before.replace("select_context(initial, backend.count_tokens)",
                                              "select_context(question, initial, backend.count_tokens, context_text)"))
        self.assertEqual(config["module_sources"]["experiments.search_agent.fast"], old)

    def test_unknown_runner_shape_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unexpected_fast_runner"):
            patch_runner("# incompatible runner")

    def test_missing_usage_ledger_is_reported_separately_not_known_zero(self):
        from experiments.evaluation.fast_context_eval import usage_summary
        summary = usage_summary([{"status": "worker_error"}, {"calls": [{"usage": {"total_tokens": 10}}]}])
        self.assertEqual(summary["known_tokens"], 10)
        self.assertEqual(summary["unreported_records"], 1)
        self.assertIsNone(summary["total_tokens"])

    def test_empty_scored_group_and_unknown_tokens_remain_missing(self):
        from experiments.evaluation.fast_context_eval import arm_summary
        value = arm_summary("a", ["q"], {"a": {"q": {"latency_ms": 10, "calls": [{"status": "pending", "reserve_tokens": 100}]}}},
                            {"a": {"q": 90}}, {"q": {"difficulty": "Easy"}})
        self.assertEqual(value["difficulty"]["Hard"]["n"], 0)
        self.assertIsNone(value["difficulty"]["Hard"]["quality_mean"])
        self.assertIsNone(value["tokens"]["mean"])

    def test_modified_grading_input_fails_integrity_verification(self):
        from experiments.evaluation.fast_context_eval import verify_quality
        from experiments.evaluation.fast_rag_eval import SHA
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "quality").mkdir()
            config, jobs, source = p / "quality/config.json", p / "quality/jobs.json", p / "records.jsonl"
            for item in (config, jobs, source):
                item.write_text("{}")
            manifest = {"config_sha256": SHA(config), "jobs_sha256": SHA(jobs), "input_sha256": {str(source): SHA(source)}}
            (p / "quality/manifest.json").write_text(json.dumps(manifest))
            verify_quality(p)
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "grading_input_changed"):
                verify_quality(p)


if __name__ == "__main__":
    unittest.main()
