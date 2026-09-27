import unittest

from experiments.evaluation.fast_rag_eval import paired_ids, unique_rows, freeze_runtime


class ProtocolTests(unittest.TestCase):
    def test_selection_uses_all_paired_dev_even_when_baseline_failed(self):
        dev = {"d1": {}, "d2": {}}
        rag = {"d1": {"status": "ok"}, "d2": {"status": "timeout"}}
        agent = {"d1": {}, "d2": {}}
        self.assertEqual(paired_ids(dev, rag, agent), ["d1", "d2"])
        with self.assertRaisesRegex(ValueError, "outside_development"):
            paired_ids(dev, rag, {**agent, "test1": {}})

    def test_duplicate_answers_are_not_cherry_picked(self):
        with self.assertRaisesRegex(ValueError, "duplicate_answer"):
            unique_rows([[{"question_id": "q", "status": "timeout"}], [{"question_id": "q", "status": "ok"}]])

    def test_frozen_backend_changes_only_runner_import_and_loads_fast_first(self):
        sources = {"experiments.search_agent.core": "# core",
                   "experiments.search_agent.runtime": "from .core import AgentLimits, run\n# rest"}
        frozen = freeze_runtime(sources, "# fast")
        self.assertEqual(frozen["experiments.search_agent.runtime"], "from .core import AgentLimits\nfrom .fast import run\n# rest")
        self.assertLess(list(frozen).index("experiments.search_agent.fast"), list(frozen).index("experiments.search_agent.runtime"))
        self.assertEqual(sources["experiments.search_agent.runtime"], "from .core import AgentLimits, run\n# rest")


if __name__ == "__main__":
    unittest.main()
