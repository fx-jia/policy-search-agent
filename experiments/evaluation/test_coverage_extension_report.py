import importlib
import importlib.util
import unittest


class ExtensionReportTests(unittest.TestCase):
    def test_paired_gains_regressions_and_strata_keep_failed_questions(self):
        name = "experiments.evaluation.coverage_extension_report"
        self.assertIsNotNone(importlib.util.find_spec(name), "extension analysis missing")
        m = importlib.import_module(name)
        rubric = {"a": {"policy_group_id": "g1", "subgroup": "multiple_requirements"}, "b": {"policy_group_id": "g2", "subgroup": "single_requirement"}, "failed": {"policy_group_id": "g2", "subgroup": "multiple_requirements"}}
        scored = []
        for qid, outcomes in [("a", [False, False, True]), ("b", [True, True, False]), ("failed", [False, False, False])]:
            for policy, good in zip(["draft", "generic", "coverage"], outcomes):
                scored.append({"question_id": qid, "policy": policy, "answer_correct": good, "grounded_correct": good})
        result = m.paired_analysis(scored, rubric)
        pair = result["generic_to_coverage"]["grounded_correct"]
        self.assertEqual(pair["gained"], ["a"])
        self.assertEqual(pair["lost"], ["b"])
        self.assertEqual(pair["delta"], 0)
        self.assertEqual(pair["questions"], 3)
        self.assertLess(pair["cluster_ci95"][0], 0)
        self.assertGreater(pair["cluster_ci95"][1], 0)
        self.assertEqual(pair["subgroups"]["multiple_requirements"], {"questions": 2, "left_correct": 0, "right_correct": 1})


if __name__ == "__main__":
    unittest.main()
