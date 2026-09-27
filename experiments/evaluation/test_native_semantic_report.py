import json
from pathlib import Path
import tempfile
import unittest

from experiments.evaluation import native_semantic_report as report


RUBRIC = json.loads((Path(__file__).resolve().parents[1] / "results/private/native-corpus1000-v2-preflight/combined_answer_rubric_v2.json").read_text())
POINTS = {q["question_id"]: len(q["required_points"]) for q in RUBRIC["questions"]}


def annotation(qid, *, pass_values=None, support=None, wrong=None, unsupported=None, citations=None, uncertain=False, mixed=False):
    count = POINTS[qid]
    return {
        "question_id": qid,
        "points_pass": [True] * count if pass_values is None else pass_values,
        "point_reasons": ["Answer supplies this point."] * count,
        "point_prompt_support": [True] * count if support is None else support,
        "incorrect_extra_claims": [] if wrong is None else wrong,
        "unsupported_extra_claims": [] if unsupported is None else unsupported,
        "citation_issues": [] if citations is None else citations,
        "failure_category": "none",
        "notes": "Assistant review against the captured prompt.",
        "applicability_uncertain": uncertain,
        "mixed_refusal": mixed,
    }


class NativeSemanticReportTests(unittest.TestCase):
    def test_correct_unsupported_and_wrong_extras_have_distinct_scores(self):
        rows = [
            annotation("med-001", support=[True, False], citations=["Citation 1 is unrelated."], uncertain=True),
            annotation("med-002", wrong=["Extra policy date is incorrect."]),
            annotation("med-003", unsupported=["Extra eligibility claim is absent from prompt."], mixed=True),
        ]
        summary, scored = report.aggregate(RUBRIC, rows)
        by_id = {row["question_id"]: row for row in scored}
        self.assertTrue(by_id["med-001"]["rubric_content_complete"])
        self.assertFalse(by_id["med-001"]["rubric_content_complete_and_grounded"])
        self.assertFalse(by_id["med-002"]["rubric_content_complete"])
        self.assertTrue(by_id["med-003"]["rubric_content_complete"])
        self.assertFalse(by_id["med-003"]["rubric_content_complete_and_grounded"])
        self.assertEqual(summary["all150"]["planned_questions"], 150)
        self.assertEqual(summary["all150"]["planned_points"], 319)
        self.assertEqual(summary["all150"]["assessed_questions"], 3)
        self.assertEqual(summary["all150"]["semantic_citation_issue_questions"], 1)
        self.assertEqual(summary["all150"]["rubric_content_complete_count"], 2)
        self.assertEqual(summary["all150"]["rubric_content_complete_and_grounded_count"], 0)
        self.assertEqual(summary["all150"]["prompt_fully_supports_required_points_count"], 2)
        self.assertEqual(summary["all150"]["unsupported_extra_claim_question_count"], 1)
        self.assertEqual(summary["all150"]["applicability_uncertain_count"], 1)
        self.assertEqual(summary["all150"]["mixed_refusal_count"], 1)
        self.assertTrue(summary["all150"]["partial_run"])
        self.assertEqual(len(scored), 150)
        self.assertFalse(by_id["med-004"]["assessed"])

    def test_rejects_malformed_rows_duplicates_unknowns_and_required_omissions(self):
        good = annotation("med-001")
        for field, invalid in [
            ("points_pass", [1, True]),
            ("points_pass", [True]),
            ("point_reasons", ["one"]),
            ("point_prompt_support", [True, 1]),
            ("point_prompt_support", [True]),
        ]:
            with self.subTest(field=field, invalid=invalid):
                bad = {**good, field: invalid}
                with self.assertRaises(ValueError):
                    report.aggregate(RUBRIC, [bad])
        with self.assertRaises(ValueError):
            report.aggregate(RUBRIC, [good, good])
        with self.assertRaises(ValueError):
            report.aggregate(RUBRIC, [{**good, "question_id": "unknown"}])
        with self.assertRaises(ValueError):
            report.aggregate(RUBRIC, [{**good, "applicability_uncertain": 1}])
        with self.assertRaises(ValueError):
            report.aggregate(RUBRIC, [{**good, "mixed_refusal": "yes"}])
        with self.assertRaises(ValueError):
            report.aggregate(RUBRIC, [good], require_complete=True)
        complete_rows = [annotation(q["question_id"]) for q in RUBRIC["questions"]]
        summary, _ = report.aggregate(RUBRIC, complete_rows, require_complete=True)
        self.assertFalse(summary["all150"]["partial_run"])
        self.assertEqual(summary["all150"]["point_pass_fraction_planned"], 1.0)

    def test_only_explicit_annotation_globs_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            selected = folder / "selected.jsonl"
            unrelated = folder / "unrelated.jsonl"
            selected.write_text(json.dumps(annotation("med-001")) + "\n")
            unrelated.write_text(json.dumps(annotation("med-002")) + "\n")
            rows, paths = report.load_annotations([str(selected)])
            self.assertEqual([r["question_id"] for r in rows], ["med-001"])
            self.assertEqual(paths, [selected.resolve()])
            with self.assertRaises(ValueError):
                report.load_annotations([str(selected), str(selected)])


if __name__ == "__main__":
    unittest.main()
