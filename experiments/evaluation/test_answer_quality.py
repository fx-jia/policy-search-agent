"""Offline contracts for anonymous, source-grounded answer grading."""
import copy
import importlib.util
from pathlib import Path
import unittest


class QualityTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name("answer_quality.py")
        self.assertTrue(path.exists(), "answer quality grader missing")
        spec = importlib.util.spec_from_file_location("quality_under_test", path)
        self.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)
        self.question = {
            "question_id": "hidden-q", "question": "多久完成？", "difficulty": "Hard",
            "reference_answer": "20个工作日", "required_points": [{"text": "20个工作日"}],
            "evidence": [{"source_document_id": "doc", "source_title": "政策", "text": "受理后20个工作日内完成。"}],
        }
        self.record = {
            "status": "ok", "arm": "ab", "outcome": "complete", "trace": ["secret decision"],
            "answer": "受理后20个工作日内完成。[ID:0]",
            "reference": {"chunks": [{"id": "c"}]},
            "seen_chunks": {"c": {"doc_id": "doc", "docnm_kwd": "政策", "content_with_weight": "受理后20个工作日内完成。"}},
        }

    def packet(self):
        return self.api.build_packet(self.question, self.record, "只评分题干范围")

    def result(self):
        return {"scores": {k: 4 for k in self.api.WEIGHTS},
                "reasons": {k: "要求和原文一致" for k in self.api.WEIGHTS},
                "citation_checks": [{"citation_id": 0, "support": "supports", "claim": "20个工作日",
                                     "quote": "受理后20个工作日内完成。", "reason": "直接支持"}],
                "critical_error": False, "review_needed": False}

    def test_packet_does_not_expose_arm_history_or_grade(self):
        packet = self.packet()
        self.assertEqual(set(packet), {"question", "reference_answer", "required_points", "reference_evidence", "scope", "answer", "citations"})
        self.assertNotIn("secret decision", str(packet))
        self.assertNotIn("Hard", str(packet))

    def test_cited_source_must_have_been_observed(self):
        self.record["seen_chunks"] = {}
        with self.assertRaisesRegex(ValueError, "unseen"):
            self.packet()

    def test_technical_failure_cannot_be_scored_as_answer(self):
        self.record["status"] = "technical_error"
        with self.assertRaisesRegex(ValueError, "no_answer"):
            self.packet()

    def test_out_of_range_citation_is_not_silently_dropped(self):
        self.record["answer"] += "[ID:9]"
        with self.assertRaisesRegex(ValueError, "citation"):
            self.packet()

    def test_weighted_total_is_computed_not_trusted_from_judge(self):
        result = self.result()
        result["scores"]["accuracy"] = 2
        result["scores"]["clarity"] = 0
        self.assertEqual(self.api.validate_result(result, self.packet())["total"], 77.5)

    def test_uncertain_score_stays_missing_not_zero(self):
        result = self.result()
        result["scores"]["grounding"] = None
        result["review_needed"] = True
        self.assertIsNone(self.api.validate_result(result, self.packet())["total"])

    def test_fabricated_supporting_quote_is_rejected(self):
        result = self.result()
        result["citation_checks"][0]["quote"] = "文件没有这些文字"
        with self.assertRaisesRegex(ValueError, "quote"):
            self.api.validate_result(result, self.packet())

    def test_missing_citation_review_is_rejected(self):
        result = self.result()
        result["citation_checks"] = []
        with self.assertRaisesRegex(ValueError, "citation_coverage"):
            self.api.validate_result(result, self.packet())

    def test_invalid_scores_cannot_inflate_average(self):
        for invalid in (True, 5, -1, 3.7, "4"):
            result = self.result()
            result["scores"]["accuracy"] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.api.validate_result(result, self.packet())

    def test_critical_error_is_retained_even_with_high_style(self):
        result = self.result()
        result["critical_error"] = True
        output = self.api.validate_result(result, self.packet())
        self.assertTrue(output["critical_error"])
        self.assertTrue(output["review_needed"])

    def test_validation_does_not_mutate_raw_judge_result(self):
        result = self.result()
        original = copy.deepcopy(result)
        self.api.validate_result(result, self.packet())
        self.assertEqual(result, original)

    def test_reference_crossing_chunks_can_be_verified(self):
        self.assertTrue(self.api.covered_by_sources("受理后20个工作日内完成。名单需要公示七个自然日。", ["受理后20个工作日内完成。", "名单需要公示七个自然日。其他规定"]))

    def test_unverified_reference_number_is_rejected(self):
        self.assertFalse(self.api.covered_by_sources("受理后30个工作日内完成。", ["受理后20个工作日内完成。其他规定"]))

    def test_reference_preserves_decimal_sign_and_percentage(self):
        for reference, source in [("35%", "3.5%"), ("100", "-100"), ("35", "35%"), ("1:2", "12")]:
            with self.subTest(reference=reference, source=source):
                self.assertFalse(self.api.covered_by_sources("待遇补助标准为" + reference + "元的具体规定。", ["待遇补助标准为" + source + "元的具体规定。"]))

    def test_reference_allows_full_width_numbers_and_whitespace(self):
        self.assertTrue(self.api.covered_by_sources("待遇补助比例为３．５％的具体规定。", ["待遇补助比例为3.5% 的具体规定。其他条款。"]))

    def test_short_reference_must_match_whole_phrase(self):
        self.assertTrue(self.api.covered_by_sources("医院收费票据", ["材料名称：医院收费票据，原件。"] ))
        self.assertFalse(self.api.covered_by_sources("医院收费票据", ["医院结算方式", "费用清单票据"]))


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        from experiments.evaluation import answer_quality_eval
        self.api = answer_quality_eval

    def test_unfinished_attempt_blocks_restart(self):
        with self.assertRaisesRegex(ValueError, "incomplete_prior_attempt"):
            self.api.validate_prior_attempts([{"id": "a"}], [])

    def test_unknown_usage_blocks_restart(self):
        with self.assertRaisesRegex(ValueError, "unresolved_prior_usage"):
            self.api.validate_prior_attempts([{"id": "a"}], [{"id": "a", "status": "provider_error", "ledger": {"pending_calls": ["a:0"]}}])

    def test_fatal_guard_stops_even_after_usage_settled(self):
        self.assertTrue(self.api.stop_required({"status": "technical_error", "ledger": {"pending_calls": [], "fatal_reason": "usage_exceeded_reservation"}}))

    def test_confirmed_not_sent_failure_allows_other_jobs(self):
        row = {"id": "a", "status": "technical_error", "reason": "provider_connect_failed", "ledger": {"pending_calls": [], "fatal_reason": None}}
        self.assertFalse(self.api.stop_required(row))
        self.assertEqual(self.api.validate_prior_attempts([{"id": "a"}], [row]), {"a"})


if __name__ == "__main__":
    unittest.main()
