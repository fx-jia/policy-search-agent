import unittest

from experiments.evaluation.generation_report import score_record


class GenerationReportTests(unittest.TestCase):
    def row(self):
        return {"request_id": "q-ranked", "status": "ok", "parsed": {"claims": [{"statement": "340元", "evidence": [{"chunk_id": "c", "quote": "340元"}]}]}, "citation_checks": [{"claim_index": 0, "valid_quote": True}]}

    def test_partial_answer_is_not_fully_correct_even_with_valid_citation(self):
        assessment = {"points": [True, False], "extra_errors": [], "faithful": True, "all_answer_facts_in_claims": True, "citation_support": [True], "reason": "缺第二个子问"}
        r = score_record(self.row(), assessment, 2)
        self.assertFalse(r["answer_correct"])
        self.assertFalse(r["grounded_correct"])
        self.assertEqual(r["correct_points"], 1)

    def test_real_quote_does_not_imply_semantic_citation_support(self):
        assessment = {"points": [True], "extra_errors": [], "faithful": False, "all_answer_facts_in_claims": True, "citation_support": [False], "reason": "引用另一个年份"}
        r = score_record(self.row(), assessment, 1)
        self.assertTrue(r["answer_correct"])
        self.assertEqual(r["valid_quotes"], 1)
        self.assertEqual(r["supported_cited_claims"], 0)
        self.assertFalse(r["grounded_correct"])

    def test_missing_review_or_missing_claim_cannot_be_full_grounded_pass(self):
        with self.assertRaises(ValueError):
            score_record(self.row(), None, 1)
        row = self.row(); row["parsed"]["claims"] = []; row["citation_checks"] = []
        r = score_record(row, {"points": [True], "extra_errors": [], "faithful": True, "all_answer_facts_in_claims": False, "citation_support": [], "reason": "无引用"}, 1)
        self.assertFalse(r["grounded_correct"])

    def test_failed_request_is_preserved_without_claiming_answer_was_judged(self):
        r = score_record({"status": "error"}, None, 2)
        self.assertFalse(r["answer_correct"])
        self.assertFalse(r["assessed"])

    def test_uncited_fact_in_answer_prevents_complete_grounded_pass(self):
        row = self.row()
        row['parsed']['answer'] = '2022年340元；2024年420元。'
        assessment = {"points": [True, True], "extra_errors": [], "faithful": True, "citation_support": [True], "all_answer_facts_in_claims": False, "reason": "第二个金额虽正确且在上下文中，但漏了对应claim和引用"}
        r = score_record(row, assessment, 2)
        self.assertTrue(r['answer_correct'])
        self.assertFalse(r['grounded_correct'])


if __name__ == "__main__":
    unittest.main()
