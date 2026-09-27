import hashlib
import unittest

from experiments.evaluation import heldout_generation as subject


class HeldoutGenerationTests(unittest.TestCase):
    def test_generation_failures_stay_in_planned_accuracy_denominator(self):
        from experiments.evaluation.heldout_generation_report import summarize

        rows = [{"policy": "a", "answer_correct": v, "grounded_correct": v, "faithful": v, "assessed": v, "correct_points": int(v), "point_count": 1, "claims": int(v), "supported_cited_claims": int(v), "quotes": int(v), "valid_quotes": int(v)} for v in [True, False]]
        result = summarize(rows, ["a"])["a"]
        self.assertEqual(result["answer_accuracy_all_planned"], 0.5)
        self.assertEqual(result["planned_questions"], 2)

    def test_inline_contacts_are_removed_but_policy_requirements_remain(self):
        raw = "申请须持身份证和医保卡。\n咨询：联系人：张三，电话：0516-12345678，邮箱：person@example.com\n李四委员：\n待遇为每月900元。"
        self.assertEqual(subject.sanitize(raw), "申请须持身份证和医保卡。\n待遇为每月900元。")

    def test_individual_record_is_blocked_before_redaction(self):
        with self.assertRaises(ValueError):
            subject.sanitize("患者姓名：测试人\n住院病历号：12345\n待遇900元")

    def test_reviewed_spans_are_redacted_without_dropping_the_chunk(self):
        text = "前文。个人案例。每年900元。"
        self.assertEqual(subject.redact(text, [[3, 8]]), "前文。[已省略非政策个人信息]每年900元。")
        with self.assertRaises(ValueError):
            subject.redact(text, [[3, 9], [8, 10]])

    def test_request_uses_saved_order_and_never_includes_gold(self):
        chunks = {"one": {"id": "one", "source_document_id": "S", "content": "每年900元。", "tokens": 20}}
        review = {"one": {"source_document_id": "S", "raw_sha256": hashlib.sha256("每年900元。".encode()).hexdigest(), "safe_sha256": hashlib.sha256("每年900元。".encode()).hexdigest()}}
        queries = [{"question_id": "q", "question": "每年多少？", "reference_answer": "GOLD_SECRET"}]
        sels = [{"question_id": "q", "method": m, "policy": p, "selected_ids": ["one"], "tokens": 20} for m, p in [("dense", "boundary"), ("hybrid", "ranked")]]
        requests = subject.assemble(queries, sels, chunks, review)
        self.assertEqual([r["policy"] for r in requests], ["dense_boundary", "hybrid_ranked"])
        self.assertEqual(requests[0]["selected_ids"], ["one"])
        self.assertNotIn("GOLD_SECRET", str(requests))
        with self.assertRaises(ValueError):
            subject.assemble(queries, sels + [sels[0]], chunks, review)
        chunks["one"]["content"] = "每年800元。"
        with self.assertRaises(ValueError):
            subject.assemble(queries, sels, chunks, review)

    def test_missing_condition_cannot_silently_shrink_denominator(self):
        with self.assertRaises(ValueError):
            subject.assemble([{"question_id": "q", "question": "问题"}], [], {}, {})

    def test_quota_and_protocol_errors_stop_without_retries(self):
        self.assertTrue(subject.stop_after({"error_category": "quota"}, 1))
        self.assertTrue(subject.stop_after({"error_category": "worker_protocol"}, 1))
        self.assertFalse(subject.stop_after({"error_category": "transport_or_service"}, 1))
        self.assertTrue(subject.stop_after({"error_category": "transport_or_service"}, 2))


if __name__ == "__main__":
    unittest.main()
