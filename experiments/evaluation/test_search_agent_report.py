import unittest

from experiments.evaluation.search_agent_report import summarize, paired_changes, diagnostics


class ReportTests(unittest.TestCase):
    def test_missing_physical_embedding_telemetry_is_not_zero_calls(self):
        row = {"question_id": "a", "status": "ok", "latency_ms": 1000, "query_embedding_calls": 5}
        self.assertIsNone(summarize(["a"], [row], {})["query_embedding_physical_calls"])
        row["embedding_physical"] = {"query_physical_calls": 0}
        self.assertEqual(summarize(["a"], [row], {})["query_embedding_physical_calls"], 0)
        row["embedding_physical"]["query_physical_calls"] = 5
        missing = {"question_id": "b", "status": "timeout", "latency_ms": 1000}
        self.assertIsNone(summarize(["a", "b"], [row, missing], {})["query_embedding_physical_calls"])

    def test_prepared_but_budget_blocked_context_is_not_counted_as_sent(self):
        question = {"evidence": [{"text": "条款", "source_document_id": "source"}], "known_relevant_chunk_ids": []}
        body = {"messages": [{"role": "system", "content": "ID: 0\n├── Title: 标题\n└── Content:\n条款"}]}
        record = {"seen_chunks": {"a": {"doc_id": "d", "docnm_kwd": "标题", "content_with_weight": "条款"}},
                  "selected": [{"chunk_id": "a", "document_id": "d"}], "final_request": body, "calls": []}
        blocked = diagnostics(question, record, {"d": "source"}, {})
        self.assertEqual(blocked["final_literal_evidence"], [False])
        self.assertEqual(blocked["final_request_delivery"], "not_sent")
        record["calls"] = [{"request": body, "reserve_tokens": 100}]
        unknown = diagnostics(question, record, {"d": "source"}, {})
        self.assertEqual(unknown["final_request_delivery"], "unknown")
        record["calls"][0]["usage"] = {"total_tokens": 10}
        confirmed = diagnostics(question, record, {"d": "source"}, {})
        self.assertEqual(confirmed["final_literal_evidence"], [True])
        self.assertEqual(confirmed["final_request_delivery"], "confirmed")

    def test_fallback_answers_do_not_hide_controller_failures(self):
        rows = [{"question_id": "a", "status": "ok", "answer": "答案", "latency_ms": 1000,
                 "controller_error": "invalid_action", "fallback_used": True},
                {"question_id": "b", "status": "budget_censored", "latency_ms": 1000,
                 "controller_error": "unknown_evidence_id", "fallback_used": True}]
        result = summarize(["a", "b"], rows, {})
        self.assertEqual(result["controller_failures"], 2)
        self.assertEqual(result["fallback_questions"], 2)
        self.assertEqual(result["fallback_answered_questions"], 1)

    def test_incomplete_timeout_telemetry_is_reported_as_unknown(self):
        result = summarize(["a"], [{"question_id": "a", "status": "host_timeout", "latency_ms": 60000,
                                    "control_trace_complete": False}], {})
        self.assertEqual(result["controller_telemetry_incomplete"], 1)

    def test_pending_review_is_not_counted_as_a_regression(self):
        result = paired_changes(["a"], {"a": "complete"}, {"a": "pending_review"})
        self.assertEqual(result["regressed"], [])
        self.assertEqual(result["unpaired"], ["a"])

    def test_failed_attempts_remain_in_denominator_and_pending_is_not_wrong(self):
        rows = [{"question_id": "a", "status": "ok", "latency_ms": 1000, "calls": []},
                {"question_id": "b", "status": "timeout", "latency_ms": 60000, "calls": []}]
        result = summarize(["a", "b", "c"], rows, {"a": {"outcome": "complete"}})
        self.assertEqual(result["planned"], 3)
        self.assertEqual(result["attempted"], 2)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["unassessed"], 2)

    def test_pairing_counts_both_improvements_and_regressions(self):
        result = paired_changes(["a", "b", "c"], {"a": "complete", "b": "incomplete", "c": "complete"},
                                {"a": "incomplete", "b": "complete"})
        self.assertEqual(result["improved"], ["b"])
        self.assertEqual(result["regressed"], ["a"])
        self.assertEqual(result["unpaired"], ["c"])


if __name__ == "__main__":
    unittest.main()
