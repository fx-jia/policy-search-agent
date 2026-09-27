"""Offline contracts for the integrated policy QA path; no service calls."""

import hashlib
import json
import unittest

from experiments import qa
from experiments.evaluation.generation_pilot import sanitize_content


def fixture():
    chunks = [
        {"id": "a", "document_id": "d", "source_document_id": "s", "order": 0, "content": "个人缴费提高至", "tokens": 1000},
        {"id": "b", "document_id": "d", "source_document_id": "s", "order": 1, "content": "380元。", "tokens": 500},
        {"id": "c", "document_id": "x", "source_document_id": "t", "order": 0, "content": "其他内容。", "tokens": 1100},
    ]
    snapshot = {c["id"]: c for c in chunks}
    ranks = [{k: snapshot[cid][k] for k in ["id", "document_id", "source_document_id"]} | {"rank": i} for i, cid in enumerate(["a", "c", "b"], 1)]
    review = {c["id"]: {"source_document_id": c["source_document_id"], "raw_sha256": hashlib.sha256(c["content"].encode()).hexdigest(), "safe_sha256": hashlib.sha256(sanitize_content(c["content"]).encode()).hexdigest()} for c in chunks}
    return ranks, snapshot, review


class PolicyQATest(unittest.TestCase):
    def test_boundary_selection_is_used_before_short_citation_mapping(self):
        self.assertTrue(hasattr(qa, "prepare_request"), "integrated context preparation missing")
        ranks, snapshot, review = fixture()
        request = qa.prepare_request("个人缴费多少？", ranks, snapshot, review, set(), "short_ids")
        self.assertEqual(request["selected_ids"], ["a", "b"])
        self.assertEqual(request["selection_tokens_cl100k"], 1500)
        self.assertEqual(request["citation_map"]["E2"]["original_chunk_id"], "b")
        self.assertNotIn("其他内容", request["messages"][1]["content"])

    def test_unreviewed_or_blocked_selected_text_cannot_build_a_prompt(self):
        self.assertTrue(hasattr(qa, "prepare_request"), "privacy gate missing")
        ranks, snapshot, review = fixture()
        for allowed, blocked in [({}, set()), (review, {"s"})]:
            with self.subTest(blocked=blocked), self.assertRaises(ValueError):
                qa.prepare_request("个人缴费多少？", ranks, snapshot, allowed, blocked, "short_ids")

    def test_empty_retrieval_does_not_request_generation(self):
        self.assertTrue(hasattr(qa, "prepare_request"), "empty context handling missing")
        request = qa.prepare_request("政策？", [], {}, {}, set(), "short_ids")
        self.assertEqual(request["status"], "no_context")
        self.assertEqual(request["messages"], [])

    def test_model_input_excludes_labels_and_rejects_personal_identifiers(self):
        self.assertTrue(hasattr(qa, "prepare_request"), "query validation missing")
        ranks, snapshot, review = fixture()
        ranks[0]["answer"] = "不应发送的金标"
        req = qa.prepare_request("政策？", ranks, snapshot, review, set(), "parts")
        self.assertNotIn("不应发送的金标", str(req["messages"]))
        with self.assertRaises(ValueError):
            qa.prepare_request("手机号13800138000对应的医保情况", ranks, snapshot, review, set(), "parts")

    def test_structured_parts_render_every_answer_and_preserve_refusal(self):
        self.assertTrue(hasattr(qa, "parse_answer"), "structured answer parsing missing")
        data = {"parts": [
            {"subquestion": "标准？", "answer": "个人缴费380元。", "claims": [{"statement": "个人缴费380元。", "evidence": [{"chunk_id": "E2", "quote": "380元。"}]}], "insufficient_evidence": False},
            {"subquestion": "实施日期？", "answer": "现有材料无法确认实施日期。", "claims": [], "insufficient_evidence": True},
        ]}
        parsed = qa.parse_answer(json.dumps(data, ensure_ascii=False), "parts")
        self.assertIn("380元", parsed["answer"])
        self.assertIn("无法确认实施日期", parsed["answer"])
        self.assertEqual(len(parsed["claims"]), 1)
        self.assertTrue(parsed["insufficient_evidence"])
        self.assertEqual(len(parsed["parts"]), 2)

    def test_empty_parts_duplicate_subquestions_and_uncited_assertions_are_invalid(self):
        self.assertTrue(hasattr(qa, "parse_answer"), "structured schema validation missing")
        part = {"subquestion": "标准？", "answer": "380元", "claims": [], "insufficient_evidence": False}
        for data in [{"parts": []}, {"parts": [part]}, {"parts": [{**part, "insufficient_evidence": True}] * 2}]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                qa.parse_answer(json.dumps(data), "parts")

    def test_invalid_citation_is_not_presented_as_a_valid_answer(self):
        self.assertTrue(hasattr(qa, "process_response"), "citation validation missing")
        ranks, snapshot, review = fixture()
        req = qa.prepare_request("标准？", ranks, snapshot, review, set(), "short_ids")
        answer = {"answer": "380元。", "claims": [{"statement": "380元。", "evidence": [{"chunk_id": "E9", "quote": "380元。"}]}], "insufficient_evidence": False}
        result = qa.process_response({"status": "ok", "text": json.dumps(answer)}, req)
        self.assertEqual(result["status"], "invalid_citations")
        self.assertNotIn("380元", qa.render_answer(result, req))
        answer["claims"][0]["evidence"][0]["chunk_id"] = "E2"
        valid = qa.process_response({"status": "ok", "text": json.dumps(answer)}, req)
        self.assertEqual(valid["status"], "ok")
        self.assertIn("380元", qa.render_answer(valid, req))
        self.assertIn("E2", qa.render_answer(valid, req))

    def test_provider_failure_is_not_parsed_or_silently_retried(self):
        self.assertTrue(hasattr(qa, "process_response"), "provider failure handling missing")
        result = qa.process_response({"status": "output_truncated", "text": "partial"}, {"mode": "parts"})
        self.assertEqual(result["status"], "output_truncated")
        self.assertNotIn("parsed", result)


if __name__ == "__main__":
    unittest.main()
