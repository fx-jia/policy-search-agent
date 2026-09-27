import hashlib
import json
import unittest

from experiments.evaluation import generation_pilot
from experiments.evaluation.generation_pilot import build_messages, check_citations, safe_chunk, sanitize_content, validate_response


class GenerationPilotTests(unittest.TestCase):
    def test_short_ids_preserve_evidence_and_document_grouping_without_exposing_two_id_types(self):
        self.assertTrue(hasattr(generation_pilot, "citation_view"), "short citation mapping is missing")
        chunks = [{"id": "hash-a", "source_document_id": "source-x", "content": "上半句30"}, {"id": "hash-b", "source_document_id": "source-y", "content": "其他政策"}, {"id": "hash-c", "source_document_id": "source-x", "content": "元，标准380元。"}]
        messages = build_messages("标准？", chunks, short_ids=True)
        prompt = messages[1]['content']
        self.assertNotIn('hash-a', prompt)
        self.assertNotIn('source-x', prompt)
        self.assertIn('[chunk_id=E1]', prompt)
        self.assertIn('[chunk_id=E3]', prompt)
        self.assertIn('[["E1", "E3"], ["E2"]]', prompt)
        for c in chunks:
            self.assertIn(c['content'], prompt)
        view = generation_pilot.citation_view(chunks, True)
        self.assertEqual([c['id'] for c in view], ['E1', 'E2', 'E3'])
        self.assertEqual([c['original_chunk_id'] for c in view], ['hash-a', 'hash-b', 'hash-c'])
        self.assertEqual(chunks[0]['id'], 'hash-a')

    def test_short_citations_are_scoped_to_the_current_request_without_guessing_invalid_ids(self):
        self.assertTrue(hasattr(generation_pilot, "citation_view"), "short citation mapping is missing")
        chunks = [{"id": "hash-a", "source_document_id": "source-x", "content": "2024年标准380元。"}, {"id": "hash-b", "source_document_id": "source-y", "content": "2023年标准350元。"}]
        response = {'claims': [{'evidence': [{'chunk_id': cid, 'quote': quote} for cid, quote in [('E1', '2024年标准380元。'), ('E2', '380元'), ('E1 ', '380元'), ('hash-a', '380元'), ('E1', '2024年...380元。')]]}]}
        checks = check_citations(response, generation_pilot.citation_view(chunks, True))
        self.assertEqual([c['valid_quote'] for c in checks], [True, False, False, False, False])
        self.assertEqual(checks[0]['original_chunk_id'], 'hash-a')
        self.assertIsNone(checks[2]['original_chunk_id'])

    def test_blocked_source_is_rejected_even_if_allowlisted(self):
        c = {"id": "c", "source_document_id": "79870", "content": "普通文字"}
        reviewed = {"c": {"source_document_id": "79870", "raw_sha256": hashlib.sha256(c["content"].encode()).hexdigest()}}
        with self.assertRaises(ValueError):
            safe_chunk(c, reviewed, {"79870"})

    def test_unreviewed_or_changed_chunk_cannot_leave_the_machine(self):
        c = {"id": "c", "source_document_id": "a", "content": "规定"}
        with self.assertRaises(ValueError):
            safe_chunk(c, {}, set())
        with self.assertRaises(ValueError):
            safe_chunk(c, {"c": {"source_document_id": "a", "raw_sha256": "wrong"}}, set())

    def test_contact_lines_and_image_links_removed_but_policy_preserved(self):
        raw = "参保缴费340元。\n联系人：测试人\n联系电话：0516-12345678\n联系地址：测试街道\n![](https://example.com/image.jpg)\n与用人单位解除劳动关系者可参保。"
        actual = sanitize_content(raw)
        self.assertNotIn("测试人", actual)
        self.assertNotIn("12345678", actual)
        self.assertNotIn("https://", actual)
        self.assertIn("参保缴费340元", actual)
        self.assertIn("劳动关系者可参保", actual)

    def test_prompt_uses_question_and_approved_context_without_gold(self):
        messages = build_messages("问什么？", [{"id": "c", "source_document_id": "a", "content": "政策正文"}])
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("问什么？", messages[1]["content"])
        self.assertIn("chunk=c", messages[1]["content"])
        self.assertIn("政策正文", messages[1]["content"])

    def test_citation_requires_a_sent_chunk_and_unchanged_quote(self):
        response = {"answer": "340元", "claims": [{"statement": "340元", "evidence": [{"chunk_id": "a", "quote": "340元"}, {"chunk_id": "absent", "quote": "340元"}, {"chunk_id": "b", "quote": "340元"}]}], "insufficient_evidence": False}
        checks = check_citations(response, [{"id": "a", "content": "缴费340\n元。"}, {"id": "b", "content": "缴费350元。"}])
        self.assertEqual([r["valid_quote"] for r in checks], [True, False, False])

    def test_valid_json_is_not_enough_without_required_fields(self):
        with self.assertRaises(ValueError):
            validate_response('{"answer":"金额340元"}')
        valid = {"answer": "材料不足，无法确认", "claims": [], "insufficient_evidence": True}
        self.assertEqual(validate_response(json.dumps(valid)), valid)


if __name__ == "__main__":
    unittest.main()
