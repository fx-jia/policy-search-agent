import unittest

from experiments.evaluation.audit_context import find_evidence


class ContextAuditTests(unittest.TestCase):
    def test_formatting_is_ignored_but_source_identity_is_required(self):
        evidence = [{"source_document_id": "a", "text": "比例为 60%"}]
        chunks = [{"id": "one", "source_document_id": "b", "content": "比例为60%"}, {"id": "two", "source_document_id": "a", "content": "<p>比例为\n60%</p>"}]
        self.assertEqual(find_evidence(evidence, chunks), [["two"]])

    def test_different_number_does_not_match(self):
        evidence = [{"source_document_id": "a", "text": "比例为60%"}]
        chunks = [{"id": "one", "source_document_id": "a", "content": "比例为65%"}]
        self.assertEqual(find_evidence(evidence, chunks), [[]])

    def test_split_span_is_left_for_inspection(self):
        evidence = [{"source_document_id": "a", "text": "缴费满6个月，生育当月正常参保"}]
        chunks = [{"id": "one", "source_document_id": "a", "content": "缴费满6个月"}, {"id": "two", "source_document_id": "a", "content": "生育当月正常参保"}]
        self.assertEqual(find_evidence(evidence, chunks), [[]])


if __name__ == "__main__":
    unittest.main()
