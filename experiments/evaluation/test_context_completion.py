"""Behavioral checks for offline context selection and evidence evaluation."""

import unittest

from experiments.evaluation.context_completion import Candidate, select_context, evidence_coverage


def chunk(cid, rank, order, text, tokens=10, source="a", document="doc-a"):
    return Candidate(cid, document, source, rank, order, text, tokens)


class ContextCompletionTests(unittest.TestCase):
    def test_split_completion_promotes_only_adjacent_candidate_and_stays_in_budget(self):
        rows = [chunk("a0", 1, 0, "个人缴费30"), chunk("b0", 2, 0, "其他。", source="b", document="doc-b"), chunk("a1", 34, 1, "元，合计380元。")]
        self.assertEqual([c.id for c in select_context(rows, "ranked", 20)], ["a0", "b0"])
        self.assertEqual([c.id for c in select_context(rows, "boundary", 20)], ["a0", "a1"])

    def test_complete_sentence_does_not_trigger_boundary_completion(self):
        rows = [chunk("a0", 1, 0, "缴费380元。"), chunk("b0", 2, 0, "其他。", source="b", document="doc-b"), chunk("a1", 9, 1, "下一款。")]
        self.assertEqual([c.id for c in select_context(rows, "boundary", 20)], ["a0", "b0"])
        self.assertEqual([c.id for c in select_context(rows, "neighbors", 20)], ["a0", "a1"])

    def test_completion_does_not_jump_missing_neighbor_or_recurse(self):
        rows = [chunk("a0", 1, 0, "开始"), chunk("a1", 7, 1, "继续"), chunk("a2", 8, 2, "结束。"), chunk("a4", 9, 4, "隔块。")]
        self.assertEqual([c.id for c in select_context(rows, "neighbors", 30, seed_top_k=1)], ["a0", "a1", "a2"])
        # Rank 2 unrelated content must beat an a2 recursively expanded from a1.
        rows.append(chunk("b0", 2, 0, "其他。", source="b", document="doc-b"))
        self.assertEqual([c.id for c in select_context(rows, "neighbors", 30, seed_top_k=1)], ["a0", "a1", "b0"])

    def test_oversized_chunks_are_skipped_and_duplicates_rejected(self):
        rows = [chunk("a0", 1, 0, "长。", 21), chunk("a1", 2, 1, "短。", 8), chunk("a2", 3, 2, "短。", 12)]
        self.assertEqual([c.id for c in select_context(rows, "ranked", 20)], ["a1", "a2"])
        with self.assertRaises(ValueError):
            select_context([rows[1], rows[1]], "ranked", 20)

    def test_joined_evidence_requires_same_source_and_contiguous_order(self):
        evidence = [{"source_document_id": "a", "text": "缴费380元。"}]
        a = chunk("a0", 1, 0, "缴费3")
        b = chunk("a1", 2, 1, "80元。")
        self.assertEqual(evidence_coverage(evidence, [b, a]), [True])
        self.assertEqual(evidence_coverage(evidence, [a, chunk("a2", 2, 2, "80元。")]), [False])
        self.assertEqual(evidence_coverage(evidence, [a, chunk("b1", 2, 1, "80元。", source="b", document="doc-b")]), [False])
        self.assertEqual(evidence_coverage(evidence, [a, chunk("a1", 2, 1, "90元。")]), [False])

    def test_source_order_ambiguity_is_rejected(self):
        with self.assertRaises(ValueError):
            select_context([chunk("a", 1, 0, "一"), chunk("b", 2, 0, "二")], "neighbors", 20)

    def test_boundary_start_uses_preceding_chunk_end(self):
        rows = [chunk("a1", 1, 1, "元。"), chunk("a0", 8, 0, "缴费380")]
        self.assertEqual([c.id for c in select_context(rows, "boundary", 20)], ["a1", "a0"])


if __name__ == "__main__":
    unittest.main()
