"""Hand-calculated cases; no services, embeddings or relevance labels."""

import math
import unittest

from experiments.retrieval.fusion import Hit, fuse
from experiments.retrieval.bm25_worker import bm25_body


class FusionTests(unittest.TestCase):
    def test_rrf_rewards_shared_candidate_using_one_based_ranks(self):
        rows = fuse([Hit('a', 100), Hit('b', 3)], [Hit('c', .9), Hit('b', .8)], 'rrf')
        self.assertEqual([r['id'] for r in rows], ['b', 'a', 'c'])
        self.assertAlmostEqual(rows[0]['score'], 2 / 62)
        self.assertAlmostEqual(rows[1]['score'], 1 / 61)
        self.assertIsNone(rows[1]['dense_rank'])

    def test_weighted_uses_separate_minmax_and_zero_for_missing(self):
        rows = fuse([Hit('a', 100), Hit('b', 40), Hit('d', 20)], [Hit('b', .9), Hit('c', .6), Hit('d', .3)], 'weighted')
        self.assertEqual([r['id'] for r in rows], ['b', 'a', 'c', 'd'])
        self.assertAlmostEqual(rows[0]['score'], .625)
        self.assertAlmostEqual(rows[1]['score'], .5)
        self.assertAlmostEqual(rows[2]['score'], .25)
        self.assertEqual(rows[3]['score'], 0)

    def test_affine_score_scale_changes_neither_method(self):
        a, b = [Hit('a', 8), Hit('b', 2)], [Hit('b', .8), Hit('c', -.4)]
        for method in ['rrf', 'weighted']:
            original = fuse(a, b, method)
            scaled = fuse([Hit(h.id, h.score * 5 + 42) for h in a], b, method)
            self.assertEqual([(r['id'], r['score']) for r in original], [(r['id'], r['score']) for r in scaled])

    def test_constant_and_empty_lists_have_defined_behavior(self):
        self.assertEqual(fuse([], [], 'rrf'), [])
        rows = fuse([Hit('z', 2), Hit('a', 2)], [], 'weighted')
        self.assertEqual([r['id'] for r in rows], ['a', 'z'])
        self.assertEqual([r['score'] for r in rows], [.5, .5])
        self.assertTrue(all(r['dense_contribution'] == 0 for r in rows))

    def test_fixed_limit_is_applied_after_scoring_union(self):
        rows = fuse([Hit('a', 2), Hit('b', 1)], [Hit('c', 2), Hit('b', 1)], 'rrf', limit=1)
        self.assertEqual([r['id'] for r in rows], ['b'])

    def test_rejects_duplicate_nonfinite_and_out_of_order_scores(self):
        for hits in [[Hit('a', 2), Hit('a', 1)], [Hit('a', math.nan)], [Hit('a', math.inf)], [Hit('a', 1), Hit('b', 2)]]:
            with self.assertRaises(ValueError):
                fuse(hits, [], 'rrf')

    def test_rejects_invalid_hyperparameters(self):
        for args in [{'method':'unknown'}, {'method':'rrf','rrf_k':0}, {'method':'weighted','dense_weight':1.2}, {'method':'rrf','limit':0}]:
            with self.assertRaises(ValueError):
                fuse([], [], **args)


class BM25QueryTests(unittest.TestCase):
    def test_pure_content_bm25_has_fixed_scope_and_no_vector_or_title_scoring(self):
        one = bm25_body('缴费 标准', ['k1', 'k2'], ['d1', 'd2'])
        two = bm25_body('退休 医保', ['k1', 'k2'], ['d1', 'd2'])
        self.assertEqual(one['query']['bool']['filter'], two['query']['bool']['filter'])
        self.assertEqual(one['query']['bool']['filter'], [{'terms':{'kb_id':['k1','k2']}}, {'terms':{'doc_id':['d1','d2']}}])
        self.assertEqual(one['query']['bool']['must'], [{'match':{'content_ltks':{'query':'缴费 标准','operator':'or','minimum_should_match':1}}}])
        self.assertEqual(one['query']['bool']['must_not'], [{'range':{'available_int':{'lt':1}}}])
        self.assertEqual(one['size'], 64)
        self.assertEqual(one['sort'], [{'_score':'desc'}, {'id':'asc'}])
        self.assertNotIn('knn', str(one))
        self.assertNotIn('title', str(one))

    def test_rejects_empty_query_or_scope(self):
        for args in [('', ['k'], ['d']), ('词', [], ['d']), ('词', ['k'], [])]:
            with self.assertRaises(ValueError):
                bm25_body(*args)


if __name__ == '__main__':
    unittest.main()
