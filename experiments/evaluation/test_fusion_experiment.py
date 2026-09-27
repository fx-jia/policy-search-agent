import unittest

from experiments.evaluation.context_completion import Candidate
from experiments.evaluation.fusion_experiment import evidence_prefix_rank, project_saved, validate_candidates


class FusionExperimentTests(unittest.TestCase):
    def test_prefix_requires_all_evidence_including_contiguous_chunks(self):
        chunks = [Candidate('c1','d','s',1,0,'缴费',10), Candidate('noise','x','x',2,0,'其他',10), Candidate('c2','d','s',3,1,'380元',10)]
        refs = [{'source_document_id':'s','text':'缴费380元'}]
        self.assertEqual(evidence_prefix_rank(refs, chunks), 3)
        self.assertIsNone(evidence_prefix_rank(refs, chunks[:2]))

    def test_projection_strips_labels_and_checks_identity(self):
        cached = {'c': {'document_id':'d','source_document_id':'s'}}
        rows = [{'id':'c','document_id':'d','source_document_id':'s','rank':1,'similarity':.8,'gold':'answer'}]
        self.assertEqual(project_saved(rows,cached), [{'id':'c','score':.8}])
        rows[0]['document_id']='wrong'
        with self.assertRaises(ValueError):
            project_saved(rows,cached)

    def test_rejects_missing_query_or_out_of_snapshot_candidate(self):
        valid = [{'question_id':'q','status':'ok','bm25':[{'id':'c','score':2}], 'dense':[{'id':'c','score':.8}]}]
        validate_candidates(valid, ['q'], {'c': {}})
        with self.assertRaises(ValueError):
            validate_candidates(valid, ['other'], {'c': {}})
        with self.assertRaises(ValueError):
            validate_candidates(valid, ['q'], {})

    def test_failed_bm25_cannot_silently_carry_successful_candidates(self):
        with self.assertRaises(ValueError):
            validate_candidates([{'question_id':'q','status':'error','bm25':[{'id':'c','score':2}], 'dense':[]}], ['q'], {'c': {}})


if __name__ == '__main__':
    unittest.main()
