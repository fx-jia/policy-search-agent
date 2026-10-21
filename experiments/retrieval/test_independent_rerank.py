import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.retrieval.independent_rerank import eligible_source, sanitize_outbound, reorder_slots, validate_scores
from experiments.retrieval.rerank_worker import make_payload, parse_result
from experiments.retrieval import rerank_worker


class RerankTests(unittest.TestCase):
    def test_runner_rejects_unapproved_job_before_writes_or_docker(self):
        from experiments.evaluation import independent_rerank_experiment as runner
        config = {"endpoint": rerank_worker.ENDPOINT, "model_id": "1" * 32}
        for approved in (None, "2" * 32):
            with self.subTest(approved=approved), tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
                if approved is None:
                    os.environ.pop("POLICY_APPROVED_RERANK_MODEL_ID", None)
                else:
                    os.environ["POLICY_APPROVED_RERANK_MODEL_ID"] = approved
                output = Path(directory) / "run"
                with patch.object(runner, "verified", return_value=(config, [], {})), patch.object(runner.subprocess, "Popen", side_effect=AssertionError("Docker forbidden")):
                    with self.assertRaises(ValueError):
                        runner.run(output, True)
                self.assertFalse(output.exists())

    def test_runner_forwards_approved_environment_separately_from_job_input(self):
        from experiments.evaluation import independent_rerank_experiment as runner
        config = {"endpoint": rerank_worker.ENDPOINT, "model_id": "1" * 32}
        commands = []

        def stop_before_docker(command, **kwargs):
            commands.append(command)
            raise InterruptedError("offline process boundary")

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"POLICY_APPROVED_RERANK_MODEL_ID": "1" * 32}):
            root = Path(directory)
            (root / "pre_call_freeze.json").write_text("{}")
            with patch.object(runner, "ROOT", root), patch.object(runner, "verified", return_value=(config, [], {})), patch.object(runner.subprocess, "Popen", side_effect=stop_before_docker):
                with self.assertRaises(InterruptedError):
                    runner.run(root / "run", True)
            self.assertEqual(commands[0][:5], ["docker", "exec", "-e", "POLICY_APPROVED_RERANK_MODEL_ID", "-i"])

    def test_worker_requires_separate_model_approval(self):
        config = {"endpoint": rerank_worker.ENDPOINT, "model_id": "1" * 32, "max_calls": 60}
        for value in (None, "", "replace-with-private-id", "2" * 32):
            with self.subTest(value=value), patch.dict(os.environ):
                if value is None:
                    os.environ.pop("POLICY_APPROVED_RERANK_MODEL_ID", None)
                else:
                    os.environ["POLICY_APPROVED_RERANK_MODEL_ID"] = value
                with self.assertRaises(ValueError):
                    rerank_worker.validate_config(config)
        with patch.dict(os.environ, {"POLICY_APPROVED_RERANK_MODEL_ID": "1" * 32}):
            rerank_worker.validate_config(config)

    def test_approval_preserves_endpoint_and_call_budget_guards(self):
        config = {"endpoint": rerank_worker.ENDPOINT, "model_id": "1" * 32, "max_calls": 60}
        with patch.dict(os.environ, {"POLICY_APPROVED_RERANK_MODEL_ID": "1" * 32}):
            for changed in ({"endpoint": "https://provider.invalid"}, {"max_calls": 0}, {"max_calls": 61}, {"max_calls": True}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    rerank_worker.validate_config(config | changed)
            with self.assertRaises(ValueError):
                rerank_worker.validate_config(config, "https://provider.invalid/api/v1")
            rerank_worker.validate_config(config, "https://dashscope.aliyuncs.com/api/v1/")

    def test_only_sent_slots_move_and_tail_is_preserved(self):
        ids=list('abcdef')
        self.assertEqual(reorder_slots(ids,[0,2,3],[.1,.9,.4]),list('cbd aef'.replace(' ','')))
        self.assertEqual(ids,list('abcdef'))

    def test_ties_keep_dense_order_and_empty_is_identity(self):
        self.assertEqual(reorder_slots(list('abc'),[0,2],[.3,.3]),list('abc'))
        self.assertEqual(reorder_slots(list('abc'),[],[]),list('abc'))

    def test_rejects_invalid_or_out_of_window_slots(self):
        for slots,scores in [([0,0],[.1,.2]),([1,0],[.1,.2]),([32],[.1]),([5],[.1]),([0],[]),([0],[math.nan])]:
            with self.assertRaises(ValueError):reorder_slots(list('abcde'),slots,scores)

    def test_scores_require_all_indices_once_not_provider_order(self):
        self.assertEqual(validate_scores([{'index':1,'relevance_score':.8},{'index':0,'relevance_score':.2}],2),[.2,.8])
        for rows,count in [([{'index':0,'relevance_score':.3}],2), ([{'index':0,'relevance_score':.3}]*2,2), ([{'index':True,'relevance_score':.3}],1), ([{'index':0,'relevance_score':math.inf}],1), ([{'index':0,'relevance_score':1.1}],1)]:
            with self.assertRaises(ValueError):validate_scores(rows,count)

    def test_document_scope_uses_privacy_metadata_not_gold(self):
        policy={'是否公开发布':'是','文件名称':'关于调整医疗保险待遇的通知'}
        self.assertTrue(eligible_source('a',policy,set()))
        self.assertFalse(eligible_source('a',policy,{'a'}))
        for title in ['01234567-abcd-4321-0000-0123456789ab','招聘人员公示','对人大建议的答复','劳动关系解除送达公告']:
            self.assertFalse(eligible_source('a',{**policy,'文件名称':title},set()))
        self.assertFalse(eligible_source('a',{**policy,'是否公开发布':'否'},set()))

    def test_outbound_strips_contacts_ids_links_and_caps_preview(self):
        text='居民缴费380元。\n联系人：测试姓名，电话13800138000\n证件号码110101199001010011\n监督电话：0516-12345678\n网址https://example.com/policy\n'+'规则'*1000
        result=sanitize_outbound(text)
        self.assertNotIn('测试姓名',result)
        self.assertNotIn('13800138000',result)
        self.assertNotIn('110101199001010011',result)
        self.assertNotIn('0516-12345678',result)
        self.assertNotIn('https://',result)
        self.assertTrue(result.startswith('居民缴费380元。'))
        self.assertLessEqual(len(result),1800)

    def test_payload_contains_only_query_and_sanitized_text_not_labels(self):
        p=make_payload('缴费多少？',['380元','420元'])
        self.assertEqual(p,{'model':'gte-rerank-v2','input':{'query':'缴费多少？','documents':['380元','420元']},'parameters':{'top_n':2,'return_documents':False}})
        for query,docs in [('问题',[]),('问题',['a']*33),('问题',['a'*1801]),('q'*201,['a'])]:
            with self.assertRaises(ValueError):make_payload(query,docs)

    def test_incomplete_success_and_raw_error_messages_are_not_scored_or_logged(self):
        r=parse_result({'output':{'results':[{'index':0,'relevance_score':.8}]},'usage':{'total_tokens':123},'request_id':'r'},200,2)
        self.assertEqual(r['status'],'error')
        r=parse_result({'code':'InvalidApiKey','message':'secret-token-and-document'},401,1)
        self.assertEqual(r['error_category'],'auth')
        self.assertNotIn('secret-token',str(r))


if __name__=='__main__':
    unittest.main()
