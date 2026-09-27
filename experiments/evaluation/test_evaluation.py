"""Self-contained tests; no API keys, network calls or production services."""

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from experiments.analysis.profile_dataset import fingerprint, privacy_flags
from experiments.datasets.build_benchmark import fingerprint_text, validate_sample
from experiments.evaluation.metrics import aggregate, retrieval_metrics
from experiments.retrieval.local import LocalBM25
from experiments.retrieval.ragflow import RAGFlowClient, RetrievalError, build_payload, parse_response


def chunks(*documents):
    return [{"source_document_id": doc} for doc in documents]


class MetricsTest(unittest.TestCase):
    def test_multi_positive_recall_differs_from_hit(self):
        result = retrieval_metrics(chunks("a", "x"), ["a", "b"])
        self.assertEqual(result["recall@5"], 0.5)
        self.assertEqual(result["hit_rate@5"], 1)

    def test_document_rank_and_chunk_context_are_different(self):
        result = retrieval_metrics(chunks("x", "x", "x", "x", "x", "a"), ["a"])
        self.assertEqual(result["mrr@10"], 0.5)
        self.assertEqual(result["recall@5"], 1)
        self.assertEqual(result["context_document_recall@5_chunks"], 0)

    def test_unanswerable_is_not_a_zero_recall_sample(self):
        self.assertIsNone(retrieval_metrics(chunks("x"), []))

    def test_empty_answerable_results_are_zero(self):
        self.assertTrue(all(v == 0 for v in retrieval_metrics([], ["a"]).values()))

    def test_mrr_cutoff(self):
        self.assertEqual(retrieval_metrics(chunks(*map(str, range(11))), ["10"])["mrr@10"], 0)

    def test_errors_stay_in_accuracy_denominator(self):
        summary = aggregate(
            [
                {"status": "ok", "metrics": retrieval_metrics(chunks("a"), ["a"]), "latency_ms": 4},
                {"status": "error", "metrics": retrieval_metrics([], ["a"]), "latency_ms": 100},
            ]
        )
        self.assertEqual(summary["metrics"]["recall@1"], 0.5)
        self.assertEqual(summary["latency_ms"]["mean"], 4)
        self.assertEqual(summary["errors"], 1)


class APITest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("experiments/configs/ragflow_hybrid.json").read_text())
        self.config["dataset_ids"] = ["kb"]
        self.response = {"code": 0, "data": {"chunks": [{"id": "chunk", "document_id": "rf-doc", "content": "do not persist", "similarity": 0.7}]}}

    def test_runtime_response_document_mapping_and_privacy(self):
        parsed = parse_response(self.response, {"rf-doc": "source"})
        self.assertEqual(parsed[0]["source_document_id"], "source")
        self.assertNotIn("content", parsed[0])

    def test_application_error_with_http_200(self):
        with self.assertRaises(RetrievalError):
            parse_response({"code": 102, "message": "may contain a credential"}, {})

    def test_quota_error_is_recognized_without_retaining_message(self):
        with self.assertRaisesRegex(RetrievalError, "^quota_error$"):
            parse_response({"code": 500, "message": "AllocationQuota.FreeTierOnly: private details"}, {})

    def test_nonobject_response_is_a_controlled_error(self):
        with self.assertRaises(RetrievalError):
            parse_response([], {})

    def test_corpus_filter_is_identical_for_different_questions(self):
        self.config["scope_document_ids"] = ["source-replica-1", "source-replica-2"]
        one = build_payload("徐州居民缴费", self.config)
        two = build_payload("苏州生育津贴", self.config)
        self.assertEqual(one["document_ids"], two["document_ids"])
        self.assertEqual(one["document_ids"], self.config["scope_document_ids"])

    def test_unknown_document_is_rejected(self):
        with self.assertRaisesRegex(RetrievalError, "unmapped"):
            parse_response(self.response, {})

    def test_missing_document_id_is_rejected(self):
        del self.response["data"]["chunks"][0]["document_id"]
        with self.assertRaises(RetrievalError):
            parse_response(self.response, {})

    def test_reject_nonfinite_score(self):
        self.response["data"]["chunks"][0]["similarity"] = float("nan")
        with self.assertRaises(RetrievalError):
            parse_response(self.response, {"rf-doc": "source"})

    def test_malformed_id_does_not_crash_as_unhashable(self):
        self.response["data"]["chunks"][0]["id"] = []
        with self.assertRaises(RetrievalError):
            parse_response(self.response, {"rf-doc": "source"})

    def test_nonfinite_weight_rejected(self):
        self.config["retrieval_config"]["vector_similarity_weight"] = float("nan")
        with self.assertRaises(ValueError):
            build_payload("问题", self.config)

    def test_no_oracle_document_filter_or_llm_expansion(self):
        payload = build_payload("问题", self.config)
        self.assertNotIn("document_ids", payload)
        self.assertFalse(payload["keyword"])
        self.assertEqual(payload["rerank_id"], "")
        self.assertEqual(payload["page_size"], payload["rerank_candidates_count"])

    def test_dense_cannot_be_mislabelled(self):
        self.config["method"] = "dense"
        with self.assertRaises(ValueError):
            build_payload("问题", self.config)

    def test_rerank_requires_explicit_model(self):
        self.config["method"] = "hybrid_rerank"
        with self.assertRaises(ValueError):
            build_payload("问题", self.config)

    @patch.dict("os.environ", {"RAGFLOW_API_KEY": "test-only-not-a-real-key"})
    def test_http_transport_contract_without_network(self):
        client = RAGFlowClient(self.config, {"rf-doc": "source"})
        with patch.object(client.opener, "open", return_value=io.BytesIO(json.dumps(self.response).encode())) as op:
            result = client.retrieve("医保问题", 64)
        request = op.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:9380/api/v1/retrieval")
        self.assertEqual(json.loads(request.data)["question"], "医保问题")
        self.assertEqual(result[0]["rank"], 1)

    @patch.dict("os.environ", {"RAGFLOW_API_KEY": "test-only-not-a-real-key"})
    def test_transport_error_redacts_response_body(self):
        client = RAGFlowClient(self.config, {})
        error = HTTPError(client.url, 401, "private message", {}, io.BytesIO(b"private body"))
        with patch.object(client.opener, "open", side_effect=error):
            with self.assertRaisesRegex(RetrievalError, "^http_401$"):
                client.retrieve("问题", 64)


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.records = {"a": {"政策正文": "原文证据"}}
        self.sample = {
            "question": "问题",
            "dataset_version": "v",
            "answerable": True,
            "relevant_document_ids": ["a"],
            "evidence": [{"source_document_id": "a", "start": 0, "end": 4, "text": "原文证据", "body_sha256": fingerprint_text("原文证据")}],
        }

    def test_valid_evidence(self):
        validate_sample(self.sample, self.records, "v")

    def test_changed_evidence_rejected(self):
        sample = copy.deepcopy(self.sample)
        sample["evidence"][0]["text"] = "编造证据"
        with self.assertRaises(ValueError):
            validate_sample(sample, self.records, "v")

    def test_changed_source_version_rejected(self):
        with self.assertRaises(ValueError):
            validate_sample(self.sample, self.records, "other")

    def test_missing_relevance_rejected(self):
        self.sample["relevant_document_ids"] = []
        with self.assertRaises(ValueError):
            validate_sample(self.sample, self.records, "v")

    def test_privacy_screen_is_not_policy_term_blocklist(self):
        self.assertEqual(privacy_flags("城乡居民医保报销比例65%"), [])
        self.assertIn("personal_record_context", privacy_flags("患者姓名：测试"))

    def test_local_corpus_not_question_answers(self):
        records = [(1, {"系统ID": 1, "政策正文": "居民医保报销标准", "文件名称": "医保"}), (2, {"系统ID": 2, "政策正文": "天气晴朗", "文件名称": "天气"})]
        client = LocalBM25(records)
        self.assertEqual(client.retrieve("居民医保", 10)[0]["source_document_id"], "1")

    def cli_fixture(self, directory, allow_unreviewed):
        from argparse import Namespace

        path = Path(directory)
        source = path / "source.jsonl"
        source.write_text(json.dumps({"系统ID": "a", "文件名称": "证据", "政策正文": "原文证据"}))
        config = json.loads(Path("experiments/configs/local_smoke.json").read_text())
        config["source_sha256"] = fingerprint(source)
        (path / "config.json").write_text(json.dumps(config))
        sample = dict(self.sample, dataset_version=fingerprint(source), question_id="test-1", question_type="test", region=None, review_status="pending_human_review")
        (path / "benchmark.jsonl").write_text(json.dumps(sample))
        return Namespace(
            config=str(path / "config.json"),
            source=str(source),
            benchmark=str(path / "benchmark.jsonl"),
            split_manifest=None,
            limit=1,
            allow_unreviewed=allow_unreviewed,
            experiment_id="test",
            output_root=directory,
        )

    def test_cli_requires_review_before_any_execution(self):
        from experiments.evaluation.run import run

        with tempfile.TemporaryDirectory() as directory:
            args = self.cli_fixture(directory, False)
            with self.assertRaisesRegex(ValueError, "Unreviewed"):
                run(args)

    def test_cli_saves_real_local_results(self):
        from experiments.evaluation.run import run

        with tempfile.TemporaryDirectory() as directory:
            args = self.cli_fixture(directory, True)
            with patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(run(args), 0)
            summary = json.loads((Path(directory) / "test/summary.json").read_text())
            self.assertEqual(summary["successful"], 1)
            self.assertEqual(summary["generation"]["status"], "Not Run")

    def test_cli_api_needs_explicit_authorization(self):
        from experiments.evaluation.run import run

        with tempfile.TemporaryDirectory() as directory:
            args = self.cli_fixture(directory, True)
            config = json.loads(Path(args.config).read_text())
            config["method"] = "hybrid"
            Path(args.config).write_text(json.dumps(config))
            args.allow_model_api = False
            with self.assertRaisesRegex(ValueError, "disabled"):
                run(args)


if __name__ == "__main__":
    unittest.main()
