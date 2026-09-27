import copy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


BASE = {
    "status": "prepared", "question_id": "dev-example", "question": "甲和乙各是多少？",
    "mode": "short_ids", "selected_ids": ["c1"], "selection_tokens_cl100k": 20,
    "chunks": [{"id": "c1", "source_document_id": "d1", "content": "甲为1，乙为2。"}],
    "citation_map": {"E1": {"original_chunk_id": "c1", "source_document_id": "d1"}},
    "messages": [{"role": "system", "content": "仅依据证据输出JSON"}, {"role": "user", "content": "甲和乙各是多少？\n[E1]甲为1，乙为2。"}],
    "prompt_sha256": "base-sha", "request_id": "dev-example-draft", "policy": "draft",
}
ANSWER = {"answer": "甲为1。", "claims": [{"statement": "甲为1", "evidence": [{"chunk_id": "E1", "quote": "甲为1"}]}], "insufficient_evidence": False}


class CoverageRevisionTests(unittest.TestCase):
    def module(self):
        name = "experiments.evaluation.coverage_revision"
        self.assertIsNotNone(importlib.util.find_spec(name), "coverage revision experiment is missing")
        return importlib.import_module(name)

    def test_both_review_arms_receive_same_draft_and_evidence_without_gold(self):
        module = self.module()
        base = copy.deepcopy(BASE)
        base["required_points"] = ["GOLD_MUST_NOT_LEAK"]
        draft = {"parsed": ANSWER, "status": "ok"}
        requests = [module.revision_request(base, draft, p) for p in ["generic", "coverage"]]
        for req in requests:
            self.assertEqual(req["messages"][:2], BASE["messages"])
            self.assertEqual(json.loads(req["messages"][2]["content"]), ANSWER)
            self.assertEqual(req["chunks"], BASE["chunks"])
            self.assertNotIn("GOLD_MUST_NOT_LEAK", json.dumps(req["messages"]))
            self.assertEqual(req["parent_draft_sha256"], requests[0]["parent_draft_sha256"])
        self.assertNotEqual(requests[0]["prompt_sha256"], requests[1]["prompt_sha256"])
        self.assertEqual(base["messages"], BASE["messages"])

    def test_coverage_trace_rejects_added_question_scope_and_unknown_evidence(self):
        module = self.module()
        good = [{"span": "甲", "status": "covered", "evidence_ids": ["E1"]}, {"span": "乙", "status": "repaired", "evidence_ids": ["E1"]}]
        self.assertTrue(module.valid_coverage(good, BASE))
        for trace in [[{"span": "丙", "status": "covered", "evidence_ids": ["E1"]}], [{"span": "乙", "status": "repaired", "evidence_ids": ["E9"]}], [], [{"span": "乙", "status": "covered", "evidence_ids": []}]]:
            self.assertFalse(module.valid_coverage(trace, BASE))

    def test_call_budget_and_shared_draft_are_preserved_in_saved_outputs(self):
        module = self.module()
        bases = [{**copy.deepcopy(BASE), "question_id": f"dev-{i}"} for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            module.execute(bases, {"max_calls": 15}, out, FakeClient)
            rows = [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]
            reqs = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 30)
            self.assertEqual(json.loads((out / "execution_summary.json").read_text())["actual_calls"], 30)
            self.assertEqual([r["policy"] for r in rows[:6]], ["draft", "generic", "coverage", "draft", "coverage", "generic"])
            for i in range(10):
                pair = [r for r in reqs if r["question_id"] == f"dev-{i}" and r["policy"] != "draft"]
                self.assertEqual(pair[0]["messages"][2], pair[1]["messages"][2])
                self.assertEqual(pair[0]["parent_draft_sha256"], pair[1]["parent_draft_sha256"])

    def test_failed_draft_skips_dependent_calls_but_keeps_planned_denominator(self):
        module = self.module()

        class InvalidClient(FakeClient):
            def generate(self, request):
                self.calls += 1
                return {"status": "ok", "text": "{}", "usage": {"total_tokens": 2}}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            module.execute([BASE], {"max_calls": 15}, out, InvalidClient)
            rows = [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]
            self.assertEqual([r["status"] for r in rows], ["invalid_output_schema", "not_run_invalid_draft", "not_run_invalid_draft"])
            self.assertEqual(json.loads((out / "execution_summary.json").read_text())["actual_calls"], 1)

    def test_quota_error_stops_without_retry_or_losing_records(self):
        module = self.module()

        class QuotaClient(FakeClient):
            def generate(self, request):
                self.calls += 1
                return {"status": "error", "error_category": "quota"}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            module.execute([BASE, {**BASE, "question_id": "dev-2"}], {"max_calls": 15}, out, QuotaClient)
            summary = json.loads((out / "execution_summary.json").read_text())
            rows = [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]
            self.assertEqual(summary["actual_calls"], 1)
            self.assertEqual(len(rows), 6)
            self.assertEqual([r["status"] for r in rows], ["error"] + ["not_run_after_error"] * 5)
            self.assertIsNone(summary["reported_usage"]["total_tokens"])


class FakeClient:
    def __init__(self, config):
        self.calls = 0
        self.max_calls = config["max_calls"]

    def generate(self, request):
        assert self.calls < self.max_calls
        self.calls += 1
        return {"status": "ok", "text": json.dumps(ANSWER, ensure_ascii=False), "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}, "latency_ms": 1}

    def close(self):
        pass


if __name__ == "__main__":
    unittest.main()
