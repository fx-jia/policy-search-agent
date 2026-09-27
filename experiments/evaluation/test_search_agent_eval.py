import unittest

from experiments.evaluation.search_agent_eval import runtime_questions, check_stage, bootstrap_source, bounded_worker_stop, prior_generation_tokens, remaining_ids, committed_tokens, verify_frozen_inputs, diagnostic_commitment
from experiments.search_agent.core import action_signature


class EvaluationTests(unittest.TestCase):
    def test_timeout_preserves_pre_generation_controller_error_and_fallback(self):
        from experiments.evaluation import search_agent_eval
        state = {}
        search_agent_eval._track_control_event(state, {"event": "controller_error", "error": "invalid_action"})
        search_agent_eval._track_control_event(state, {"event": "fallback", "mode": "native_initial"})
        self.assertEqual(state, {"controller_error": "invalid_action", "fallback_used": True})

    def test_direct_dev30_uses_remaining_full_budget_after_pilot_budget_spent(self):
        from experiments.evaluation import search_agent_eval
        budgets = search_agent_eval._stage_budgets(233339, direct_dev30=True)
        self.assertEqual(budgets, {"pilot": 0, "full": 216661})
        with self.assertRaisesRegex(ValueError, "study_budget_exhausted"):
            search_agent_eval._stage_budgets(450000, direct_dev30=True)
        with self.assertRaisesRegex(ValueError, "study_budget_exhausted"):
            search_agent_eval._stage_budgets(150000, direct_dev30=False)

    def test_diagnostic_usage_and_unknown_attempt_are_deducted(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "provider_diagnostic.json").write_text(json.dumps({"usage": {"total_tokens": 17}}))
            (path / "native_provider_diagnostic.json").write_text(json.dumps({"calls": [{"reserve_tokens": 4382}]}))
            self.assertEqual(diagnostic_commitment(path), 4399)

    def test_worker_restart_deducts_unknown_reservations_and_actual_usage(self):
        rows = [{"calls": [{"usage": {"total_tokens": 6459}}, {"reserve_tokens": 18969}]}]
        self.assertEqual(committed_tokens(rows), 25428)
        with self.assertRaises(ValueError):
            committed_tokens([{"calls": [{}]}])

    def test_resume_executes_frozen_modules_but_requires_unchanged_data(self):
        import hashlib
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "x.py").write_text("edited after freeze")
            (root / "data.json").write_text("original")
            sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
            config = {"module_sources": {"x": "frozen"}, "input_hashes": {"x.py": sha("frozen"), "data.json": sha("original")}}
            verify_frozen_inputs(config, root)
            (root / "data.json").write_text("changed")
            with self.assertRaises(ValueError):
                verify_frozen_inputs(config, root)

    def test_unknown_usage_does_not_requeue_failed_question_or_block_remaining(self):
        records = [{"question_id": "a", "status": "guard_stopped", "ledger": {"pending_calls": ["a:1"]},
                    "calls": [{"call_id": "a:1", "reserve_tokens": 18969}]}]
        self.assertEqual(remaining_ids(["a", "b"], ["a"], records), ["b"])
        with self.assertRaises(ValueError):
            remaining_ids(["a", "b"], ["a", "b"], records)

    def test_pilot_can_expand_after_one_isolated_transport_failure(self):
        rows = [{"question_id": str(i), "status": "ok", "ledger": {"pending_calls": []}} for i in range(6)]
        rows[1].update(status="guard_stopped", stop_reason="provider_usage_unknown", ledger={"pending_calls": ["1:1"]},
                       calls=[{"reserve_tokens": 18969}])
        check_stage("full", [str(i) for i in range(6)], rows)
        rows[2].update(status="guard_stopped", stop_reason="provider_usage_unknown")
        with self.assertRaises(ValueError):
            check_stage("full", [str(i) for i in range(6)], rows)

    def test_version_restart_carries_actual_and_unknown_committed_cost(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            rows = [{"calls": [{"usage": {"total_tokens": 6357}}]}]
            (path / "records.jsonl").write_text(json.dumps(rows[0]) + "\n")
            self.assertEqual(prior_generation_tokens([path]), 6357)
            rows[0]["calls"].append({"reserve_tokens": 100})
            (path / "records.jsonl").write_text(json.dumps(rows[0]) + "\n")
            self.assertEqual(prior_generation_tokens([path]), 6457)

    def test_effective_action_identity_ignores_need_index_and_scope_order(self):
        a = {"type": "search_policy", "query": "  期限  ", "need_index": 0, "document_ids": ["b", "a"]}
        b = {**a, "query": "期限", "need_index": 1, "document_ids": ["a", "b"]}
        self.assertEqual(action_signature(a), action_signature(b))

    def test_shutdown_timeout_terminates_isolated_process_before_return(self):
        import subprocess
        class Worker:
            stdin = None
            killed = False
            def wait(self, timeout):
                if not self.killed:
                    raise subprocess.TimeoutExpired("worker", timeout)
            def kill(self):
                self.killed = True
        calls = []
        worker = Worker()
        bounded_worker_stop(worker, lambda: calls.append("stop_remote"))
        self.assertEqual(calls, ["stop_remote"])
        self.assertTrue(worker.killed)

    def test_only_question_and_id_cross_runtime_boundary(self):
        rows = [{"question_id": "x", "question": "问题", "split": "dev", "reference_answer": "hidden", "difficulty": "Hard"}]
        self.assertEqual(runtime_questions(rows, ["x"]), [{"question_id": "x", "question": "问题"}])
        rows[0]["split"] = "test"
        with self.assertRaises(ValueError):
            runtime_questions(rows, ["x"])

    def test_expansion_requires_all_six_technically_valid(self):
        rows = [{"question_id": str(i), "status": "ok", "ledger": {"pending_calls": []}} for i in range(6)]
        check_stage("full", [str(i) for i in range(6)], rows)
        rows[0]["status"] = "invalid_decision"
        with self.assertRaises(ValueError):
            check_stage("full", [str(i) for i in range(6)], rows)

    def test_bootstrap_compiles_as_a_standalone_python_program(self):
        compile(bootstrap_source(), "bootstrap", "exec")


if __name__ == "__main__":
    unittest.main()
