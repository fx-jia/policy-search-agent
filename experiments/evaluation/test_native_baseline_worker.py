import copy
import importlib.util
from pathlib import Path
import unittest


class NativeRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name("native_baseline_worker.py")
        cls.module = None
        if path.exists():
            spec = importlib.util.spec_from_file_location("native_baseline_worker", path)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def test_only_question_and_full_configured_scope_enter_native_request(self):
        self.assertIsNotNone(self.module, "native baseline worker is missing")
        result = self.module.make_payload(
            {"chat_id": "chat", "scope_document_ids": ["d1", "d2"]},
            {"question_id": "q1", "question": "缴费金额？", "reference_answer": "secret gold", "relevant_document_ids": ["d1"]},
        )
        self.assertEqual(result["doc_ids"], "d1,d2")
        self.assertEqual(result["messages"], [{"role": "user", "content": "缴费金额？"}])
        self.assertNotIn("secret gold", str(result))
        self.assertFalse(result["store_history_messages"])

    def test_changed_prompt_or_generation_setting_is_blocked_before_authorization(self):
        self.assertIsNotNone(self.module, "native baseline worker is missing")
        expected = {"model": "model", "system": "safe", "history": [{"role": "user", "content": "query"}], "gen_conf": {"max_tokens": 1200}}
        for key, value in (("system", "raw private"), ("gen_conf", {"max_tokens": 12000}), ("model", "other")):
            actual = copy.deepcopy(expected)
            actual[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.module.validate_outbound(expected, actual)
        self.module.validate_outbound(expected, copy.deepcopy(expected))

    def test_repeated_or_unknown_question_cannot_authorize_an_extra_model_call(self):
        self.assertIsNotNone(self.module, "native baseline worker is missing")
        guard = self.module.CallGuard(["q1", "q2"])
        guard.claim("q1")
        with self.assertRaises(ValueError):
            guard.claim("q1")
        with self.assertRaises(ValueError):
            guard.claim("unplanned")
        guard.claim("q2")
        self.assertEqual(guard.calls, 2)

    def test_budget_reserves_context_before_call_and_rejects_unknown_usage(self):
        self.module.check_budget(100, 8192, 9000)
        with self.assertRaises(ValueError):
            self.module.check_budget(809, 8192, 9000)
        for value in (None, {}, {"total_tokens": 0}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module.usage_tokens(value)
        self.assertEqual(self.module.usage_tokens({"total_tokens": 130}), 130)

    def test_response_validation_rejects_empty_or_out_of_scope_references(self):
        state = {"outbound_request": {"system": "safe"}, "retrieved_chunks": [{"doc_id": "d1"}]}
        body = {"code": 0, "data": {"answer": "answer", "reference": {"chunks": [{"document_id": "d1"}]}}}
        self.assertTrue(self.module.valid_response(200, body, state, {"d1"}))
        body["data"]["reference"]["chunks"][0]["document_id"] = "other"
        self.assertFalse(self.module.valid_response(200, body, state, {"d1"}))
        body["data"]["answer"] = ""
        self.assertFalse(self.module.valid_response(200, body, state, {"d1"}))


if __name__ == "__main__":
    unittest.main()
