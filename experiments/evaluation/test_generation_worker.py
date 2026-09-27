import unittest

from experiments.evaluation.generation_worker import completion_payload, parse_response, validate_provider


class GenerationWorkerTests(unittest.TestCase):
    def test_user_selected_model_uses_provider_connection_without_model_aliasing(self):
        config = {"model_name": "qwen3.7-flash", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "max_calls": 20, "parameters": {"max_tokens": 1200}}
        provider = {"api_key": "test-secret", "extra": '{"base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1/"}'}
        validate_provider(config, provider)
        self.assertEqual(completion_payload(config, [])['model'], 'qwen3.7-flash')
        for changed in [{"model_name": "qwen3.8-max"}, {"max_calls": 21}, {"max_calls": 0}, {"parameters": {"max_tokens": 1201}}, {"base_url": "https://different.example/v1"}]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_provider({**config, **changed}, provider)
        with self.assertRaises(ValueError):
            validate_provider(config, None)

    def test_native_http_payload_preserves_model_and_generation_controls(self):
        config = {"model_name": "fixed-model", "parameters": {"temperature": 0, "max_tokens": 1200, "response_format": {"type": "json_object"}, "extra_body": {"enable_thinking": False}}}
        messages = [{"role": "user", "content": "问题"}]
        self.assertEqual(completion_payload(config, messages), {"model": "fixed-model", "messages": messages, "stream": False, "temperature": 0, "max_tokens": 1200, "response_format": {"type": "json_object"}, "enable_thinking": False})

    def test_provider_permission_code_is_kept_without_raw_error_message(self):
        result = parse_response({"error": {"code": "AccessDenied.Unpurchased", "type": "PermissionDenied", "message": "potentially sensitive detail"}}, 403)
        self.assertEqual(result["error_code"], "AccessDenied.Unpurchased")
        self.assertEqual(result["error_category"], "auth")
        self.assertNotIn("potentially sensitive detail", str(result))

    def test_truncation_is_not_reported_as_success(self):
        result = parse_response({"model": "fixed-model", "id": "request", "created": 1, "usage": {"total_tokens": 12}, "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}, 200)
        self.assertEqual(result["status"], "output_truncated")
        self.assertEqual(result["usage"]["total_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
