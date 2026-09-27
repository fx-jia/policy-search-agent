"""Run in the existing runtime with httpx/aiohttp; no network calls."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.evaluation.native_agent_guard import CallLedger
from experiments.search_agent.transport import monitored_send


class TransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ledger = CallLedger(Path(self.temp.name) / "ledger", run_limit=100000, question_limit=60000, call_limit=4)
        self.ledger.start("q")
        body = {"model": "model", "messages": [{"role": "user", "content": "question"}], "enable_thinking": False, "max_tokens": 1200}
        self.request = SimpleNamespace(method="POST", url=SimpleNamespace(host="provider.test", path="/chat/completions"), content=json.dumps(body).encode())
        self.config = {"model": "model", "provider_hosts": ["provider.test"]}

    async def test_two_unsent_reconnects_are_counted_within_physical_call_limit(self):
        import aiohttp
        import httpx
        sent = []
        async def send(*args):
            sent.append(1)
            if len(sent) <= 2:
                cause = aiohttp.ClientConnectorError(None, ConnectionResetError())
                raise httpx.ConnectError("connect") from cause
            return SimpleNamespace(is_stream_consumed=True, headers={"content-type": "application/json"}, content=b'{"usage":{"total_tokens":20}}')
        wrapped = monitored_send(send, self.ledger, {**self.config, "connect_retries": 2}, lambda _: None, lambda _: None)
        await wrapped(None, self.request)
        self.assertEqual(len(sent), 3)
        self.assertEqual(self.ledger.summary()["provider_not_sent_calls"], 2)
        self.assertEqual(self.ledger.summary()["provider_total_tokens"], 20)
        self.assertFalse(self.ledger.summary()["pending_calls"])

    async def test_reconnects_stop_after_two_and_preserve_call_cap(self):
        import aiohttp
        import httpx
        sent = []
        async def send(*args):
            sent.append(1)
            raise httpx.ConnectError("connect") from aiohttp.ClientConnectorError(None, ConnectionResetError())
        wrapped = monitored_send(send, self.ledger, {**self.config, "connect_retries": 2}, lambda _: None, lambda _: None)
        with self.assertRaises(httpx.ConnectError):
            await wrapped(None, self.request)
        self.assertEqual(len(sent), 3)
        self.assertEqual(self.ledger.summary()["provider_not_sent_calls"], 3)
        self.assertFalse(self.ledger.summary()["pending_calls"])

    async def test_no_retry_and_unknown_usage_keeps_reservation(self):
        sent = []
        events = []
        async def send(*args):
            sent.append(1)
            raise OSError("failed")
        wrapped = monitored_send(send, self.ledger, self.config, lambda _: None, events.append)
        with self.assertRaises(OSError):
            await wrapped(None, self.request)
        self.assertEqual(len(sent), 1)
        self.assertEqual(len(self.ledger.summary()["pending_calls"]), 1)
        failure = next(e for e in events if e["event"] == "provider_send_failure")
        self.assertEqual(failure["exception_type"], "OSError")
        self.assertNotIn("failed", str(failure))
        with self.assertRaises(ValueError):
            self.ledger.start("next")

    async def test_http_error_keeps_status_and_code_without_logging_body(self):
        events = []
        async def send(*args):
            return SimpleNamespace(is_stream_consumed=True, status_code=429,
                                   headers={"content-type": "application/json"},
                                   content=json.dumps({"error": {"code": "rate_limit", "message": "private text"}}).encode())
        wrapped = monitored_send(send, self.ledger, self.config, lambda _: None, events.append)
        await wrapped(None, self.request)
        failure = next(e for e in events if e["event"] == "provider_usage_failure")
        self.assertEqual(failure["http_status"], 429)
        self.assertEqual(failure["error_code"], "rate_limit")
        self.assertNotIn("private text", str(events))
        self.assertTrue(self.ledger.summary()["pending_calls"])

    async def test_actual_usage_replaces_reservation(self):
        async def send(*args):
            return SimpleNamespace(is_stream_consumed=True, headers={"content-type": "application/json"},
                                   content=json.dumps({"usage": {"total_tokens": 20}}).encode())
        wrapped = monitored_send(send, self.ledger, self.config, lambda _: None, lambda _: None)
        await wrapped(None, self.request)
        self.assertEqual(self.ledger.summary()["provider_total_tokens"], 20)
        self.assertFalse(self.ledger.summary()["pending_calls"])

    async def test_generation_fifth_call_blocked_before_send(self):
        sent = []
        async def send(*args):
            sent.append(1)
            return SimpleNamespace(is_stream_consumed=True, headers={"content-type": "application/json"},
                                   content=json.dumps({"usage": {"total_tokens": 20}}).encode())
        wrapped = monitored_send(send, self.ledger, self.config, lambda _: None, lambda _: None)
        for _ in range(4):
            await wrapped(None, self.request)
        with self.assertRaises(ValueError):
            await wrapped(None, self.request)
        self.assertEqual(len(sent), 4)


if __name__ == "__main__":
    unittest.main()
