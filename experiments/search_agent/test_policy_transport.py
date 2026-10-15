"""Physical HTTP attempts, including missing usage, are bounded and recorded."""

from pathlib import Path
import gzip
import json
import tempfile
import unittest

import httpx

from experiments.evaluation.native_agent_guard import CallLedger


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_compressed_provider_response_decodes_once_and_keeps_actual_usage(self):
        from experiments.search_agent.policy_transport import AuditedCalls, AsyncAuditTransport, SyncAuditTransport

        payload = {"usage": {"total_tokens": 19}, "data": [{"embedding": [0.1, 0.2]}]}
        compressed = gzip.compress(json.dumps(payload).encode())
        with tempfile.TemporaryDirectory() as directory:
            ledger = CallLedger(Path(directory) / "calls.jsonl", run_limit=60000, question_limit=60000, call_limit=7)
            ledger.start("q")
            audit = AuditedCalls(ledger)

            def response(request):
                return httpx.Response(200, content=compressed, headers={"Content-Type": "application/json", "Content-Encoding": "gzip"})

            async with httpx.AsyncClient(transport=AsyncAuditTransport(httpx.MockTransport(response), audit, "generation", "https://model.test/v1", "m")) as client:
                result = await client.post("https://model.test/v1/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100})
                self.assertEqual(result.json(), payload)
                self.assertNotIn("content-encoding", result.headers)
            with httpx.Client(transport=SyncAuditTransport(httpx.MockTransport(response), audit, "embedding", "https://model.test/v1", "m")) as client:
                result = client.post("https://model.test/v1/embeddings", json={"model": "m", "input": ["question"]})
                self.assertEqual(result.json(), payload)
            self.assertEqual(ledger.summary()["provider_calls"], 2)
            self.assertEqual(ledger.summary()["provider_total_tokens"], 38)

    async def test_missing_usage_stops_following_calls(self):
        from experiments.search_agent.policy_transport import AuditedCalls, AsyncAuditTransport

        with tempfile.TemporaryDirectory() as directory:
            ledger = CallLedger(Path(directory) / "calls.jsonl", run_limit=60000, question_limit=60000, call_limit=7)
            ledger.start("q")
            audit = AuditedCalls(ledger)
            inner = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": []}))
            async with httpx.AsyncClient(transport=AsyncAuditTransport(inner, audit, "generation", "https://model.test/v1", "m")) as client:
                with self.assertRaises(ValueError):
                    await client.post("https://model.test/v1/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100, "stream": False})
                with self.assertRaises(ValueError):
                    await client.post("https://model.test/v1/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100})
            self.assertEqual(ledger.summary()["provider_calls"], 1)
            self.assertEqual(ledger.summary()["pending_calls"], ["q:1"])

    async def test_generation_and_embedding_have_independent_ceiling(self):
        from experiments.search_agent.policy_transport import AuditedCalls, AsyncAuditTransport

        with tempfile.TemporaryDirectory() as directory:
            ledger = CallLedger(Path(directory) / "calls.jsonl", run_limit=60000, question_limit=60000, call_limit=7)
            ledger.start("q")
            audit = AuditedCalls(ledger)
            inner = httpx.MockTransport(lambda request: httpx.Response(200, json={"usage": {"total_tokens": 12}}))
            async with httpx.AsyncClient(transport=AsyncAuditTransport(inner, audit, "generation", "https://model.test/v1", "m")) as client:
                for _ in range(4):
                    await client.post("https://model.test/v1/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100})
                self.assertEqual(audit.remaining_generation(), 0)
                with self.assertRaises(ValueError):
                    await client.post("https://model.test/v1/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100})
            self.assertEqual(ledger.summary()["provider_total_tokens"], 48)

    async def test_cross_endpoint_is_rejected_before_send(self):
        from experiments.search_agent.policy_transport import AuditedCalls, AsyncAuditTransport

        with tempfile.TemporaryDirectory() as directory:
            ledger = CallLedger(Path(directory) / "calls.jsonl", run_limit=60000, question_limit=60000, call_limit=7)
            ledger.start("q")
            inner = httpx.MockTransport(lambda request: self.fail("unexpected send"))
            async with httpx.AsyncClient(transport=AsyncAuditTransport(inner, AuditedCalls(ledger), "generation", "https://model.test/v1", "m")) as client:
                with self.assertRaises(ValueError):
                    await client.post("https://other.test/v1/chat/completions", json={"model": "m"})
            self.assertEqual(ledger.summary()["provider_calls"], 0)


if __name__ == "__main__":
    unittest.main()
