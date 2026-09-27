import contextlib
from types import SimpleNamespace
from threading import Lock
import unittest
import ssl

from experiments.search_agent.embedding import single_request


class EmbeddingTests(unittest.TestCase):
    def test_reconnects_only_when_tls_handshake_proves_no_http_request_sent(self):
        incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
        tls = ssl.create_default_context().wrap_bio(incoming, outgoing, server_hostname="provider.test")
        incoming.write_eof()
        try:
            tls.do_handshake()
        except ssl.SSLEOFError as exc:
            handshake_error = exc
        sent = []
        def send(*args, **kw):
            sent.append(1)
            if len(sent) == 1:
                raise handshake_error
        session = SimpleNamespace(send=send)
        client = SimpleNamespace(_session=session, _session_lock=Lock(), _dashscope_http_api_url=None, model_name="qwen", key="private")
        def sdk(**kw):
            kw["session"].send(SimpleNamespace(method="POST", url="https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"))
            return SimpleNamespace(status_code=200)
        stats = {}
        single_request(client, "问题", "query", sdk, lambda _: contextlib.nullcontext(), stats, connect_retries=2)
        self.assertEqual(len(sent), 2)
        self.assertEqual(stats["query_physical_calls"], 2)
        self.assertEqual(stats["query_not_sent_calls"], 1)
        self.assertNotIn("unknown_embedding_usage", stats)

    def test_read_phase_tls_eof_is_not_assumed_unsent(self):
        def send(*args, **kw):
            raise ssl.SSLEOFError("EOF while reading a response")
        client = SimpleNamespace(_session=SimpleNamespace(send=send), _session_lock=Lock(), _dashscope_http_api_url=None, model_name="qwen", key="private")
        def sdk(**kw):
            kw["session"].send(SimpleNamespace(method="POST", url="https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"))
        stats = {}
        with self.assertRaises(ssl.SSLEOFError):
            single_request(client, "问题", "query", sdk, lambda _: contextlib.nullcontext(), stats, connect_retries=2)
        self.assertEqual(stats["query_physical_calls"], 1)
        self.assertTrue(stats["unknown_embedding_usage"])

    def test_sdk_retry_is_blocked_before_second_physical_send(self):
        sent = []
        session = SimpleNamespace(send=lambda request, **kw: sent.append(request))
        client = SimpleNamespace(_session=session, _session_lock=Lock(), _dashscope_http_api_url="https://dashscope.aliyuncs.com/api/v1", model_name="qwen", key="private")
        stats = {}
        def sdk(**kwargs):
            kwargs["session"].send(SimpleNamespace(method="POST", url="https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"))
            kwargs["session"].send(SimpleNamespace(method="POST", url="https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"))
        with self.assertRaises(ValueError):
            single_request(client, "问题", "query", sdk, lambda _: contextlib.nullcontext(), stats)
        self.assertEqual(len(sent), 1)
        self.assertEqual(stats["query_physical_calls"], 1)

    def test_http_error_is_not_retried(self):
        sent = []
        session = SimpleNamespace(send=lambda request, **kw: sent.append(request))
        client = SimpleNamespace(_session=session, _session_lock=Lock(), _dashscope_http_api_url=None, model_name="qwen", key="private")
        def sdk(**kw):
            kw["session"].send(SimpleNamespace(method="POST", url="https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"))
            return SimpleNamespace(status_code=429)
        with self.assertRaises(ValueError):
            single_request(client, "问题", "query", sdk, lambda _: contextlib.nullcontext(), {})
        self.assertEqual(len(sent), 1)


if __name__ == "__main__":
    unittest.main()
