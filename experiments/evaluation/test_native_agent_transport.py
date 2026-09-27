"""Exercise physical HTTP accounting with a local transport, never cloud calls."""

import asyncio
import errno
import gzip
import importlib.util
import json
from pathlib import Path

import aiohttp
import httpx
from openai import AsyncOpenAI
import pytest

from experiments.evaluation.native_agent_guard import CallLedger


@pytest.fixture
def api():
    path = Path(__file__).with_name("native_agent_transport.py")
    assert path.exists(), "native_agent_transport implementation missing"
    spec = importlib.util.spec_from_file_location("native_agent_transport", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sse_split_bytes_usage_only_chunk_and_duplicate_cumulative(api):
    stream = api.UsageCapture("text/event-stream", "")
    payload = ('data: {"choices":[{"delta":{"content":"医保"}}]}\n\n'
               'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
               'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
               'data: [DONE]\n\n').encode()
    for n in range(0, len(payload), 2):
        stream.feed(payload[n:n + 2])
    assert stream.finish()["total_tokens"] == 15


def test_physical_request_records_usage_and_preserves_response(api, tmp_path):
    async def scenario():
        ledger = CallLedger(tmp_path / "calls.jsonl", run_limit=20000, question_limit=20000, call_limit=3)
        ledger.start("q")
        monitor = api.ProviderMonitor(ledger, model="test-model", hosts={"provider.test"})
        original = httpx.AsyncClient._send_single_request
        httpx.AsyncClient._send_single_request = monitor.wrap_send(original)
        try:
            payload = b'{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}'
            transport = httpx.MockTransport(lambda request: httpx.Response(200, content=payload, headers={"content-type": "application/json"}))
            async with httpx.AsyncClient(transport=transport) as client:
                response = await client.post("https://provider.test/v1/chat/completions", json={"model": "test-model", "messages": [{"role": "user", "content": "q"}], "max_tokens": 100})
                assert response.content == payload
            assert ledger.summary()["provider_calls"] == 1
            assert ledger.summary()["provider_total_tokens"] == 15
        finally:
            httpx.AsyncClient._send_single_request = original
    asyncio.run(scenario())


def test_unexpected_model_is_blocked_before_transport(api, tmp_path):
    async def scenario():
        ledger = CallLedger(tmp_path / "calls.jsonl", run_limit=20000, question_limit=20000, call_limit=3)
        ledger.start("q")
        monitor = api.ProviderMonitor(ledger, model="planned", hosts={"provider.test"})
        async def forbidden_send(client, request):
            pytest.fail("unexpected model reached physical transport")
        with pytest.raises(ValueError, match="model"):
            await monitor.wrap_send(forbidden_send)(None, httpx.Request("POST", "https://provider.test/v1/chat/completions", json={"model": "unplanned", "messages": []}))
        assert ledger.summary()["provider_calls"] == 0
    asyncio.run(scenario())


def test_missing_usage_keeps_call_pending(api, tmp_path):
    async def scenario():
        ledger = CallLedger(tmp_path / "calls.jsonl", run_limit=20000, question_limit=20000, call_limit=3)
        ledger.start("q")
        monitor = api.ProviderMonitor(ledger, model="m", hosts={"provider.test"})
        async def send(client, request):
            return httpx.Response(200, content=b'data: [DONE]\n\n', headers={"content-type": "text/event-stream"}, request=request)
        response = await monitor.wrap_send(send)(None, httpx.Request("POST", "https://provider.test/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100}))
        await response.aread()
        assert ledger.summary()["pending_calls"]
        with pytest.raises(ValueError, match="usage"):
            ledger.assert_open()
    asyncio.run(scenario())


def test_provider_error_records_status_and_code_without_body_or_releasing_budget(api, tmp_path):
    async def scenario():
        ledger = CallLedger(tmp_path / "calls.jsonl", run_limit=20000, question_limit=20000, call_limit=3)
        ledger.start("q")
        events = []
        monitor = api.ProviderMonitor(ledger, model="m", hosts={"provider.test"}, event=events.append)
        async def send(client, request):
            return httpx.Response(429, json={"error": {"code": "QuotaExhausted", "message": "private diagnostic detail"}}, request=request)
        await monitor.wrap_send(send)(None, httpx.Request("POST", "https://provider.test/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100}))
        failed = next(e for e in events if e["event"] == "provider_usage_failure")
        assert failed["http_status"] == 429
        assert failed["error_code"] == "QuotaExhausted"
        assert "private diagnostic detail" not in str(events)
        assert ledger.summary()["pending_calls"] == ["q:1"]
        assert ledger.summary()["fatal_reason"] == "provider_usage_unknown"
    asyncio.run(scenario())


@pytest.mark.parametrize("compressed", [False, True])
def test_real_sdk_done_early_close_accounts_once(api, tmp_path, compressed):
    async def scenario():
        ledger = CallLedger(tmp_path / "calls.jsonl", run_limit=20000, question_limit=20000, call_limit=3)
        ledger.start("q")
        monitor = api.ProviderMonitor(ledger, model="m", hosts={"provider.test"})
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                if compressed:
                    yield gzip.compress(b'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\ndata: [DONE]\n\n')
                    return
                yield b'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
                yield b'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
                yield b'data: [DONE]\n\n'
        original = httpx.AsyncClient._send_single_request
        httpx.AsyncClient._send_single_request = monitor.wrap_send(original)
        try:
            transport = httpx.MockTransport(lambda req: httpx.Response(200, stream=Stream(), headers={"content-type": "text/event-stream", **({"content-encoding": "gzip"} if compressed else {})}))
            async with AsyncOpenAI(api_key="local-test-only", base_url="https://provider.test/v1", http_client=httpx.AsyncClient(transport=transport), max_retries=0) as sdk:
                stream = await sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], stream=True, max_tokens=100)
                async for chunk in stream:
                    pass
            assert ledger.summary()["provider_total_tokens"] == 15
            assert ledger.summary()["pending_calls"] == []
            assert ledger.summary()["fatal_reason"] is None
        finally:
            httpx.AsyncClient._send_single_request = original
    asyncio.run(scenario())


def test_local_completion_cannot_bypass_sticky_abort(api, tmp_path):
    async def scenario():
        ledger = CallLedger(tmp_path / "calls.jsonl", run_limit=100, question_limit=100, call_limit=3)
        ledger.start("q")
        ledger.abort("unreviewed_chunk")
        monitor = api.ProviderMonitor(ledger, model="m", hosts={"provider.test"})
        async def forbidden(client, request):
            pytest.fail("unreviewed request escaped via localhost")
        with pytest.raises(ValueError):
            await monitor.wrap_send(forbidden)(None, httpx.Request("POST", "http://localhost:9999/v1/chat/completions", json={"model": "m"}))
    asyncio.run(scenario())


def test_concurrent_reservations_wait_for_usage_instead_of_false_budget_stop(api, tmp_path):
    async def scenario():
        ledger = CallLedger(tmp_path / "calls.jsonl", run_limit=7000, question_limit=7000, call_limit=3)
        ledger.start("q")
        monitor = api.ProviderMonitor(ledger, model="m", hosts={"provider.test"})
        async def send(client, request):
            await asyncio.sleep(0.01)
            return httpx.Response(200, content=b'{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}', headers={"content-type": "application/json"}, request=request)
        wrapped = monitor.wrap_send(send)
        def request():
            return httpx.Request("POST", "https://provider.test/v1/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100})
        await asyncio.gather(wrapped(None, request()), wrapped(None, request()))
        assert ledger.summary()["provider_total_tokens"] == 30
        assert ledger.summary()["fatal_reason"] is None
    asyncio.run(scenario())


def test_connect_failure_retries_same_request_once_and_records_unsent_usage(api, tmp_path, monkeypatch):
    async def scenario():
        path = tmp_path / "calls.jsonl"
        limits = {"run_limit": 5000, "question_limit": 5000, "call_limit": 3}
        ledger = CallLedger(path, **limits)
        ledger.start("q")
        events, requests = [], []
        monitor = api.ProviderMonitor(ledger, model="m", hosts={"provider.test"}, event=events.append)
        original = httpx.AsyncClient._send_single_request
        monkeypatch.setattr(httpx.AsyncClient, "_send_single_request", monitor.wrap_send(original))

        def handler(request):
            requests.append((request.method, str(request.url), list(request.headers.raw), request.content))
            if len(requests) == 1:
                socket_error = OSError(errno.ECONNREFUSED, "private cause detail")
                connector_error = aiohttp.ClientConnectorError(None, socket_error)
                connector_error.__cause__ = socket_error
                raise httpx.ConnectError("private diagnostic detail", request=request) from connector_error
            return httpx.Response(200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await client.post("https://provider.test/v1/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100})
        assert response.status_code == 200
        assert len(requests) == 2 and requests[0] == requests[1]
        assert ledger.calls["q:1"]["transport_status"] == "not_sent"
        assert ledger.calls["q:1"]["transport_error"] == "ConnectError"
        assert ledger.calls["q:1"]["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        assert ledger.calls["q:1"]["request"] == ledger.calls["q:2"]["request"]
        assert "transport_status" not in ledger.calls["q:2"]
        assert ledger.summary()["provider_calls"] == 2
        assert ledger.summary()["provider_not_sent_calls"] == 1
        assert ledger.summary()["provider_total_tokens"] == 15
        assert ledger.summary()["pending_calls"] == []
        assert ledger.summary()["fatal_reason"] is None
        assert CallLedger(path, **limits).summary() == ledger.summary()
        failure = next(row for row in events if row["event"] == "provider_send_failure")
        assert failure["exception_chain"] == [
            {"exception_type": "ConnectError"},
            {"exception_type": "ClientConnectorError", "errno": errno.ECONNREFUSED},
            {"exception_type": "ConnectionRefusedError", "errno": errno.ECONNREFUSED},
        ]
        assert failure["transport_status"] == "not_sent"
        assert failure["transport_error"] == "ConnectError"
        finished = [row for row in events if row["event"] == "provider_finished"]
        assert len(finished) == 2
        assert finished[0]["transport_status"] == "not_sent"
        assert finished[0]["transport_error"] == "ConnectError"
        assert finished[0]["usage"]["total_tokens"] == 0
        assert "private diagnostic detail" not in json.dumps(events) + path.read_text()
        assert "private cause detail" not in json.dumps(events) + path.read_text()

    asyncio.run(scenario())


def test_recovers_after_two_proven_unsent_failures_without_restarting_graph(api, tmp_path):
    async def scenario():
        ledger = CallLedger(tmp_path / "calls.jsonl", run_limit=5000, question_limit=5000, call_limit=4)
        ledger.start("q")
        monitor = api.ProviderMonitor(ledger, model="m", hosts={"provider.test"})
        requests = []

        async def send(client, request):
            requests.append(request)
            if len(requests) <= 2:
                raise httpx.ConnectError("not sent", request=request) from aiohttp.ClientConnectorError(None, OSError(errno.ECONNREFUSED, "not sent"))
            return httpx.Response(200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

        request = httpx.Request("POST", "https://provider.test/v1/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100})
        response = await monitor.wrap_send(send)(None, request)
        assert response.status_code == 200
        assert len(requests) == 3 and all(r is request for r in requests)
        assert ledger.summary()["provider_not_sent_calls"] == 2
        assert ledger.summary()["provider_total_tokens"] == 15
        assert ledger.summary()["pending_calls"] == []
        assert ledger.summary()["fatal_reason"] is None

    asyncio.run(scenario())


@pytest.mark.parametrize("call_limit, expected_attempts, fatal", [(5, 4, "provider_connect_failed"), (3, 3, "question_call_limit"), (1, 1, "question_call_limit")])
def test_connect_recovery_is_bounded_and_respects_attempt_limit(api, tmp_path, call_limit, expected_attempts, fatal):
    async def scenario():
        ledger = CallLedger(tmp_path / "calls.jsonl", run_limit=5000, question_limit=5000, call_limit=call_limit)
        ledger.start("q")
        monitor = api.ProviderMonitor(ledger, model="m", hosts={"provider.test"})
        attempts = 0

        async def send(client, request):
            nonlocal attempts
            attempts += 1
            raise httpx.ConnectError("unavailable", request=request) from aiohttp.ClientConnectorError(None, OSError(errno.ECONNREFUSED, "unavailable"))

        error = httpx.ConnectError if fatal == "provider_connect_failed" else ValueError
        with pytest.raises(error):
            await monitor.wrap_send(send)(None, httpx.Request("POST", "https://provider.test/v1/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100}))
        assert attempts == expected_attempts
        assert ledger.summary()["provider_calls"] == expected_attempts
        assert ledger.summary()["provider_not_sent_calls"] == expected_attempts
        assert ledger.summary()["provider_total_tokens"] == 0
        assert ledger.summary()["pending_calls"] == []
        assert ledger.summary()["fatal_reason"] == fatal
        with pytest.raises(ValueError, match=fatal):
            ledger.assert_open()

    asyncio.run(scenario())


@pytest.mark.parametrize("cause", [None, aiohttp.ClientOSError(errno.EPIPE, "writing request body failed")], ids=["no_cause", "request_body_write_error"])
def test_connect_error_without_connector_cause_keeps_usage_unknown(api, tmp_path, cause):
    async def scenario():
        ledger = CallLedger(tmp_path / "calls.jsonl", run_limit=20000, question_limit=20000, call_limit=3)
        ledger.start("q")
        monitor = api.ProviderMonitor(ledger, model="m", hosts={"provider.test"})
        attempts = 0

        async def send(client, request):
            nonlocal attempts
            attempts += 1
            raise httpx.ConnectError("mapped transport error", request=request) from cause

        with pytest.raises(httpx.ConnectError):
            await monitor.wrap_send(send)(None, httpx.Request("POST", "https://provider.test/v1/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100}))
        assert attempts == 1
        assert ledger.summary()["pending_calls"] == ["q:1"]
        assert ledger.summary()["provider_not_sent_calls"] == 0
        assert "transport_status" not in ledger.calls["q:1"]
        assert "usage" not in ledger.calls["q:1"]
        assert ledger.summary()["fatal_reason"] == "provider_usage_unknown"

    asyncio.run(scenario())


@pytest.mark.parametrize("error", [httpx.ReadError, httpx.WriteError, httpx.ConnectTimeout])
def test_other_send_errors_are_not_retried_or_recorded_as_zero(api, tmp_path, error):
    async def scenario():
        ledger = CallLedger(tmp_path / "calls.jsonl", run_limit=20000, question_limit=20000, call_limit=3)
        ledger.start("q")
        monitor = api.ProviderMonitor(ledger, model="m", hosts={"provider.test"})
        attempts = 0

        async def send(client, request):
            nonlocal attempts
            attempts += 1
            raise error("failure", request=request)

        with pytest.raises(error):
            await monitor.wrap_send(send)(None, httpx.Request("POST", "https://provider.test/v1/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100}))
        assert attempts == 1
        assert ledger.summary()["pending_calls"] == ["q:1"]
        assert "transport_status" not in ledger.calls["q:1"]
        assert "usage" not in ledger.calls["q:1"]
        assert ledger.summary()["fatal_reason"] == "provider_usage_unknown"

    asyncio.run(scenario())


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadError])
def test_stream_error_after_response_does_not_retry_or_release_reservation(api, tmp_path, error):
    async def scenario():
        ledger = CallLedger(tmp_path / "calls.jsonl", run_limit=20000, question_limit=20000, call_limit=3)
        ledger.start("q")
        monitor = api.ProviderMonitor(ledger, model="m", hosts={"provider.test"})

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                raise error("stream failed")

        async def send(client, request):
            return httpx.Response(200, stream=Stream(), headers={"content-type": "text/event-stream"}, request=request)

        response = await monitor.wrap_send(send)(None, httpx.Request("POST", "https://provider.test/v1/chat/completions", json={"model": "m", "messages": [], "max_tokens": 100}))
        with pytest.raises(error):
            await response.aread()
        assert ledger.summary()["provider_calls"] == 1
        assert ledger.summary()["pending_calls"] == ["q:1"]
        assert "transport_status" not in ledger.calls["q:1"]
        assert "usage" not in ledger.calls["q:1"]
        assert ledger.summary()["fatal_reason"] == "provider_usage_unknown"

    asyncio.run(scenario())
