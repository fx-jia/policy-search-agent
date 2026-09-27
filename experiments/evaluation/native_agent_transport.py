"""Account for physical provider attempts without consuming response streams."""

import gzip
import json
import zlib
import asyncio
import re

import aiohttp
import httpx


def _exception_chain(exc):
    """Keep connection diagnostics without arbitrary exception text."""
    chain, seen = [], set()
    while exc is not None and id(exc) not in seen and len(chain) < 8:
        seen.add(id(exc))
        item = {"exception_type": type(exc).__name__}
        error_number = getattr(exc, "errno", None)
        if isinstance(error_number, int):
            item["errno"] = error_number
        chain.append(item)
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return chain


class UsageCapture:
    def __init__(self, content_type, encoding):
        self.content_type = content_type
        self.encoding = encoding
        self.body = bytearray()

    def feed(self, data):
        if len(self.body) + len(data) > 4_000_000:
            raise ValueError("provider_response_size_limit")
        self.body.extend(data)

    def _events(self):
        raw = bytes(self.body)
        if self.encoding == "gzip":
            raw = gzip.decompress(raw)
        elif self.encoding == "deflate":
            raw = zlib.decompress(raw)
        elif self.encoding not in ("", "identity"):
            raise ValueError("unknown_response_encoding")
        text = raw.decode("utf-8")
        events = []
        if "event-stream" in self.content_type:
            for line in text.splitlines():
                if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                    events.append(json.loads(line[5:]))
        else:
            events = [json.loads(text)]
        return events

    def error_code(self):
        try:
            for event in self._events():
                error = event.get("error") if isinstance(event, dict) else None
                code = error.get("code") if isinstance(error, dict) else None
                if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", code):
                    return code
        except (ValueError, UnicodeError, OSError, EOFError, zlib.error):
            pass
        return None

    def finish(self):
        usages = [event["usage"] for event in self._events() if isinstance(event, dict) and isinstance(event.get("usage"), dict)]
        if not usages:
            raise ValueError("provider_usage_unknown")
        return usages[-1]

    def terminal_seen(self):
        raw = bytes(self.body)
        try:
            if self.encoding == "gzip":
                raw = gzip.decompress(raw)
            elif self.encoding == "deflate":
                raw = zlib.decompress(raw)
        except (EOFError, OSError, zlib.error):
            return False
        return any(line.strip() == b"data: [DONE]" for line in raw.splitlines())


class _Tee(httpx.AsyncByteStream):
    def __init__(self, original, capture, completed, failed):
        self.original = original
        self.capture = capture
        self.completed = completed
        self.failed = failed
        self.done = False

    async def __aiter__(self):
        try:
            async for part in self.original:
                self.capture.feed(part)
                if not self.done and self.capture.terminal_seen():
                    self.completed(self.capture)
                    self.done = True
                yield part
            if not self.done:
                self.completed(self.capture)
                self.done = True
        except BaseException:
            if not self.done:
                self.failed()
            raise

    async def aclose(self):
        try:
            await self.original.aclose()
        finally:
            if not self.done:
                self.failed()


class ProviderMonitor:
    def __init__(self, ledger, *, model, hosts, before_send=None, event=None):
        self.ledger = ledger
        self.model = model
        self.hosts = set(hosts)
        self.before_send = before_send
        self.event = event or (lambda row: None)

    async def _wait_for_reservations(self, reserve):
        deadline = asyncio.get_running_loop().time() + 120
        while True:
            with self.ledger.lock:
                self.ledger.assert_open()
                rows = list(self.ledger.calls.values())
                current = [r for r in rows if r["question_id"] == self.ledger.question_id]
                run_limit, question_limit, call_limit = self.ledger.limits

                def spent(items):
                    return sum(r.get("usage", {}).get("total_tokens", 0) for r in items)

                def committed(items):
                    return sum(r["usage"]["total_tokens"] if "usage" in r else r["reserve_tokens"] for r in items)

                can_fit_after_pending = spent(rows) + reserve <= run_limit and spent(current) + reserve <= question_limit
                waits_on_pending = committed(rows) + reserve > run_limit or committed(current) + reserve > question_limit
                if len(current) >= call_limit or not can_fit_after_pending or not waits_on_pending:
                    return
            if asyncio.get_running_loop().time() >= deadline:
                self.ledger.abort("budget_reservation_wait_timeout")
                raise ValueError("budget_reservation_wait_timeout")
            await asyncio.sleep(0.05)

    def wrap_send(self, original):
        async def send(client, request):
            host = request.url.host
            self.ledger.assert_open()
            if host not in self.hosts or request.method != "POST" or not request.url.path.endswith("/chat/completions"):
                self.ledger.abort("unexpected_provider_endpoint")
                raise ValueError("unexpected_provider_endpoint")
            body = json.loads(request.content)
            if body.get("model") != self.model:
                self.ledger.abort("unexpected_provider_model")
                raise ValueError("unexpected_provider_model")
            if not isinstance(body.get("messages"), list) or not isinstance(body.get("max_tokens"), int) or not 0 < body["max_tokens"] <= 1200:
                self.ledger.abort("unbounded_provider_request")
                raise ValueError("unbounded_provider_request")
            if self.before_send:
                self.before_send(body)
            # Byte-based upper reservation includes messages, tools and generous framing.
            reserve = len(json.dumps(body, ensure_ascii=False).encode()) + 4096 + body["max_tokens"]
            for attempt in range(4):
                await self._wait_for_reservations(reserve)
                call_id = f"{self.ledger.question_id}:{len(self.ledger.calls) + 1}"
                try:
                    self.ledger.reserve(call_id, reserve, body)
                except ValueError:
                    self.event({"event": "provider_budget_blocked", "request": body, "reserve_tokens": reserve})
                    raise
                self.event({"event": "provider_reserved", "call_id": call_id, "request": body, "reserve_tokens": reserve})
                try:
                    response = await original(client, request)
                    break
                except BaseException as exc:
                    failure = {"event": "provider_send_failure", "call_id": call_id,
                               "exception_type": type(exc).__name__, "exception_chain": _exception_chain(exc)}
                    if isinstance(exc, httpx.ConnectError) and isinstance(exc.__cause__, aiohttp.ClientConnectorError):
                        # LiteLLM also maps post-send ClientOSError to ConnectError; require connector evidence.
                        self.ledger.finish_not_sent(call_id)
                        result = {key: self.ledger.calls[call_id][key] for key in ("transport_status", "transport_error", "usage")}
                        self.event({**failure, **result})
                        self.event({"event": "provider_finished", "call_id": call_id, **result})
                        if attempt < 3:
                            await asyncio.sleep(0.25 * 2 ** attempt)
                            continue
                        self.ledger.abort("provider_connect_failed")
                    else:
                        self.event(failure)
                        self.ledger.abort("provider_usage_unknown")
                    raise

            def failed(capture, reason):
                self.event({"event": "provider_usage_failure", "call_id": call_id, "http_status": response.status_code,
                            "content_type": capture.content_type, "response_bytes": len(capture.body),
                            "error_code": capture.error_code(), "reason": reason})
                self.ledger.abort("provider_usage_unknown")

            def completed(capture):
                try:
                    usage = capture.finish()
                    self.ledger.finish(call_id, usage)
                    self.event({"event": "provider_finished", "call_id": call_id, "usage": usage})
                except (ValueError, KeyError, UnicodeError, OSError, EOFError, zlib.error) as exc:
                    failed(capture, type(exc).__name__)

            if response.is_stream_consumed:
                capture = UsageCapture(response.headers.get("content-type", ""), "")
                capture.feed(response.content)
                completed(capture)
            else:
                capture = UsageCapture(response.headers.get("content-type", ""), response.headers.get("content-encoding", ""))
                response.stream = _Tee(response.stream, capture, completed, lambda: failed(capture, "stream_incomplete"))
            return response
        return send
