"""Request-scoped OpenAI transports with actual physical-call accounting."""

import json
from threading import RLock

import httpx

from experiments.evaluation.native_agent_transport import UsageCapture


class AuditedCalls:
    def __init__(self, ledger):
        self.ledger = ledger
        self.lock = RLock()
        self.kinds = {}
        self.consumed = set()

    def remaining_generation(self):
        with self.lock:
            self.ledger.assert_open()
            return 4 - sum(kind == "generation" for kind in self.kinds.values())

    def reserve(self, kind, body):
        with self.lock:
            self.ledger.assert_open()
            ceiling = 4 if kind == "generation" else 3
            if sum(value == kind for value in self.kinds.values()) >= ceiling:
                raise ValueError(kind + "_call_limit")
            if kind == "generation" and (body.get("stream") or type(body.get("max_tokens")) is not int or not 0 < body["max_tokens"] <= 1200):
                raise ValueError("unbounded_generation_request")
            call_id = f"{self.ledger.question_id}:{len(self.ledger.calls) + 1}"
            reserve = len(json.dumps(body, ensure_ascii=False).encode()) + 4096 + (body.get("max_tokens", 0) if kind == "generation" else 0)
            self.ledger.reserve(call_id, reserve, dict(body, call_kind=kind))
            self.kinds[call_id] = kind
            return call_id

    def complete(self, call_id, response, blob):
        try:
            capture = UsageCapture(response.headers.get("content-type", ""), "")
            capture.feed(blob)
            self.ledger.finish(call_id, capture.finish())
            response.raise_for_status()
        except BaseException:
            self.ledger.abort("provider_usage_unknown" if "usage" not in self.ledger.calls[call_id] else "provider_http_error")
            raise

    def audit(self, stage, result):
        with self.lock:
            self.ledger.assert_open()
            kind = "embedding" if stage == "search" else "generation"
            rows = [(cid, row) for cid, row in self.ledger.calls.items() if cid not in self.consumed and self.kinds.get(cid) == kind]
            if not rows or any("usage" not in row for _, row in rows):
                self.ledger.abort("provider_usage_unknown")
                raise ValueError("provider_usage_unknown")
            self.consumed.update(cid for cid, _ in rows)
            return {"total_tokens": sum(row["usage"]["total_tokens"] for _, row in rows), "physical_calls": len(rows)}


def _validate(request, kind, base_url, model):
    base = httpx.URL(base_url)
    endpoint = base.path.rstrip("/") + ("/chat/completions" if kind == "generation" else "/embeddings")
    if request.url.scheme != base.scheme or request.url.host != base.host or request.url.port != base.port or request.url.path != endpoint or request.method != "POST":
        raise ValueError("unexpected_provider_endpoint")
    body = json.loads(request.content)
    if body.get("model") != model:
        raise ValueError("unexpected_provider_model")
    return body


class AsyncAuditTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner, audit, kind, base_url, model):
        self.inner, self.audit, self.kind, self.base_url, self.model = inner, audit, kind, base_url, model

    async def handle_async_request(self, request):
        call_id = self.audit.reserve(self.kind, _validate(request, self.kind, self.base_url, self.model))
        try:
            response = await self.inner.handle_async_request(request)
            capture, parts = UsageCapture("", ""), []
            async for part in response.aiter_bytes():
                capture.feed(part)
                parts.append(part)
            blob = b"".join(parts)
            # The raw transport response has no associated Request yet.
            response.request = request
            self.audit.complete(call_id, response, blob)
            headers = {key: value for key, value in response.headers.items() if key.lower() not in {"content-encoding", "content-length", "transfer-encoding"}}
            return httpx.Response(response.status_code, headers=headers, content=blob, request=request)
        except BaseException:
            if "usage" not in self.audit.ledger.calls[call_id]:
                self.audit.ledger.abort("provider_usage_unknown")
            raise
        finally:
            if "response" in locals():
                await response.aclose()

    async def aclose(self):
        await self.inner.aclose()


class SyncAuditTransport(httpx.BaseTransport):
    def __init__(self, inner, audit, kind, base_url, model):
        self.inner, self.audit, self.kind, self.base_url, self.model = inner, audit, kind, base_url, model

    def handle_request(self, request):
        call_id = self.audit.reserve(self.kind, _validate(request, self.kind, self.base_url, self.model))
        try:
            response = self.inner.handle_request(request)
            capture, parts = UsageCapture("", ""), []
            for part in response.iter_bytes():
                capture.feed(part)
                parts.append(part)
            blob = b"".join(parts)
            response.request = request
            self.audit.complete(call_id, response, blob)
            headers = {key: value for key, value in response.headers.items() if key.lower() not in {"content-encoding", "content-length", "transfer-encoding"}}
            return httpx.Response(response.status_code, headers=headers, content=blob, request=request)
        except BaseException:
            if "usage" not in self.audit.ledger.calls[call_id]:
                self.audit.ledger.abort("provider_usage_unknown")
            raise
        finally:
            if "response" in locals():
                response.close()

    def close(self):
        self.inner.close()
